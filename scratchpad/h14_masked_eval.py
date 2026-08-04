"""
H14 — masked-region targets: the four pre-registered metrics + the standard
aligned/permuted/blank ablation, on the ``masked-randstr`` lane.

The lane (``data/preprocessing/masked_regions.py``) occludes random runs of words
on a rendered page and supervises a target that carries ONE ``<mask>`` sentinel
per occluded run.  Because the mask is drawn i.i.d. per row, no language prior can
supply the target; and to know a span is hidden the model must LOCALIZE it.

What this script measures (doc: docs/hypothesis/todo/H14-masked-region-targets.md)

1. **span-boundary accuracy** — teacher-forced accuracy at the sentinel tokens and
   the ``--after-k`` tokens immediately after them.  The stored
   ``sentinel_char_starts`` are **character** offsets into the target string, so
   they are mapped to token indices with ``return_offsets_mapping`` and the map is
   ASSERTED against the supervised label ids (a silent off-by-one here would
   fabricate the headline number, so every row is checked and a mismatch voids the
   row rather than being averaged in).
   MEASURED CAVEAT: Qwen3 splits ``<mask>`` into THREE tokens (``" <"``,
   ``"mask"``, ``">"``), so with ``--after-k 1`` a sentinel contributes 4 scored
   tokens of which 2 are near-deterministic continuations.  A model that never
   decides *where* a mask goes but is fluent still scores ~50% on the
   pre-registered combined metric — only 10 points under its 60% bar.  The script
   therefore also reports **first-sentinel-token** accuracy (the actual
   localization decision) and the measured degenerate floor, and downgrades a
   combined PASS that does not clear that floor.
2. **hallucinated-content rate** — fraction of masked words (ground truth
   ``words[mask_word_indices]``) that appear in the greedy decode anyway, against
   the analytic guess floor from ``floor.json`` AND against a **donor null**
   (the same test run with another row's masked words), which is the empirical
   "could this rate arise by chance" baseline.
3. **mask-permutation control** — the same words re-rendered under the stored
   SECOND mask (``mask_b_*``) and decoded again: what fraction of rows change
   output?  Both sides are built by the same ``render_masked_pages`` call, and the
   rebuild of mask A is checked pixel-for-pixel against the stored PNG, so an
   A-vs-B difference cannot be a rendering artefact.
4. **visible-word accuracy** — the degenerate-shortcut guard the doc's "Known
   risk" section demands.  If this sits at the guess floor while span-boundary
   accuracy is high, the model learned to SEE BLACK RECTANGLES, not to read, and
   the result does **not** support H14.  The verdict says so explicitly.
5. **aligned / permuted / blank** response-only CE + teacher-forced token accuracy
   (Δperm, Δblank, reading gain), same conventions as
   ``scratchpad/hybrid_pretrained_ablation.py``.  This is the gate: if the model
   ignores the image entirely, every mask metric above is uninterpretable and the
   verdict is VOID rather than a number.

Pre-registered criterion (verbatim from the doc):
  CONFIRMS: span-boundary accuracy >= 60% AND hallucinated-content rate <= 15%
            AND mask-permutation changes the output on >= 80% of rows.
  REFUTES : hallucinated-content rate ~= the unmasked-baseline fabrication rate.

The verdict is NOT forced.  Every criterion is decided against a cluster-bootstrap
95% CI and comes back PASS / FAIL / UNDECIDABLE, and the final line can be VOID,
DEGENERATE-SHORTCUT, AMBIGUOUS or UNDECIDABLE.  Point estimates are always printed
with a per-ROW (cluster) SE, because tokens and words inside one row are not
independent.

NOTE (GPU): loading the checkpoint needs the card.  Do not run this while a
training run holds it — the project rule is that training runs SOLO.

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/h14_masked_eval.py \\
      --checkpoint data/checkpoints/h14-masked-randstr-v0/final -n 150

  # no GPU, no model, no checkpoint: exercise every piece of pure logic
  uv run python scratchpad/h14_masked_eval.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h14_masked_eval")

DEFAULT_VAL = "data/materialized/h14-masked-v0/masked-randstr/validation"
DEFAULT_FLOOR = "data/materialized/h14-masked-v0/floor.json"
DEFAULT_OUT = "data/eval/h14-masked-randstr.json"
TEXT_SOURCE = "unsloth/Qwen3-1.7B"
IMAGE_PLACEHOLDER = "<|univi_image|>"

Z95 = 1.959963985
#: two-sided 0.05 at 80% power => |effect| >= (1.96 + 0.84) * SE
Z_MDE = 2.801585

# --- pre-registered thresholds (docs/hypothesis/todo/H14-masked-region-targets.md)
SPAN_BOUNDARY_MIN = 0.60
HALLUCINATION_MAX = 0.15
PERM_CHANGE_MIN = 0.80


# ---------------------------------------------------------------------------
# stats helpers (numpy only — so --self-test runs on any box)
# ---------------------------------------------------------------------------
def _nan():
    return float("nan")


def _fin(x):
    """JSON has no NaN/Inf: emit null, so a missing number stays missing."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _fmt(x, spec=".2f", dash="--"):
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return dash
    return format(xf, spec) if math.isfinite(xf) else dash


def _pct(x, spec=".2f", dash="--"):
    return dash if x is None or not math.isfinite(float(x)) else format(float(x) * 100, spec)


def row_means(values, rows):
    """Per-row means of ``values`` (``rows`` = the row index of each value)."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    rows = np.asarray(rows)
    if len(values) == 0:
        return np.zeros(0)
    uniq, inv = np.unique(rows, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq))
    return s / c


def cluster_stats(values, rows):
    """(mean-of-row-means, cluster SE, n_rows).

    The honest SE for token/word-level rates: observations inside one row share a
    page, a mask and a model state, so the naive Bernoulli SE is optimistic.
    """
    rm = row_means(values, rows)
    n = len(rm)
    if n == 0:
        return _nan(), _nan(), 0
    if n == 1:
        return float(rm[0]), _nan(), 1
    return float(rm.mean()), float(rm.std(ddof=1) / math.sqrt(n)), n


def cluster_rate(values, rows, n_boot=1000, seed=0):
    """Token/word-level rate with a ROW-level (cluster) bootstrap CI.

    ``rate`` is the pooled (observation-weighted) estimate, which is what
    "fraction of masked words ..." means literally.  The CI comes from resampling
    ROWS, so it respects the clustering.  ``rate_rowmean`` is the unweighted
    per-row mean, reported alongside because the two differ when rows contribute
    different numbers of observations.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    rows = np.asarray(rows)
    out = {
        "n": int(len(values)), "n_rows": 0, "rate": None, "rate_rowmean": None,
        "se_cluster": None, "ci95": [None, None], "method": "cluster-bootstrap",
        "n_boot": int(n_boot),
    }
    if len(values) == 0:
        out["method"] = "empty"
        return out
    uniq, inv = np.unique(rows, return_inverse=True)
    n_rows = len(uniq)
    out["n_rows"] = int(n_rows)
    out["rate"] = float(values.mean())
    rm, se, _ = cluster_stats(values, rows)
    out["rate_rowmean"] = _fin(rm)
    out["se_cluster"] = _fin(se)
    if n_rows < 2:
        out["method"] = "single-row (no CI)"
        return out
    s = np.bincount(inv, weights=values, minlength=n_rows)
    c = np.bincount(inv, minlength=n_rows).astype(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_rows, n_rows)
        cs = c[pick].sum()
        reps[b] = s[pick].sum() / cs if cs > 0 else np.nan
    reps = reps[np.isfinite(reps)]
    if len(reps):
        out["ci95"] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    return out


def paired_mean(diffs):
    """(mean, SE, n) of a per-row paired difference; NaNs dropped."""
    import numpy as np

    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return None, None, 0
    if len(d) == 1:
        return float(d[0]), None, 1
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), int(len(d))


def decide(ci, threshold, direction):
    """PASS / FAIL / UNDECIDABLE for "estimate ``direction`` threshold".

    ``direction`` is ``">="`` (the criterion wants the value at or above the
    threshold) or ``"<="``.  The decision uses the CI, never the point estimate:
    a point estimate that clears the bar with a CI straddling it is UNDECIDABLE,
    which is the whole reason this function exists.
    """
    lo, hi = (ci or [None, None])
    if lo is None or hi is None:
        return "UNDECIDABLE"
    if direction == ">=":
        if lo >= threshold:
            return "PASS"
        if hi < threshold:
            return "FAIL"
        return "UNDECIDABLE"
    if direction == "<=":
        if hi <= threshold:
            return "PASS"
        if lo > threshold:
            return "FAIL"
        return "UNDECIDABLE"
    raise ValueError(f"direction must be '>=' or '<=', got {direction!r}")


# ---------------------------------------------------------------------------
# character -> token mapping  (the part that must not be wrong)
# ---------------------------------------------------------------------------
def token_offsets(tok, text):
    """``(ids, spans)`` for *text*, no special tokens.

    ``spans[i] == (char_start, char_end)`` of token ``i``.  Raises if the fast
    tokenizer does not return an offset mapping — falling back to a guess is
    exactly how a fabricated span-boundary number would happen.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in enc:
        raise RuntimeError(
            "tokenizer returned no offset_mapping (need a *fast* tokenizer); "
            "span-boundary accuracy cannot be located without it."
        )
    ids = list(enc["input_ids"])
    spans = [tuple(s) for s in enc["offset_mapping"]]
    if len(ids) != len(spans):
        raise RuntimeError("offset_mapping length != input_ids length")
    return ids, spans


def spans_cover_text(spans, text):
    """True when the spans tile ``text`` left to right with no gap or overlap.

    Byte-level BPE spans do exactly this (a leading space belongs to the token
    that follows it), so a False here means the mapping cannot be trusted.
    """
    cur = 0
    for a, b in spans:
        if a != cur or b < a:
            return False
        cur = b
    return cur == len(text)


def sentinel_token_indices(spans, char_start, sent_len, after_k, n_tokens):
    """``(sentinel_token_idx, after_token_idx)`` for ONE sentinel.

    A token belongs to the sentinel when its character span OVERLAPS
    ``[char_start, char_start + sent_len)``.  With byte-level BPE the first such
    token usually also carries the preceding space (" <"), which is correct: that
    token is where the model has to commit to "a region is hidden here".
    ``after_token_idx`` is the next ``after_k`` tokens, i.e. where it has to
    resume reading the page.
    """
    lo, hi = char_start, char_start + sent_len
    sent = [i for i, (a, b) in enumerate(spans) if a < hi and b > lo]
    if not sent:
        return [], []
    nxt = sent[-1] + 1
    after = [i for i in range(nxt, min(nxt + after_k, n_tokens))]
    return sent, after


def check_sentinel_columns(target, char_starts, visible_index, sentinel):
    """Cross-check the stored sentinel columns against the target STRING.

    Returns a list of human-readable problems (empty == consistent).  These are
    the stored columns the whole metric rests on, so they are verified per row
    rather than trusted:

    - every ``sentinel_char_starts`` offset must actually point at the sentinel;
    - the number of sentinels must equal the number of stored offsets;
    - ``sentinel_visible_index[k]`` (visible words before sentinel k) must equal
      the number of whitespace pieces before that offset minus the k sentinels
      already emitted.
    """
    problems = []
    n_in_text = target.count(sentinel)
    if n_in_text != len(char_starts):
        problems.append(
            f"target contains {n_in_text} sentinels but {len(char_starts)} offsets stored"
        )
    for k, s in enumerate(char_starts):
        if target[s:s + len(sentinel)] != sentinel:
            problems.append(f"offset {k} (char {s}) does not point at {sentinel!r}")
            continue
        if visible_index is not None and k < len(visible_index):
            pieces_before = len(target[:s].split())
            expect = int(visible_index[k]) + k
            if pieces_before != expect:
                problems.append(
                    f"offset {k}: {pieces_before} pieces precede it, expected "
                    f"{expect} (= sentinel_visible_index {visible_index[k]} + {k} sentinels)"
                )
    return problems


# ---------------------------------------------------------------------------
# decode-based metrics
# ---------------------------------------------------------------------------
def decode_pieces(text, sentinel):
    """``(all_pieces, non_sentinel_pieces)`` of a decoded answer.

    Whitespace split, then the sentinel token dropped.  Substring matching is
    deliberately NOT used anywhere: a 3-letter masked word occurs inside a
    7-letter visible word often enough to inflate the hallucination rate.
    """
    pieces = text.split()
    return pieces, [p for p in pieces if p != sentinel]


def hallucination_hits(decode, masked_words, sentinel):
    """Per-masked-word 0/1: did this hidden word appear in the decode at all?

    Membership is over whitespace pieces (see :func:`decode_pieces`).  A masked
    word that ALSO appears visibly on the same page is not evidence of anything
    — the model could have read the visible copy — so those are reported
    separately by the caller and excluded from the headline rate.
    """
    bag = set(decode_pieces(decode, sentinel)[1])
    return [1.0 if w in bag else 0.0 for w in masked_words]


def visible_word_hits(decode, visible_words, sentinel):
    """``(positional_hits, recall_hits)`` over the ground-truth visible words.

    ``positional`` compares the decode's non-sentinel pieces to ``visible_words``
    index by index (the doc's "exact-match accuracy over these"), and is 0 for
    every position past the end of a short decode.  ``recall`` is
    alignment-free bag membership — immune to a single dropped word shifting
    everything, and reported as the robustness check.
    """
    got = decode_pieces(decode, sentinel)[1]
    bag = set(got)
    positional = [
        1.0 if i < len(got) and got[i] == w else 0.0
        for i, w in enumerate(visible_words)
    ]
    recall = [1.0 if w in bag else 0.0 for w in visible_words]
    return positional, recall


# ---------------------------------------------------------------------------
# rendering the mask-permutation control
# ---------------------------------------------------------------------------
def spec_from_render_config(render_config):
    """Rebuild the exact :class:`LaneSpec` a row was materialized with.

    Read from the row's stored ``render_config`` rather than from the module
    defaults, so a probe run after the generator's defaults change still
    re-renders the geometry the checkpoint was TRAINED on.
    """
    from data.preprocessing.masked_regions import CANVAS, LaneSpec

    rc = json.loads(render_config) if isinstance(render_config, str) else dict(render_config)
    if int(rc.get("canvas_width", CANVAS)) != CANVAS or \
            int(rc.get("canvas_height", CANVAS)) != CANVAS:
        raise ValueError(
            f"row was rendered on a {rc.get('canvas_width')}x{rc.get('canvas_height')} "
            f"canvas but masked_regions.CANVAS is {CANVAS}; re-rendering would not "
            "reproduce the stored page."
        )
    return LaneSpec(
        group_len_min=int(rc["group_len_min"]),
        group_len_max=int(rc["group_len_max"]),
        n_groups=int(rc["n_groups"]),
        font_size=int(rc["font_size"]),
        rate_min=float(rc["mask_rate_min"]),
        rate_max=float(rc["mask_rate_max"]),
        run_min=int(rc["mask_run_min"]),
        run_max=int(rc["mask_run_max"]),
        mask_pad_x=int(rc["mask_pad_x"]),
        mask_pad_y=int(rc["mask_pad_y"]),
        fill_color=rc["mask_fill_color"],
        sentinel=rc["sentinel"],
        max_pages=int(rc.get("max_pages", 1)),
    )


def render_from_runs(words, starts, lengths, spec):
    """One masked page from stored words + run columns, in the stored image mode."""
    from data.preprocessing.masked_regions import render_masked_pages

    runs = [(int(s), int(n)) for s, n in zip(starts, lengths)]
    pages, _ = render_masked_pages(list(words), runs, spec)
    return [p.convert("L") for p in pages]


def images_identical(a, b):
    if a.size != b.size:
        return False
    return a.convert("L").tobytes() == b.convert("L").tobytes()


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def verdict(M, args):
    """Decide the pre-registered branch, or refuse to.

    ``M`` is the metrics dict built by :func:`run` (or, in the self-test, by hand).
    Nothing here forces a branch: the gates below can return VOID (the model does
    not read at all / the control render is broken), DEGENERATE-SHORTCUT (the
    doc's named risk), UNDECIDABLE (CIs straddle a threshold) or AMBIGUOUS.
    """
    lines, checks, caveats = [], {}, []

    # ---- GATE 0: does the model read the image AT ALL? -------------------
    abl = M["ablation"]
    dperm = abl["delta_perm_ce_paired"]
    gain = abl["reading_gain_pts"]
    reads = bool(
        (dperm.get("ci95") or [None])[0] is not None and dperm["ci95"][0] > 0
    ) or bool((gain.get("ci95") or [None])[0] is not None and gain["ci95"][0] > 0)
    checks["reads_image"] = reads
    lines.append(
        f"GATE  image is read at all: Dperm paired dCE "
        f"{_fmt(dperm.get('mean'), '+.4f')} +/- {_fmt(dperm.get('se'), '.4f')} nats, "
        f"reading gain {_fmt(gain.get('mean'), '+.2f')} +/- {_fmt(gain.get('se'), '.2f')} pts "
        f"=> {'READS' if reads else 'NO EVIDENCE OF READING'}"
    )

    # ---- GATE 1: is the mask-permutation control even valid? -------------
    render_ok = M["render_check"]["n_mismatch"] == 0
    checks["render_rebuild_identical"] = render_ok
    lines.append(
        f"GATE  mask-A rebuild == stored PNG on "
        f"{M['render_check']['n_checked'] - M['render_check']['n_mismatch']}/"
        f"{M['render_check']['n_checked']} rows "
        f"=> {'control is like-for-like' if render_ok else 'CONTROL INVALID'}"
    )

    # ---- the three pre-registered numbers --------------------------------
    span = M["span_boundary"]["combined"]
    hall = M["hallucination"]["excluding_visible_duplicates"]
    perm = M["mask_permutation"]["change_rate"]

    s_span = decide(span.get("ci95"), SPAN_BOUNDARY_MIN, ">=") if span.get("n") else "UNDECIDABLE"
    # A sentinel is 3 Qwen tokens; the 2nd and 3rd are near-free once the 1st is
    # committed. This is the score of a model that gets EVERY continuation token
    # and nothing else — measured from the data, not assumed.
    dfloor = M["span_boundary"].get("degenerate_floor")
    s_span_floor = (decide(span.get("ci95"), dfloor, ">=")
                    if (dfloor is not None and span.get("n")) else "UNDECIDABLE")
    checks["span_above_degenerate_floor"] = s_span_floor
    first = M["span_boundary"].get("first_sentinel_token", {})
    lines.append(
        f"C1'   span-boundary degenerate floor (sentinel continuation tokens) "
        f"{_pct(dfloor)}%; combined vs floor -> {s_span_floor}; "
        f"FIRST-sentinel-token accuracy {_pct(first.get('rate'))}% "
        f"[{_pct((first.get('ci95') or [None, None])[0])}, "
        f"{_pct((first.get('ci95') or [None, None])[1])}]"
    )
    s_hall = decide(hall.get("ci95"), HALLUCINATION_MAX, "<=") if hall.get("n") else "UNDECIDABLE"
    s_perm = decide(perm.get("ci95"), PERM_CHANGE_MIN, ">=") if perm.get("n") else "UNDECIDABLE"
    if M["decode"]["skipped"]:
        s_hall = s_perm = "UNDECIDABLE"
    if not render_ok:
        s_perm = "VOID"
    checks.update({"span_boundary": s_span, "hallucination": s_hall,
                   "mask_permutation": s_perm})

    lines.append(
        f"C1    span-boundary accuracy {_pct(span.get('rate'))}% "
        f"[{_pct((span.get('ci95') or [None, None])[0])}, "
        f"{_pct((span.get('ci95') or [None, None])[1])}] vs >= "
        f"{SPAN_BOUNDARY_MIN * 100:.0f}%  -> {s_span}"
    )
    lines.append(
        f"C2    hallucinated-content rate {_pct(hall.get('rate'), '.4f')}% "
        f"[{_pct((hall.get('ci95') or [None, None])[0], '.4f')}, "
        f"{_pct((hall.get('ci95') or [None, None])[1], '.4f')}] vs <= "
        f"{HALLUCINATION_MAX * 100:.0f}%  -> {s_hall}"
    )
    lines.append(
        f"C3    mask-permutation change rate {_pct(perm.get('rate'))}% "
        f"[{_pct((perm.get('ci95') or [None, None])[0])}, "
        f"{_pct((perm.get('ci95') or [None, None])[1])}] vs >= "
        f"{PERM_CHANGE_MIN * 100:.0f}%  -> {s_perm}"
    )

    # ---- the doc's named risk: box detection without reading -------------
    vis = M["visible_words"]["positional"]
    floor = M.get("floors", {}).get("masked_word_guess_accuracy")
    vis_ci = vis.get("ci95") or [None, None]
    vis_at_floor = (
        vis_ci[1] is not None and floor is not None
        and vis_ci[1] <= max(float(floor) * 10, 0.01)
    )
    checks["visible_word_at_floor"] = bool(vis_at_floor)
    lines.append(
        f"GUARD visible-word accuracy {_pct(vis.get('rate'))}% "
        f"[{_pct(vis_ci[0])}, {_pct(vis_ci[1])}] vs guess floor "
        f"{_fmt(floor, '.2e')}  -> "
        f"{'AT FLOOR (box detection only)' if vis_at_floor else 'above floor (reads text)'}"
    )

    # ---- assemble ---------------------------------------------------------
    statuses = [s_span, s_hall, s_perm]
    if not reads:
        final = ("VOID — the ablation shows no evidence the model reads the image at "
                 "all, so span-boundary/hallucination/permutation numbers describe a "
                 "model generating from the prompt alone and cannot bear on H14.")
    elif s_span == "PASS" and vis_at_floor and not M["decode"]["skipped"]:
        final = ("DEGENERATE-SHORTCUT — span boundaries land but visible-word accuracy "
                 "is at the guess floor. The model learned to detect black rectangles, "
                 "not to read. Per the doc's Known-risk section this does NOT support "
                 "the claim.")
    elif all(s == "PASS" for s in statuses) and s_span_floor == "PASS":
        final = ("CONFIRMS — all three pre-registered thresholds cleared, the "
                 "visible-text guard is above floor, and span-boundary accuracy "
                 "clears the sentinel-continuation degenerate floor.")
    elif all(s == "PASS" for s in statuses):
        final = ("AMBIGUOUS — the three pre-registered thresholds are cleared, but "
                 "span-boundary accuracy does not clear the degenerate floor that a "
                 "fluent non-locating model gets for free from the 2 continuation "
                 "tokens of a 3-token <mask>. Judge it on the first-sentinel-token "
                 "number instead; the pre-registered 60% bar is too close to that "
                 "floor to separate the hypotheses.")
    elif s_hall == "FAIL":
        final = ("REFUTES-shaped — the hallucinated-content rate is above 15%. NOTE the "
                 "caveat below: on the prior-proof randstr lane this indicates a MASK "
                 "LEAK (ink surviving the occlusion), not prior-driven fabrication.")
    elif "UNDECIDABLE" in statuses or "VOID" in statuses:
        pairs = list(zip(("span-boundary", "hallucination", "mask-permutation"),
                         statuses, (span, hall, perm)))
        which = []
        for name, s, r in pairs:
            if s not in ("UNDECIDABLE", "VOID"):
                continue
            se = r.get("se_cluster")
            mde = (f", MDE at 80% power = {_pct(Z_MDE * se)} pts"
                   if se not in (None, 0) and math.isfinite(float(se)) else "")
            which.append(f"{name} ({s}{mde})")
        final = ("UNDECIDABLE — " + "; ".join(which) + " could not be separated from its "
                 "threshold at this sample size (or was voided). Raise -n past the MDE "
                 "or fix the voided input; do not report a branch.")
    else:
        which = [n for n, s in zip(("span-boundary", "hallucination", "mask-permutation"),
                                   statuses) if s == "FAIL"]
        final = ("AMBIGUOUS / DOES NOT CONFIRM — " + ", ".join(which) + " below the "
                 "pre-registered bar while the others cleared it. This is not the "
                 "doc's REFUTES branch (that one is specifically about hallucination).")

    caveats = [
        "The doc's REFUTES branch ('hallucination ~= the unmasked-baseline fabrication "
        "rate') has almost NO power on the randstr lane: an occluded word is 3-7 "
        "uniform letters, so a non-reader's chance of emitting it is "
        f"{_fmt(floor, '.2e')} and a READER cannot emit it either (it is not on the "
        "page). Both hypotheses predict ~0. A high rate here means the occlusion "
        "LEAKED ink, not that the model used a language prior. The metric only "
        "becomes a prior test on the real-text `masked-text` port. The donor-null "
        "figure beside it is the empirical chance baseline.",
        "span-boundary accuracy is teacher-forced: it scores the sentinel tokens "
        "given a correct prefix. It is therefore an UPPER bound on what free decoding "
        "would place correctly, and is not comparable to the decode-based numbers "
        "beside it.",
        "Qwen3 tokenizes '<mask>' as THREE tokens (' <', 'mask', '>'). With "
        "--after-k 1 a sentinel contributes 4 scored tokens, 2 of which are "
        "near-deterministic continuations, so the measured degenerate floor printed "
        "above is ~50% and the pre-registered 60% bar sits only ~10 points over it. "
        "The first-sentinel-token number is the one that actually measures "
        "localization; treat the combined number as the doc-faithful headline, not "
        "as the discriminating one.",
        "The occlusion rectangle's width still leaks run length (measured 87% of the "
        "mutual information at the 3-7 jitter, 94% width-only MAP accuracy). Under "
        "target variant (b) the target never encodes k, so this does not inflate "
        "span-boundary accuracy — but read any *number of sentinels* statistic "
        "against that leak, not against chance.",
        "All CIs are ROW-level (cluster) bootstraps. Token- and word-level "
        "observations within a page share a mask and a model state; the naive "
        "Bernoulli SE would be roughly sqrt(obs-per-row) times too small.",
    ]
    return {"lines": lines, "checks": checks, "final": final, "caveats": caveats}


# ---------------------------------------------------------------------------
# GPU path
# ---------------------------------------------------------------------------
def build_model(checkpoint, max_soft_tokens, max_length):
    """``(model, tok, collator, device, mst)`` — heavy imports stay in here."""
    import torch
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator
    from univi.hybrid.pretrained import (
        UniViHybridPretrained, build_image_processor, resolve_max_soft_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", checkpoint, device)
    model = UniViHybridPretrained.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = True
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:
        tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
        if IMAGE_PLACEHOLDER not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            "a FAST tokenizer is required: sentinel offsets are character-indexed "
            "and must be mapped with return_offsets_mapping."
        )
    # Process images at the budget the CHECKPOINT records unless overridden — a
    # probe run at a different budget feeds a geometry the model never saw and
    # silently produces garbage that looks like a result.
    mst = resolve_max_soft_tokens(model, max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s)", mst,
                getattr(model.config, "max_soft_tokens", "<absent -> 280>"))
    col = HybridCollator(
        tokenizer=tok, image_processor=build_image_processor(mst),
        image_token_id=model.image_token_id, max_length=max_length,
        response_only=True,
    )
    return model, tok, col, device, mst


def make_scorer(model, col, device):
    """``score(row, image) -> (correct[], ce[], label_ids[])`` teacher-forced over
    the supervised positions.

    Labels are deliberately NOT passed to the model: passing them triggers the
    ``logits_to_keep`` optimization and the logits come back empty.
    """
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def score(row, image):
        r = {k: row[k] for k in row}
        r["images"] = [image]
        batch = col([r])
        labels = batch["labels"]
        fwd = {k: (v.to(device) if hasattr(v, "to") else v)
               for k, v in batch.items() if k != "labels"}
        out = model(**fwd)
        logits = out.logits[0]
        tg = labels[0].to(logits.device)[1:]
        lg = logits[:-1]
        idx = (tg != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None
        sl = lg[idx].float()
        t = tg[idx]
        corr = (sl.argmax(-1) == t).float().cpu().numpy()
        ce = F.cross_entropy(sl, t, reduction="none").cpu().numpy()
        return corr, ce, t.cpu().numpy()

    return score


def make_greedy(model, tok, col, device):
    """``greedy(row, image, max_new) -> str``.

    Uses a KV cache: the first forward carries the image and the whole prompt,
    every later step feeds ONE token with ``past_key_values`` and NO
    ``pixel_values`` (re-passing them would re-scatter into a 1-token sequence
    and raise the placeholder-count error).  If the cached path raises — the
    cache API is the most version-fragile thing here — it falls back permanently
    to the uncached full-forward loop used by
    ``scratchpad/hybrid_4lane_generate.py``, and says so once.
    """
    import torch

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    state = {"cached": True}

    def _prefix(row, image):
        r = {k: row[k] for k in row}
        r["images"] = [image]
        batch = col([r])
        labels = batch["labels"][0]
        nz = (labels != -100).nonzero()
        if len(nz) == 0:
            return None
        start = int(nz[0])                       # first supervised (answer) token
        return batch, start

    @torch.no_grad()
    def greedy(row, image, max_new):
        got = _prefix(row, image)
        if got is None:
            return ""
        batch, start = got
        ids = batch["input_ids"][:, :start].to(device)
        attn = batch["attention_mask"][:, :start].to(device)
        extra = {k: batch[k].to(device)
                 for k in ("pixel_values", "image_position_ids",
                           "num_soft_tokens_per_image") if k in batch}
        out_ids = []
        if state["cached"]:
            try:
                # NOTE: a LOCAL attention mask. Extending the outer `attn` here and
                # then falling back mid-row would hand the uncached loop a mask
                # longer than its input_ids.
                c_attn = attn
                o = model(input_ids=ids, attention_mask=c_attn, use_cache=True, **extra)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax(-1))
                for _ in range(max_new):
                    if nxt == im_end or nxt == tok.eos_token_id:
                        break
                    out_ids.append(nxt)
                    c_attn = torch.cat(
                        [c_attn, torch.ones((1, 1), dtype=c_attn.dtype, device=device)],
                        dim=1,
                    )
                    o = model(
                        input_ids=torch.tensor([[nxt]], device=device),
                        attention_mask=c_attn, past_key_values=past, use_cache=True,
                    )
                    past = o.past_key_values
                    nxt = int(o.logits[0, -1].argmax(-1))
                return tok.decode(out_ids)
            except Exception as exc:  # noqa: BLE001 — any cache-API breakage
                logger.warning(
                    "KV-cached decoding failed (%s); falling back to the uncached "
                    "full-forward loop for the rest of the run. This re-runs the "
                    "vision tower every step and is ~%dx slower — consider a smaller -n.",
                    exc, max_new,
                )
                state["cached"] = False
                out_ids = []
        cur, cur_attn = ids, attn
        for _ in range(max_new):
            o = model(input_ids=cur, attention_mask=cur_attn, **extra)
            nxt = int(o.logits[0, -1].argmax(-1))
            if nxt == im_end or nxt == tok.eos_token_id:
                break
            out_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=device)], dim=1
            )
        return tok.decode(out_ids)

    return greedy


def assistant_target(row):
    """The supervised answer string of a materialized row."""
    for msg in row["messages"]:
        if msg["role"] != "assistant":
            continue
        c = msg["content"]
        if isinstance(c, list):
            return "".join(p.get("text") or "" for p in c if p.get("type") != "image")
        return str(c)
    return ""


def collect(ds, tok, score, greedy, args, mst, floors):
    """Score every row and assemble the metrics dict (no verdict).

    ``score`` and ``greedy`` are passed IN — the whole accumulation, binning and
    JSON assembly is therefore exercisable with fakes on a CPU box, which is the
    only way any of it gets tested before a GPU run.
    """
    import numpy as np

    from univi.hybrid.data import blank_like

    n = len(ds)
    donor = [(i + 1) % n for i in range(n)]
    blank = blank_like(ds[0]["images"][0])
    logger.info("Scoring %d rows (blank %s %s)", n, blank.mode, blank.size)

    # accumulators: (value, row) pairs so every rate gets a cluster CI
    acc = {k: ([], []) for k in (
        "sent_tok", "sent_first_tok", "sent_cont_tok", "after_tok", "span_tok",
        "vis_tok", "halluc", "halluc_donor", "halluc_null", "vis_pos", "vis_recall",
    )}
    n_span_tokens = n_cont_tokens = 0
    perm_changed, perm_rows = [], []
    ce_rows = {"a": [], "p": [], "b": []}
    acc_rows = {"a": [], "p": [], "b": []}
    n_render_checked = n_render_mismatch = 0
    n_col_problem = n_map_problem = n_truncated = 0
    problems_seen = []
    n_masked_also_visible = 0
    decodes = []

    for i in range(n):
        row = ds[i]
        img = row["images"][0]
        target = assistant_target(row)
        sentinel = json.loads(row["render_config"])["sentinel"]
        words = list(row["words"])
        masked_words = [words[j] for j in row["mask_word_indices"]]
        visible_words = list(row["visible_words"])

        # -- stored-column cross-check -------------------------------------
        probs = check_sentinel_columns(
            target, list(row["sentinel_char_starts"]),
            list(row["sentinel_visible_index"]), sentinel,
        )
        if probs:
            n_col_problem += 1
            problems_seen.extend(probs[:2])
            continue

        # -- char -> token map, asserted against the supervised labels ------
        ids, spans = token_offsets(tok, target)
        if not spans_cover_text(spans, target):
            n_map_problem += 1
            problems_seen.append(f"row {i}: offset spans do not tile the target")
            continue

        res_a = score(row, img)
        if res_a is None:
            continue
        a_c, a_ce, lab = res_a
        if len(lab) < len(ids):
            n_truncated += 1
            problems_seen.append(
                f"row {i}: {len(lab)} supervised labels < {len(ids)} target tokens "
                "(max_length truncation)"
            )
            continue
        if list(lab[:len(ids)]) != list(ids):
            n_map_problem += 1
            problems_seen.append(
                f"row {i}: supervised label ids != tokenizer(target) ids — the "
                "char->token map does not address the scored positions"
            )
            continue

        res_p = score(row, ds[donor[i]]["images"][0])
        res_b = score(row, blank)
        if res_p is None or res_b is None:
            continue
        ce_rows["a"].append(float(a_ce.mean()))
        ce_rows["p"].append(float(res_p[1].mean()))
        ce_rows["b"].append(float(res_b[1].mean()))
        acc_rows["a"].append(float(a_c.mean()))
        acc_rows["p"].append(float(res_p[0].mean()))
        acc_rows["b"].append(float(res_b[0].mean()))

        # -- 1. span-boundary accuracy (teacher-forced) ---------------------
        sent_idx, after_idx, first_idx, cont_idx = [], [], [], []
        for s in row["sentinel_char_starts"]:
            si, ai = sentinel_token_indices(
                spans, int(s), len(sentinel), args.after_k, len(ids)
            )
            sent_idx += si
            after_idx += ai
            if si:
                first_idx.append(si[0])     # the localization DECISION
                cont_idx += si[1:]          # near-free continuations of "<mask>"
        after_idx = [j for j in after_idx if j not in set(sent_idx)]
        n_span_tokens += len(sent_idx) + len(after_idx)
        n_cont_tokens += len(cont_idx)
        for j in first_idx:
            acc["sent_first_tok"][0].append(float(a_c[j]))
            acc["sent_first_tok"][1].append(i)
        for j in cont_idx:
            acc["sent_cont_tok"][0].append(float(a_c[j]))
            acc["sent_cont_tok"][1].append(i)
        for j in sent_idx:
            acc["sent_tok"][0].append(float(a_c[j])); acc["sent_tok"][1].append(i)
        for j in after_idx:
            acc["after_tok"][0].append(float(a_c[j])); acc["after_tok"][1].append(i)
        for j in sent_idx + after_idx:
            acc["span_tok"][0].append(float(a_c[j])); acc["span_tok"][1].append(i)
        # visible-text tokens = everything that is neither sentinel nor the
        # resume token right after one (the box-detection guard, teacher-forced)
        excl = set(sent_idx) | set(after_idx)
        for j in range(len(ids)):
            if j not in excl:
                acc["vis_tok"][0].append(float(a_c[j])); acc["vis_tok"][1].append(i)

        # -- rebuild check + the mask-permutation control -------------------
        if not args.skip_decode:
            spec = spec_from_render_config(row["render_config"])
            page_a = render_from_runs(
                words, row["mask_run_starts"], row["mask_run_lengths"], spec
            )[0]
            n_render_checked += 1
            if not images_identical(page_a, img):
                n_render_mismatch += 1
            page_b = render_from_runs(
                words, row["mask_b_run_starts"], row["mask_b_run_lengths"], spec
            )[0]
            # BOTH sides come from render_masked_pages, so a difference cannot be
            # a rendering artefact; the stored-vs-rebuilt check above is separate.
            dec_a = greedy(row, page_a, args.max_new)
            dec_b = greedy(row, page_b, args.max_new)
            perm_changed.append(1.0 if dec_a != dec_b else 0.0)
            perm_rows.append(i)

            # -- 2. hallucinated-content rate -----------------------------
            vis_set = set(visible_words)
            dupes = [w for w in masked_words if w in vis_set]
            n_masked_also_visible += len(dupes)
            clean = [w for w in masked_words if w not in vis_set]
            for h in hallucination_hits(dec_a, masked_words, sentinel):
                acc["halluc"][0].append(h)  # all masked words (upper bound)
                acc["halluc"][1].append(i)
            for h in hallucination_hits(dec_a, clean, sentinel):
                acc["halluc_donor"][0].append(h)
                acc["halluc_donor"][1].append(i)
            # donor null: another row's hidden words, same decode
            dwords = [ds[donor[i]]["words"][j]
                      for j in ds[donor[i]]["mask_word_indices"]]
            for h in hallucination_hits(dec_a, dwords, sentinel):
                acc["halluc_null"][0].append(h); acc["halluc_null"][1].append(i)

            # -- 4. visible-word accuracy ---------------------------------
            pos, rec = visible_word_hits(dec_a, visible_words, sentinel)
            for v in pos:
                acc["vis_pos"][0].append(v); acc["vis_pos"][1].append(i)
            for v in rec:
                acc["vis_recall"][0].append(v); acc["vis_recall"][1].append(i)

            if len(decodes) < args.keep_decodes:
                decodes.append({"row": i, "target": target[:400],
                                "decode_mask_a": dec_a[:400],
                                "decode_mask_b": dec_b[:400],
                                "target_mask_b": str(row["mask_b_target"])[:400]})

        if (i + 1) % 25 == 0:
            logger.info("  %d/%d rows", i + 1, n)

    def rate(key):
        v, r = acc.get(key, ([], []))
        return cluster_rate(v, r, args.bootstrap, args.seed)

    a_ce_m = float(np.mean(ce_rows["a"])) if ce_rows["a"] else _nan()
    p_ce_m = float(np.mean(ce_rows["p"])) if ce_rows["p"] else _nan()
    b_ce_m = float(np.mean(ce_rows["b"])) if ce_rows["b"] else _nan()
    dp_m, dp_se, dp_n = paired_mean(
        [p - a for p, a in zip(ce_rows["p"], ce_rows["a"])])
    db_m, db_se, db_n = paired_mean(
        [b - a for b, a in zip(ce_rows["b"], ce_rows["a"])])
    rg_m, rg_se, rg_n = paired_mean(
        [(a - b) * 100 for a, b in zip(acc_rows["a"], acc_rows["b"])])

    def _ci(m, se):
        if m is None or se is None:
            return [None, None]
        return [_fin(m - Z95 * se), _fin(m + Z95 * se)]

    perm_rate = cluster_rate(perm_changed, perm_rows, args.bootstrap, args.seed) \
        if perm_changed else cluster_rate([], [], args.bootstrap, args.seed)

    M = {
        "meta": {
            "checkpoint": args.checkpoint, "val_path": args.val_path,
            "n_rows_available": n,
            "n_requested": args.max_samples, "n_rows_scored": len(ce_rows["a"]),
            "seed": args.seed, "max_soft_tokens": mst, "max_length": args.max_length,
            "after_k": args.after_k, "max_new": args.max_new,
            "bootstrap": args.bootstrap, "skip_decode": bool(args.skip_decode),
        },
        "data_integrity": {
            "n_rows_with_column_problem": n_col_problem,
            "n_rows_with_map_problem": n_map_problem,
            "n_rows_truncated": n_truncated,
            "examples": problems_seen[:10],
        },
        "render_check": {
            "n_checked": n_render_checked, "n_mismatch": n_render_mismatch,
            "note": "stored PNG vs render_masked_pages(words, mask_a_runs); the "
                    "permutation control feeds the REBUILT page on both sides so a "
                    "mismatch here does not silently become an A-vs-B difference.",
        },
        "decode": {"skipped": bool(args.skip_decode), "samples": decodes},
        "span_boundary": {
            "sentinel_tokens": rate("sent_tok"),
            "first_sentinel_token": rate("sent_first_tok"),
            "sentinel_continuation_tokens": rate("sent_cont_tok"),
            "after_tokens": rate("after_tok"),
            "combined": rate("span_tok"),
            # Score of a model that gets every "<mask>" continuation token right
            # and NOTHING else. Measured, not assumed: Qwen3 splits the sentinel
            # into 3 tokens, so with after_k=1 this floor is ~0.5.
            "degenerate_floor": _fin(n_cont_tokens / n_span_tokens) if n_span_tokens else None,
            "n_span_tokens": n_span_tokens,
            "n_continuation_tokens": n_cont_tokens,
        },
        "hallucination": {
            "all_masked_words": rate("halluc"),
            "excluding_visible_duplicates": rate("halluc_donor"),
            "donor_null": rate("halluc_null"),
            "n_masked_words_also_visible_on_page": n_masked_also_visible,
        },
        "mask_permutation": {"change_rate": perm_rate},
        "visible_words": {
            "positional": rate("vis_pos"),
            "recall_alignment_free": rate("vis_recall"),
            "teacher_forced_token_acc": rate("vis_tok"),
        },
        "ablation": {
            "response_ce": {"aligned": _fin(a_ce_m), "permuted": _fin(p_ce_m),
                            "blank": _fin(b_ce_m)},
            "tok_acc": {
                "aligned": _fin(np.mean(acc_rows["a"]) if acc_rows["a"] else None),
                "permuted": _fin(np.mean(acc_rows["p"]) if acc_rows["p"] else None),
                "blank": _fin(np.mean(acc_rows["b"]) if acc_rows["b"] else None),
            },
            "delta_perm_rel": _fin((p_ce_m - a_ce_m) / a_ce_m if a_ce_m else None),
            "delta_blank_rel": _fin((b_ce_m - a_ce_m) / a_ce_m if a_ce_m else None),
            "delta_perm_ce_paired": {"mean": _fin(dp_m), "se": _fin(dp_se),
                                     "n_rows": dp_n, "ci95": _ci(dp_m, dp_se)},
            "delta_blank_ce_paired": {"mean": _fin(db_m), "se": _fin(db_se),
                                      "n_rows": db_n, "ci95": _ci(db_m, db_se)},
            "reading_gain_pts": {"mean": _fin(rg_m), "se": _fin(rg_se),
                                 "n_rows": rg_n, "ci95": _ci(rg_m, rg_se)},
        },
        "floors": {
            "no_reading_floor_nats_per_token":
                floors.get("no_reading_floor_nats_per_token"),
            "no_reading_floor_with_length_nats_per_token":
                floors.get("no_reading_floor_with_length_nats_per_token"),
            "masked_word_guess_accuracy": floors.get("masked_word_guess_accuracy"),
        },
    }
    return M


def run(args):
    """Load the checkpoint, score, verdict. The only function that needs a GPU."""
    from datasets import load_from_disk

    model, tok, col, device, mst = build_model(
        args.checkpoint, args.max_soft_tokens, args.max_length
    )
    ds = load_from_disk(args.val_path)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))

    floors = {}
    try:
        floors = json.loads(Path(args.floor).read_text())
    except Exception:
        logger.warning("no floor.json at %s — analytic floors will be null", args.floor)

    M = collect(ds, tok, make_scorer(model, col, device),
                make_greedy(model, tok, col, device), args, mst, floors)
    M["verdict"] = verdict(M, args)
    return M


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def print_report(M):
    m = M["meta"]
    print(f"\n{'=' * 96}")
    print(f"H14 MASKED-REGION EVAL — {m['checkpoint']}")
    print(f"{'=' * 96}")
    print(f"rows scored {m['n_rows_scored']}/{m['n_requested']}   soft-tokens "
          f"{m['max_soft_tokens']}   max_length {m['max_length']}   after_k {m['after_k']}")
    di = M["data_integrity"]
    if di["n_rows_with_column_problem"] or di["n_rows_with_map_problem"] or di["n_rows_truncated"]:
        print(f"  !! dropped rows: {di['n_rows_with_column_problem']} stored-column "
              f"problems, {di['n_rows_with_map_problem']} char->token map problems, "
              f"{di['n_rows_truncated']} truncated")
        for e in di["examples"][:5]:
            print(f"     - {e}")
    rc = M["render_check"]
    print(f"  rebuild check: {rc['n_checked'] - rc['n_mismatch']}/{rc['n_checked']} "
          f"rows rebuild pixel-identical to the stored PNG")

    def line(label, r, spec=".2f"):
        ci = r.get("ci95") or [None, None]
        print(f"  {label:<38}{_pct(r.get('rate'), spec):>9}%  "
              f"[{_pct(ci[0], spec):>7}, {_pct(ci[1], spec):>7}]  "
              f"n={r.get('n')} obs / {r.get('n_rows')} rows")

    print("\n-- 1. SPAN-BOUNDARY ACCURACY (teacher-forced)")
    line("sentinel tokens (all)", M["span_boundary"]["sentinel_tokens"])
    line("  first sentinel token (localizes)",
         M["span_boundary"]["first_sentinel_token"])
    line("  sentinel continuations (near-free)",
         M["span_boundary"]["sentinel_continuation_tokens"])
    line("tokens immediately after", M["span_boundary"]["after_tokens"])
    line("COMBINED (pre-registered, >= 60%)", M["span_boundary"]["combined"])
    print(f"  {'degenerate floor (continuations only)':<38}"
          f"{_pct(M['span_boundary'].get('degenerate_floor')):>9}%  "
          f"({M['span_boundary'].get('n_continuation_tokens')}/"
          f"{M['span_boundary'].get('n_span_tokens')} scored tokens are free)")

    print("\n-- 2. HALLUCINATED-CONTENT RATE (greedy decode)")
    line("all masked words", M["hallucination"]["all_masked_words"], ".4f")
    line("excl. words also visible (headline)",
         M["hallucination"]["excluding_visible_duplicates"], ".4f")
    line("donor null (another row's words)", M["hallucination"]["donor_null"], ".4f")
    print(f"  analytic guess floor E_L[26^-L]      "
          f"{_fmt(M['floors'].get('masked_word_guess_accuracy'), '.3e'):>9}")
    print(f"  masked words that were also visible : "
          f"{M['hallucination']['n_masked_words_also_visible_on_page']}")

    print("\n-- 3. MASK-PERMUTATION CONTROL (same words, second stored mask)")
    line("rows whose output changed (>= 80%)", M["mask_permutation"]["change_rate"])

    print("\n-- 4. VISIBLE-WORD ACCURACY (degenerate-shortcut guard)")
    line("positional exact match", M["visible_words"]["positional"])
    line("alignment-free recall", M["visible_words"]["recall_alignment_free"])
    line("teacher-forced token acc", M["visible_words"]["teacher_forced_token_acc"])

    a = M["ablation"]
    print("\n-- 5. ALIGNED / PERMUTED / BLANK")
    print(f"  response-only CE : aligned {_fmt(a['response_ce']['aligned'], '.4f')} | "
          f"permuted {_fmt(a['response_ce']['permuted'], '.4f')} "
          f"(Dperm {_pct(a['delta_perm_rel'], '+.2f')}%) | "
          f"blank {_fmt(a['response_ce']['blank'], '.4f')} "
          f"(Dblank {_pct(a['delta_blank_rel'], '+.2f')}%)")
    print(f"  floor (visible letters, lower bound) "
          f"{_fmt(M['floors'].get('no_reading_floor_nats_per_token'), '.4f')} "
          f"[+length {_fmt(M['floors'].get('no_reading_floor_with_length_nats_per_token'), '.4f')}]")
    print(f"  tok-acc          : aligned {_pct(a['tok_acc']['aligned'])}% | "
          f"blank {_pct(a['tok_acc']['blank'])}% | reading gain "
          f"{_fmt(a['reading_gain_pts']['mean'], '+.2f')} +/- "
          f"{_fmt(a['reading_gain_pts']['se'], '.2f')} pts")

    v = M["verdict"]
    print(f"\n{'=' * 96}")
    for ln in v["lines"]:
        print(ln)
    print(f"\n>>> VERDICT: {v['final']}")
    print("\nCAVEATS (read before quoting the verdict):")
    for i, c in enumerate(v["caveats"], 1):
        print(f"  {i}. {c}")


# ---------------------------------------------------------------------------
# self-test (no torch, no GPU, no checkpoint)
# ---------------------------------------------------------------------------
class _FakeTok:
    """Whitespace-ish tokenizer with EXACT offsets, for testing the char->token map.

    Splits on spaces and emits the leading space as part of the following token,
    which is precisely the byte-level-BPE behaviour the real mapping must handle.
    """

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans, cur = [], 0
        for k, piece in enumerate(text.split(" ")):
            start = cur - (1 if k else 0)
            end = cur + len(piece)
            spans.append((start, end))
            cur = end + 1
        ids = list(range(len(spans)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def self_test():
    import numpy as np

    n_assert = 0

    def ok(msg):
        print(f"  ok  {msg}")

    # ---- 1. char -> token mapping ---------------------------------------
    from data.preprocessing.masked_regions import build_target

    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    runs = [(1, 2), (5, 1)]          # hide bravo+charlie, and foxtrot
    b = build_target(words, runs, "<mask>")
    assert b["target"] == "alpha <mask> delta echo <mask> golf", b["target"]
    assert b["visible_words"] == ["alpha", "delta", "echo", "golf"]
    assert b["sentinel_visible_index"] == [1, 3]
    assert b["masked_indices"] == [1, 2, 5]
    n_assert += 4
    ok("build_target: one sentinel per RUN, visible words and indices as documented")

    # offsets point exactly at the sentinels
    for k, s in enumerate(b["sentinel_char_starts"]):
        assert b["target"][s:s + 6] == "<mask>", (k, s)
    n_assert += len(b["sentinel_char_starts"])
    assert check_sentinel_columns(
        b["target"], b["sentinel_char_starts"], b["sentinel_visible_index"], "<mask>"
    ) == []
    n_assert += 1
    ok("stored sentinel columns cross-check clean on a known target")

    # a deliberately corrupted column must be CAUGHT, not averaged in
    bad = check_sentinel_columns(
        b["target"], [x + 1 for x in b["sentinel_char_starts"]],
        b["sentinel_visible_index"], "<mask>",
    )
    assert bad and "does not point at" in bad[0], bad
    off_by_one_vis = check_sentinel_columns(
        b["target"], b["sentinel_char_starts"], [0, 3], "<mask>"
    )
    assert off_by_one_vis and "pieces precede it" in off_by_one_vis[0], off_by_one_vis
    n_assert += 2
    ok("an off-by-one in EITHER stored column is detected (not silently used)")

    tokf = _FakeTok()
    ids, spans = token_offsets(tokf, b["target"])
    assert spans_cover_text(spans, b["target"])
    assert len(ids) == 6                       # 6 space-separated pieces
    n_assert += 2
    si, ai = sentinel_token_indices(spans, b["sentinel_char_starts"][0], 6, 1, len(ids))
    assert si == [1] and ai == [2], (si, ai)   # "<mask>" is piece 1, "delta" piece 2
    si2, ai2 = sentinel_token_indices(spans, b["sentinel_char_starts"][1], 6, 2, len(ids))
    assert si2 == [4] and ai2 == [5], (si2, ai2)   # after_k clipped at the end
    n_assert += 4
    ok("sentinel char offsets map to the right TOKEN indices (incl. end clipping)")

    # a sentinel split across several tokens must select ALL of them
    split_spans = [(0, 5), (5, 7), (7, 11), (11, 12), (12, 18)]
    si3, ai3 = sentinel_token_indices(split_spans, 5, 6, 1, len(split_spans))
    assert si3 == [1, 2] and ai3 == [3], (si3, ai3)
    n_assert += 2
    ok("a sentinel spanning several BPE tokens selects all of them, then the next")

    assert not spans_cover_text([(0, 3), (4, 7)], "abc def")
    n_assert += 1
    ok("spans that do not tile the target are rejected")

    # ---- 2. decode metrics ----------------------------------------------
    target_words = ["abcde", "fghij", "klmno", "pqrst"]
    dec = "abcde <mask> klmno pqrst"
    pieces, non_sent = decode_pieces(dec, "<mask>")
    assert non_sent == ["abcde", "klmno", "pqrst"]
    n_assert += 1
    hits = hallucination_hits(dec, ["fghij"], "<mask>")
    assert hits == [0.0]
    hits2 = hallucination_hits("abcde fghij klmno", ["fghij"], "<mask>")
    assert hits2 == [1.0]
    n_assert += 2
    ok("hallucination: a hidden word is a hit only when it is actually emitted")

    # substring must NOT count: "abc" inside "abcde" is not an emission
    assert hallucination_hits("abcde", ["abc"], "<mask>") == [0.0]
    n_assert += 1
    ok("hallucination uses whole-word membership, not substring (no inflation)")

    pos, rec = visible_word_hits(dec, ["abcde", "klmno", "pqrst"], "<mask>")
    assert pos == [1.0, 1.0, 1.0] and rec == [1.0, 1.0, 1.0]
    pos2, rec2 = visible_word_hits("abcde zzzzz klmno", ["abcde", "klmno", "pqrst"], "<mask>")
    assert pos2 == [1.0, 0.0, 0.0], pos2      # misalignment punishes positional
    assert rec2 == [1.0, 1.0, 0.0], rec2      # recall is alignment-free
    n_assert += 4
    ok("visible-word accuracy: positional is strict, recall is alignment-free")

    pos3, _ = visible_word_hits("", ["abcde"], "<mask>")
    assert pos3 == [0.0]
    n_assert += 1
    ok("an empty decode scores 0, not NaN")

    # ---- 3. statistics ---------------------------------------------------
    m, se, nr = cluster_stats(np.array([1.0, 1, 0, 0]), np.array([0, 0, 1, 1]))
    assert nr == 2 and abs(m - 0.5) < 1e-12 and abs(se - 0.5) < 1e-12
    n_assert += 3
    ok("cluster SE is computed over ROWS (n_rows=2 here), not over observations")

    # 100 rows, half all-hit and half all-miss: the row-level bootstrap must show
    # real spread. (Pairing a hit and a miss INSIDE every row would make every row
    # mean exactly 0.5 and the CI legitimately degenerate — that is the difference
    # a cluster bootstrap is there to express.)
    r = cluster_rate([1.0] * 50 + [0.0] * 50, list(range(100)), 200, 0)
    assert abs(r["rate"] - 0.5) < 1e-12 and r["n_rows"] == 100
    assert r["ci95"][0] < 0.5 < r["ci95"][1]
    r_within = cluster_rate([1.0, 0.0] * 50, [i // 2 for i in range(100)], 200, 0)
    assert r_within["ci95"] == [0.5, 0.5], r_within["ci95"]
    n_assert += 5
    r_all = cluster_rate([1.0] * 20, list(range(20)), 200, 0)
    assert r_all["rate"] == 1.0 and r_all["ci95"] == [1.0, 1.0]
    n_assert += 2
    ok("cluster bootstrap: CI brackets the point estimate and degenerates correctly")

    assert cluster_rate([], [], 10, 0)["method"] == "empty"
    n_assert += 1
    ok("an empty metric returns nulls, never 0")

    assert decide([0.7, 0.9], 0.6, ">=") == "PASS"
    assert decide([0.3, 0.5], 0.6, ">=") == "FAIL"
    assert decide([0.5, 0.7], 0.6, ">=") == "UNDECIDABLE"
    assert decide([0.01, 0.10], 0.15, "<=") == "PASS"
    assert decide([0.20, 0.40], 0.15, "<=") == "FAIL"
    assert decide([0.10, 0.30], 0.15, "<=") == "UNDECIDABLE"
    assert decide(None, 0.6, ">=") == "UNDECIDABLE"
    n_assert += 7
    ok("threshold decisions use the CI: straddling the bar is UNDECIDABLE, not a PASS")

    mm, mse, mn = paired_mean([1.0, 2.0, 3.0])
    assert mn == 3 and abs(mm - 2.0) < 1e-12 and abs(mse - 0.5773502691896258) < 1e-9
    n_assert += 3
    ok("paired per-row mean/SE")

    # ---- 4. the verdict tree --------------------------------------------
    def fake(span, hall, perm, vis, reads=True, render_mismatch=0, skip=False,
             floor=1.18e-05, dfloor=0.5, first=(0.7, 0.05)):
        def R(rate, half, n=1000, rows=100):
            return {"n": n, "n_rows": rows, "rate": rate, "rate_rowmean": rate,
                    "se_cluster": half / Z95 if half else 0.0,
                    "ci95": [rate - half, rate + half]}
        gain = 20.0 if reads else 0.0
        gse = 2.0
        return {
            "meta": {"n_rows_scored": 100},
            "data_integrity": {},
            "render_check": {"n_checked": 100, "n_mismatch": render_mismatch},
            "decode": {"skipped": skip},
            "span_boundary": {"combined": R(*span), "degenerate_floor": dfloor,
                              "first_sentinel_token": R(*first),
                              "n_span_tokens": 4000, "n_continuation_tokens": 2000},
            "hallucination": {"excluding_visible_duplicates": R(*hall)},
            "mask_permutation": {"change_rate": R(*perm, n=100, rows=100)},
            "visible_words": {"positional": R(*vis)},
            "ablation": {
                "delta_perm_ce_paired": {"mean": 1.0 if reads else 0.0, "se": 0.1,
                                         "ci95": [0.8, 1.2] if reads else [-0.2, 0.2]},
                "reading_gain_pts": {"mean": gain, "se": gse,
                                     "ci95": [gain - Z95 * gse, gain + Z95 * gse]
                                     if reads else [-4.0, 4.0]},
            },
            "floors": {"masked_word_guess_accuracy": floor},
        }

    args = argparse.Namespace()

    v = verdict(fake((0.75, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05)), args)
    assert v["final"].startswith("CONFIRMS"), v["final"]
    assert v["checks"]["span_boundary"] == "PASS"
    assert v["checks"]["span_above_degenerate_floor"] == "PASS"
    n_assert += 3
    ok("verdict: all three thresholds cleared, guard above floor -> CONFIRMS")

    # 62% combined clears the 60% bar but NOT the measured 50% continuation floor
    # once the CI is honest -> must not be reported as CONFIRMS
    v = verdict(fake((0.62, 0.015), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05),
                     dfloor=0.62, first=(0.05, 0.02)), args)
    assert v["checks"]["span_boundary"] == "PASS"
    assert v["checks"]["span_above_degenerate_floor"] != "PASS"
    assert v["final"].startswith("AMBIGUOUS"), v["final"]
    assert "degenerate floor" in v["final"]
    n_assert += 4
    ok("verdict: combined clears 60% but not the 3-token-sentinel degenerate floor "
       "-> AMBIGUOUS")

    v = verdict(fake((0.75, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05),
                     reads=False), args)
    assert v["final"].startswith("VOID"), v["final"]
    n_assert += 1
    ok("verdict: the model does not read the image at all -> VOID (no branch forced)")

    v = verdict(fake((0.85, 0.04), (0.02, 0.01), (0.95, 0.03), (0.00002, 0.00001)), args)
    assert v["final"].startswith("DEGENERATE-SHORTCUT"), v["final"]
    assert v["checks"]["visible_word_at_floor"] is True
    n_assert += 2
    ok("verdict: sentinels right, visible words at floor -> DEGENERATE-SHORTCUT")

    v = verdict(fake((0.55, 0.10), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05)), args)
    assert v["final"].startswith("UNDECIDABLE"), v["final"]
    assert v["checks"]["span_boundary"] == "UNDECIDABLE"
    n_assert += 2
    ok("verdict: a CI straddling a threshold -> UNDECIDABLE, not a rounded PASS")

    v = verdict(fake((0.30, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05)), args)
    assert v["final"].startswith("AMBIGUOUS"), v["final"]
    n_assert += 1
    ok("verdict: span-boundary clearly below bar -> AMBIGUOUS (NOT the REFUTES branch)")

    v = verdict(fake((0.75, 0.05), (0.40, 0.05), (0.95, 0.03), (0.60, 0.05)), args)
    assert v["final"].startswith("REFUTES-shaped"), v["final"]
    assert "MASK LEAK" in v["final"]
    n_assert += 2
    ok("verdict: hallucination above the bar -> REFUTES-shaped, flagged as a leak")

    v = verdict(fake((0.75, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05),
                     render_mismatch=3), args)
    assert v["checks"]["mask_permutation"] == "VOID", v["checks"]
    assert v["final"].startswith("UNDECIDABLE"), v["final"]
    n_assert += 2
    ok("verdict: a broken control render VOIDs the permutation criterion")

    v = verdict(fake((0.75, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05),
                     skip=True), args)
    assert v["checks"]["hallucination"] == "UNDECIDABLE"
    assert v["checks"]["mask_permutation"] == "UNDECIDABLE"
    n_assert += 2
    ok("verdict: --skip-decode leaves the decode-based criteria UNDECIDABLE")

    assert any("REFUTES branch" in c for c in v["caveats"])
    n_assert += 1
    ok("the low-power REFUTES caveat travels with every verdict")

    # ---- 5. JSON schema --------------------------------------------------
    M = fake((0.75, 0.05), (0.02, 0.01), (0.95, 0.03), (0.60, 0.05))
    M["verdict"] = verdict(M, args)
    blob = json.dumps(M)
    back = json.loads(blob)
    for key in ("span_boundary", "hallucination", "mask_permutation",
                "visible_words", "ablation", "render_check", "verdict", "floors"):
        assert key in back, key
    n_assert += 8
    assert set(back["verdict"]) == {"lines", "checks", "final", "caveats"}
    n_assert += 1
    ok("result dict is JSON round-trippable and carries every documented section")

    # ---- 6. LaneSpec round-trip (real render_config) ---------------------
    from data.preprocessing.masked_regions import LaneSpec, _render_config_json

    spec0 = LaneSpec()
    spec1 = spec_from_render_config(_render_config_json(spec0))
    for f in ("group_len_min", "group_len_max", "n_groups", "font_size", "rate_min",
              "rate_max", "run_min", "run_max", "mask_pad_x", "mask_pad_y",
              "fill_color", "sentinel", "max_pages"):
        assert getattr(spec1, f) == getattr(spec0, f), f
    n_assert += 13
    ok("LaneSpec rebuilt from a row's stored render_config matches the generator")

    print(f"\nSELF-TEST PASSED ({n_assert} assertions: char->token map, stored-column "
          f"cross-checks, decode metrics, cluster bootstrap, threshold decisions, "
          f"verdict tree, JSON schema, LaneSpec round-trip)")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", help="trained UniViHybridPretrained checkpoint dir")
    ap.add_argument("-n", "--max-samples", type=int, default=150,
                    help="validation rows to score")
    ap.add_argument("--val-path", default=DEFAULT_VAL,
                    help="validation split (load_from_disk)")
    ap.add_argument("--out", "--output", dest="out", default=DEFAULT_OUT,
                    help="where to write the JSON result")
    ap.add_argument("--floor", default=DEFAULT_FLOOR,
                    help="floor.json for the analytic floors")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="per-image soft-token budget; default = whatever the "
                         "CHECKPOINT records (280 pre-H17). Overriding this to a "
                         "budget the model was not trained at produces garbage.")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="collator truncation budget (H14 trains at 1024)")
    ap.add_argument("--after-k", type=int, default=1,
                    help="tokens after each sentinel counted as span boundary")
    ap.add_argument("--max-new", type=int, default=280,
                    help="greedy decode budget (H14 targets are ~209 tokens)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="row-level bootstrap replicates for every CI")
    ap.add_argument("--keep-decodes", type=int, default=8,
                    help="decoded samples stored in the JSON for eyeballing")
    ap.add_argument("--skip-decode", action="store_true",
                    help="teacher-forced metrics + ablation only (fast); leaves the "
                         "hallucination and permutation criteria UNDECIDABLE")
    ap.add_argument("--self-test", action="store_true",
                    help="run the CPU-only checks on every piece of pure logic and exit")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (no model, no GPU, no checkpoint)")
        self_test()
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required (or use --self-test)")

    M = run(a)
    print_report(M)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(M, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
