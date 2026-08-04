"""
H13 — density-ladder discriminator.

For each rung of ``data/materialized/h13-density-v0`` (randstr-d1..d5) this
computes, from ONE teacher-forced pass per image condition:

1. per-ANSWER-TOKEN-POSITION reading gain  = acc(aligned) - acc(blank), fine bins;
2. the same gain per CHARACTER position, using the *measured* char offset of each
   target token (chars/token is measured per rung, not assumed);
3. **the position-0 (and 0-4) reading gain per rung** — the PRIMARY discriminator,
   because it survives the case where the dense rungs never read at all and K is
   therefore undefined (see below);
4. **K** as pre-registered: "the character position beyond which positional
   reading gain drops below 50% of the 0-100-char gain" — reported in characters
   AND in the equivalent token index, with the degenerate cases named rather than
   silently turned into a number (see ``k_from_buckets``);
5. Delta-perm / Delta-blank (aggregate response-only CE, aligned vs another row's
   image vs blank).  **NOT comparable across densities** — this script's own data
   refutes that (the docstring used to claim it was): the aggregate averages over
   ALL supervised tokens while reading lives in the first ~5-12, so it is a
   function of TARGET LENGTH.  Measured here for IDENTICAL reading: +113.84% at 17
   supervised tok/row, +14.31% at 60, +3.05% at 234, +0.63% at 932.  Read it as
   context and compare it only WITHIN one target length; grounding strength across
   rungs comes from the position-resolved read-outs (``pos0_gain``,
   ``pos0_4_gain``, ``char_bins``);
6. aggregate CE vs each rung's ``floor.json`` no-reading floor, plus the fraction
   of ROWS whose aligned accuracy beats blank at all (with an exact sign test) —
   a cheap "does this rung read anything" gate to pass before any K is believed.

and then prints which PRE-REGISTERED BRANCH the numbers support:

    d5 >> d3                        => lines-per-patch demux => fix render geometry

    -- RETIRED (2026-07-29), kept only so old artifacts remain readable ----------
    K shrinks as chars/page rises  => (A) bandwidth  => H17 is the fix
    K fixed at the same token index => (C) scan      => H18, more tokens won't help
    K >= 400 chars                  => capacity not binding at mixture density

    K is STRUCTURALLY FLOORED: ``k_from_buckets`` scans buckets strictly PAST the
    100-char baseline window, so ``k_chars >= baseline_chars`` always, and every K
    this repo ever reported sat on that floor (an identical 48.4913 tokens at soft-
    token budgets 280 AND 560, for rungs whose real reading differed 2.8x).  A
    floored K carries no information about where reading decays, so the three
    K-keyed rows above cannot be evaluated.  ``K["at_floor"]`` now flags it, the
    K-trend step refuses to name a branch when the extremes are floored, and the
    position-resolved read-outs (pos-0 / pos-0-4 / char_bins) are the trustworthy
    substitute.

PRIMARY vs SECONDARY. On the evals available at launch only d1 clearly reads and
d3/d4/d5 sit above their floors. A rung that never reads has no "position where
reading decays", so K is UNDEFINED there — and the across-rung K comparison can
collapse to one usable point. The position-0 profile discriminates anyway:

    (C) scan failure => the model reads the START of the image at every density
        and only then loses its place => position-0 gain stays HIGH on d4 too.
    (A) bandwidth    => the representation is lossy even at the beginning of a
        dense page => position-0 gain DEGRADES monotonically as chars/page rises.

So the script leads with position-0 and treats K as secondary; if fewer than two
rungs yield a defined K it says the K branch is underpowered and falls back to
the position-0 profile rather than reporting a trend through one point.

Statistics. Every bin carries its token population, and gains carry BOTH a naive
Bernoulli SE and a **cluster (per-row) SE**, because tokens inside one row are
not independent — the cluster SE is the honest one and is what the CIs use. K
carries a row-level bootstrap CI. The d5-vs-d3 comparison is run **paired**: those
two rungs share a spec and therefore an RNG stream, so row i has a byte-identical
target in both and only the font differs (the script verifies this row-by-row and
falls back to unpaired if it does not hold). Rungs of DIFFERENT density do not
share targets, so cross-density comparisons are unpaired and weaker. When a
comparison comes back null the minimum detectable effect is printed beside it, so
"d5 == d3" can be judged as a real null or as underpowered.

Known ambiguity in the pre-registered criterion (printed with the verdict):
chars/token is ~2.06 on EVERY rung, so "fixed character position" and "fixed
token index" are proportional. A K that is constant across rungs therefore cannot
distinguish (C) a fixed token-index scan from a fixed-absolute-capacity flavour of
(A); only a K that MOVES with density is diagnostic on its own.

NOTE (GPU): this loads the checkpoint. Do not run it while H13 training holds the
card.

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/h13_analyze.py \
      --checkpoint data/checkpoints/h13-density-v0/final -n 150

  # no GPU, no model: exercise the K / binning / verdict logic on synthetic data
  uv run python scratchpad/h13_analyze.py --self-test
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
logger = logging.getLogger("h13_analyze")

DATA_ROOT = "data/materialized/h13-density-v0"
ALL_RUNGS = ["randstr-d1", "randstr-d2", "randstr-d3", "randstr-d4", "randstr-d5"]
OUT_DEFAULT = "data/eval/h13-density-ladder.json"

#: Fine answer-token-position bins. d1 targets are ~14.5 tokens and d4 ~929, so a
#: single fixed bin set cannot serve both: these resolve the first few positions
#: one-by-one (where H10/H11 found the gain lives) and then widen geometrically.
TOKEN_BINS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, 40),
    (40, 80), (80, 160), (160, 320), (320, 640), (640, 10**9),
]

#: Coarse character-position bins, for the printed table only. K is computed on
#: the uniform fine grid (``--char-bin-width``), not on these.
CHAR_BINS_REPORT = [
    (0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400),
    (400, 800), (800, 1600), (1600, 10**9),
]

Z95 = 1.959963985
#: two-sided 0.05, 80% power  =>  |effect| >= (1.96 + 0.84) * SE
Z_MDE = 2.801585


# ----------------------------------------------------------------------------
# small stats helpers (numpy-only; no torch — so --self-test runs on any box)
# ----------------------------------------------------------------------------
def _nan():
    return float("nan")


def _fin(x):
    """JSON cannot hold NaN/Inf; emit null instead so a missing number is a
    missing number and not a zero."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _fmt(x, spec=".2f", dash="--"):
    return dash if x is None or not math.isfinite(float(x)) else format(float(x), spec)


def row_means(values, rows):
    """Per-row means of ``values`` (both 1-D arrays, ``rows`` = row index)."""
    import numpy as np

    if len(values) == 0:
        return np.zeros(0)
    uniq, inv = np.unique(rows, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(uniq))
    c = np.bincount(inv, minlength=len(uniq))
    return s / c


def cluster_stats(values, rows):
    """(mean-of-row-means, cluster SE, n_rows) — the honest SE for token-level
    quantities, which are correlated within a row."""

    rm = row_means(values, rows)
    n = len(rm)
    if n == 0:
        return _nan(), _nan(), 0
    if n == 1:
        return float(rm[0]), _nan(), 1
    return float(rm.mean()), float(rm.std(ddof=1) / math.sqrt(n)), n


def naive_se(values):
    import numpy as np

    n = len(values)
    if n < 2:
        return _nan()
    return float(np.std(values, ddof=1) / math.sqrt(n))


def sign_test_p(n_pos, n_neg):
    """Exact two-sided binomial sign test (ties dropped, p=0.5). Distribution-free,
    so it holds even though per-row accuracies are neither normal nor independent
    *within* a row — the unit here is the row."""
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


# ----------------------------------------------------------------------------
# binning
# ----------------------------------------------------------------------------
def summarize_bins(bins, x, rows, a_corr, b_corr, a_ce, b_ce, min_count):
    """Per-bin accuracies / gain / CE with populations and SEs.

    ``x`` is the binning coordinate (token position or character offset).
    Empty bins are emitted with ``n_tokens: 0`` and **null** statistics — never 0 —
    so d1's absent deep positions cannot be misread as "no gain there".
    """

    out = []
    for lo, hi in bins:
        m = (x >= lo) & (x < hi)
        n = int(m.sum())
        label = f"{lo}-{hi - 1}" if hi < 10**9 else f"{lo}+"
        if n == 0:
            out.append({
                "bin": label, "lo": lo, "hi": None if hi >= 10**9 else hi,
                "n_tokens": 0, "n_rows": 0, "status": "empty",
                "acc_aligned": None, "acc_blank": None,
                "gain_pts": None, "gain_pts_rowmean": None,
                "se_naive_pts": None, "se_cluster_pts": None,
                "ci95_lo_pts": None, "ci95_hi_pts": None,
                "ce_aligned": None, "ce_blank": None, "underpowered": True,
            })
            continue
        d = a_corr[m] - b_corr[m]
        g_rm, se_cl, n_rows = cluster_stats(d, rows[m])
        se_nv = naive_se(d)
        # Point estimate = token-weighted (matches the acc_* columns beside it);
        # dispersion = the cluster SE, because tokens within a row are correlated
        # and the naive Bernoulli SE would be optimistic. gain_pts_rowmean is the
        # unweighted per-row version, kept for anyone who prefers it.
        gain = float(d.mean() * 100)
        out.append({
            "bin": label, "lo": lo, "hi": None if hi >= 10**9 else hi,
            "n_tokens": n, "n_rows": n_rows,
            "status": "ok",
            "acc_aligned": float(a_corr[m].mean()),
            "acc_blank": float(b_corr[m].mean()),
            "gain_pts": gain,
            "gain_pts_rowmean": _fin(g_rm * 100) if math.isfinite(g_rm) else None,
            "se_naive_pts": _fin(se_nv * 100),
            "se_cluster_pts": _fin(se_cl * 100),
            "ci95_lo_pts": _fin(gain - Z95 * se_cl * 100),
            "ci95_hi_pts": _fin(gain + Z95 * se_cl * 100),
            "ce_aligned": float(a_ce[m].mean()),
            "ce_blank": float(b_ce[m].mean()),
            "underpowered": bool(n < min_count),
        })
    return out


def bucketize(char_off, width, n_buckets):
    import numpy as np

    idx = np.minimum((char_off // width).astype(int), n_buckets - 1)
    return idx


def per_row_bucket_tables(char_off, rows, diff, width, n_buckets):
    """(sum_d, cnt) of shape [n_rows, n_buckets] — the bootstrap resamples ROWS,
    so pre-aggregating per row makes each replicate a couple of array sums."""
    import numpy as np

    uniq, inv = np.unique(rows, return_inverse=True)
    b = bucketize(char_off, width, n_buckets)
    flat = inv * n_buckets + b
    size = len(uniq) * n_buckets
    s = np.bincount(flat, weights=diff, minlength=size).reshape(len(uniq), n_buckets)
    c = np.bincount(flat, minlength=size).reshape(len(uniq), n_buckets)
    return s, c


# ----------------------------------------------------------------------------
# K — the pre-registered discriminator
# ----------------------------------------------------------------------------
def k_from_buckets(sum_d, cnt, width, baseline_chars, min_count, frac=0.5):
    """K = first character position past ``baseline_chars`` where the reading
    gain drops below ``frac`` x the 0-``baseline_chars``-char gain AND STAYS below.

    ``sum_d`` / ``cnt`` are per-uniform-bucket sums/counts of the paired
    per-token gain (aligned correct - blank correct).

    Degenerate cases are returned as a STATUS, never as a number:

    - ``no_baseline_tokens`` — fewer than ``min_count`` tokens inside the
      baseline window (K is not defined; the reference gain does not exist).
    - ``no_reading``         — the baseline gain is <= 0. There is no reading to
      decay, so "50% of it" is meaningless. K is null. (The caller additionally
      downgrades a baseline that is positive but not significantly so.)
    - ``censored_short``     — no eligible bucket exists beyond the baseline
      window: the targets END inside (or barely past) it. This is the expected
      outcome for d1 (~29 chars) and near-expected for d2 (~119 chars): the
      pre-registered 0-100-char reference window swallows the whole target. K is
      null and ``k_lower_bound_chars`` records how far we could actually see.
    - ``censored_right``     — the gain never falls below the threshold out to the
      last populated bucket. K is null and ``k_lower_bound_chars`` = the end of
      the last eligible bucket, i.e. "K is at least this".
    - ``ok``                 — ``k_chars`` is the START of the crossing bucket.

    THE SIXTH DEGENERATE CASE, MISSING UNTIL 2026-07-29: **K AT ITS STRUCTURAL
    FLOOR.**  The scan starts at bucket ``nbase = baseline_chars // width``, so
    ``k_chars >= baseline_chars`` BY CONSTRUCTION.  When the crossing lands on that
    very first eligible bucket, ``k_chars == baseline_chars`` is not a measurement
    of anything — it is the smallest value the statistic can take, and it is what
    every K ever reported in this repo turned out to be (an identical 48.4913
    tokens at soft-token budgets 280 AND 560, for rungs whose real reading differed
    2.8x).  ``status`` stays ``"ok"`` so the bootstrap and the downstream
    ``status == "ok"`` consumers keep working, but ``at_floor`` is set True,
    ``status_detail`` becomes ``"ok_at_structural_floor"`` and ``floor_warning``
    spells it out.  A floored K must be read as "reading had already decayed by
    ``baseline_chars``, i.e. K <= baseline_chars", NOT as "K = baseline_chars".
    To resolve it you must SHRINK ``--baseline-chars`` (see ``K_alt_baseline25``)
    or use the position-resolved read-outs instead.

    ``k_chars_first_crossing`` reports the first sub-threshold bucket WITHOUT the
    "stays below" requirement, as a sensitivity check on that requirement.
    """

    nb = len(cnt)
    nbase = int(baseline_chars // width)
    base_cnt = float(cnt[:nbase].sum())
    base_sum = float(sum_d[:nbase].sum())
    res = {
        "k_chars": None, "k_bucket": None, "status": None,
        # `at_floor` is the sixth degenerate case (see the docstring): a K sitting
        # on the smallest value the scan can return is not a measurement.
        "status_detail": None, "at_floor": False, "floor_warning": None,
        "floor_chars": int(nbase * width), "first_eligible_bucket": int(nbase),
        "baseline_gain_pts": None, "threshold_pts": None,
        "baseline_n_tokens": int(base_cnt),
        "k_chars_first_crossing": None, "k_lower_bound_chars": None,
        "n_eligible_buckets_past_baseline": 0,
    }
    if base_cnt < min_count:
        res["status"] = res["status_detail"] = "no_baseline_tokens"
        return res
    base_gain = base_sum / base_cnt
    res["baseline_gain_pts"] = base_gain * 100
    if base_gain <= 0:
        res["status"] = res["status_detail"] = "no_reading"
        return res
    thr = frac * base_gain
    res["threshold_pts"] = thr * 100

    eligible = [b for b in range(nbase, nb) if cnt[b] >= min_count]
    res["n_eligible_buckets_past_baseline"] = len(eligible)
    if not eligible:
        # Nothing measurable past the reference window: the targets end there.
        # The bound is where the DATA ends, which may be well short of the
        # window itself (d1: ~29 chars vs a 100-char window).
        populated = [b for b in range(nb) if cnt[b] > 0]
        res["status"] = res["status_detail"] = "censored_short"
        res["k_lower_bound_chars"] = (
            (max(populated) + 1) * width if populated else 0
        )
        return res

    gains = {b: sum_d[b] / cnt[b] for b in eligible}
    below = [b for b in eligible if gains[b] < thr]
    if below:
        res["k_chars_first_crossing"] = below[0] * width
    # first bucket that is below AND is followed only by below-threshold buckets
    k_bucket = None
    for i, b in enumerate(eligible):
        if gains[b] < thr and all(gains[c] < thr for c in eligible[i + 1:]):
            k_bucket = b
            break
    if k_bucket is None:
        res["status"] = "censored_right"
        res["status_detail"] = "censored_right"
        res["k_lower_bound_chars"] = (eligible[-1] + 1) * width
        return res
    res["status"] = "ok"
    res["k_bucket"] = int(k_bucket)
    res["k_chars"] = int(k_bucket * width)
    # AT THE FLOOR: the crossing is the FIRST bucket the scan is allowed to look
    # at, so k_chars == baseline_chars is the minimum the statistic can return and
    # carries no information about where reading decayed. Status stays "ok" for the
    # bootstrap / downstream consumers; the flags below are what must be read.
    res["at_floor"] = bool(k_bucket == nbase)
    res["status_detail"] = "ok_at_structural_floor" if res["at_floor"] else "ok"
    if res["at_floor"]:
        res["floor_warning"] = (
            f"K = {res['k_chars']} chars is the STRUCTURAL FLOOR of this statistic: "
            f"the scan starts at the first bucket past the {baseline_chars}-char "
            f"baseline window, so k_chars >= {baseline_chars} always. Read it as "
            f"'reading had already decayed by {baseline_chars} chars, i.e. "
            f"K <= {baseline_chars}', NOT as a measured collapse point. Shrink "
            f"--baseline-chars or use the position-resolved read-outs (pos0_gain, "
            f"pos0_4_gain, char_bins) instead.")
    return res


def bootstrap_k(sum_rows, cnt_rows, width, baseline_chars, min_count, n_boot, seed,
                frac=0.5):
    """Row-level bootstrap CI for K. Returns percentiles over the replicates that
    produced a numeric K, plus the status histogram — a K whose bootstrap only
    resolves 40% of the time is not a measurement, and the histogram says so."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_rows = sum_rows.shape[0]
    ks, statuses = [], {}
    n_at_floor = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_rows, n_rows)
        s = sum_rows[pick].sum(axis=0)
        c = cnt_rows[pick].sum(axis=0)
        r = k_from_buckets(s, c, width, baseline_chars, min_count, frac)
        # the histogram distinguishes a resolved K from one sitting on the
        # structural floor; `n_resolved` still counts every status == "ok"
        # replicate so the CI percentiles are unchanged.
        key = r["status_detail"] or r["status"]
        statuses[key] = statuses.get(key, 0) + 1
        if r["status"] == "ok":
            ks.append(r["k_chars"])
            n_at_floor += int(bool(r["at_floor"]))
    out = {"n_boot": n_boot, "status_counts": statuses,
           "n_resolved": len(ks), "n_resolved_at_floor": n_at_floor,
           "frac_resolved_at_floor": (n_at_floor / len(ks)) if ks else None,
           "ci_lo_chars": None, "ci_hi_chars": None,
           "median_chars": None}
    if ks:
        arr = np.array(ks, dtype=float)
        out["ci_lo_chars"] = float(np.percentile(arr, 5))
        out["ci_hi_chars"] = float(np.percentile(arr, 95))
        out["median_chars"] = float(np.percentile(arr, 50))
    return out


# ----------------------------------------------------------------------------
# token -> character offset mapping
# ----------------------------------------------------------------------------
class TokenChars:
    """Decoded length of each token id, cached, and the token -> character offset
    map for a supervised sequence.

    "Content" = everything BEFORE the supervised trailer (the first special token,
    ``<|im_end|>``, and the ``"\\n"`` after it). The trailer is dropped from the
    positional statistics: it is trivially predictable under every image
    condition, it always lands at the deepest positions, and on d1 it is 2 of ~16
    supervised tokens — keeping it would push d1's (and only d1's) deep-position
    gain toward 0 as an artefact.

    Whitespace-only tokens INSIDE the target are kept: they are part of the page
    text, their character offsets matter, and dropping them would break the
    reconstruction check below. They are counted separately (``n_format_tokens``)
    because they are format, not content, and carry ~0 gain — the 5-letter group
    layout makes them periodic, which is the likeliest source of any wobble in
    the single-position bins.

    Positions are indices into the FULL supervised sequence, so dropping the
    trailer never renumbers anything.
    """

    def __init__(self, tok):
        self.tok = tok
        self.special = set(getattr(tok, "all_special_ids", []) or [])
        self._cache = {}

    def piece(self, tid):
        tid = int(tid)
        s = self._cache.get(tid)
        if s is None:
            s = "" if tid in self.special else self.tok.decode(
                [tid], skip_special_tokens=True
            )
            self._cache[tid] = s
        return s

    def offsets(self, ids):
        """``(char_off, char_len, is_content, total_chars)`` for one supervised
        token id sequence."""
        first_special = None
        for k, tid in enumerate(ids):
            if int(tid) in self.special:
                first_special = k
                break
        offs, lens, content = [], [], []
        cur = 0
        for k, tid in enumerate(ids):
            p = self.piece(tid)
            offs.append(cur)
            lens.append(len(p))
            content.append(first_special is None or k < first_special)
            cur += len(p)
        return offs, lens, content, cur


def assistant_text(row):
    parts = []
    for msg in row["messages"]:
        if msg["role"] != "assistant":
            continue
        c = msg["content"]
        if isinstance(c, list):
            parts += [p.get("text") or "" for p in c if p.get("type") != "image"]
        else:
            parts.append(str(c))
    return "".join(parts)


# ----------------------------------------------------------------------------
# GPU pass
# ----------------------------------------------------------------------------
def make_scorer(model, col, device, logit_chunk):
    """``score(row, image) -> (correct[], ce[], target_ids[], seq_len)``, teacher
    forced over the supervised (response-only) positions.

    Labels are deliberately NOT passed to the model: passing them triggers the
    ``logits_to_keep`` optimization and the logits come back empty. Logits are
    sliced to the supervised positions and cast to fp32 in chunks so a
    931-token d4 row does not materialise a full [S, 151936] fp32 tensor.
    """
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def score(row, image):
        r = {k: row[k] for k in row}
        r["images"] = [image]
        batch = col([r])
        labels = batch["labels"]
        seq_len = int(batch["input_ids"].shape[1])
        fwd = {k: (v.to(device) if hasattr(v, "to") else v)
               for k, v in batch.items() if k != "labels"}
        out = model(**fwd)
        logits = out.logits[0]
        tg = labels[0].to(logits.device)[1:]
        lg = logits[:-1]
        idx = (tg != -100).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None
        corr = torch.empty(idx.numel())
        ce = torch.empty(idx.numel())
        for s in range(0, idx.numel(), logit_chunk):
            j = idx[s:s + logit_chunk]
            sl = lg[j].float()
            t = tg[j]
            corr[s:s + logit_chunk] = (sl.argmax(-1) == t).float().cpu()
            ce[s:s + logit_chunk] = F.cross_entropy(sl, t, reduction="none").cpu()
            del sl
        ids = tg[idx].cpu().numpy()
        del out, logits, lg
        return corr.numpy(), ce.numpy(), ids, seq_len

    return score


def collect_rung(score, tokchars, ds, max_length, skip_perm):
    """One teacher-forced pass per row per condition (aligned / permuted / blank).

    Returns ``(arrays, rows_info, meta)``: ``arrays`` holds one record per scored
    ANSWER TOKEN (row, token position, character offset, correctness and CE under
    each condition); ``rows_info`` one record per row. Takes ``score`` as an
    argument so the accumulation is exercisable without a model.
    """
    import numpy as np
    from univi.hybrid.data import blank_like

    n = len(ds)
    donor = [(i + 1) % n for i in range(n)]
    ref = ds[0]["images"][0]
    blank = blank_like(ref)

    A = {k: [] for k in ("row", "pos", "char_off", "a_c", "b_c", "p_c",
                         "a_ce", "b_ce", "p_ce")}
    rows_info = []
    n_trunc = 0
    n_charlen_mismatch = 0
    n_format_tokens = 0
    kept = 0
    for i in range(n):
        row = ds[i]
        img = row["images"][0]
        ra = score(row, img)
        rb = score(row, blank)
        rp = None if skip_perm else score(row, ds[donor[i]]["images"][0])
        if ra is None or rb is None or (not skip_perm and rp is None):
            continue
        a_c, a_ce, ids, seq_len = ra
        b_c, b_ce, ids_b, _ = rb
        m = min(len(a_c), len(b_c))
        if not skip_perm:
            m = min(m, len(rp[0]))
        ids = ids[:m]
        offs, lens, content, total_chars = tokchars.offsets(ids)
        text = assistant_text(row)
        # The decoded CONTENT tokens must reconstruct the target exactly. (The
        # supervised trailer is <|im_end|> -> "" plus "\n", both non-content, so
        # total_chars is len(text)+1; comparing that would flag every row.)
        content_chars = sum(l for l, c in zip(lens, content) if c)
        if content_chars != len(text):
            n_charlen_mismatch += 1
        truncated = seq_len >= max_length
        n_trunc += int(truncated)
        for p in range(m):
            if not content[p]:
                continue
            if not tokchars.piece(ids[p]).strip():
                n_format_tokens += 1
            A["row"].append(kept)
            A["pos"].append(p)
            A["char_off"].append(offs[p])
            A["a_c"].append(a_c[p]); A["b_c"].append(b_c[p])
            A["a_ce"].append(a_ce[p]); A["b_ce"].append(b_ce[p])
            A["p_c"].append(rp[0][p] if not skip_perm else float("nan"))
            A["p_ce"].append(rp[1][p] if not skip_perm else float("nan"))
        rows_info.append({
            "n_supervised_tokens": int(m),
            "n_content_tokens": int(sum(content[:m])),
            # chars/page = the target text itself; the decoded-content count is
            # kept alongside so a mismatch is visible rather than silent.
            "n_target_chars": len(text),
            "n_decoded_content_chars": int(content_chars),
            "truncated": bool(truncated),
            # aggregate, ALL supervised tokens (incl. the <|im_end|> trailer) so
            # these CE numbers stay comparable with the earlier ablation scripts
            "ce_aligned_all": float(a_ce[:m].mean()),
            "ce_blank_all": float(b_ce[:m].mean()),
            "ce_perm_all": (float(rp[1][:m].mean()) if not skip_perm else float("nan")),
            "acc_aligned_all": float(a_c[:m].mean()),
            "acc_blank_all": float(b_c[:m].mean()),
            "acc_perm_all": (float(rp[0][:m].mean()) if not skip_perm else float("nan")),
            "target_text": text,
        })
        kept += 1
        if (i + 1) % 25 == 0:
            logger.info("    %d/%d rows", i + 1, n)
    arrays = {k: np.array(v, dtype=(np.int64 if k in ("row", "pos", "char_off")
                                    else np.float64)) for k, v in A.items()}
    return arrays, rows_info, {"n_truncated": n_trunc,
                               "n_charlen_mismatch": n_charlen_mismatch,
                               "n_format_tokens": n_format_tokens}


# ----------------------------------------------------------------------------
# per-rung analysis
# ----------------------------------------------------------------------------
def analyse_rung(arrays, rows_info, meta, floor, spec, args):
    import numpy as np

    n_rows = len(rows_info)
    a_c, b_c = arrays["a_c"], arrays["b_c"]
    diff = a_c - b_c
    pos, ch, rows = arrays["pos"], arrays["char_off"], arrays["row"]

    tgt_tokens = [r["n_supervised_tokens"] for r in rows_info]
    tgt_content = [r["n_content_tokens"] for r in rows_info]
    tgt_chars = [r["n_target_chars"] for r in rows_info]
    total_chars = float(sum(tgt_chars))
    total_content_tok = float(sum(tgt_content))
    cpt = total_chars / total_content_tok if total_content_tok else _nan()

    def _fmean(xs):
        xs = [x for x in xs if math.isfinite(x)]
        return float(sum(xs) / len(xs)) if xs else _nan()

    ce_a = float(np.mean([r["ce_aligned_all"] for r in rows_info]))
    ce_b = float(np.mean([r["ce_blank_all"] for r in rows_info]))
    ce_p = _fmean([r["ce_perm_all"] for r in rows_info])
    dperm = (ce_p - ce_a) / ce_a if ce_a else _nan()
    dblank = (ce_b - ce_a) / ce_a if ce_a else _nan()
    # paired per-row CE deltas + cluster SE, so Dperm/Dblank get error bars too
    dp_rows = np.array([r["ce_perm_all"] - r["ce_aligned_all"] for r in rows_info])
    db_rows = np.array([r["ce_blank_all"] - r["ce_aligned_all"] for r in rows_info])

    def _pair(x):
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return None, None
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))

    dp_m, dp_se = _pair(dp_rows)
    db_m, db_se = _pair(db_rows)

    # --- PRIMARY discriminator: gain at the very start of the answer.
    # A rung whose page is too dense to encode at all (A) loses it here; a rung
    # that encodes the start fine and then loses its place (C) keeps it here.
    def _posgain(lo, hi):
        m = (pos >= lo) & (pos < hi)
        n = int(m.sum())
        if n == 0:
            return {"n_tokens": 0, "n_rows": 0, "gain_pts": None, "se_pts": None,
                    "ci95": [None, None], "acc_aligned": None, "acc_blank": None,
                    "significant": False, "status": "empty"}
        _, se, nr = cluster_stats(diff[m], rows[m])
        g = float(diff[m].mean())
        return {
            "n_tokens": n, "n_rows": nr,
            "gain_pts": _fin(g * 100), "se_pts": _fin(se * 100),
            "ci95": [_fin((g - Z95 * se) * 100), _fin((g + Z95 * se) * 100)],
            "acc_aligned": float(a_c[m].mean()), "acc_blank": float(b_c[m].mean()),
            "significant": bool(math.isfinite(g) and math.isfinite(se)
                                and g - Z95 * se > 0),
            "status": "ok",
        }

    pos0 = _posgain(0, 1)
    pos04 = _posgain(0, 5)

    # --- row-level sanity: does this rung read ANYTHING? (unit = row, not token)
    ra, rb = row_means(a_c, rows), row_means(b_c, rows)
    rd = ra - rb
    n_gt, n_lt = int((rd > 0).sum()), int((rd < 0).sum())
    row_sanity = {
        "n_rows": int(len(rd)),
        "n_rows_aligned_gt_blank": n_gt,
        "n_rows_aligned_lt_blank": n_lt,
        "n_rows_tied": int((rd == 0).sum()),
        "frac_rows_aligned_gt_blank": (float(n_gt / len(rd)) if len(rd) else None),
        "sign_test_p": sign_test_p(n_gt, n_lt),
        "mean_row_gain_pts": _fin(float(rd.mean() * 100)) if len(rd) else None,
    }

    tok_bins = summarize_bins(TOKEN_BINS, pos, rows, a_c, b_c,
                              arrays["a_ce"], arrays["b_ce"], args.min_bin_tokens)
    char_bins = summarize_bins(CHAR_BINS_REPORT, ch, rows, a_c, b_c,
                               arrays["a_ce"], arrays["b_ce"], args.min_bin_tokens)

    # --- fine uniform char grid (the K substrate)
    width = args.char_bin_width
    max_char = int(ch.max()) + 1 if len(ch) else 0
    n_buckets = max(1, math.ceil(max_char / width))
    n_buckets = max(n_buckets, math.ceil(args.baseline_chars / width) + 1)
    s_rows, c_rows = per_row_bucket_tables(ch, rows, diff, width, n_buckets)
    s_tot, c_tot = s_rows.sum(axis=0), c_rows.sum(axis=0)
    fine = []
    for b in range(n_buckets):
        n = int(c_tot[b])
        if n == 0:
            fine.append({"lo": b * width, "hi": (b + 1) * width, "n_tokens": 0,
                         "gain_pts": None, "se_cluster_pts": None,
                         "underpowered": True})
            continue
        m = bucketize(ch, width, n_buckets) == b
        _, se_cl, _ = cluster_stats(diff[m], rows[m])
        fine.append({"lo": b * width, "hi": (b + 1) * width, "n_tokens": n,
                     "gain_pts": float(s_tot[b] / c_tot[b] * 100),
                     "se_cluster_pts": _fin(se_cl * 100),
                     "underpowered": bool(n < args.min_bin_tokens)})

    # --- baseline gain over the pre-registered 0-BASELINE_CHARS window
    bm = ch < args.baseline_chars
    if bm.any():
        _, base_se, base_rows = cluster_stats(diff[bm], rows[bm])
        # token-weighted, so it equals the reference gain K itself is built from
        base_gain = float(diff[bm].mean())
    else:
        base_gain, base_se, base_rows = _nan(), _nan(), 0
    base_sig = bool(math.isfinite(base_gain) and math.isfinite(base_se)
                    and base_gain - Z95 * base_se > 0)

    K = k_from_buckets(s_tot, c_tot, width, args.baseline_chars,
                       args.min_bin_tokens)
    if K.get("k_lower_bound_chars") is not None and len(ch):
        # never claim a bound past the last character we actually scored
        K["k_lower_bound_chars"] = int(min(K["k_lower_bound_chars"],
                                           int(ch.max()) + 1))
    if K["status"] in ("ok", "censored_right", "censored_short") and not base_sig:
        # positive but not distinguishable from zero => there is no established
        # reading gain for "50% of" to refer to. Say so instead of reporting a K.
        K = dict(K)
        K["status"] = K["status_detail"] = "no_reading_ns"
        K["k_chars"] = None
        K["at_floor"] = False
    K["baseline_gain_ci95_pts"] = [_fin((base_gain - Z95 * base_se) * 100),
                                   _fin((base_gain + Z95 * base_se) * 100)]
    K["baseline_gain_significant"] = base_sig
    K["baseline_window_chars"] = args.baseline_chars
    K["bootstrap"] = bootstrap_k(s_rows, c_rows, width, args.baseline_chars,
                                 args.min_bin_tokens, args.bootstrap, args.seed) \
        if K["status"] == "ok" else {"n_boot": 0, "n_resolved": 0,
                                     "n_resolved_at_floor": 0,
                                     "frac_resolved_at_floor": None,
                                     "status_counts": {}, "ci_lo_chars": None,
                                     "ci_hi_chars": None, "median_chars": None}

    # K in tokens: (a) from the measured chars/token, (b) empirically, as the
    # median token index of the tokens actually sitting in the crossing bucket.
    k_tok_ratio = k_tok_emp = None
    if K["status"] == "ok":
        k_tok_ratio = K["k_chars"] / cpt if cpt and math.isfinite(cpt) else None
        sel = (ch >= K["k_chars"]) & (ch < K["k_chars"] + width)
        if sel.any():
            k_tok_emp = float(np.median(pos[sel]))
    K["k_tokens_from_ratio"] = _fin(k_tok_ratio)
    K["k_tokens_empirical_median"] = _fin(k_tok_emp)

    # --- secondary, NOT pre-registered: a 25-char reference window. The 0-100
    # window is wider than d1's entire target (~29 chars) and covers 84% of d2's,
    # so the pre-registered K is structurally unmeasurable there. This variant at
    # least lets d2 contribute to the density trend. Reported separately, never
    # substituted for the pre-registered number.
    alt_base = min(25, args.baseline_chars)
    K_alt = k_from_buckets(s_tot, c_tot, width, alt_base, args.min_bin_tokens)
    K_alt["baseline_window_chars"] = alt_base
    K_alt["preregistered"] = False

    return {
        "n_rows": n_rows,
        "spec": spec,
        "floor_nats": floor,
        "target_tokens_mean": float(np.mean(tgt_tokens)) if tgt_tokens else None,
        "target_content_tokens_mean": float(np.mean(tgt_content)) if tgt_content else None,
        "target_chars_mean": float(np.mean(tgt_chars)) if tgt_chars else None,
        "target_chars_max": int(np.max(tgt_chars)) if tgt_chars else None,
        "chars_per_token_realized": _fin(cpt),
        "n_tokens_scored": int(len(pos)),
        "n_format_tokens_scored": meta.get("n_format_tokens"),
        "n_truncated_rows": meta["n_truncated"],
        "n_charlen_mismatch_rows": meta["n_charlen_mismatch"],
        "ce_all_tokens": {"aligned": ce_a, "permuted": _fin(ce_p), "blank": ce_b},
        "ce_vs_floor": _fin(ce_a - floor) if floor is not None else None,
        "below_floor": (bool(ce_a < floor) if floor is not None else None),
        "acc_all_tokens": {
            "aligned": float(np.mean([r["acc_aligned_all"] for r in rows_info])),
            "permuted": _fin(_fmean([r["acc_perm_all"] for r in rows_info])),
            "blank": float(np.mean([r["acc_blank_all"] for r in rows_info])),
        },
        "delta_perm_rel": _fin(dperm),
        "delta_blank_rel": _fin(dblank),
        "delta_perm_ce_paired": {"mean": _fin(dp_m), "se": _fin(dp_se),
                                 "ci95": [_fin(dp_m - Z95 * dp_se) if dp_m is not None else None,
                                          _fin(dp_m + Z95 * dp_se) if dp_m is not None else None]},
        "delta_blank_ce_paired": {"mean": _fin(db_m), "se": _fin(db_se),
                                  "ci95": [_fin(db_m - Z95 * db_se) if db_m is not None else None,
                                           _fin(db_m + Z95 * db_se) if db_m is not None else None]},
        "baseline_gain_pts": _fin(base_gain * 100),
        "baseline_gain_se_pts": _fin(base_se * 100),
        "baseline_gain_n_rows": base_rows,
        "baseline_gain_significant": base_sig,
        "pos0_gain": pos0,
        "pos0_4_gain": pos04,
        "row_sanity": row_sanity,
        "token_bins": tok_bins,
        "char_bins": char_bins,
        "char_bins_fine": fine,
        "char_bin_width": width,
        "K": K,
        "K_alt_baseline25": K_alt,
    }


# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------
def print_rung(name, R):
    print(f"\n{'=' * 100}")
    print(f"RUNG {name}   rows={R['n_rows']}   "
          f"letters/page={R['spec'].get('n_random_letters')}   "
          f"chars/page={_fmt(R['target_chars_mean'], '.1f')}   "
          f"font={R['spec'].get('font_size')}")
    print(f"  target tokens (supervised) {_fmt(R['target_tokens_mean'], '.1f')} | "
          f"content {_fmt(R['target_content_tokens_mean'], '.1f')} | "
          f"realized chars/token {_fmt(R['chars_per_token_realized'], '.3f')} | "
          f"scored {R['n_tokens_scored']} tokens "
          f"({R['n_format_tokens_scored']} of them whitespace/format)")
    ce = R["ce_all_tokens"]
    print(f"  CE aligned {ce['aligned']:.4f} | permuted {_fmt(ce['permuted'], '.4f')} | "
          f"blank {ce['blank']:.4f} | floor {_fmt(R['floor_nats'], '.4f')} | "
          f"CE-floor {_fmt(R['ce_vs_floor'], '+.4f')} "
          f"({'BELOW floor = reads' if R['below_floor'] else 'above floor'})")
    dp, db = R["delta_perm_ce_paired"], R["delta_blank_ce_paired"]
    print(f"  Dperm {_fmt(R['delta_perm_rel'] and R['delta_perm_rel'] * 100, '+.2f')}% "
          f"(paired dCE {_fmt(dp['mean'], '+.4f')} +/- {_fmt(dp['se'], '.4f')})   "
          f"Dblank {_fmt(R['delta_blank_rel'] and R['delta_blank_rel'] * 100, '+.2f')}% "
          f"(paired dCE {_fmt(db['mean'], '+.4f')} +/- {_fmt(db['se'], '.4f')})")
    if R["n_truncated_rows"]:
        print(f"  !! {R['n_truncated_rows']} rows hit the --max-length budget; deep "
              f"positions in those rows were CUT. Raise --max-length.")
    if R["n_charlen_mismatch_rows"]:
        print(f"  !! {R['n_charlen_mismatch_rows']} rows: decoded token chars != target "
              f"chars (char offsets suspect on those rows)")

    print("\n  -- reading gain by ANSWER-TOKEN position "
          "(gain = acc_aligned - acc_blank; * = n < min-bin-tokens)")
    print(f"  {'pos':>9}{'tokens':>8}{'rows':>6}{'acc_algn':>10}{'acc_blank':>11}"
          f"{'gain':>9}{'+/-SE':>8}{'95% CI':>18}")
    for b in R["token_bins"]:
        if b["n_tokens"] == 0:
            print(f"  {b['bin']:>9}{0:>8}{'--':>6}{'--':>10}{'--':>11}{'(empty)':>9}"
                  f"{'':>8}{'':>18}")
            continue
        flag = "*" if b["underpowered"] else " "
        ci = f"[{_fmt(b['ci95_lo_pts'], '+.1f')},{_fmt(b['ci95_hi_pts'], '+.1f')}]"
        print(f"  {b['bin']:>9}{b['n_tokens']:>8}{b['n_rows']:>6}"
              f"{b['acc_aligned'] * 100:>9.2f}%{b['acc_blank'] * 100:>10.2f}%"
              f"{b['gain_pts']:>8.2f}{flag}{_fmt(b['se_cluster_pts'], '.2f'):>8}{ci:>18}")

    print("\n  -- reading gain by CHARACTER position")
    print(f"  {'chars':>9}{'tokens':>8}{'rows':>6}{'acc_algn':>10}{'acc_blank':>11}"
          f"{'gain':>9}{'+/-SE':>8}{'95% CI':>18}")
    for b in R["char_bins"]:
        if b["n_tokens"] == 0:
            print(f"  {b['bin']:>9}{0:>8}{'--':>6}{'--':>10}{'--':>11}{'(empty)':>9}"
                  f"{'':>8}{'':>18}")
            continue
        flag = "*" if b["underpowered"] else " "
        ci = f"[{_fmt(b['ci95_lo_pts'], '+.1f')},{_fmt(b['ci95_hi_pts'], '+.1f')}]"
        print(f"  {b['bin']:>9}{b['n_tokens']:>8}{b['n_rows']:>6}"
              f"{b['acc_aligned'] * 100:>9.2f}%{b['acc_blank'] * 100:>10.2f}%"
              f"{b['gain_pts']:>8.2f}{flag}{_fmt(b['se_cluster_pts'], '.2f'):>8}{ci:>18}")

    rs = R["row_sanity"]
    p0, p04 = R["pos0_gain"], R["pos0_4_gain"]
    print("\n  -- PRIMARY: start-of-answer reading gain, and does this rung read at all")
    print(f"     position 0   gain {_fmt(p0['gain_pts'], '+.2f')} +/- "
          f"{_fmt(p0['se_pts'], '.2f')} pts  95% CI "
          f"[{_fmt(p0['ci95'][0], '+.2f')}, {_fmt(p0['ci95'][1], '+.2f')}]  "
          f"(aligned {_fmt(p0['acc_aligned'] and p0['acc_aligned'] * 100, '.1f')}% vs "
          f"blank {_fmt(p0['acc_blank'] and p0['acc_blank'] * 100, '.1f')}%, "
          f"n={p0['n_rows']} rows)  "
          f"{'SIGNIFICANT' if p0['significant'] else 'not significant'}")
    print(f"     positions 0-4 gain {_fmt(p04['gain_pts'], '+.2f')} +/- "
          f"{_fmt(p04['se_pts'], '.2f')} pts  95% CI "
          f"[{_fmt(p04['ci95'][0], '+.2f')}, {_fmt(p04['ci95'][1], '+.2f')}]  "
          f"({p04['n_tokens']} tokens)  "
          f"{'SIGNIFICANT' if p04['significant'] else 'not significant'}")
    print(f"     rows with aligned acc > blank acc: {rs['n_rows_aligned_gt_blank']}/"
          f"{rs['n_rows']} ({_fmt(rs['frac_rows_aligned_gt_blank'] and rs['frac_rows_aligned_gt_blank'] * 100, '.1f')}%)"
          f"  [worse on {rs['n_rows_aligned_lt_blank']}, tied on {rs['n_rows_tied']}]"
          f"  sign-test p = {_fmt(rs['sign_test_p'], '.2e')}"
          f"  mean row gain {_fmt(rs['mean_row_gain_pts'], '+.2f')} pts")

    K = R["K"]
    print(f"\n  -- SECONDARY: K (pre-registered: gain < 50% of the "
          f"0-{K['baseline_window_chars']}-char gain, and stays below)")
    print(f"     baseline gain {_fmt(R['baseline_gain_pts'], '+.2f')} pts "
          f"(95% CI [{_fmt(K['baseline_gain_ci95_pts'][0], '+.2f')}, "
          f"{_fmt(K['baseline_gain_ci95_pts'][1], '+.2f')}], "
          f"{K['baseline_n_tokens']} tokens) "
          f"{'SIGNIFICANT' if K['baseline_gain_significant'] else 'NOT significant'}")
    print(f"     status {K.get('status_detail') or K['status']}   K = "
          f"{K['k_chars'] if K['k_chars'] is not None else 'UNDEFINED'} chars"
          + (f"  (~token {_fmt(K['k_tokens_from_ratio'], '.1f')} by ratio, "
             f"median token {_fmt(K['k_tokens_empirical_median'], '.0f')} empirically)"
             if K["status"] == "ok" else ""))
    if K.get("at_floor"):
        print(f"     !! K IS AT ITS STRUCTURAL FLOOR — {K['floor_warning']}")
    if K.get("k_lower_bound_chars") is not None:
        print(f"     K is CENSORED: all we can say is K >= {K['k_lower_bound_chars']} "
              f"chars (that is where the measurable data ends)")
    if K["status"] == "ok":
        bs = K["bootstrap"]
        print(f"     bootstrap 90% CI [{_fmt(bs['ci_lo_chars'], '.0f')}, "
              f"{_fmt(bs['ci_hi_chars'], '.0f')}] chars from "
              f"{bs['n_resolved']}/{bs['n_boot']} resolved replicates "
              f"(statuses {bs['status_counts']})")
        if bs.get("n_resolved_at_floor"):
            print(f"     !! {bs['n_resolved_at_floor']}/{bs['n_resolved']} resolved "
                  f"replicates sat ON THE FLOOR — a tight CI here measures the "
                  f"floor, not the collapse point")
        if K["k_chars_first_crossing"] is not None and \
                K["k_chars_first_crossing"] != K["k_chars"]:
            print(f"     (first sub-threshold bucket without the stays-below rule: "
                  f"{K['k_chars_first_crossing']} chars)")
    ka = R["K_alt_baseline25"]
    print(f"     [not pre-registered] same statistic with a 0-25-char reference "
          f"window: status {ka.get('status_detail') or ka['status']}, K = "
          f"{ka['k_chars'] if ka['k_chars'] is not None else 'UNDEFINED'} chars"
          + ("  !! also AT ITS FLOOR (25 chars) — reading decays before the 25-char "
             "window ends, so even this variant cannot locate it"
             if ka.get("at_floor") else ""))


# ----------------------------------------------------------------------------
# the discriminator
# ----------------------------------------------------------------------------
def paired_d5_d3(res, arrays_by_rung, rows_by_rung):
    """d5 - d3 reading gain, PAIRED on row (identical targets, font differs)."""

    d3, d5 = "randstr-d3", "randstr-d5"
    if d3 not in arrays_by_rung or d5 not in arrays_by_rung:
        return {"available": False, "reason": "both d3 and d5 must be in --rungs"}
    t3 = [r["target_text"] for r in rows_by_rung[d3]]
    t5 = [r["target_text"] for r in rows_by_rung[d5]]
    n = min(len(t3), len(t5))
    paired = n > 0 and all(t3[i] == t5[i] for i in range(n))

    a3, a5 = arrays_by_rung[d3], arrays_by_rung[d5]
    g3 = a3["a_c"] - a3["b_c"]
    g5 = a5["a_c"] - a5["b_c"]
    m3, se3, _ = cluster_stats(g3, a3["row"])
    m5, se5, _ = cluster_stats(g5, a5["row"])

    out = {"available": True, "paired": bool(paired),
           "n_rows_compared": int(n),
           "gain_d3_pts": _fin(m3 * 100), "gain_d5_pts": _fin(m5 * 100)}
    if paired:
        r3 = row_means(g3, a3["row"])[:n]
        r5 = row_means(g5, a5["row"])[:n]
        d = (r5 - r3) * 100
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else _nan()
        method = "paired-by-row (identical targets, font 40 vs 14)"
    else:
        mean = float((m5 - m3) * 100)
        se = float(math.sqrt(se3 ** 2 + se5 ** 2) * 100)
        method = "UNPAIRED (targets did not match row-by-row)"
    out.update({
        "method": method,
        "diff_pts": _fin(mean), "se_pts": _fin(se),
        "ci95": [_fin(mean - Z95 * se), _fin(mean + Z95 * se)],
        "mde_80pct_power_pts": _fin(Z_MDE * se),
        "significant": bool(math.isfinite(se) and abs(mean) > Z95 * se),
    })
    return out


def posrange_by_row(arrays, n_rows, lo, hi):
    """Per-row mean gain (aligned correct - blank correct) over answer-token
    positions ``[lo, hi)``; NaN for rows with no token in that range."""
    import numpy as np

    m = (arrays["pos"] >= lo) & (arrays["pos"] < hi)
    v = np.full(n_rows, np.nan)
    if not m.any():
        return v
    d = (arrays["a_c"] - arrays["b_c"])[m]
    uniq, inv = np.unique(arrays["row"][m], return_inverse=True)
    v[uniq] = np.bincount(inv, weights=d) / np.bincount(inv)
    return v


def pos0_by_row(arrays, n_rows):
    return posrange_by_row(arrays, n_rows, 0, 1)


def _compare_pos(name_a, name_b, arrays_by_rung, rows_by_rung, lo=0, hi=1):
    """Start-of-answer gain of rung B minus rung A over positions ``[lo, hi)``.

    Pairing is OPPORTUNISTIC and checked per row, never assumed. It holds for
    **d3 vs d5**, which share a spec and therefore an RNG stream — byte-identical
    targets, only the font differs — and there it removes the content variance
    entirely. It does NOT hold across densities: each rung draws a different
    number of letters per row from one shared ``random.Random`` stream, so the
    streams diverge after row 0. When the check fails the comparison falls back
    to two independent means with a pooled SE and says so in ``method``.
    """
    import numpy as np

    if name_a not in arrays_by_rung or name_b not in arrays_by_rung:
        return None
    ta = [r["target_text"] for r in rows_by_rung[name_a]]
    tb = [r["target_text"] for r in rows_by_rung[name_b]]
    n = min(len(ta), len(tb))
    paired = n > 0 and all(
        ta[i].startswith(tb[i]) or tb[i].startswith(ta[i]) for i in range(n)
    )
    va = posrange_by_row(arrays_by_rung[name_a], len(ta), lo, hi)
    vb = posrange_by_row(arrays_by_rung[name_b], len(tb), lo, hi)
    fa, fb = va[np.isfinite(va)], vb[np.isfinite(vb)]
    if len(fa) < 2 or len(fb) < 2:
        return None
    ga, gb = float(fa.mean() * 100), float(fb.mean() * 100)
    sea = float(fa.std(ddof=1) / math.sqrt(len(fa)) * 100)
    seb = float(fb.std(ddof=1) / math.sqrt(len(fb)) * 100)
    if paired:
        d = (vb[:n] - va[:n]) * 100
        d = d[np.isfinite(d)]
        if len(d) < 2:
            return None
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        method = "paired-by-row (identical targets)"
        n_used = int(len(d))
    else:
        mean, se = gb - ga, float(math.hypot(sea, seb))
        method = "UNPAIRED (different rungs draw different targets)"
        n_used = n
    return {
        "a": name_a, "b": name_b, "positions": [lo, hi], "method": method,
        "n_rows": n_used, "gain_a_pts": _fin(ga), "gain_b_pts": _fin(gb),
        "diff_pts": _fin(mean), "se_pts": _fin(se),
        "ci95": [_fin(mean - Z95 * se), _fin(mean + Z95 * se)],
        "mde_80pct_power_pts": _fin(Z_MDE * se),
        "significant": bool(math.isfinite(se) and se > 0 and abs(mean) > Z95 * se),
    }


def cross_rung_pos0(results, arrays_by_rung, rows_by_rung):
    """Position-0 (and 0-4) gain per rung, plus each rung against the SPARSEST."""
    order = [r for r in ALL_RUNGS if r in results]
    out = {"per_rung": {}, "vs_reference": {}, "vs_reference_pos0_4": {},
           "reference": None, "d5_vs_d3": None}
    for r in order:
        out["per_rung"][r] = {
            "chars_per_page": results[r]["target_chars_mean"],
            **{k: results[r]["pos0_gain"][k]
               for k in ("gain_pts", "se_pts", "ci95", "n_rows", "significant")},
            "pos0_4_gain_pts": results[r]["pos0_4_gain"]["gain_pts"],
            "pos0_4_se_pts": results[r]["pos0_4_gain"]["se_pts"],
            "pos0_4_significant": results[r]["pos0_4_gain"]["significant"],
        }
    if not order:
        return out
    ref = min(order, key=lambda r: results[r]["target_chars_mean"] or 0)
    out["reference"] = ref
    for r in order:
        if r == ref:
            continue
        c0 = _compare_pos(ref, r, arrays_by_rung, rows_by_rung, 0, 1)
        c4 = _compare_pos(ref, r, arrays_by_rung, rows_by_rung, 0, 5)
        if c0:
            out["vs_reference"][r] = c0
        if c4:
            out["vs_reference_pos0_4"][r] = c4
    if "randstr-d3" in order and "randstr-d5" in order:
        out["d5_vs_d3"] = _compare_pos("randstr-d3", "randstr-d5",
                                       arrays_by_rung, rows_by_rung, 0, 1)
    return out


def verdict(results, pairwise, pos0_cmp, args):
    """Map the numbers onto the pre-registered branches, or refuse to."""
    lines, checks = [], {}
    order = [r for r in ALL_RUNGS if r in results]
    dens = {r: results[r]["target_chars_mean"] for r in order}

    reads = {r: results[r]["baseline_gain_significant"] for r in order}
    checks["rungs_with_significant_baseline_gain"] = [r for r in order if reads[r]]
    lines.append("1. Does each rung read ANYTHING? (gate to pass before any K is "
                 "interpreted)")
    for r in order:
        R = results[r]
        rs = R["row_sanity"]
        lines.append(
            f"     {r}: 0-{args.baseline_chars}-char gain "
            f"{_fmt(R['baseline_gain_pts'], '+.2f')} pts "
            f"(95% CI [{_fmt(R['K']['baseline_gain_ci95_pts'][0], '+.2f')}, "
            f"{_fmt(R['K']['baseline_gain_ci95_pts'][1], '+.2f')}]) -> "
            f"{'READS' if reads[r] else 'no significant reading'} | rows aligned>blank "
            f"{rs['n_rows_aligned_gt_blank']}/{rs['n_rows']} "
            f"({_fmt(rs['frac_rows_aligned_gt_blank'] and rs['frac_rows_aligned_gt_blank'] * 100, '.0f')}%, "
            f"sign p {_fmt(rs['sign_test_p'], '.1e')}) | "
            f"Dperm {_fmt(R['delta_perm_rel'] and R['delta_perm_rel'] * 100, '+.2f')}% | "
            f"CE-floor {_fmt(R['ce_vs_floor'], '+.3f')}")

    # ---------------- PRIMARY: the position-0 profile across density -----------
    lines.append("2. PRIMARY DISCRIMINATOR — position-0 reading gain vs chars/page")
    lines.append("     (C) scan predicts it STAYS HIGH at every density; "
                 "(A) bandwidth predicts it DEGRADES as density rises.")
    p0 = pos0_cmp["per_rung"]
    prof = sorted(((p0[r]["chars_per_page"] or 0, r) for r in order))
    for d, r in prof:
        e = p0[r]
        lines.append(f"     {r} ({d:.0f} chars/page): pos-0 gain "
                     f"{_fmt(e['gain_pts'], '+.2f')} +/- {_fmt(e['se_pts'], '.2f')} pts "
                     f"95% CI [{_fmt(e['ci95'][0], '+.2f')}, {_fmt(e['ci95'][1], '+.2f')}] "
                     f"n={e['n_rows']} -> "
                     f"{'significant' if e['significant'] else 'NOT significant'}")
    ref = pos0_cmp["reference"]
    for r, c in pos0_cmp["vs_reference"].items():
        lines.append(f"     pos0  {r} - {ref}: {_fmt(c['diff_pts'], '+.2f')} +/- "
                     f"{_fmt(c['se_pts'], '.2f')} pts, 95% CI "
                     f"[{_fmt(c['ci95'][0], '+.2f')}, {_fmt(c['ci95'][1], '+.2f')}], "
                     f"MDE(80%) {_fmt(c['mde_80pct_power_pts'], '.2f')} pts "
                     f"[{c['method']}]")
    for r, c in pos0_cmp.get("vs_reference_pos0_4", {}).items():
        lines.append(f"     pos0-4 {r} - {ref}: {_fmt(c['diff_pts'], '+.2f')} +/- "
                     f"{_fmt(c['se_pts'], '.2f')} pts, 95% CI "
                     f"[{_fmt(c['ci95'][0], '+.2f')}, {_fmt(c['ci95'][1], '+.2f')}], "
                     f"MDE(80%) {_fmt(c['mde_80pct_power_pts'], '.2f')} pts "
                     f"(wider window: 5x the tokens, so more power than pos0 alone)")

    sig0 = [r for r in order if p0[r]["significant"]]
    checks["rungs_with_significant_pos0_gain"] = sig0
    gains0 = [(d, p0[r]["gain_pts"], r) for d, r in prof if p0[r]["gain_pts"] is not None]
    mono_down = all(gains0[i][1] >= gains0[i + 1][1] for i in range(len(gains0) - 1)) \
        if len(gains0) > 1 else False
    densest = prof[-1][1] if prof else None
    drop = pos0_cmp["vs_reference"].get(densest) if densest and densest != ref else None
    drop4 = pos0_cmp.get("vs_reference_pos0_4", {}).get(densest) \
        if densest and densest != ref else None
    drop_sig = bool(drop and drop["significant"] and (drop["diff_pts"] or 0) < 0)
    drop4_sig = bool(drop4 and drop4["significant"] and (drop4["diff_pts"] or 0) < 0)
    checks["pos0_monotone_decreasing_in_density"] = bool(mono_down)
    checks["pos0_densest_vs_sparsest_significant_drop"] = drop_sig
    checks["pos0_4_densest_vs_sparsest_significant_drop"] = drop4_sig

    primary = None
    if not sig0:
        primary = "VOID"
        lines.append("     => NO rung has a significant position-0 gain: the model "
                     "does not read even the START of any page. Neither (A) nor (C) "
                     "is supported — there is no reading to characterise.")
    elif densest in sig0 and not drop_sig and drop4_sig:
        primary = "AMBIGUOUS"
        lines.append("     => CONFLICTING WINDOWS: position 0 alone shows no "
                     "significant drop at the densest rung, but the wider (and more "
                     "powerful) 0-4 window does "
                     f"({_fmt(drop4['diff_pts'], '+.2f')} +/- "
                     f"{_fmt(drop4['se_pts'], '.2f')} pts). The single-token test is "
                     "probably underpowered rather than null, so (C) is NOT "
                     "established. Treat as ambiguous, leaning (A).")
    elif densest in sig0 and not drop_sig:
        primary = "C"
        lines.append(f"     => position-0 gain SURVIVES at the densest rung "
                     f"({densest}: {_fmt(p0[densest]['gain_pts'], '+.2f')} pts) and is "
                     f"not significantly below the sparsest ({ref}: "
                     f"{_fmt(p0[ref]['gain_pts'], '+.2f')} pts)"
                     + (f"; difference {_fmt(drop['diff_pts'], '+.2f')} +/- "
                        f"{_fmt(drop['se_pts'], '.2f')} pts, and this n could only "
                        f"have detected {_fmt(drop['mde_80pct_power_pts'], '.2f')} pts"
                        if drop else "")
                     + (f"; the wider 0-4 window agrees "
                        f"({_fmt(drop4['diff_pts'], '+.2f')} +/- "
                        f"{_fmt(drop4['se_pts'], '.2f')} pts)" if drop4 else "")
                     + " => the start of a dense page IS encoded; what fails is "
                       "keeping place afterwards => (C) SCAN FAILURE.")
    elif drop_sig and mono_down:
        primary = "A"
        lines.append(f"     => position-0 gain DEGRADES monotonically with density "
                     f"({ref} {_fmt(p0[ref]['gain_pts'], '+.2f')} -> {densest} "
                     f"{_fmt(p0[densest]['gain_pts'], '+.2f')} pts, difference "
                     f"{_fmt(drop['diff_pts'], '+.2f')} +/- {_fmt(drop['se_pts'], '.2f')}"
                     f") => even the beginning of a dense page is not encoded "
                     f"=> (A) BANDWIDTH.")
    elif drop_sig:
        primary = "A?"
        lines.append("     => the densest rung's position-0 gain is significantly "
                     "below the sparsest, but the profile is NOT monotone in density "
                     "— consistent with (A) but the ladder is not behaving as a "
                     "clean density axis; inspect the per-rung numbers.")
    else:
        primary = "AMBIGUOUS"
        lines.append("     => mixed: the densest rung has no significant position-0 "
                     "gain, yet the drop from the sparsest is not significant either "
                     "(underpowered). Neither pattern is established.")
    checks["pos0_pattern"] = primary

    # ---------------- SECONDARY: K --------------------------------------------
    measurable = [r for r in order if results[r]["K"]["status"] == "ok"]
    at_floor = [r for r in measurable if results[r]["K"].get("at_floor")]
    checks["rungs_with_measurable_K"] = measurable
    checks["rungs_with_K_at_structural_floor"] = at_floor
    lines.append("3. SECONDARY — K, measurable on: " +
                 (", ".join(measurable) if measurable else "NO RUNG")
                 + (f"   !! AT THE STRUCTURAL FLOOR on: {', '.join(at_floor)} "
                    f"(k_chars == the {args.baseline_chars}-char baseline window = "
                    f"the smallest value the scan can return; NOT a measurement)"
                    if at_floor else ""))
    for r in order:
        K = results[r]["K"]
        why = {
            "no_reading": "the rung shows no positive reading gain to decay from",
            "no_reading_ns": "the 0-%d-char gain is not significantly > 0" % args.baseline_chars,
            "no_baseline_tokens": "too few tokens inside the reference window",
            "censored_short": "the targets end inside/near the reference window",
            "censored_right": "the gain never falls below half the reference gain",
            "ok": "",
        }.get(K["status"], "")
        lines.append(f"     {r}: status {K.get('status_detail') or K['status']}, K = "
                     f"{K['k_chars'] if K['k_chars'] is not None else 'UNDEFINED'} chars"
                     + (" [AT THE STRUCTURAL FLOOR — read as K <= "
                        f"{K['baseline_window_chars']}, not K = {K['k_chars']}]"
                        if K.get("at_floor") else "")
                     + (f", ~token {_fmt(K['k_tokens_from_ratio'], '.1f')}"
                        if K["status"] == "ok" else "")
                     + (f" (K >= {K['k_lower_bound_chars']} chars, censored)"
                        if K.get("k_lower_bound_chars") is not None else "")
                     + (f" — {why}" if why else ""))

    # --- branch: d5 vs d3 (line demux) — independent of the K trend
    lines.append("4. d5 vs d3 (same 400 chars/page, font 40 vs 14 => ~1.0 vs ~2.8 "
                 "text lines per 48px patch):")
    if not pairwise.get("available"):
        lines.append(f"     not evaluated ({pairwise.get('reason')})")
        demux = None
    else:
        p = pairwise
        lines.append(f"     overall gain d3 {_fmt(p['gain_d3_pts'], '+.2f')} pts | "
                     f"d5 {_fmt(p['gain_d5_pts'], '+.2f')} pts | "
                     f"d5-d3 {_fmt(p['diff_pts'], '+.2f')} +/- {_fmt(p['se_pts'], '.2f')} "
                     f"pts, 95% CI [{_fmt(p['ci95'][0], '+.2f')}, "
                     f"{_fmt(p['ci95'][1], '+.2f')}]  [{p['method']}]")
        lines.append(f"     minimum effect this n could detect at 80% power: "
                     f"{_fmt(p['mde_80pct_power_pts'], '.2f')} pts")
        c53 = pos0_cmp.get("d5_vs_d3") or None
        if c53:
            lines.append(f"     position-0 only: d5-d3 {_fmt(c53['diff_pts'], '+.2f')} "
                         f"+/- {_fmt(c53['se_pts'], '.2f')} pts, 95% CI "
                         f"[{_fmt(c53['ci95'][0], '+.2f')}, {_fmt(c53['ci95'][1], '+.2f')}]"
                         f", MDE {_fmt(c53['mde_80pct_power_pts'], '.2f')} pts")
        demux = bool(p["significant"] and p["diff_pts"] and p["diff_pts"] > 0)
        if demux:
            lines.append("     => d5 >> d3: the mechanism is LINES-PER-PATCH DEMUX. "
                         "Fix render geometry (free) before token budget (costly).")
        elif p["significant"]:
            lines.append("     => d5 is significantly WORSE than d3 — not a "
                         "pre-registered branch; treat as a render-geometry side effect.")
        else:
            lines.append(f"     => NULL: d5 == d3 within noise. This is a real null "
                         f"only for effects larger than "
                         f"{_fmt(p['mde_80pct_power_pts'], '.2f')} pts; anything "
                         f"smaller is simply below this run's resolution.")
    checks["d5_gg_d3"] = demux

    # --- branch: K trend across density
    lines.append("5. Does K move with chars/page?  [K-KEYED BRANCHES RETIRED "
                 "2026-07-29 — see the module docstring]")
    trend = None
    if measurable and len(at_floor) == len(measurable):
        # Every K that "resolved" is pinned to the smallest value the statistic can
        # return, so a flat trend through them says nothing about density. This is
        # the defect that produced an identical 48.4913-token K at soft-token
        # budgets 280 AND 560 for rungs whose real reading differed 2.8x.
        trend = "VOID_K_AT_FLOOR"
        lines.append(f"     => VOID: every resolved K ({', '.join(at_floor)}) sits ON "
                     f"THE STRUCTURAL FLOOR (k_chars == the {args.baseline_chars}-char "
                     f"baseline window). A trend through floored values is not a "
                     f"measurement of anything — it would report 'K is CONSTANT "
                     f"across density => (C)' no matter what the model does. The "
                     f"call rests on the position-0 profile; to revive K, shrink "
                     f"--baseline-chars (K_alt_baseline25) so the scan can resolve "
                     f"below the window.")
    elif len(measurable) < 2:
        lines.append(f"     UNDECIDABLE / UNDERPOWERED: K resolved on "
                     f"{len(measurable)} rung(s) "
                     f"({', '.join(measurable) if measurable else 'none'}), and the "
                     f"pre-registered discriminator is a comparison ACROSS rungs. "
                     f"A trend cannot be drawn through one point — falling back to "
                     f"the position-0 profile above.")
    else:
        pts = sorted(((dens[r], results[r]["K"]["k_chars"], r) for r in measurable))
        lines.append("     " + " | ".join(
            f"{r} {d:.0f} chars/page -> K {k} chars "
            f"[boot {_fmt(results[r]['K']['bootstrap']['ci_lo_chars'], '.0f')}, "
            f"{_fmt(results[r]['K']['bootstrap']['ci_hi_chars'], '.0f')}]"
            for d, k, r in pts))
        lo_r, hi_r = pts[0][2], pts[-1][2]
        lo_K, hi_K = results[lo_r]["K"], results[hi_r]["K"]
        floored_extremes = [r for r in (lo_r, hi_r) if results[r]["K"].get("at_floor")]
        lo_ci = (lo_K["bootstrap"]["ci_lo_chars"], lo_K["bootstrap"]["ci_hi_chars"])
        hi_ci = (hi_K["bootstrap"]["ci_lo_chars"], hi_K["bootstrap"]["ci_hi_chars"])
        disjoint = (None not in lo_ci and None not in hi_ci
                    and (hi_ci[1] < lo_ci[0] or lo_ci[1] < hi_ci[0]))
        decreasing = all(pts[i][1] >= pts[i + 1][1] for i in range(len(pts) - 1))
        checks["K_monotone_decreasing_in_density"] = decreasing
        checks["K_extremes_bootstrap_CIs_disjoint"] = bool(disjoint)
        if floored_extremes:
            # a floored K cannot move DOWN, so comparing it with a free K biases the
            # trend toward "constant" (=> a spurious (C)); refuse instead.
            trend = "VOID_K_AT_FLOOR"
            lines.append(f"     => VOID: the extreme rung(s) "
                         f"{', '.join(floored_extremes)} report a K ON THE STRUCTURAL "
                         f"FLOOR ({args.baseline_chars} chars), which cannot move "
                         f"downward, so any comparison against them is biased toward "
                         f"'K is constant' => a spurious (C). No K branch is named; "
                         f"the call rests on the position-0 profile.")
        elif decreasing and disjoint and pts[-1][1] < pts[0][1]:
            trend = "A"
            lines.append(f"     => K SHRINKS with density ({lo_r} K={pts[0][1]} -> "
                         f"{hi_r} K={pts[-1][1]}), bootstrap CIs disjoint "
                         f"=> (A) BANDWIDTH => H17 (raise the soft-token budget) is the fix.")
        elif not disjoint:
            trend = "C?"
            lines.append(f"     => K is CONSTANT across density within noise "
                         f"({lo_r} K={pts[0][1]} vs {hi_r} K={pts[-1][1]}, bootstrap "
                         f"CIs overlap) => the collapse point does not care how much "
                         f"text is on the page => (C) SCAN FAILURE => H18; more soft "
                         f"tokens will not help.")
        else:
            trend = "AMBIGUOUS"
            lines.append("     => K moves but not monotonically in density; the "
                         "pre-registered mapping does not cover this shape.")
    checks["K_trend"] = trend

    lines.append("6. K >= 400 chars anywhere (capacity not binding at mixture density)?"
                 + ("   [UNANSWERABLE: the resolved Ks are floored at "
                    f"{args.baseline_chars} chars, so 'K < 400' here is a property of "
                    "the reference window, not of capacity]" if at_floor else ""))
    ge400 = [r for r in measurable if results[r]["K"]["k_chars"] >= 400]
    ge400_lb = [r for r in order
                if results[r]["K"].get("k_lower_bound_chars") is not None
                and results[r]["K"]["k_lower_bound_chars"] >= 400
                and results[r]["baseline_gain_significant"]]
    checks["rungs_with_K_ge_400"] = ge400
    lines.append("     " + (", ".join(f"{r} K={results[r]['K']['k_chars']}" for r in ge400)
                            if ge400 else
                            "no rung reaches K >= 400 chars (or K is unmeasurable there)")
                 + (f"; censored-but-reading rungs whose lower bound already exceeds "
                    f"400 chars: {', '.join(ge400_lb)}" if ge400_lb else ""))

    # --- final call. PRIMARY = the position-0 profile; K is corroboration.
    k_says = {"A": "A", "C?": "C"}.get(trend)
    prim_says = {"A": "A", "A?": "A", "C": "C"}.get(primary)
    conflict = bool(k_says and prim_says and k_says != prim_says)
    checks["primary_vs_K_conflict"] = conflict

    if primary == "VOID":
        final = ("AMBIGUOUS / VOID — no rung reads even at answer position 0, so "
                 "there is no reading to characterise: K is undefined everywhere and "
                 "this run cannot separate (A) from (C). The checkpoint fails before "
                 "the question the ladder asks. (Check the training run actually "
                 "converged and that the probe's soft-token budget matches it.)")
    elif conflict:
        final = (f"AMBIGUOUS / CONFLICT — the position-0 profile points to ({prim_says}) "
                 f"but the K trend points to ({k_says}). Do not pick one; the two "
                 f"statistics disagree and the run does not settle the question.")
    elif prim_says == "C":
        final = ("(C) SCAN FAILURE — the start of the page is read at every density; "
                 "what fails is keeping place afterwards. H18 becomes the main line; "
                 "more soft tokens will not help."
                 + (" Corroborated by K: the collapse point is invariant to density."
                    if k_says == "C" else
                    " NOTE: K could not corroborate this (see step 5) — the call "
                    "rests on the position-0 profile alone."))
    elif prim_says == "A":
        final = ("(A) BANDWIDTH — reading degrades with chars/page even at the very "
                 "start of the answer. H17 (raise the soft-token budget) is the "
                 "indicated fix."
                 + (" Corroborated by K shrinking with density." if k_says == "A" else
                    " NOTE: K could not corroborate this (see step 5) — the call "
                    "rests on the position-0 profile alone."))
    elif demux:
        final = ("LINE-DEMUX — neither the position-0 profile nor K resolves (A) vs "
                 "(C), but d5 >> d3 at equal density: fix render geometry first.")
    else:
        final = ("AMBIGUOUS — the position-0 profile is inconclusive"
                 + (" and K resolved on fewer than 2 rungs"
                    if len(measurable) < 2 else
                    " and the K pattern matches no pre-registered branch")
                 + "; see the checks above. Most likely cause: the run is "
                   "underpowered at this n / this checkpoint reads too little.")
    checks["final"] = final

    caveats = [
        "K IS STRUCTURALLY FLOORED. k_from_buckets scans buckets strictly PAST the "
        f"{args.baseline_chars}-char reference window, so k_chars >= "
        f"{args.baseline_chars} BY CONSTRUCTION. Every K this repo has reported sat "
        "on that floor (an identical 48.4913 tokens at soft-token budgets 280 AND "
        "560, for rungs whose real reading differed 2.8x). K['at_floor'] flags it "
        "and the K-keyed branches are RETIRED; a floored K means 'reading had "
        f"already decayed by {args.baseline_chars} chars', i.e. K <= "
        f"{args.baseline_chars}, and nothing more.",
        "AGGREGATE Delta-perm IS LENGTH-DILUTED and is NOT comparable across rungs "
        "of different target length: measured here at +113.84% (17 supervised "
        "tok/row), +14.31% (60), +3.05% (234), +0.63% (932) for the SAME reading. "
        "Any bar of the form 'Delta-perm >= 30%' is above the observed ceiling at "
        "T >= 60 and unreachable by construction at T >= 234. Use pos0_gain / "
        "pos0_4_gain / char_bins.",
        "chars/token is ~constant across ALL rungs (see chars_per_token_realized), "
        "so within a target 'character position' and 'token index' are proportional. "
        "A K that is constant across rungs is therefore equally consistent with (C) a "
        "fixed token-index scan AND with a fixed-ABSOLUTE-CAPACITY reading of (A) "
        "bandwidth. The pre-registered mapping calls it (C); what the data strictly "
        "supports is 'the collapse point is invariant to page density', which rules "
        "out a density-proportional bandwidth account but not a fixed-capacity one. "
        "Extra leverage: if a short rung whose ENTIRE page fits inside the implied "
        "capacity still collapses at the same position, fixed-capacity is excluded.",
        "The position-0 profile is the primary discriminator here BECAUSE K is "
        "undefined on any rung that never reads. It is not itself pre-registered; it "
        "is the pre-registered logic (does the collapse point move with density?) "
        "evaluated at the one position every rung can supply. Its weakness is the "
        "mirror image of K's: it says nothing about what happens deeper in the page, "
        "so a rung that reads position 0 and nothing else scores the same as one "
        "that reads throughout. Read it together with the per-position tables.",
        "The pre-registered 0-100-char reference window is wider than d1's whole "
        "target (~29 chars) and covers ~84% of d2's (~119 chars): K is structurally "
        "UNMEASURABLE on those rungs, not merely null. Only d3/d4/d5 can carry the "
        "trend. K_alt_baseline25 is reported as a non-pre-registered sensitivity.",
        "d1..d4 vary chars/page and target LENGTH together; only d3-vs-d5 varies one "
        "factor (font/lines-per-patch) at fixed density. A K trend across d1..d4 "
        "cannot separate 'denser page' from 'longer target'.",
        "d3 and d5 draw byte-identical targets (same spec, same RNG stream), so "
        "that comparison is run PAIRED and is the sharpest in the ladder. Rungs of "
        "DIFFERENT density do not share targets beyond row 0 — each consumes a "
        "different number of letters from the one shared stream — so cross-density "
        "comparisons are unpaired and correspondingly less powerful; the printed "
        "MDE is the number to judge a null against.",
    ]
    return {"lines": lines, "checks": checks, "final": final, "caveats": caveats}


# ----------------------------------------------------------------------------
# self-test (no torch, no GPU)
# ----------------------------------------------------------------------------
def self_test():
    import numpy as np

    w, base, minc = 25, 100, 50
    nb = 40

    def synth(decay_bucket, base_gain=0.5, per_bucket=200, n_rows=50):
        s = np.zeros((n_rows, nb))
        c = np.zeros((n_rows, nb))
        per_row = per_bucket // n_rows
        for b in range(nb):
            g = base_gain if b < decay_bucket else base_gain * 0.1
            c[:, b] = per_row
            s[:, b] = g * per_row
        return s, c

    s, c = synth(decay_bucket=8)          # gain collapses at bucket 8 => 200 chars
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["status"] == "ok" and r["k_chars"] == 200, r
    assert r["at_floor"] is False and r["status_detail"] == "ok", r
    print(f"  ok  crossing detected at {r['k_chars']} chars (expected 200), "
          f"at_floor False")

    # THE STRUCTURAL FLOOR: the crossing lands on the FIRST eligible bucket, so
    # k_chars == baseline_chars is the smallest value the scan can return. This is
    # what every K in this repo actually was (identical 48.4913 tokens at soft-token
    # budgets 280 and 560), and it used to be reported as a plain status "ok".
    s, c = synth(decay_bucket=4)          # decays exactly at the window edge
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["status"] == "ok" and r["k_chars"] == base, r
    assert r["at_floor"] is True, r
    assert r["status_detail"] == "ok_at_structural_floor", r
    assert "STRUCTURAL FLOOR" in (r["floor_warning"] or ""), r
    print(f"  ok  K == baseline ({base} chars) is flagged at_floor / "
          f"{r['status_detail']} (was silently 'ok')")

    bs = bootstrap_k(s, c, w, base, minc, 50, 0)
    assert bs["n_resolved"] == 50 and bs["n_resolved_at_floor"] == 50, bs
    assert bs["status_counts"].get("ok_at_structural_floor") == 50, bs
    print("  ok  the bootstrap histogram separates ok_at_structural_floor from ok, "
          "so a tight CI on the floor cannot masquerade as a measurement")

    s, c = synth(decay_bucket=nb)         # never decays
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["status"] == "censored_right" and r["k_chars"] is None, r
    print(f"  ok  never-decays -> {r['status']}, K >= {r['k_lower_bound_chars']} chars")

    s, c = synth(decay_bucket=0, base_gain=0.0)   # no reading at all
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["status"] == "no_reading" and r["k_chars"] is None, r
    print(f"  ok  zero baseline gain -> {r['status']}, K is null")

    s, c = synth(decay_bucket=2)          # short target: nothing past the window
    c[:, 4:] = 0
    s[:, 4:] = 0
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["status"] == "censored_short" and r["k_chars"] is None, r
    print(f"  ok  d1-like short target -> {r['status']}, K >= "
          f"{r['k_lower_bound_chars']} chars (undefined, NOT 0)")

    s, c = synth(decay_bucket=8)          # thin buckets must not trigger a crossing
    c[:, 5] = 1
    s[:, 5] = 0.0
    r = k_from_buckets(s.sum(0), c.sum(0), w, base, minc)
    assert r["k_chars"] == 200, r
    print("  ok  an under-populated bucket does not trigger a spurious crossing")

    r = k_from_buckets(np.zeros(nb), np.zeros(nb), w, base, minc)
    assert r["status"] == "no_baseline_tokens"
    print("  ok  empty data -> no_baseline_tokens")

    s, c = synth(decay_bucket=8)
    bs = bootstrap_k(s, c, w, base, minc, 50, 0)
    assert bs["n_resolved"] == 50 and bs["median_chars"] == 200, bs
    print(f"  ok  bootstrap resolved {bs['n_resolved']}/50, median "
          f"{bs['median_chars']} chars")

    # binning: empty bins must be null, gains must be paired diffs
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    rows = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([1.0, 1, 1, 1, 0, 0, 0, 0])
    b = np.zeros(8)
    out = summarize_bins([(0, 4), (4, 8), (8, 12)], x, rows, a, b, b, b, min_count=2)
    assert out[0]["gain_pts"] == 100.0 and out[1]["gain_pts"] == 0.0
    assert out[2]["n_tokens"] == 0 and out[2]["gain_pts"] is None
    print("  ok  empty bins emit null (not 0); gains are paired aligned-blank diffs")

    m, se, n = cluster_stats(np.array([1.0, 1, 0, 0]), np.array([0, 0, 1, 1]))
    assert n == 2 and abs(m - 0.5) < 1e-9 and abs(se - 0.5) < 1e-9
    print("  ok  cluster SE uses per-row means (n_rows, not n_tokens)")

    assert abs(sign_test_p(5, 0) - 0.0625) < 1e-12
    assert sign_test_p(75, 75) == 1.0 and sign_test_p(0, 0) is None
    print("  ok  exact two-sided sign test (5/0 -> p=0.0625; 75/75 -> p=1)")

    # --- position-0 machinery + the verdict decision tree ---------------------
    import argparse as _ap

    def fake_rung(chars, pos0_gain, pos0_se, k_status="censored_short", k=None,
                  boot=(None, None), reads=True, n_rows=150, k_at_floor=False):
        return {
            "target_chars_mean": chars, "target_tokens_mean": chars / 2.06,
            "chars_per_token_realized": 2.06,
            "baseline_gain_pts": pos0_gain, "baseline_gain_significant": reads,
            "ce_all_tokens": {"aligned": 5.0}, "ce_vs_floor": -0.1,
            "delta_perm_rel": 0.05, "delta_blank_rel": 0.05,
            "pos0_gain": {"gain_pts": pos0_gain, "se_pts": pos0_se,
                          "ci95": [pos0_gain - Z95 * pos0_se,
                                   pos0_gain + Z95 * pos0_se],
                          "n_rows": n_rows,
                          "significant": pos0_gain - Z95 * pos0_se > 0,
                          "n_tokens": n_rows, "acc_aligned": 0.5, "acc_blank": 0.1,
                          "status": "ok"},
            "row_sanity": {"n_rows": n_rows, "n_rows_aligned_gt_blank": 100,
                           "n_rows_aligned_lt_blank": 20, "n_rows_tied": 30,
                           "frac_rows_aligned_gt_blank": 100 / n_rows,
                           "sign_test_p": 1e-9, "mean_row_gain_pts": pos0_gain},
            "K": {"status": k_status, "at_floor": k_at_floor,
                  "status_detail": ("ok_at_structural_floor" if k_at_floor
                                    else k_status),
                  "k_chars": k, "k_tokens_from_ratio":
                  (k / 2.06 if k is not None else None),
                  "k_tokens_empirical_median": None,
                  "k_lower_bound_chars": (None if k_status == "ok" else int(chars)),
                  "baseline_gain_ci95_pts": [pos0_gain - Z95 * pos0_se,
                                             pos0_gain + Z95 * pos0_se],
                  "baseline_gain_significant": reads, "baseline_window_chars": 100,
                  "baseline_n_tokens": 1000, "k_chars_first_crossing": k,
                  "bootstrap": {"n_boot": 300, "n_resolved": 300,
                                "ci_lo_chars": boot[0], "ci_hi_chars": boot[1],
                                "median_chars": k, "status_counts": {"ok": 300}}},
        }

    def fake_pos0_cmp(res, diffs):
        out = {"per_rung": {}, "vs_reference": {}, "reference": "randstr-d1",
               "d5_vs_d3": None}
        for r, R in res.items():
            out["per_rung"][r] = {"chars_per_page": R["target_chars_mean"],
                                  **{k: R["pos0_gain"][k] for k in
                                     ("gain_pts", "se_pts", "ci95", "n_rows",
                                      "significant")}}
        for r, (d, se) in diffs.items():
            out["vs_reference"][r] = {
                "a": "randstr-d1", "b": r, "method": "paired-by-row (synthetic)",
                "n_rows": 150, "gain_a_pts": None, "gain_b_pts": None,
                "diff_pts": d, "se_pts": se, "ci95": [d - Z95 * se, d + Z95 * se],
                "mde_80pct_power_pts": Z_MDE * se,
                "significant": abs(d) > Z95 * se}
        return out

    args = _ap.Namespace(baseline_chars=100, min_bin_tokens=50)
    no_pair = {"available": False, "reason": "synthetic"}

    # scenario (C): pos-0 gain survives at the densest rung
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0),
           "randstr-d4": fake_rung(1919, 52.0, 4.0)}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-3.0, 5.0)}), args)
    assert v["checks"]["pos0_pattern"] == "C" and v["final"].startswith("(C)"), v["final"]
    print("  ok  verdict: pos-0 gain survives at the densest rung -> (C) scan failure")

    # scenario (A): pos-0 gain collapses with density
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0),
           "randstr-d4": fake_rung(1919, 2.0, 2.0, reads=False)}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-53.0, 5.0)}), args)
    assert v["checks"]["pos0_pattern"] == "A" and v["final"].startswith("(A)"), v["final"]
    print("  ok  verdict: pos-0 gain collapses with density -> (A) bandwidth")

    # scenario VOID: nothing reads anywhere
    res = {"randstr-d1": fake_rung(29, 0.4, 1.5, reads=False),
           "randstr-d4": fake_rung(1919, 0.2, 1.5, reads=False)}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-0.2, 2.0)}), args)
    assert v["checks"]["pos0_pattern"] == "VOID" and "VOID" in v["final"], v["final"]
    print("  ok  verdict: no rung reads -> VOID (no forced A/C branch)")

    # scenario CONFLICT: pos-0 says (C), K says (A)
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0, k_status="ok", k=400,
                                   boot=(375, 425)),
           "randstr-d4": fake_rung(1919, 52.0, 4.0, k_status="ok", k=50,
                                   boot=(25, 75))}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-3.0, 5.0)}), args)
    assert v["checks"]["primary_vs_K_conflict"] and "CONFLICT" in v["final"], v["final"]
    print("  ok  verdict: pos-0 and K disagreeing -> AMBIGUOUS / CONFLICT")

    # every resolved K on the structural floor -> the K trend must refuse to name a
    # branch (it would otherwise report "K is CONSTANT across density => (C)" from
    # two values that CANNOT differ)
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0, k_status="ok", k=100,
                                   boot=(100, 100), k_at_floor=True),
           "randstr-d4": fake_rung(1919, 52.0, 4.0, k_status="ok", k=100,
                                   boot=(100, 100), k_at_floor=True)}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-3.0, 5.0)}), args)
    assert v["checks"]["K_trend"] == "VOID_K_AT_FLOOR", v["checks"]["K_trend"]
    assert v["checks"]["rungs_with_K_at_structural_floor"] == ["randstr-d1",
                                                              "randstr-d4"]
    assert not v["checks"]["primary_vs_K_conflict"]
    assert any("STRUCTURAL FLOOR" in ln for ln in v["lines"])
    print("  ok  verdict: every resolved K on the structural floor -> K trend VOID "
          "(no spurious '(C) K is constant'), call rests on pos-0")

    # MIXED: one extreme floored, one free -> still refuse (a floored K cannot move
    # down, so the comparison is biased toward "constant" => a spurious (C))
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0, k_status="ok", k=100,
                                   boot=(100, 100), k_at_floor=True),
           "randstr-d4": fake_rung(1919, 52.0, 4.0, k_status="ok", k=400,
                                   boot=(375, 425))}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-3.0, 5.0)}), args)
    assert v["checks"]["K_trend"] == "VOID_K_AT_FLOOR", v["checks"]["K_trend"]
    print("  ok  verdict: ONE floored extreme is enough to void the K trend "
          "(a floor cannot move down, so 'constant' would be an artefact)")

    # only one rung yields a K -> the K branch must declare itself underpowered
    res = {"randstr-d1": fake_rung(29, 55.0, 4.0, k_status="ok", k=25, boot=(25, 50)),
           "randstr-d4": fake_rung(1919, 52.0, 4.0)}
    v = verdict(res, no_pair, fake_pos0_cmp(res, {"randstr-d4": (-3.0, 5.0)}), args)
    assert v["checks"]["K_trend"] is None
    assert any("UNDERPOWERED" in ln for ln in v["lines"])
    print("  ok  verdict: one measurable K -> K branch UNDERPOWERED, falls back to pos-0")

    # pos0_by_row / paired cross-rung comparison
    arr = {"pos": np.array([0, 1, 0, 1]), "row": np.array([0, 0, 1, 1]),
           "a_c": np.array([1.0, 1, 1, 0]), "b_c": np.array([0.0, 0, 0, 0])}
    v0 = pos0_by_row(arr, 2)
    assert list(v0) == [1.0, 1.0]
    res = {"randstr-d1": fake_rung(29, 100.0, 0.0), "randstr-d3": fake_rung(479, 50.0, 0.0)}
    arrays = {"randstr-d1": arr,
              "randstr-d3": {**arr, "a_c": np.array([1.0, 1, 0, 0])}}
    rowsi = {"randstr-d1": [{"target_text": "ab"}, {"target_text": "cd"}],
             "randstr-d3": [{"target_text": "abxx"}, {"target_text": "cdxx"}]}
    c = _compare_pos("randstr-d1", "randstr-d3", arrays, rowsi)
    assert c["method"].startswith("paired") and abs(c["diff_pts"] + 50.0) < 1e-9, c
    rowsi["randstr-d3"] = [{"target_text": "zz"}, {"target_text": "yy"}]
    c = _compare_pos("randstr-d1", "randstr-d3", arrays, rowsi)
    assert c["method"].startswith("UNPAIRED"), c
    print("  ok  cross-rung comparison pairs rows only when the targets match, "
          "else falls back to UNPAIRED")

    # pos-0 says (C) but the wider, more powerful 0-4 window says the dense rung
    # degrades => must NOT be called (C)
    res = {"randstr-d1": fake_rung(29, 55.0, 8.0),
           "randstr-d4": fake_rung(1919, 45.0, 8.0)}
    pc = fake_pos0_cmp(res, {"randstr-d4": (-10.0, 11.0)})
    pc["vs_reference_pos0_4"] = {"randstr-d4": {
        "diff_pts": -12.0, "se_pts": 3.0, "ci95": [-17.9, -6.1],
        "mde_80pct_power_pts": 8.4, "significant": True, "method": "unpaired",
        "n_rows": 150, "a": "randstr-d1", "b": "randstr-d4", "positions": [0, 5],
        "gain_a_pts": 55.0, "gain_b_pts": 43.0}}
    v = verdict(res, no_pair, pc, args)
    assert v["checks"]["pos0_pattern"] == "AMBIGUOUS", v["checks"]["pos0_pattern"]
    assert any("CONFLICTING WINDOWS" in ln for ln in v["lines"])
    print("  ok  verdict: pos-0 null but 0-4 shows a drop -> AMBIGUOUS, not (C)")

    print("\nSELF-TEST PASSED (K logic, bins, bootstrap, cluster SE, sign test, "
          "pos-0 pairing, verdict tree)")


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", help="trained UniViHybridPretrained checkpoint dir")
    ap.add_argument("--rungs", nargs="*", default=ALL_RUNGS,
                    help="rung names or shorthands (d1 .. d5)")
    ap.add_argument("-n", type=int, default=150, help="validation rows per rung")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="per-image soft-token budget; default = whatever the "
                         "checkpoint records (280 pre-H17)")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="collator truncation budget (H13 trained at 2048); raise "
                         "together with --max-soft-tokens")
    ap.add_argument("--char-bin-width", type=int, default=25,
                    help="width of the uniform character grid K is computed on")
    ap.add_argument("--baseline-chars", type=int, default=100,
                    help="pre-registered reference window for K (0-100 chars)")
    ap.add_argument("--min-bin-tokens", type=int, default=50,
                    help="bins under this many tokens are flagged and are ignored "
                         "when scanning for K")
    ap.add_argument("--bootstrap", type=int, default=300,
                    help="row-level bootstrap replicates for the K CI")
    ap.add_argument("--logit-chunk", type=int, default=256,
                    help="positions scored per chunk (caps the float32 logit slice)")
    ap.add_argument("--skip-perm", action="store_true",
                    help="skip the permuted-image condition (drops Dperm, 1/3 faster)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the CPU-only checks on the K/binning logic and exit")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (no model, no GPU)")
        self_test()
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required (or use --self-test)")
    if a.baseline_chars % a.char_bin_width:
        ap.error(f"--baseline-chars ({a.baseline_chars}) must be a multiple of "
                 f"--char-bin-width ({a.char_bin_width}) so the reference window is "
                 f"exactly a whole number of buckets")

    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator
    from univi.hybrid.pretrained import (
        UniViHybridPretrained, build_image_processor, resolve_max_soft_tokens,
    )

    rungs = [r if r.startswith("randstr-") else f"randstr-{r}" for r in a.rungs]
    root = Path(a.data_root)
    manifest = {}
    mpath = root / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    specs = manifest.get("lane_specs", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", a.checkpoint, device)
    model = UniViHybridPretrained.from_pretrained(a.checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    mst = resolve_max_soft_tokens(model, a.max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s); max_length %d",
                mst, getattr(model.config, "max_soft_tokens", "<absent -> 280>"),
                a.max_length)
    col = HybridCollator(tokenizer=tok, image_processor=build_image_processor(mst),
                         image_token_id=model.image_token_id,
                         max_length=a.max_length, response_only=True)
    tokchars = TokenChars(tok)

    results, arrays_by_rung, rows_by_rung = {}, {}, {}
    for rung in rungs:
        vpath = root / rung / "validation"
        if not vpath.exists():
            logger.warning("[skip] %s: no validation split at %s", rung, vpath)
            continue
        floor = None
        fpath = root / rung / "floor.json"
        if fpath.exists():
            floor = json.loads(fpath.read_text()).get("no_reading_floor_nats_per_token")
        ds = load_from_disk(str(vpath))
        ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
        logger.info("[%s] %d rows", rung, len(ds))
        arrays, rows_info, meta = collect_rung(
            make_scorer(model, col, device, a.logit_chunk),
            tokchars, ds, a.max_length, a.skip_perm)
        results[rung] = analyse_rung(arrays, rows_info, meta, floor,
                                     specs.get(rung, {}), a)
        arrays_by_rung[rung] = arrays
        rows_by_rung[rung] = rows_info
        print_rung(rung, results[rung])

    if not results:
        logger.error("no rung produced data")
        return

    pairwise = paired_d5_d3(results, arrays_by_rung, rows_by_rung)
    pos0_cmp = cross_rung_pos0(results, arrays_by_rung, rows_by_rung)
    v = verdict(results, pairwise, pos0_cmp, a)

    print_summary_table(results)
    print(f"\n{'=' * 118}")
    print("DISCRIMINATOR")
    print(f"{'=' * 118}")
    for line in v["lines"]:
        print(line)
    print(f"\n>>> VERDICT: {v['final']}")
    print("\nCAVEATS (read before quoting the verdict):")
    for i, c in enumerate(v["caveats"], 1):
        print(f"  {i}. {c}")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "checkpoint": a.checkpoint, "n_per_rung": a.n, "seed": a.seed,
            "max_soft_tokens": mst, "max_length": a.max_length,
            "char_bin_width": a.char_bin_width, "baseline_chars": a.baseline_chars,
            "min_bin_tokens": a.min_bin_tokens, "bootstrap": a.bootstrap,
            "skip_perm": a.skip_perm, "data_root": a.data_root,
            "token_bins": [[lo, (None if hi >= 10**9 else hi)] for lo, hi in TOKEN_BINS],
        },
        "rungs": results,
        "pos0_profile": pos0_cmp,
        "d5_vs_d3": pairwise,
        "verdict": {"final": v["final"], "checks": v["checks"],
                    "lines": v["lines"], "caveats": v["caveats"]},
    }, indent=2))
    logger.info("Wrote %s", out)


def print_summary_table(results):
    print(f"\n{'=' * 132}")
    print("H13 DENSITY LADDER — SUMMARY")
    print(f"{'=' * 132}")
    hdr = (f"{'rung':<12}{'chars/page':>11}{'tgt tok':>9}{'ch/tok':>8}{'CE':>9}"
           f"{'floor':>9}{'CE-floor':>10}{'Dperm%':>9}{'Dblank%':>9}"
           f"{'pos0 gain':>16}{'rows a>b':>10}{'K(chars)':>13}{'K(tokens)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for rung in [r for r in ALL_RUNGS if r in results]:
        R = results[rung]
        K = R["K"]
        # a floored K prints as "<=100!" — never as a bare number, which is what
        # let an unmeasurable 48.5-token K be quoted as a finding for three rungs.
        kc = (("<=%d!" % K["baseline_window_chars"] if K.get("at_floor")
               else str(K["k_chars"])) if K["k_chars"] is not None
              else (f">={K['k_lower_bound_chars']}"
                    if K.get("k_lower_bound_chars") is not None else K["status"]))
        kt = (_fmt(K["k_tokens_from_ratio"], ".0f") if K["status"] == "ok"
              else "UNDEF")
        p0 = R["pos0_gain"]
        p0s = (f"{_fmt(p0['gain_pts'], '+.1f')}+/-{_fmt(p0['se_pts'], '.1f')}"
               + ("*" if p0["significant"] else " "))
        rs = R["row_sanity"]
        rsf = _fmt(rs["frac_rows_aligned_gt_blank"] and
                   rs["frac_rows_aligned_gt_blank"] * 100, ".0f") + "%"
        print(f"{rung:<12}{_fmt(R['target_chars_mean'], '.0f'):>11}"
              f"{_fmt(R['target_tokens_mean'], '.0f'):>9}"
              f"{_fmt(R['chars_per_token_realized'], '.2f'):>8}"
              f"{R['ce_all_tokens']['aligned']:>9.3f}{_fmt(R['floor_nats'], '.3f'):>9}"
              f"{_fmt(R['ce_vs_floor'], '+.3f'):>10}"
              f"{_fmt(R['delta_perm_rel'] and R['delta_perm_rel'] * 100, '+.2f'):>9}"
              f"{_fmt(R['delta_blank_rel'] and R['delta_blank_rel'] * 100, '+.2f'):>9}"
              f"{p0s:>16}{rsf:>10}{kc:>13}{kt:>11}")
    print("\npos0 gain: acc(aligned)-acc(blank) at answer token 0, +/- cluster SE; "
          "* = 95% CI excludes 0.")
    print("rows a>b : fraction of rows whose aligned accuracy beats blank at all "
          "(sanity gate; see per-rung sign test).")
    print("K(chars) : a number = measured; '<=x!' = AT THE STRUCTURAL FLOOR (the scan "
          "cannot return anything smaller — not a measurement); '>=x' = censored (the "
          "measurable data ends there); a status word = K does not exist for that "
          "rung, and 'UNDEF' in K(tokens) means exactly that — NOT zero.")


if __name__ == "__main__":
    main()
