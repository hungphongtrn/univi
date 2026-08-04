"""Prefix-matched Δperm / Δblank + pure-poisoned-token reading gain.

This produces the two H15 read-outs that **no existing probe in this repo can**,
and it is multi-image safe, which is why the existing single-image ablation cannot
simply be pointed at these splits.

1. **The VOID guard** — `docs/hypothesis/todo/H15-prior-poisoned-text.md:213`:
   *"If prefix-matched Δperm over the first 48 answer tokens is < +10% on any arm,
   that arm did not read at all and none of its position numbers may be
   interpreted."*  Aggregate Δperm cannot stand in: H13 §4 measured it collapsing
   14.31% → 0.63% for **identical** reading purely as targets lengthen, and this
   lane's targets average ~1061 tokens.  Aggregate Δperm is reported here too, but
   only as the context H13 §4 says it is — it decides nothing.
   **The 48 in that sentence is the retired depth** (H13's K, structurally floored),
   so `--prefix-tokens` now defaults to **10**: at 48 a ~5-token effect is diluted
   ~10x, which is how an arm reading at +54.0 pts at position 0 was declared not to
   read.  Pass `--prefix-tokens 48` to reproduce the criterion as written.
2. **Pure-poisoned-token gain** — `…/H15-prior-poisoned-text.md:205-211`: tokens
   lying *entirely* inside poisoned characters, obtained by mapping `poison_mask`
   (CHARACTER indices into the assistant target) onto the decoded length of each
   supervised token.  Prior-proof by construction: without reading pixels no model
   can beat ln 26 = 3.2581 nats or 1/26 = 3.85% accuracy there
   (`h15-deepval-*/floor.json`, metric (1) — the TIGHT one).

Why not `scratchpad/hybrid_pretrained_ablation.py`: it forces `r["images"] =
[image]` while `HybridCollator` indexes `soft_counts` once per `{"type": "image"}`
content part, so a **2-page row raises IndexError**.  Measured: H15's p00 arm has
112/500 two-page validation rows (p15 has 9, p50 has 0) and H20's wide librispeech
validation has 567/2703 multi-page rows.  The donor/blank construction here is
copied from `scratchpad/hybrid_4lane_ablation.py`, which is multi-image correct:
exactly `len(row["images"])` donor images (cycling the donor's list) and one blank
per image at the original size/mode, so the soft-token count is identical across
the three conditions and only image CONTENT varies.

Everything else is borrowed from `scratchpad/h13_analyze.py` (`TokenChars`,
`assistant_text`, `cluster_stats`, the chunked-logit scorer) so the numbers come
from the code that produced H13, and the position bins are imported from
`scratchpad/hybrid_4lane_position_decay.py` so they are the SAME bins as the
primary instrument.

Per-row records are written into the JSON (`rows`), so the cross-arm
row-clustered bootstrap H15's CONFIRMS branch asks for can be run afterwards
without touching the GPU again.

CE and `floor.json`'s LOOSE per-token number decide nothing here (H13 §7).

GPU: loads a checkpoint. Never run while training holds the card.

    HF_HUB_OFFLINE=1 uv run python scratchpad/prefix_dperm_probe.py \
        --checkpoint data/checkpoints/h15-poisoned-p50-v0/final \
        --val-path data/materialized/h15-deepval-p50/poisoned-text/validation \
        -n 150 --max-length 4096 --tag h15-p50 \
        --out data/eval/h15-void-guard-p50.json

    # no GPU, no model: exercise the mapping / statistics / verdict logic
    uv run python scratchpad/prefix_dperm_probe.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h13_analyze import (  # noqa: E402
    Z95,
    TokenChars,
    _fin,
    _nan,
    assistant_text,
    cluster_stats,
)
from hybrid_4lane_position_decay import BINS  # noqa: E402  (identical bins)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("prefix_dperm")

#: H15's VOID guard bar, verbatim: prefix Δperm < +10% ⇒ the arm did not read.
VOID_DPERM_REL = 0.10
#: H15's prefix window as WRITTEN was 48 answer tokens — chosen when the readable
#: depth was believed to be ~48 (H13's K, since shown to be structurally floored at
#: its 100-char baseline window). The position-resolved read-outs put reading in the
#: first ~5-12 answer tokens, so a 48-token window DILUTES a ~5-token effect by
#: ~10x: it declared an arm "did not read" that shows +54.0 pts of reading gain at
#: position 0. The default is therefore 10; ``--prefix-tokens 48`` reproduces H15's
#: window verbatim (and the existing data/eval/h15-void-guard-*.json artifacts).
PREFIX_TOKENS = 10
PREFIX_TOKENS_H15_AS_WRITTEN = 48


def _bin_label(lo, hi):
    return f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"


# ---------------------------------------------------------------------------
# pure logic (self-tested)
# ---------------------------------------------------------------------------
def pure_poisoned_flags(offs, lens, content, poison):
    """(pure, touch) per supervised token.

    ``pure`` = every character the token covers is in ``poison`` (so the token is
    prior-proof: uniform over 26 letters); ``touch`` = at least one is.  Tokens of
    zero decoded length (specials) and non-content tokens (the supervised
    ``<|im_end|>\\n`` trailer) are neither.
    """
    pure, touch = [], []
    for off, ln, is_c in zip(offs, lens, content):
        if not is_c or ln <= 0:
            pure.append(False)
            touch.append(False)
            continue
        idx = range(off, off + ln)
        hits = sum(1 for k in idx if k in poison)
        pure.append(hits == ln)
        touch.append(hits > 0)
    return pure, touch


def rel_delta(rows_a, rows_x):
    """Relative CE delta (x vs aligned) + the PAIRED per-row absolute delta.

    The relative number is the one every doc quotes (``(CE_x − CE_a) / CE_a`` over
    row means); the paired absolute delta is what carries an honest error bar,
    because the two conditions share the row.
    """
    import numpy as np

    a = np.array([x for x in rows_a], dtype=float)
    x = np.array([x for x in rows_x], dtype=float)
    ok = np.isfinite(a) & np.isfinite(x)
    a, x = a[ok], x[ok]
    if len(a) == 0:
        return {"n_rows": 0, "ce_aligned": None, "ce_other": None,
                "rel": None, "paired_nats": None, "se_nats": None, "ci95": None}
    ma, mx = float(a.mean()), float(x.mean())
    d = x - a
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {
        "n_rows": int(len(a)),
        "ce_aligned": ma,
        "ce_other": mx,
        "rel": ((mx - ma) / ma) if ma else None,
        "paired_nats": float(d.mean()),
        "se_nats": _fin(se),
        "ci95": [_fin(float(d.mean()) - Z95 * se), _fin(float(d.mean()) + Z95 * se)]
        if math.isfinite(se) else None,
    }


def d50_from_bins(bins, ref_label="0-0", frac=0.5):
    """H15's D50: the DEEPEST bin whose all-token gain is >= ``frac`` of the
    position-0 gain.  Returns the degenerate cases by name instead of a number —
    a reference gain at or below 0 means the arm never read position 0, and then
    "half of it" is meaningless (H13's ``k_from_buckets`` lesson).
    """
    ref = next((b for b in bins if b["bin"] == ref_label), None)
    if ref is None or ref.get("gain_pts") is None:
        return {"status": "UNDEFINED", "reason": f"no {ref_label} bin"}
    r = ref["gain_pts"]
    if r <= 0:
        return {"status": "UNDEFINED", "reason": "position-0 gain <= 0: the arm did "
                                                "not read at position 0, so a "
                                                "fraction of it is meaningless",
                "pos0_gain_pts": r}
    bar = frac * r
    deepest = None
    for b in bins:
        if b.get("gain_pts") is not None and b["gain_pts"] >= bar:
            deepest = b
    if deepest is None:
        return {"status": "UNDEFINED", "reason": "no bin reaches the bar",
                "pos0_gain_pts": r, "bar_pts": bar}
    return {"status": "ok", "pos0_gain_pts": r, "bar_pts": bar,
            "d50_bin": deepest["bin"], "d50_bin_lo_token": deepest["lo"],
            "d50_bin_hi_token": deepest["hi"], "d50_gain_pts": deepest["gain_pts"]}


def void_verdict(prefix_dperm_rel, bar=VOID_DPERM_REL):
    if prefix_dperm_rel is None:
        return "NO-DATA"
    return "VOID" if prefix_dperm_rel < bar else "PASS"


def summarise_bins(acc):
    """Token-level bin table from the accumulator."""
    out = []
    for (lo, hi), d in zip(BINS, acc):
        if d["n"] == 0:
            continue
        n = d["n"]
        row = {
            "bin": _bin_label(lo, hi), "lo": lo,
            "hi": (None if hi >= 10 ** 9 else hi),
            "n_tokens": n,
            "acc_aligned": d["a_c"] / n, "acc_blank": d["b_c"] / n,
            "acc_permuted": d["p_c"] / n,
            "gain_pts": (d["a_c"] - d["b_c"]) / n * 100,
            "gain_vs_perm_pts": (d["a_c"] - d["p_c"]) / n * 100,
            "ce_aligned": d["a_ce"] / n, "ce_blank": d["b_ce"] / n,
            "ce_permuted": d["p_ce"] / n,
        }
        if d["pure_n"]:
            pn = d["pure_n"]
            row["pure_poisoned"] = {
                "n_tokens": pn,
                "acc_aligned": d["pure_a_c"] / pn,
                "acc_blank": d["pure_b_c"] / pn,
                "acc_permuted": d["pure_p_c"] / pn,
                "gain_pts": (d["pure_a_c"] - d["pure_b_c"]) / pn * 100,
                "ce_aligned": d["pure_a_ce"] / pn,
                "ce_blank": d["pure_b_ce"] / pn,
            }
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# GPU pass
# ---------------------------------------------------------------------------
def make_multi_scorer(model, col, device, logit_chunk):
    """``score(row, images) -> (correct[], ce[], ids[], seq_len)`` over the
    supervised (response-only) positions, for a LIST of images.

    Identical to ``h13_analyze.make_scorer`` except that it accepts the row's full
    image list — the whole reason this file exists.  Labels are deliberately not
    passed to the model (that triggers ``logits_to_keep`` and the logits come back
    empty), and logits are cast to fp32 in chunks so a 4-page row does not
    materialise a full ``[S, 151936]`` fp32 tensor.
    """
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def score(row, images):
        r = {k: row[k] for k in row}
        r["images"] = list(images)
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


def run(a) -> int:
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator, blank_like
    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", a.checkpoint, device)
    model = UniViHybridPretrained.from_pretrained(
        a.checkpoint, dtype=torch.bfloat16).to(device)
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
    score = make_multi_scorer(model, col, device, a.logit_chunk)

    ds = load_from_disk(a.val_path)
    has_mask = a.poison_mask_column in ds.column_names
    if not has_mask:
        logger.warning("no `%s` column in %s — pure-poisoned-token read-out is "
                       "SKIPPED (prefix Δperm is unaffected)",
                       a.poison_mask_column, a.val_path)
    ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
    n = len(ds)
    donor = [(i + 1) % n for i in range(n)]
    logger.info("%d rows from %s (seed %d)", n, a.val_path, a.seed)

    acc = [{k: 0.0 for k in ("a_c", "b_c", "p_c", "a_ce", "b_ce", "p_ce",
                             "pure_a_c", "pure_b_c", "pure_p_c",
                             "pure_a_ce", "pure_b_ce")} for _ in BINS]
    for d in acc:
        d["n"] = 0
        d["pure_n"] = 0

    rows_out = []
    n_trunc = n_mismatch = n_skipped = 0
    for i in range(n):
        row = ds[i]
        imgs = list(row["images"])
        k = len(imgs)
        d_imgs = list(ds[donor[i]]["images"])
        p_imgs = [d_imgs[j % len(d_imgs)] for j in range(k)]
        b_imgs = [blank_like(im) for im in imgs]

        ra = score(row, imgs)
        rp = score(row, p_imgs)
        rb = score(row, b_imgs)
        if ra is None or rp is None or rb is None:
            n_skipped += 1
            continue
        a_c, a_ce, ids, seq_len = ra
        p_c, p_ce, _, _ = rp
        b_c, b_ce, _, _ = rb
        m = min(len(a_c), len(p_c), len(b_c))
        n_trunc += int(seq_len >= a.max_length)

        offs, lens, content, _ = tokchars.offsets(ids[:m])
        text = assistant_text(row)
        content_chars = sum(l for l, c in zip(lens, content) if c)
        # The decoded content tokens must reconstruct the target exactly, or the
        # CHARACTER indices in poison_mask do not line up with these tokens. A
        # silent off-by-one here would fabricate the headline number, so the row
        # is dropped from the pure-poisoned statistics rather than averaged in.
        mapped = content_chars == len(text)
        n_mismatch += int(not mapped)

        if has_mask and mapped:
            poison = set(int(x) for x in (row[a.poison_mask_column] or []))
            pure, _touch = pure_poisoned_flags(offs, lens, content, poison)
        else:
            pure = [False] * m

        # per-row means: the prefix window (H15's VOID guard) and all tokens
        pre = min(a.prefix_tokens, m)
        rec = {
            "n_images": k, "n_supervised_tokens": int(m),
            "n_target_chars": len(text), "truncated": bool(seq_len >= a.max_length),
            "char_map_ok": bool(mapped),
            "prefix_tokens": int(pre),
            "prefix_ce_aligned": float(a_ce[:pre].mean()),
            "prefix_ce_permuted": float(p_ce[:pre].mean()),
            "prefix_ce_blank": float(b_ce[:pre].mean()),
            "prefix_acc_aligned": float(a_c[:pre].mean()),
            "prefix_acc_blank": float(b_c[:pre].mean()),
            "all_ce_aligned": float(a_ce[:m].mean()),
            "all_ce_permuted": float(p_ce[:m].mean()),
            "all_ce_blank": float(b_ce[:m].mean()),
            "all_acc_aligned": float(a_c[:m].mean()),
            "all_acc_blank": float(b_c[:m].mean()),
        }
        # content-only prefix, for lanes whose targets are shorter than the window
        cpre = [p for p in range(m) if content[p]][:a.prefix_tokens]
        if cpre:
            rec["prefix_content_tokens"] = len(cpre)
            rec["prefix_content_ce_aligned"] = float(np.mean([a_ce[p] for p in cpre]))
            rec["prefix_content_ce_permuted"] = float(np.mean([p_ce[p] for p in cpre]))
            rec["prefix_content_ce_blank"] = float(np.mean([b_ce[p] for p in cpre]))
        # per-row, per-bin sums so a cross-arm row-clustered bootstrap is possible
        rec["bins"] = {}
        for p in range(m):
            for bi, (lo, hi) in enumerate(BINS):
                if lo <= p < hi:
                    d = acc[bi]
                    d["a_c"] += float(a_c[p]); d["b_c"] += float(b_c[p])
                    d["p_c"] += float(p_c[p])
                    d["a_ce"] += float(a_ce[p]); d["b_ce"] += float(b_ce[p])
                    d["p_ce"] += float(p_ce[p])
                    d["n"] += 1
                    lbl = _bin_label(lo, hi)
                    br = rec["bins"].setdefault(
                        lbl, {"n": 0, "a_c": 0.0, "b_c": 0.0, "p_c": 0.0,
                              "pure_n": 0, "pure_a_c": 0.0, "pure_b_c": 0.0})
                    br["n"] += 1
                    br["a_c"] += float(a_c[p]); br["b_c"] += float(b_c[p])
                    br["p_c"] += float(p_c[p])
                    if pure[p]:
                        d["pure_n"] += 1
                        d["pure_a_c"] += float(a_c[p]); d["pure_b_c"] += float(b_c[p])
                        d["pure_p_c"] += float(p_c[p])
                        d["pure_a_ce"] += float(a_ce[p])
                        d["pure_b_ce"] += float(b_ce[p])
                        br["pure_n"] += 1
                        br["pure_a_c"] += float(a_c[p])
                        br["pure_b_c"] += float(b_c[p])
                    break
        rec["n_pure_poisoned_tokens"] = int(sum(1 for x in pure if x))
        rows_out.append(rec)
        if (i + 1) % 25 == 0:
            logger.info("  %d/%d rows", i + 1, n)

    if not rows_out:
        logger.error("no row produced data")
        return 1

    bins = summarise_bins(acc)
    prefix_perm = rel_delta([r["prefix_ce_aligned"] for r in rows_out],
                            [r["prefix_ce_permuted"] for r in rows_out])
    prefix_blank = rel_delta([r["prefix_ce_aligned"] for r in rows_out],
                             [r["prefix_ce_blank"] for r in rows_out])
    all_perm = rel_delta([r["all_ce_aligned"] for r in rows_out],
                         [r["all_ce_permuted"] for r in rows_out])
    all_blank = rel_delta([r["all_ce_aligned"] for r in rows_out],
                          [r["all_ce_blank"] for r in rows_out])
    cprefix_perm = None
    if all("prefix_content_ce_aligned" in r for r in rows_out):
        cprefix_perm = rel_delta(
            [r["prefix_content_ce_aligned"] for r in rows_out],
            [r["prefix_content_ce_permuted"] for r in rows_out])

    # pure-poisoned aggregate, cluster SE over rows
    pure_stats = None
    n_pure = sum(r["n_pure_poisoned_tokens"] for r in rows_out)
    if n_pure:
        # One value per ROW (the row's own pure-token gain), so the SE is the
        # honest cluster SE — tokens inside a page share a model state and an
        # image and are not independent.
        gains = []
        for r in rows_out:
            pa = sum(b["pure_a_c"] for b in r["bins"].values())
            pb = sum(b["pure_b_c"] for b in r["bins"].values())
            pn = sum(b["pure_n"] for b in r["bins"].values())
            if pn:
                gains.append((pa - pb) / pn * 100)
        g = np.array(gains, dtype=float)
        se = float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else _nan()
        pure_stats = {
            "n_tokens": int(n_pure),
            "n_rows_with_pure_tokens": int(len(gains)),
            "gain_pts": float(g.mean()),
            "cluster_se_pts": _fin(se),
            "ci95_pts": [_fin(float(g.mean()) - Z95 * se),
                         _fin(float(g.mean()) + Z95 * se)]
            if math.isfinite(se) else None,
            "prior_proof_floor": {
                "nats": math.log(a.poison_charset_size),
                "accuracy": 1.0 / a.poison_charset_size,
                "note": "i.i.d. uniform over the charset ⇒ a Jensen bound that "
                        "holds for ANY predictive distribution; no model beats it "
                        "without reading pixels.",
            },
        }

    verdict = void_verdict(prefix_perm["rel"])
    d50 = d50_from_bins(bins)

    rep = {
        "meta": {
            "checkpoint": a.checkpoint, "val_path": a.val_path, "tag": a.tag,
            "n_rows_requested": a.n, "n_rows_scored": len(rows_out),
            "seed": a.seed, "max_soft_tokens": mst, "max_length": a.max_length,
            "prefix_tokens": a.prefix_tokens,
            "poison_mask_column": a.poison_mask_column if has_mask else None,
            "n_truncated": n_trunc, "n_char_map_mismatch": n_mismatch,
            "n_rows_skipped": n_skipped,
            "mean_images_per_row": sum(r["n_images"] for r in rows_out) / len(rows_out),
            "mean_supervised_tokens": sum(r["n_supervised_tokens"] for r in rows_out)
            / len(rows_out),
        },
        "void_guard": {
            "bar_rel": VOID_DPERM_REL,
            "prefix_delta_perm": prefix_perm,
            "prefix_delta_blank": prefix_blank,
            "prefix_content_delta_perm": cprefix_perm,
            "verdict": verdict,
            "note": "H15:213 — prefix-matched Δperm over the first "
                    f"{a.prefix_tokens} answer tokens < +{VOID_DPERM_REL:.0%} ⇒ the "
                    "arm did not read and NONE of its position numbers may be "
                    "interpreted.",
        },
        "aggregate_context_only": {
            "all_token_delta_perm": all_perm,
            "all_token_delta_blank": all_blank,
            "note": "CONTEXT, NOT A CRITERION. H13 §4: aggregate Δperm falls "
                    "14.31% → 0.63% for IDENTICAL reading as targets lengthen, so "
                    "on a ~1061-token target it measures length, not grounding.",
        },
        "bins": bins,
        "d50": d50,
        "pure_poisoned": pure_stats,
        "rows": rows_out,
    }
    out = Path(a.out or f"data/eval/prefix-dperm-{a.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))

    print(f"\n===== PREFIX-MATCHED Δperm ({a.tag}) =====")
    print(f"checkpoint : {a.checkpoint}")
    print(f"val split  : {a.val_path}   n={len(rows_out)} rows  "
          f"({rep['meta']['mean_images_per_row']:.2f} img/row, "
          f"{rep['meta']['mean_supervised_tokens']:.0f} sup tok/row)")
    print(f"truncated rows {n_trunc}   char-map mismatches {n_mismatch}   "
          f"skipped {n_skipped}")
    pp = prefix_perm
    print(f"\nVOID GUARD (first {a.prefix_tokens} answer tokens, prefix-matched)")
    print(f"  CE aligned {pp['ce_aligned']:.4f} | permuted {pp['ce_other']:.4f}"
          f"  Δperm {pp['rel'] * 100:+.2f}%"
          f"  (paired {pp['paired_nats']:+.4f} ± {pp['se_nats']:.4f} nats)")
    print(f"  CE blank   {prefix_blank['ce_other']:.4f}"
          f"  Δblank {prefix_blank['rel'] * 100:+.2f}%")
    print(f"  >>> {verdict} (bar +{VOID_DPERM_REL * 100:.0f}%)")
    if verdict == "VOID":
        print("  !! NONE of the position numbers below may be interpreted (H15:213).")
    print(f"\nAGGREGATE (context only): Δperm {all_perm['rel'] * 100:+.2f}%  "
          f"Δblank {all_blank['rel'] * 100:+.2f}%")
    print(f"\n{'pos':>10}{'tokens':>9}{'acc_algn':>10}{'acc_blank':>11}{'gain':>8}"
          f"{'ce_algn':>9}{'ce_blank':>10}{'pure_n':>8}{'pure_gain':>11}")
    for r in bins:
        p = r.get("pure_poisoned")
        pure_n = p["n_tokens"] if p else 0
        pure_gain = f"{p['gain_pts']:+.2f}p" if p else "--"
        print(f"{r['bin']:>10}{r['n_tokens']:>9}{r['acc_aligned'] * 100:>9.2f}%"
              f"{r['acc_blank'] * 100:>10.2f}%{r['gain_pts']:>7.2f}p"
              f"{r['ce_aligned']:>9.3f}{r['ce_blank']:>10.3f}"
              f"{pure_n:>8}{pure_gain:>11}")
    print(f"\nD50: {d50}")
    if pure_stats:
        print(f"PURE-POISONED tokens: n={pure_stats['n_tokens']} over "
              f"{pure_stats['n_rows_with_pure_tokens']} rows, gain "
              f"{pure_stats['gain_pts']:+.2f} ± {pure_stats['cluster_se_pts']:.2f} pts "
              f"(CI95 {pure_stats['ci95_pts']}), prior-proof floor "
              f"{pure_stats['prior_proof_floor']['accuracy'] * 100:.2f}% / "
              f"{pure_stats['prior_proof_floor']['nats']:.4f} nats")
    print("\nCE and floor.json decide nothing here (H13 §7).")
    logger.info("Wrote %s", out)
    return 0


# ---------------------------------------------------------------------------
def self_test() -> None:
    # --- poison-mask -> token mapping --------------------------------------
    # tokens "ab" | "cd" | "ef"; poison covers chars 2,3 (all of "cd") and 4 only
    offs, lens, content = [0, 2, 4], [2, 2, 2], [True, True, True]
    pure, touch = pure_poisoned_flags(offs, lens, content, {2, 3, 4})
    assert pure == [False, True, False], pure
    assert touch == [False, True, True], touch
    print("  ok  a token counts as PURE only when EVERY char it covers is poisoned")

    # the supervised trailer (non-content) and zero-length specials never count
    pure, touch = pure_poisoned_flags([0, 2], [2, 0], [True, False], {0, 1, 2, 3})
    assert pure == [True, False] and touch == [True, False]
    print("  ok  non-content / zero-length tokens are excluded from both flags")

    # --- relative vs paired delta -----------------------------------------
    d = rel_delta([1.0, 1.0, 1.0], [1.2, 1.1, 1.3])
    assert abs(d["rel"] - 0.2) < 1e-12, d
    assert abs(d["paired_nats"] - 0.2) < 1e-12, d
    assert d["se_nats"] > 0 and d["ci95"][0] < d["paired_nats"] < d["ci95"][1]
    print("  ok  Δ is relative-to-aligned and the error bar is PAIRED per row")

    d = rel_delta([1.0, float("nan")], [1.2, 5.0])
    assert d["n_rows"] == 1, d
    print("  ok  non-finite rows are dropped, not propagated as nan")

    # --- VOID guard -------------------------------------------------------
    assert void_verdict(0.0999) == "VOID"
    assert void_verdict(0.10) == "PASS"
    assert void_verdict(None) == "NO-DATA"
    print("  ok  VOID fires strictly below +10% (H15:213), NO-DATA when absent")

    # --- D50 --------------------------------------------------------------
    bins = [
        {"bin": "0-0", "lo": 0, "hi": 1, "gain_pts": 30.0},
        {"bin": "1-1", "lo": 1, "hi": 2, "gain_pts": 25.0},
        {"bin": "20-49", "lo": 20, "hi": 50, "gain_pts": 16.0},
        {"bin": "50-99", "lo": 50, "hi": 100, "gain_pts": 4.0},
    ]
    d = d50_from_bins(bins)
    assert d["status"] == "ok" and d["d50_bin"] == "20-49", d
    print("  ok  D50 = deepest bin still at >= 50% of the position-0 gain")

    d = d50_from_bins([{"bin": "0-0", "lo": 0, "hi": 1, "gain_pts": 0.0}])
    assert d["status"] == "UNDEFINED", d
    print("  ok  D50 is UNDEFINED (named, not numbered) when pos-0 gain <= 0")

    # a non-monotone profile takes the DEEPEST qualifying bin, not the first
    bins2 = bins + [{"bin": "200-399", "lo": 200, "hi": 400, "gain_pts": 20.0}]
    assert d50_from_bins(bins2)["d50_bin"] == "200-399"
    print("  ok  D50 takes the deepest qualifying bin even if the profile dips")

    # --- bin table ---------------------------------------------------------
    acc = [{k: 0.0 for k in ("a_c", "b_c", "p_c", "a_ce", "b_ce", "p_ce",
                             "pure_a_c", "pure_b_c", "pure_p_c",
                             "pure_a_ce", "pure_b_ce")} for _ in BINS]
    for d_ in acc:
        d_["n"] = 0
        d_["pure_n"] = 0
    acc[0]["n"] = 2
    acc[0]["a_c"] = 2.0
    acc[0]["b_c"] = 0.0
    acc[0]["pure_n"] = 1
    acc[0]["pure_a_c"] = 1.0
    t = summarise_bins(acc)
    assert len(t) == 1 and abs(t[0]["gain_pts"] - 100.0) < 1e-9, t
    assert t[0]["pure_poisoned"]["gain_pts"] == 100.0, t
    print("  ok  empty bins are omitted; pure-poisoned sub-table only when non-empty")
    print("\nSELF-TEST PASSED (mask mapping, paired deltas, VOID bar, D50, bins)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint")
    ap.add_argument("--val-path", help="load_from_disk path of the validation split")
    ap.add_argument("-n", type=int, default=150)
    # Seed 42 is scratchpad/hybrid_4lane_position_decay.py's DEFAULT, so the two
    # probes score the SAME rows and their numbers are row-comparable. Do not
    # change it without changing both.
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="prefix-dperm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix-tokens", type=int, default=PREFIX_TOKENS,
                    help=f"answer-token window the VOID guard is computed over. "
                         f"Default {PREFIX_TOKENS} (reading lives in the first ~5-12 "
                         f"tokens); pass {PREFIX_TOKENS_H15_AS_WRITTEN} to reproduce "
                         f"H15's window as written — it dilutes a ~5-token effect "
                         f"~10x and voided an arm reading at +54 pts at position 0.")
    ap.add_argument("--poison-mask-column", default="poison_mask")
    ap.add_argument("--poison-charset-size", type=int, default=26)
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="default: whatever the checkpoint records")
    ap.add_argument("--max-length", type=int, default=4096,
                    help="collator truncation budget; 4096 for rendered text, "
                         "8192 for audio (the 2048 default truncates the deep "
                         "positions this probe exists to measure)")
    ap.add_argument("--logit-chunk", type=int, default=256)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        print("SELF-TEST (no model, no GPU)")
        self_test()
        return 0
    if not a.checkpoint or not a.val_path:
        ap.error("--checkpoint and --val-path are required (or use --self-test)")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
