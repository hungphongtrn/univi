"""
Positional decay of reading: teacher-forced reading gain as a function of
ANSWER-TOKEN POSITION.

The greedy decodes showed the model reproduces a page's OPENING words + layout
and then fabricates. If reading is bandwidth-limited (280 soft tokens can't carry
a full page), the aligned-vs-blank accuracy gain should be large at answer
position 0 and decay toward 0. If the gain is flat-but-small across positions,
the model reads a little everywhere and the limit is optimization, not bandwidth.

Reports per position-bin: token accuracy (aligned / blank), gain in points, and
CE (aligned / blank).

TRUNCATION IS THE ONE WAY THIS INSTRUMENT CAN LIE (fixed 2026-07-29)
-------------------------------------------------------------------
``--max-length`` used to default to 2048.  The collator TRUNCATES (it does not
drop) the assembled sequence, and on the ``--val-path`` branch the rows are not
length-filtered at all, so a long row's deep supervised positions simply vanish:
the 200-400 and 400+ bins get silently under-populated by exactly the rows that
should populate them, and nothing in the output said so.  The default is now 4096
(8192 for audio), and every bin carries ``n_tokens_from_truncated_rows`` plus a
loud per-lane warning naming the affected bins.  Pass ``--max-length 2048`` to
reproduce the older artifacts.

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/hybrid_4lane_position_decay.py \
      --checkpoint data/checkpoints/hybrid-4lane-v0/final --lanes fineweb-edu librispeech -n 150
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("position_decay")

SPLIT_ROOT = "data/materialized/univi-3M-v0-split"
BINS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100),
        (100, 200), (200, 400), (400, 10**9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lanes", nargs="*", default=["fineweb-edu", "librispeech", "densefusion", "smoltalk"])
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="4lane-final")
    ap.add_argument("--val-path", default=None,
                    help="Load this split directly (skips the lane/filters path). "
                         "Use for the prior-proof lanes, e.g. "
                         "data/materialized/h3-randstr-v0/random-strings/validation")
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="Per-image soft-token budget. Default: whatever the checkpoint "
                         "records (280 for pre-H17 checkpoints).")
    # 2048 silently truncated the deep positions this probe exists to measure (the
    # collator cuts, it does not drop, and --val-path applies no length filter).
    # 4096 for rendered text, 8192 for audio. 2048 reproduces the old artifacts.
    ap.add_argument("--max-length", type=int, default=4096,
                    help="Collator truncation budget; 4096 for rendered text, 8192 "
                         "for audio. Raise with --max-soft-tokens. At 2048 the deep "
                         "bins (200-400, 400+) are under-populated by truncation.")
    # Truncation budget and ROW-POOL selection used to be the same number, so
    # raising one silently changed which rows were scored. On the lane branch the
    # pool must stay at the run's training max_length (2048 for hybrid-4lane).
    ap.add_argument("--filter-max-length", type=int, default=2048,
                    help="Budget used to SELECT the row pool via the trainer's own "
                         "length filter (lane branch only; --val-path is never "
                         "filtered). Keep it at the run's training max_length.")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from datasets import load_from_disk
    from PIL import Image
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator, blank_like
    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )
    from univi.trainer import _filter_training_images, _filter_training_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniViHybridPretrained.from_pretrained(a.checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    # Process images at the budget the CHECKPOINT was trained at unless overridden.
    mst = resolve_max_soft_tokens(model, a.max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s)", mst,
                getattr(model.config, "max_soft_tokens", "<absent → 280>"))
    col = HybridCollator(
        tokenizer=tok,
        image_processor=build_image_processor(mst),
        image_token_id=model.image_token_id,
        max_length=a.max_length,
        response_only=True,
    )

    @torch.no_grad()
    def per_position(row, images):
        """(correct[per answer pos], ce[per answer pos], truncated) teacher-forced.

        ``truncated`` is True when the assembled sequence hit ``--max-length``: the
        collator CUT it, so this row's deepest supervised positions are missing from
        every bin and the deep bins are under-populated, not empty.
        """
        r = {k: row[k] for k in row}
        r["images"] = images
        batch = col([r])
        labels = batch["labels"]
        seq_len = int(batch["input_ids"].shape[1])
        fwd = {k: (v.to(device) if hasattr(v, "to") else v)
               for k, v in batch.items() if k != "labels"}
        out = model(**fwd)
        logits = out.logits[0].float()[:-1]
        tg = labels[0].to(logits.device)[1:]
        valid = tg != -100
        if int(valid.sum()) == 0:
            return None, None, seq_len >= a.max_length
        sl = logits[valid]
        t = tg[valid]
        correct = (sl.argmax(-1) == t).float().cpu()
        ce = F.cross_entropy(sl, t, reduction="none").cpu()
        return correct, ce, seq_len >= a.max_length

    results = {}
    for lane in a.lanes:
        if a.val_path:
            ds = load_from_disk(a.val_path)
        else:
            ds = load_from_disk(f"{SPLIT_ROOT}/{lane}/validation")
            ds = _filter_training_tokens(_filter_training_images(ds, lane, 4), lane,
                                         a.filter_max_length)
        ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
        # accumulators indexed by bin; `trunc_n` counts tokens contributed by rows
        # the collator CUT, and `trunc_deepest` the deepest position such a row
        # reached — together they say which bins are under-populated.
        acc = {i: {"a_c": 0.0, "b_c": 0.0, "a_ce": 0.0, "b_ce": 0.0, "n": 0,
                   "trunc_n": 0}
               for i in range(len(BINS))}
        n_rows = 0
        n_trunc = 0
        trunc_last_bins = set()
        for i in range(len(ds)):
            row = ds[i]
            imgs = list(row["images"])
            blanks = [blank_like(im)
                      for im in imgs]
            ac, ace, a_trunc = per_position(row, imgs)
            bc, bce, b_trunc = per_position(row, blanks)
            if ac is None or bc is None:
                continue
            m = min(len(ac), len(bc))
            n_rows += 1
            truncated = bool(a_trunc or b_trunc)
            n_trunc += int(truncated)
            for p in range(m):
                for bi, (lo, hi) in enumerate(BINS):
                    if lo <= p < hi:
                        d = acc[bi]
                        d["a_c"] += float(ac[p]); d["b_c"] += float(bc[p])
                        d["a_ce"] += float(ace[p]); d["b_ce"] += float(bce[p])
                        d["n"] += 1
                        if truncated:
                            d["trunc_n"] += 1
                            if p == m - 1:
                                # the row STOPPED here because of the cut: every
                                # deeper bin lost this row entirely
                                trunc_last_bins.add(bi)
                        break
            if (i + 1) % 50 == 0:
                logger.info("  [%s] %d/%d", lane, i + 1, len(ds))

        rows = []
        for bi, (lo, hi) in enumerate(BINS):
            d = acc[bi]
            if d["n"] == 0:
                continue
            n = d["n"]
            rows.append({
                "bin": f"{lo}-{hi-1}" if hi < 10**9 else f"{lo}+",
                "n_tokens": n,
                "n_tokens_from_truncated_rows": d["trunc_n"],
                "acc_aligned": d["a_c"] / n,
                "acc_blank": d["b_c"] / n,
                "gain_pts": (d["a_c"] - d["b_c"]) / n * 100,
                "ce_aligned": d["a_ce"] / n,
                "ce_blank": d["b_ce"] / n,
            })
        cut_after = sorted(
            (f"{BINS[bi][0]}-{BINS[bi][1] - 1}" if BINS[bi][1] < 10**9
             else f"{BINS[bi][0]}+") for bi in trunc_last_bins)
        results[lane] = {
            "n_rows": n_rows, "bins": rows,
            "max_length": a.max_length,
            "filter_max_length": (None if a.val_path else a.filter_max_length),
            "n_rows_truncated_at_max_length": n_trunc,
            "frac_rows_truncated": (n_trunc / n_rows) if n_rows else None,
            "bins_where_a_truncated_row_ran_out": cut_after,
            "row_filter_applied": bool(not a.val_path),
            "truncation_note": (
                "rows hitting max_length are CUT, not dropped: every bin deeper than "
                "the cut loses those rows, so a low gain there may be a population "
                "artefact. Raise --max-length (8192 for audio) and re-check."),
        }

        print(f"\n===== {lane} (n={n_rows} rows, max_length {a.max_length}) =====")
        print(f"{'pos':>10}{'tokens':>9}{'trunc_tok':>11}{'acc_algn':>10}"
              f"{'acc_blank':>11}{'gain':>8}{'ce_algn':>9}{'ce_blank':>10}")
        for r in rows:
            print(f"{r['bin']:>10}{r['n_tokens']:>9}"
                  f"{r['n_tokens_from_truncated_rows']:>11}"
                  f"{r['acc_aligned']*100:>9.2f}%"
                  f"{r['acc_blank']*100:>10.2f}%{r['gain_pts']:>7.2f}p"
                  f"{r['ce_aligned']:>9.3f}{r['ce_blank']:>10.3f}")
        if n_trunc:
            print(f"  !! TRUNCATION: {n_trunc}/{n_rows} rows "
                  f"({100 * n_trunc / max(n_rows, 1):.1f}%) hit --max-length "
                  f"{a.max_length}. Their supervised sequence was CUT"
                  + (f" — it ran out inside bin(s) {', '.join(cut_after)}, so EVERY "
                     f"DEEPER BIN is under-populated by exactly the long rows that "
                     f"should fill it" if cut_after else "")
                  + ". Raise --max-length (8192 for audio) before reading the deep "
                    "bins."
                  + ("  NOTE: --val-path applies NO length filter, so nothing else "
                     "protects you here." if a.val_path else ""))
        else:
            print(f"  ok  no row hit --max-length {a.max_length}: the deep bins are "
                  f"fully populated")

    out = Path(f"data/eval/hybrid-4lane-position-decay-{a.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checkpoint": a.checkpoint, "max_soft_tokens": mst,
         "max_length": a.max_length,
         "filter_max_length": (None if a.val_path else a.filter_max_length),
         "lanes": results},
        indent=2,
    ))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
