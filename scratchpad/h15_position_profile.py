"""H15 — where do the PRIOR-PROOF (poisoned) positions actually land?

Two questions the H15 doc does not answer, both needed to re-express its
criteria after H13:

1. **Power.** The pre-registered criterion scores "poisoned-position reading
   gain at answer positions > 50".  How many scored token positions per row
   actually exist in each position bin?  A criterion with no tokens in its bin
   is unfalsifiable.
2. **Purity.** `poison_mask` is CHARACTER-indexed.  A token straddling a masked
   and an unmasked character is not prior-proof.  What fraction of tokens
   touching a poisoned character are *entirely* inside poisoned characters?

Also recomputes the lane's LOOSE per-token floor under H13 §7's correction
(response-only masking supervises `target + <|im_end|>\n`, i.e. T+2 tokens, not
T), which matters at the short target lengths H13 forces on this lane.

CPU only, no model, no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing.poisoned_text import (  # noqa: E402
    LaneSpec,
    _fit_token_budget,
    _poison,
    _visible_lines,
)

SCAN_DEPTH = 48
BINS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100),
        (100, 200), (200, 400), (400, 10**9)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--source", default="data/materialized/univi-3M-v0-split/fineweb-edu")
    ap.add_argument("--tokenizer", default="data/checkpoints/encoder-free-v0/best")
    ap.add_argument("--seed", type=int, default=3407 + 10**6)
    ap.add_argument("--poison-rate", type=float, default=0.15)
    ap.add_argument("--max-target-tokens", type=int, default=1400)
    ap.add_argument("--out", default="data/eval/h15-position-profile.json")
    a = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    src = load_from_disk(f"{a.source}/{a.split}").select(range(a.n)).select_columns(["messages"])
    spec = LaneSpec(poison_rate=a.poison_rate, max_pages=4,
                    max_target_tokens=a.max_target_tokens)

    bin_pure = [0] * len(BINS)      # tokens fully inside poisoned chars
    bin_touch = [0] * len(BINS)     # tokens touching >=1 poisoned char
    bin_all = [0] * len(BINS)       # all supervised target tokens
    rows_with_pure = [0] * len(BINS)
    per_row_tokens, per_row_poison, floors, floors_corr = [], [], [], []

    for i, row in enumerate(src):
        raw = row["messages"][1]["content"][0]["text"]
        rng = random.Random(f"{a.seed}|{spec.subset_name}|{a.split}|{i}")
        lines = _visible_lines(raw, spec)
        if not lines:
            continue
        poisoned, mask = _poison("\n".join(lines), spec.poison_rate, rng)
        plines = poisoned.split("\n")
        target, mask, otl = _fit_token_budget(plines, mask, tokenizer, spec.max_target_tokens)
        if not target:
            continue
        mset = set(mask)
        enc = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        per_row_tokens.append(len(offs))
        per_row_poison.append(len(mask))
        # LOOSE per-token floor, published vs H13-§7-corrected (T+2 supervised)
        nats = math.log(26) * len(mask)
        floors.append(nats / max(len(offs), 1))
        floors_corr.append(nats / max(len(offs) + 2, 1))
        seen_pure = [False] * len(BINS)
        for t, (s, e) in enumerate(offs):
            b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= t < hi)
            bin_all[b] += 1
            span = range(s, e)
            hit = [c in mset for c in span]
            if any(hit):
                bin_touch[b] += 1
            if hit and all(hit):
                bin_pure[b] += 1
                seen_pure[b] = True
        for b, v in enumerate(seen_pure):
            rows_with_pure[b] += int(v)

    n = len(per_row_tokens)
    print(f"n={n} rows, poison_rate={a.poison_rate}, max_target_tokens={a.max_target_tokens}")
    print(f"target tokens/row: mean {statistics.fmean(per_row_tokens):.1f}  "
          f"poisoned chars/row: mean {statistics.fmean(per_row_poison):.1f}")
    print(f"LOOSE per-token floor: published {statistics.fmean(floors):.4f} nats  "
          f"H13-corrected (T+2) {statistics.fmean(floors_corr):.4f} nats "
          f"({100 * (1 - statistics.fmean(floors_corr) / statistics.fmean(floors)):.2f}% lower)")
    print(f"{'bin':>12} {'all tok/row':>12} {'touch/row':>10} {'PURE/row':>9} "
          f"{'pure frac':>10} {'rows w/ pure':>12}")
    out_bins = []
    for k, (lo, hi) in enumerate(BINS):
        label = f"{lo}-{hi if hi < 10**9 else ''}"
        print(f"{label:>12} {bin_all[k]/n:12.2f} {bin_touch[k]/n:10.2f} {bin_pure[k]/n:9.2f} "
              f"{(bin_pure[k]/bin_touch[k] if bin_touch[k] else 0):10.3f} "
              f"{100*rows_with_pure[k]/n:11.1f}%")
        out_bins.append({
            "bin": label, "tokens_per_row": bin_all[k] / n,
            "poison_touching_per_row": bin_touch[k] / n,
            "poison_pure_per_row": bin_pure[k] / n,
            "pure_frac_of_touching": (bin_pure[k] / bin_touch[k]) if bin_touch[k] else 0.0,
            "pct_rows_with_a_pure_token": 100 * rows_with_pure[k] / n,
        })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "n_rows": n, "poison_rate": a.poison_rate,
        "max_target_tokens": a.max_target_tokens,
        "target_tokens_per_row_mean": statistics.fmean(per_row_tokens),
        "poisoned_chars_per_row_mean": statistics.fmean(per_row_poison),
        "loose_floor_published_nats_per_token": statistics.fmean(floors),
        "loose_floor_corrected_T_plus_2": statistics.fmean(floors_corr),
        "bins": out_bins,
    }, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
