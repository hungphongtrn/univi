"""H15 — measure the token-weighted READABLE FRACTION of the poisoned-text lane.

H13's verdict: a run whose supervised tokens fall mostly beyond the readable depth
collapses into the uniform-marginal basin (H13 died at 14.2% readable; H07
succeeded at 100%; H14 was repaired to 75.2% before launch — all three measured at
depth 48).

THE DEPTH IS NOW A FLAG, NOT A CONSTANT (2026-07-29).  ~48 came from H13's K, and K
is structurally floored (see ``SCAN_DEPTH`` below); position-resolved read-outs put
the readable depth at ~5-12 answer tokens.  ``--scan-depth`` defaults to 48 so this
script's existing artifact stays reproducible, but at 48 both the readable fraction
AND the poisoned-in-depth column are upper bounds.

This script measures where H15 sits, using the REAL Qwen3 tokenizer on REAL
fineweb-edu source rows pushed through `poisoned_text.py`'s own target-building
path (`_visible_lines` -> `_poison` -> `_fit_token_budget`).  Rendering is
skipped: it does not affect the target string except through the `max_pages`
shrink guard, which cannot fire when the target is built from the first
`max_pages` pages' wrapped lines (checked separately with --check-pages).

CPU only, no model, no GPU.

  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
      scratchpad/h15_readable_fraction.py -n 200
"""
from __future__ import annotations

import argparse
import json
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

#: DEFAULT ONLY, AND IT IS THE RETIRED NUMBER.  48 came from H13 §4's K
#: (48.8 / 48.5 / 48.5 answer tokens over a 16x density span), and K is now known to
#: be STRUCTURALLY FLOORED: `h13_analyze.k_from_buckets` scans buckets strictly past
#: a 100-char baseline window, so k_chars >= 100 BY CONSTRUCTION — the identical
#: 48.4913 at soft-token budgets 280 AND 560 was that floor, not a measurement.
#: Position-resolved read-outs put the readable depth at ~5-12 answer tokens.
#: Kept at 48 so `data/eval/h15-readable-fraction.json` stays reproducible; use
#: `--scan-depth` for the honest gate. It also slices the POISON MASK (see
#: `_poison_within_depth`), so the "poisoned-in-depth" column moves with it.
SCAN_DEPTH = 48
SCAN_DEPTH_RETIRED = 48
SCAN_DEPTH_MEASURED_RANGE = (5, 12)
TRAILER_TOKENS = 2  # response-only masking also supervises `<|im_end|>\n` (H13 §7)


def build_target(raw: str, index: int, split: str, seed: int, tokenizer, spec: LaneSpec):
    """`poisoned_text._make_row`'s target path, minus rasterisation."""
    import random

    rng = random.Random(f"{seed}|{spec.subset_name}|{split}|{index}")
    lines = _visible_lines(raw, spec)
    if not lines:
        return None
    poisoned, mask = _poison("\n".join(lines), spec.poison_rate, rng)
    plines = poisoned.split("\n")
    if spec.max_target_tokens is not None:
        target, mask, otl = _fit_token_budget(plines, mask, tokenizer, spec.max_target_tokens)
        if not target:
            return None
    else:
        target = poisoned
        otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])
    return target, mask, otl, len(lines)


def summarise(name: str, rows: list[dict], spec: LaneSpec,
              scan_depth: int = SCAN_DEPTH) -> dict:
    tok = [r["tokens"] for r in rows]
    sup = [t + TRAILER_TOKENS for t in tok]
    mean_tok = statistics.fmean(tok)
    # H13/H14 convention: readable = min(1, depth / mean target tokens per row)
    readable_conv = min(1.0, scan_depth / mean_tok)
    # Exact per-row token-weighted version (what the gradient actually sees)
    readable_exact = sum(min(scan_depth, t) for t in tok) / sum(tok)
    readable_sup = sum(min(scan_depth, s) for s in sup) / sum(sup)
    chars = [r["chars"] for r in rows]
    return {
        "name": name,
        "scan_depth": scan_depth,
        "poison_rate": spec.poison_rate,
        "max_pages": spec.max_pages,
        "max_target_tokens": spec.max_target_tokens,
        "font_size": spec.font_size,
        "n_rows": len(rows),
        "target_tokens_mean": mean_tok,
        "target_tokens_p50": statistics.median(tok),
        "target_tokens_min": min(tok),
        "target_tokens_max": max(tok),
        "target_chars_mean": statistics.fmean(chars),
        "chars_per_token": statistics.fmean(chars) / mean_tok,
        "poisoned_chars_mean": statistics.fmean(r["n_poison"] for r in rows),
        "poisoned_within_depth_mean": statistics.fmean(r["poison_in_depth"] for r in rows),
        "rendered_lines_mean": statistics.fmean(r["lines"] for r in rows),
        "readable_frac_h13_convention": readable_conv,
        "readable_frac_token_weighted": readable_exact,
        "readable_frac_incl_trailer": readable_sup,
        "rows_fully_readable": sum(t <= scan_depth for t in tok) / len(tok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--split", default="validation")
    ap.add_argument(
        "--source", default="data/materialized/univi-3M-v0-split/fineweb-edu"
    )
    ap.add_argument("--tokenizer", default="data/checkpoints/encoder-free-v0/best")
    ap.add_argument("--seed", type=int, default=3407 + 10**6)
    ap.add_argument("--out", default="data/eval/h15-readable-fraction.json")
    ap.add_argument("--scan-depth", type=int, default=SCAN_DEPTH,
                    help=f"answer-token depth for the readable fraction AND for the "
                         f"poison-mask slice. Default {SCAN_DEPTH} reproduces the "
                         f"existing artifact but is the RETIRED K floor; the measured "
                         f"readable depth is ~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
                         f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens.")
    a = ap.parse_args()

    depth = a.scan_depth
    if depth == SCAN_DEPTH_RETIRED:
        print(f"WARNING: scan depth {depth} is the RETIRED K floor (H13's K is "
              f"structurally floored at its 100-char baseline window, so every "
              f"reported K WAS that floor). Measured readable depth is "
              f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-{SCAN_DEPTH_MEASURED_RANGE[1]} "
              f"answer tokens, so every fraction below — and the "
              f"poisoned-in-depth column — is an UPPER BOUND. Re-run with "
              f"--scan-depth 5 / 12 before using this as a launch gate.")
    else:
        print(f"scan depth {depth} (the default {SCAN_DEPTH_RETIRED} is the retired "
              f"K floor; the H13/H14/H07 reference points were all measured at 48 "
              f"and are NOT comparable with this number)")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    src = load_from_disk(f"{a.source}/{a.split}").select(range(a.n))
    src = src.select_columns(["messages", "row_id"])
    raws = [r["messages"][1]["content"][0]["text"] for r in src]

    # The candidate ladder: the design as written, plus shorter variants.
    specs = {
        "as-designed (4 pages, no cap)": LaneSpec(max_pages=4, max_target_tokens=None),
        "as-decided (4 pages, cap 1400)": LaneSpec(max_pages=4, max_target_tokens=1400),
        "1 page, no cap": LaneSpec(max_pages=1, max_target_tokens=None),
        "cap 400": LaneSpec(max_pages=4, max_target_tokens=400),
        "cap 200": LaneSpec(max_pages=4, max_target_tokens=200),
        "cap 128": LaneSpec(max_pages=4, max_target_tokens=128),
        "cap 96": LaneSpec(max_pages=4, max_target_tokens=96),
        "cap 64": LaneSpec(max_pages=4, max_target_tokens=64),
        "cap 48": LaneSpec(max_pages=4, max_target_tokens=48),
    }

    results = []
    for name, spec in specs.items():
        rows = []
        for i, raw in enumerate(raws):
            built = build_target(raw, i, a.split, a.seed, tokenizer, spec)
            if built is None:
                continue
            target, mask, otl, n_lines = built
            rows.append(
                {
                    "tokens": otl,
                    "chars": len(target),
                    "n_poison": len(mask),
                    # poison indices are CHARACTER indices; how many land inside
                    # the first SCAN_DEPTH answer tokens?
                    "poison_in_depth": _poison_within_depth(
                        target, mask, tokenizer, depth),
                    "lines": n_lines,
                }
            )
        s = summarise(name, rows, spec, depth)
        results.append(s)
        print(
            f"{name:34s} tok/row {s['target_tokens_mean']:8.1f}  "
            f"chars {s['target_chars_mean']:8.1f}  ch/tok {s['chars_per_token']:4.2f}  "
            f"READABLE {100 * s['readable_frac_token_weighted']:5.1f}%  "
            f"(h13-conv {100 * s['readable_frac_h13_convention']:5.1f}%)  "
            f"poisoned-in-depth {s['poisoned_within_depth_mean']:5.1f}"
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(
            {
                "scan_depth_tokens": depth,
                "scan_depth_is_retired_default": bool(depth == SCAN_DEPTH_RETIRED),
                "scan_depth_note": (
                    f"48 is H13's K, which is structurally floored at its 100-char "
                    f"baseline window; the position-resolved measurement puts the "
                    f"readable depth at ~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
                    f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens. The reference "
                    f"points below were all computed at depth 48."),
                "source": f"{a.source}/{a.split}",
                "tokenizer": a.tokenizer,
                "n_rows": a.n,
                "reference_points": {
                    "H13_died_at": 0.142,
                    "H14_repaired_to": 0.752,
                    "H07_succeeded_at": 1.0,
                    "measured_at_scan_depth": SCAN_DEPTH_RETIRED,
                },
                "specs": results,
            },
            indent=2,
        )
    )
    print(f"wrote {a.out}")


def _poison_within_depth(target: str, mask: list[int], tokenizer,
                         scan_depth: int = SCAN_DEPTH) -> int:
    """Number of poisoned CHARACTERS inside the first ``scan_depth`` answer tokens."""
    if not mask:
        return 0
    enc = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(offs) <= scan_depth:
        return len(mask)
    char_limit = offs[scan_depth - 1][1]
    return sum(1 for i in mask if i < char_limit)


if __name__ == "__main__":
    main()
