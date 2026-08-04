#!/usr/bin/env python
"""H13 density-ladder DATA INTEGRITY audit (CPU-only, no model).

Question under audit: does each row's STORED IMAGE actually encode that row's
STORED TARGET TEXT?  A grounding ablation on this lane found
loss(aligned) == loss(permuted) to 5 sig figs, which is exactly what you get if
images and targets were shuffled independently at materialization time.

Tests
  1. schema        — dump the actual columns/структура of every rung+split.
  2. re-render     — deterministically re-render each sampled row's OWN target
                     with its OWN stored render_config and compare pixels to the
                     stored image.  The renderer (PIL truetype) is deterministic,
                     so a correct pairing must be byte-identical.
  3. negative ctl  — re-render row j's target and compare against row i's image
                     (j != i).  Establishes the metric actually discriminates.
  4. d3 vs d5      — the two rungs are built from the same spec + same RNG seed
                     stream and should have byte-identical targets row-for-row.
  5. ink sanity    — ink-pixel volume per rung vs expected chars/page, plus a
                     per-row ink-vs-target-length correlation and blank-image
                     detection.
  6. messages      — is the target present in the assistant turn, is exactly one
                     image referenced, glyph height estimate.

Usage:
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python scratchpad/h13_data_integrity.py \
      --out data/eval/h13-data-integrity.json --n 25
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python scratchpad/h13_data_integrity.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing.render_utils import render_text_pages  # noqa: E402

ROOT = Path("data/materialized/h13-density-v0")
RUNGS = ["randstr-d1", "randstr-d2", "randstr-d3", "randstr-d4", "randstr-d5"]
SPLITS = ["train", "validation"]


# --------------------------------------------------------------------------
# row accessors
# --------------------------------------------------------------------------
def row_target(row) -> str:
    """Assistant-turn text = the transcription target."""
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            return "".join(c["text"] or "" for c in msg["content"])
    raise KeyError("no assistant turn")


def row_prompt(row) -> str:
    for msg in row["messages"]:
        if msg["role"] == "user":
            return "".join(c["text"] or "" for c in msg["content"])
    raise KeyError("no user turn")


def row_n_image_placeholders(row) -> int:
    n = 0
    for msg in row["messages"]:
        for c in msg["content"]:
            if c["type"] == "image":
                n += 1
    return n


def rerender(target: str, rcfg: dict) -> np.ndarray:
    """Re-run the materializer's exact render path for *target*."""
    pages = render_text_pages(
        target,
        canvas_width=int(rcfg["canvas_width"]),
        canvas_height=int(rcfg["canvas_height"]),
        font_size=int(rcfg["font_size"]),
    )
    if len(pages) != 1:
        raise ValueError(f"re-render produced {len(pages)} pages")
    return np.asarray(pages[0].convert("L"))


def img_array(row) -> np.ndarray:
    im = row["images"][0]
    return np.asarray(im.convert("L"))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel comparison stats between stored (a) and re-rendered (b)."""
    if a.shape != b.shape:
        return {
            "shape_mismatch": True,
            "stored_shape": list(a.shape),
            "rerender_shape": list(b.shape),
            "exact": False,
            "frac_equal": 0.0,
            "mean_abs_diff": None,
            "ink_iou": 0.0,
        }
    ai = a.astype(np.int16)
    bi = b.astype(np.int16)
    diff = np.abs(ai - bi)
    ink_a = a < 128
    ink_b = b < 128
    inter = int(np.logical_and(ink_a, ink_b).sum())
    union = int(np.logical_or(ink_a, ink_b).sum())
    return {
        "shape_mismatch": False,
        "exact": bool(diff.max() == 0),
        "frac_equal": float((diff == 0).mean()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "ink_iou": float(inter / union) if union else 1.0,
        "ink_px_stored": int(ink_a.sum()),
        "ink_px_rerender": int(ink_b.sum()),
    }


def ink_stats(a: np.ndarray) -> dict:
    ink = a < 128
    n = int(ink.sum())
    rows_with_ink = np.where(ink.any(axis=1))[0]
    return {
        "ink_px": n,
        "blank": n == 0,
        "ink_top_row": int(rows_with_ink.min()) if n else -1,
        "ink_bottom_row": int(rows_with_ink.max()) if n else -1,
        "min_pixel": int(a.min()),
    }


# --------------------------------------------------------------------------
def audit(n_per_rung: int, out_path: Path) -> dict:
    from datasets import load_from_disk

    report: dict = {
        "dataset_root": str(ROOT),
        "manifest": json.loads((ROOT / "manifest.json").read_text()),
        "n_sampled_per_rung_per_split": n_per_rung,
        "schema": {},
        "rerender_test": {},
        "negative_control": {},
        "d3_vs_d5_targets": {},
        "ink_sanity": {},
        "messages_check": {},
        "notes": [],
    }

    cache: dict[tuple[str, str], object] = {}

    def get(rung: str, split: str):
        key = (rung, split)
        if key not in cache:
            cache[key] = load_from_disk(str(ROOT / rung / split))
        return cache[key]

    # ---- 1. schema ------------------------------------------------------
    for rung in RUNGS:
        report["schema"][rung] = {}
        for split in SPLITS:
            ds = get(rung, split)
            report["schema"][rung][split] = {
                "num_rows": len(ds),
                "columns": list(ds.column_names),
                "features": str(ds.features),
                "floor": json.loads((ROOT / rung / "floor.json").read_text()),
            }

    # ---- 2/3/5/6 --------------------------------------------------------
    for rung in RUNGS:
        report["rerender_test"][rung] = {}
        report["negative_control"][rung] = {}
        report["ink_sanity"][rung] = {}
        report["messages_check"][rung] = {}
        for split in SPLITS:
            ds = get(rung, split)
            N = len(ds)
            k = min(n_per_rung, N)
            # deterministic evenly-spaced sample over the whole split
            idxs = [int(round(i * (N - 1) / max(k - 1, 1))) for i in range(k)]
            idxs = sorted(set(idxs))

            rows = [ds[i] for i in idxs]
            targets = [row_target(r) for r in rows]
            rcfgs = [json.loads(r["render_config"]) for r in rows]
            stored = [img_array(r) for r in rows]

            # --- 2. matched re-render ---
            matched = []
            for t, rc, st in zip(targets, rcfgs, stored):
                matched.append(compare(st, rerender(t, rc)))
            n_exact = sum(m["exact"] for m in matched)

            # --- 3. negative control: row i image vs row j target (j = i+1 cyclic)
            mismatched = []
            for i in range(len(rows)):
                j = (i + 1) % len(rows)
                if targets[j] == targets[i]:
                    continue
                mismatched.append(compare(stored[i], rerender(targets[j], rcfgs[i])))
            n_exact_neg = sum(m["exact"] for m in mismatched)

            def agg(ms, key):
                vals = [m[key] for m in ms if m.get(key) is not None]
                return {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": float(statistics.fmean(vals)),
                } if vals else None

            report["rerender_test"][rung][split] = {
                "n_sampled": len(rows),
                "sampled_indices": idxs,
                "n_exact_pixel_match": int(n_exact),
                "frac_exact": float(n_exact / len(rows)),
                "frac_equal_pixels": agg(matched, "frac_equal"),
                "mean_abs_diff": agg(matched, "mean_abs_diff"),
                "ink_iou": agg(matched, "ink_iou"),
                "any_shape_mismatch": any(m["shape_mismatch"] for m in matched),
            }
            report["negative_control"][rung][split] = {
                "n_pairs": len(mismatched),
                "n_exact_pixel_match": int(n_exact_neg),
                "frac_equal_pixels": agg(mismatched, "frac_equal"),
                "mean_abs_diff": agg(mismatched, "mean_abs_diff"),
                "ink_iou": agg(mismatched, "ink_iou"),
            }
            # separation margin on ink_iou (the robust metric)
            m_iou = report["rerender_test"][rung][split]["ink_iou"]
            n_iou = report["negative_control"][rung][split]["ink_iou"]
            if m_iou and n_iou:
                report["negative_control"][rung][split]["separation_margin_ink_iou"] = {
                    "matched_min": m_iou["min"],
                    "mismatched_max": n_iou["max"],
                    "gap": m_iou["min"] - n_iou["max"],
                }

            # --- 5. ink sanity ---
            inks = [ink_stats(s) for s in stored]
            ink_px = [x["ink_px"] for x in inks]
            tgt_len = [len(t) for t in targets]
            n_letters = [len(t.replace(" ", "")) for t in targets]
            per_char = [p / max(c, 1) for p, c in zip(ink_px, n_letters)]
            corr = None
            if len(set(ink_px)) > 1 and len(set(tgt_len)) > 1:
                corr = float(np.corrcoef(ink_px, tgt_len)[0, 1])
            report["ink_sanity"][rung][split] = {
                "n_blank_images": int(sum(x["blank"] for x in inks)),
                "ink_px": {"min": min(ink_px), "max": max(ink_px),
                           "mean": float(statistics.fmean(ink_px))},
                "ink_px_per_letter": {"min": min(per_char), "max": max(per_char),
                                      "mean": float(statistics.fmean(per_char))},
                "target_chars": {"min": min(tgt_len), "max": max(tgt_len),
                                 "mean": float(statistics.fmean(tgt_len))},
                "target_letters_mean": float(statistics.fmean(n_letters)),
                "expected_letters_from_manifest":
                    report["manifest"]["lane_specs"][rung]["n_random_letters"],
                "corr_inkpx_vs_targetlen": corr,
                "image_shape": list(stored[0].shape),
                "ink_row_span": {
                    "top_min": min(x["ink_top_row"] for x in inks),
                    "bottom_max": max(x["ink_bottom_row"] for x in inks),
                },
            }

            # --- 6. messages / structure ---
            bad_assistant = [
                i for i, (r, t) in enumerate(zip(rows, targets))
                if t.strip() == "" or t != t.strip()
            ]
            n_imgs = [len(r["images"]) for r in rows]
            placeholders = [row_n_image_placeholders(r) for r in rows]
            charset_ok = all(set(t) <= set("abcdefghijklmnopqrstuvwxyz ") for t in targets)
            report["messages_check"][rung][split] = {
                "prompt_sample": row_prompt(rows[0]),
                "target_sample": targets[0][:120],
                "n_roles": [[m["role"] for m in r["messages"]] for r in rows[:1]][0],
                "images_per_row": {"min": min(n_imgs), "max": max(n_imgs)},
                "image_placeholders_per_row": {"min": min(placeholders),
                                               "max": max(placeholders)},
                "n_empty_or_untrimmed_targets": len(bad_assistant),
                "targets_within_charset": charset_ok,
                "target_appears_in_prompt": any(t in row_prompt(r)
                                                for r, t in zip(rows, targets)),
                "n_duplicate_targets_in_sample": len(targets) - len(set(targets)),
                "row_ids_sample": [r["row_id"] for r in rows[:3]],
                "render_config_sample": rcfgs[0],
            }

    # ---- 4. d3 vs d5 target identity ------------------------------------
    for split in SPLITS:
        d3 = get("randstr-d3", split)
        d5 = get("randstr-d5", split)
        k = min(max(30, n_per_rung), len(d3), len(d5))
        idxs = [int(round(i * (min(len(d3), len(d5)) - 1) / max(k - 1, 1))) for i in range(k)]
        idxs = sorted(set(idxs))
        t3 = [row_target(d3[i]) for i in idxs]
        t5 = [row_target(d5[i]) for i in idxs]
        same = sum(a == b for a, b in zip(t3, t5))
        # off-by-one control: d3[i] vs d5[i+1] should NOT match
        off = sum(row_target(d3[i]) == row_target(d5[min(i + 1, len(d5) - 1)])
                  for i in idxs[:-1])
        report["d3_vs_d5_targets"][split] = {
            "n_compared": len(idxs),
            "n_identical": int(same),
            "frac_identical": float(same / len(idxs)),
            "n_identical_offset_by_one_control": int(off),
            "example_d3": t3[0][:80],
            "example_d5": t5[0][:80],
        }

    # ---- glyph geometry -------------------------------------------------
    from PIL import ImageFont
    from data.preprocessing.render_utils import _find_font

    fp = _find_font(None)
    report["font"] = {"path": fp}
    for fs in sorted({int(json.loads(get(r, "train")[0]["render_config"])["font_size"])
                      for r in RUNGS}):
        f = ImageFont.truetype(fp, max(fs, 14)) if fp else None
        if f is None:
            continue
        asc, desc = f.getmetrics()
        bbox = f.getbbox("x")
        bbox_h = f.getbbox("hgb")
        report["font"][f"size_{fs}"] = {
            "effective_size": max(fs, 14),
            "ascent": asc,
            "descent": desc,
            "line_height": asc + desc,
            "advance_px": float(f.getlength("m")),
            "x_height_px": bbox[3] - bbox[1],
            "cap_plus_desc_px": bbox_h[3] - bbox_h[1],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
def self_test() -> int:
    """Prove the comparison machinery discriminates on synthetic data."""
    rc = {"canvas_width": 1024, "canvas_height": 1024, "font_size": 14}
    a = rerender("hello world abcde", rc)
    a2 = rerender("hello world abcde", rc)
    b = rerender("hello world abcdf", rc)  # ONE letter different
    c = rerender("zzzzz yyyyy xxxxx", rc)

    same = compare(a, a2)
    one_char = compare(a, b)
    diff = compare(a, c)
    print("determinism  (same string twice) exact =", same["exact"])
    print("one-char-diff exact =", one_char["exact"],
          " ink_iou =", round(one_char["ink_iou"], 4))
    print("full-diff     exact =", diff["exact"],
          " ink_iou =", round(diff["ink_iou"], 4))
    ok = same["exact"] and not one_char["exact"] and not diff["exact"]
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/eval/h13-data-integrity.json")
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    rep = audit(a.n, Path(a.out))
    # terse console table
    print(f"{'rung':<12}{'split':<12}{'exact/N':<12}{'match_iou_min':<16}"
          f"{'neg_iou_max':<14}{'gap':<10}{'blank'}")
    for rung in RUNGS:
        for split in SPLITS:
            r = rep["rerender_test"][rung][split]
            n = rep["negative_control"][rung][split]
            sep = n.get("separation_margin_ink_iou", {})
            print(f"{rung:<12}{split:<12}"
                  f"{r['n_exact_pixel_match']}/{r['n_sampled']:<9}"
                  f"{r['ink_iou']['min']:<16.6f}"
                  f"{n['ink_iou']['max']:<14.6f}"
                  f"{sep.get('gap', float('nan')):<10.4f}"
                  f"{rep['ink_sanity'][rung][split]['n_blank_images']}")
    print("d3_vs_d5:", json.dumps(rep["d3_vs_d5_targets"], indent=2)[:600])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
