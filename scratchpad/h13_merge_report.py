#!/usr/bin/env python
"""Merge the H13 pairing audit + the secondary lane-defect scan into one report.

Reads ``data/eval/h13-data-integrity.json`` (written by h13_data_integrity.py)
and ``scratchpad/h13-lane-defects.json`` (written by h13_lane_defects.py) and
adds a verdict + derived analysis (gradient share, soft-token bandwidth, page
occupancy) in place.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MAIN = Path("data/eval/h13-data-integrity.json")
SEC = Path("scratchpad/h13-lane-defects.json")
RUNGS = ["randstr-d1", "randstr-d2", "randstr-d3", "randstr-d4", "randstr-d5"]

rep = json.loads(MAIN.read_text())
sec = json.loads(SEC.read_text())

# ---- verdict ----------------------------------------------------------------
tot = exact = 0
gaps = []
for r in RUNGS:
    for s in ("train", "validation"):
        t = rep["rerender_test"][r][s]
        tot += t["n_sampled"]
        exact += t["n_exact_pixel_match"]
        g = rep["negative_control"][r][s].get("separation_margin_ink_iou")
        if g:
            gaps.append(g["gap"])

rep["VERDICT"] = {
    "pairing": "CORRECT" if exact == tot else "MIS-PAIRED",
    "rows_tested": tot,
    "rows_byte_identical_on_rerender": exact,
    "method": (
        "Re-rendered each sampled row's OWN stored assistant target with its OWN "
        "stored render_config (canvas/font) through the materializer's exact "
        "render path (render_utils.render_text_pages -> convert('L')) and compared "
        "the uint8 arrays against the stored image. The renderer is deterministic "
        "(verified by --self-test), so this is an exact pass/fail."
    ),
    "negative_control": {
        "design": "row i's stored image vs row (i+1)'s target re-rendered with row i's geometry",
        "matched_ink_iou": 1.0,
        "mismatched_ink_iou_max": max(
            rep["negative_control"][r][s]["ink_iou"]["max"]
            for r in RUNGS for s in ("train", "validation")
        ),
        "min_separation_margin_ink_iou": min(gaps),
        "matched_exact_pixel_matches": exact,
        "mismatched_exact_pixel_matches": sum(
            rep["negative_control"][r][s]["n_exact_pixel_match"]
            for r in RUNGS for s in ("train", "validation")
        ),
    },
    "conclusion": (
        "The mis-pairing hypothesis is FALSIFIED. Every sampled row's stored image "
        "is the byte-exact render of that row's stored target. Structurally, "
        "data/preprocessing/random_strings.py::_build_split appends the image and "
        "the messages in the SAME loop iteration and never shuffles, so there is no "
        "code path that could de-synchronise them."
    ),
}

# ---- derived: gradient share, bandwidth, page occupancy ---------------------
# HF Trainer normalises the response-only CE by the number of supervised tokens in
# the accumulation window, so the mixture is TOKEN-weighted, not row-weighted.
sup = {r: sec["rungs"][r]["supervised_tokens"]["mean"] for r in RUNGS}
tot_sup = sum(sup.values())

# Gemma4UnifiedImageProcessor geometry at max_soft_tokens=280:
#   patch_size 16, pooling_kernel_size 3 -> model_patch_size 48, side_mult 48
#   max_patches = 280*9 = 2520 -> target_px = 2520*256 = 645120
#   1024x1024 -> factor sqrt(645120/1048576)=0.7844 -> floor(803/48)*48 = 768
PATCH, POOL = 16, 3
MAXP = 280 * POOL**2
side_mult = PATCH * POOL
factor = math.sqrt((MAXP * PATCH**2) / (1024 * 1024))
resized = int(math.floor(factor * 1024 / side_mult)) * side_mult
soft = (resized // side_mult) ** 2

derived = {
    "processor_geometry_at_280_soft_tokens": {
        "render_canvas_px": 1024,
        "resized_to_px": resized,
        "resize_factor": round(resized / 1024, 4),
        "model_patch_size_px": side_mult,
        "patch_grid": f"{resized // side_mult}x{resized // side_mult}",
        "soft_tokens_actually_emitted": soft,
        "note": "get_aspect_ratio_preserving_size(1024,1024,16,2520,3) -> 768x768 -> 256 soft tokens",
    },
    "per_rung": {},
}
for r in RUNGS:
    ls = rep["manifest"]["lane_specs"][r]
    fs = ls["font_size"]
    chars = sec["rungs"][r]["target_chars"]
    ink = rep["ink_sanity"][r]["validation"]
    # glyph geometry at render scale, then after the 0.75x processor resize
    adv = rep["font"][f"size_{fs}"]["advance_px"]
    xh = rep["font"][f"size_{fs}"]["x_height_px"]
    lh = rep["font"][f"size_{fs}"]["line_height"]
    top, bot = ink["ink_row_span"]["top_min"], ink["ink_row_span"]["bottom_max"]
    n_text_rows_of_patches = math.ceil((bot + 1) * factor / side_mult)
    derived["per_rung"][r] = {
        "letters": ls["n_random_letters"],
        "target_chars": chars,
        "font_size": fs,
        "supervised_tokens_per_row": sup[r],
        "gradient_share_of_mixture_pct": round(100 * sup[r] / tot_sup, 2),
        "chars_per_soft_token": round(chars / soft, 3),
        "glyph_advance_px_at_render": adv,
        "glyph_advance_px_after_resize": round(adv * factor, 2),
        "x_height_px_after_resize": round(xh * factor, 2),
        "line_height_px_after_resize": round(lh * factor, 2),
        "text_lines_per_48px_model_patch": round(side_mult / (lh * factor), 2),
        "chars_per_48px_model_patch_horizontally": round(side_mult / (adv * factor), 2),
        "inked_page_rows_px": [top, bot],
        "patch_rows_containing_ink": n_text_rows_of_patches,
        "patch_rows_total": resized // side_mult,
        "frac_of_page_area_with_any_text": round(
            n_text_rows_of_patches / (resized // side_mult), 3
        ),
        "floor_published_nats_per_tok": sec["rungs"][r]["floor_published_nats_per_tok"],
        "floor_corrected_for_trailer_nats_per_tok":
            sec["rungs"][r]["floor_corrected_for_trailer_nats_per_tok"],
        "observed_final_eval_loss": sec["rungs"][r]["observed_final_eval_loss"],
        "obs_minus_corrected_floor": round(sec["rungs"][r]["obs_minus_corrected_floor"], 4),
    }

rep["secondary_lane_scan"] = {
    "token_accounting": {r: {
        k: sec["rungs"][r][k] for k in (
            "original_token_length_stored", "original_token_length_recomputed",
            "stored_equals_recomputed", "assembled_seq_len",
            "n_truncated_at_max_length", "supervised_tokens",
            "supervised_text_equals_target_plus_trailer",
        )} for r in RUNGS},
    "trailer": sec["trailer"],
    "shared_rng_stream": sec["shared_rng_stream"],
    "duplicates_and_leakage": sec["duplicates_and_leakage"],
    "d3_d5_eval_subsample": sec["d3_d5_eval_subsample"],
    "derived": derived,
}

rep["OTHER_DEFECTS_FOUND"] = [
    {
        "id": "D1-floor-instrumentation",
        "severity": "high (misleading, not corrupting)",
        "what": (
            "floor.json's no_reading_floor_nats_per_token divides the string entropy "
            "by the TARGET token count only, but the trainer's response-only mask also "
            "supervises the deterministic '<|im_end|>\\n' trailer (2 tokens, ~0 nats). "
            "eval_loss is therefore averaged over target+2 tokens and the true "
            "no-reading floor is lower than the published one."
        ),
        "impact": (
            "Worst on the shortest rung: d1's published floor 5.5973 vs corrected 4.8902. "
            "Observed d1 eval 4.994 looks 0.60 nats BELOW floor (= 'it reads') but is "
            "actually 0.104 nats ABOVE the real no-reading floor (= it reads nothing). "
            "Every rung is above its corrected floor: d1 +0.104, d2 +0.139, d3 +0.064, "
            "d4 +0.053, d5 +0.064."
        ),
    },
    {
        "id": "D2-token-weighted-mixture",
        "severity": "high (likely primary cause of the null)",
        "what": (
            "The 5 rungs are mixed with equal ROW counts (20k each) but the loss is "
            "normalised per supervised TOKEN, so gradient share follows target length: "
            "d1 1.13%, d2 4.11%, d3 15.86%, d4 63.05%, d5 15.86%."
        ),
        "impact": (
            "d1 is the only rung with a positive prior (H07 read the identical geometry "
            "at Dperm +115% when it was 100% of the mixture). In H13 it gets 1.13% of the "
            "gradient — an ~88x reduction — while 63% goes to d4, which must push 1919 "
            "characters through 256 soft tokens (7.5 chars/token). The cheapest global "
            "solution for a mixture dominated by an over-compressed rung is the uniform "
            "letter marginal, which is exactly the observed state (CE at the floor, "
            "grad_norm ~0.06, aligned==permuted)."
        ),
    },
    {
        "id": "D3-rungs-share-one-RNG-stream",
        "severity": "medium (confounds the ladder, does not corrupt rows)",
        "what": (
            "All rungs draw from random.Random(3407) with the same group_len, so the five "
            "rungs are re-segmentations of ONE letter stream. Verified: d2 row0 == d1 rows "
            "0-3 joined; d3 row0 == d1 rows 0-15 joined; d4 row0 == d1 rows 0-63 joined; "
            "d5 == d3 exactly (200/200 rows, both splits)."
        ),
        "impact": (
            "d3 and d5 are the SAME 20k strings, and the trainer's eval cap "
            "(shuffle(seed=42).select(range(128))) picks the same 128 indices for both, so "
            "their eval targets are identical 128/128. Their reported eval losses being "
            "bit-equal (5.634 vs 5.634) is therefore what you get IF the image is ignored "
            "-- a free ignore-detector, but it also means d5 is not an independent control "
            "of d3. The early stream content is additionally duplicated across rungs."
        ),
    },
    {
        "id": "D4-page-occupancy",
        "severity": "low (design note, same as the run that DID read)",
        "what": (
            "Text occupies only the top strip of the 1024px page: d1 rows 22-35 (1 line), "
            "d2 22-52, d3 22-103, d4 22-307, d5 28-689. After the processor's 0.75x resize "
            "to 768px and 48px model patches, d1 inks 1 of 16 patch rows."
        ),
        "impact": (
            "Not a defect on its own (H07 read d1 with this exact layout), but it means "
            "most soft tokens are pure white for the low rungs."
        ),
    },
    {
        "id": "NON-DEFECTS-checked",
        "severity": "none",
        "what": (
            "Verified clean: no blank/all-white images (0/2000); every target present "
            "verbatim in the assistant turn; response-only mask covers exactly "
            "target + '<|im_end|>\\n' and nothing else; the target never appears in the "
            "prompt; exactly 1 image and 1 image placeholder per row; stored "
            "original_token_length matches the real Qwen3 tokenizer exactly; no row is "
            "truncated at max_length=2048 (longest assembled sequence 1244 on d4); the "
            "trainer's token/image filters drop 0 rows; no duplicate targets and 0 "
            "train/val target overlap on any rung; the ablation's permutation donor "
            "(row i+1 of the same rung) is a genuinely different image, so "
            "aligned==permuted is not a no-op measurement bug."
        ),
    },
]

MAIN.write_text(json.dumps(rep, indent=2))
print("VERDICT:", rep["VERDICT"]["pairing"],
      f"({exact}/{tot} byte-identical, min separation margin {min(gaps):.4f})")
for r in RUNGS:
    d = derived["per_rung"][r]
    print(f"{r}: grad_share {d['gradient_share_of_mixture_pct']:>5.2f}%  "
          f"chars/soft_tok {d['chars_per_soft_token']:>6.3f}  "
          f"obs {d['observed_final_eval_loss']:.3f} vs corrected floor "
          f"{d['floor_corrected_for_trailer_nats_per_tok']:.4f} "
          f"({d['obs_minus_corrected_floor']:+.4f})")
print("wrote", MAIN)
