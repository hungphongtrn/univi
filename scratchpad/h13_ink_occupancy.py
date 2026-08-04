"""
H13 / H17 — how much of the soft-token stream actually carries INK?

Both H13's density ladder and H17's "280 soft tokens is the page ceiling" argument
reason about *characters per soft token*, dividing the page's characters by the
**budget** (280).  Two facts make that the wrong denominator:

1. The processor does not emit 280 tokens for a 1024x1024 page.  It resizes to
   768x768 (largest aspect-preserving size whose patch count fits 280*3^2 teacher
   patches and is divisible by ``pooling_kernel_size * patch_size`` = 48) and emits
   a **16x16 = 256** grid of 48px cells.  On the *original* page one cell is 64px.
2. Most of those 256 cells are blank paper.  A randstr page puts its text in the
   first couple of grid ROWS; everything below is white.  The quantity that could
   plausibly bind is therefore **characters per INKED soft token**, which is ~8x
   larger than the budget-denominated figure.

This matters most for H13's d3-vs-d5 rung pair: d5 renders the *byte-identical*
target as d3 at font 40 instead of font 14, and d5 has been numerically identical
to d3 at every eval.  That null means opposite things depending on whether d5
actually spreads its ink over more soft tokens than d3 (⇒ spreading bought
nothing) or not (⇒ the rung never varied the quantity it was designed to vary).
This script measures it.

WHAT "INK" MEANS HERE
  Pages are white paper (255) with black glyphs.  After the processor the pixel
  values are float in [0,1] (do_rescale, no normalize), so *darkness* = 1 - v.
  A pixel is "ink" when darkness > ``--ink-threshold`` (default 8/255 — bicubic
  ringing on a region with no nearby ink is exactly 0, so this only guards
  against float noise).  Because a threshold can silently do all the work, the
  script NEVER reports a bare binary count: it reports the number of inked cells
  at several *per-cell ink-pixel* cutoffs (>=1, >=10, >=50, >=1% of the 2304px
  cell, >=5%), the pooled histogram of per-cell ink-pixel counts, and the total
  page ink mass.  If the >=1 and >=1% counts agree, the threshold is not doing
  hidden work.

THE PIXEL -> PATCH MAPPING IS SELF-TESTED (``--self-test``)
  An off-by-one or an x/y transposition in "soft token m covers pixels
  [y*48,(y+1)*48) x [x*48,(x+1)*48)" would silently fabricate the headline
  number, so the mapping is asserted against synthetic images with ink at known
  coordinates, including a mutation test that a deliberately shifted mapping
  FAILS the same assertion.

CPU ONLY.  Loads no model weights and never touches the GPU (a training run
holds the card and this repo's rule is that training runs SOLO).  Heavy imports
are deferred inside functions per repo convention; ``univi.hybrid.vision`` is
loaded **by file path** because ``univi/__init__.py`` imports unsloth, which
raises without an accelerator.

Usage:
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h13_ink_occupancy.py --self-test
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h13_ink_occupancy.py --rung all -n 30
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h13_ink_occupancy.py --rung d3 --rung d5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h13_ink_occupancy")

DEFAULT_ROOT = "data/materialized/h13-density-v0"
DEFAULT_OUT = "data/eval/h13-ink-occupancy.json"
RUNGS = ("d1", "d2", "d3", "d4", "d5")
#: per-cell ink-pixel cutoffs an "inked cell" is counted at.  2304 = 48*48.
CELL_AREA = 48 * 48
INKED_CUTOFFS = (1, 10, 50, 23, 115)          # 23 = 1% of the cell, 115 = 5%
INKED_CUTOFF_LABELS = ("ge1", "ge10", "ge50", "ge1pct", "ge5pct")
Z95 = 1.959963985


# ---------------------------------------------------------------------------
# module loading (deferred / path-based — see the docstring)
# ---------------------------------------------------------------------------
def load_vision_module():
    """``univi/hybrid/vision.py`` as a standalone module, without importing ``univi``.

    ``univi/__init__.py`` pulls in unsloth, which raises ``NotImplementedError``
    when no torch accelerator is visible — and this script must run with
    ``CUDA_VISIBLE_DEVICES=""``.  ``vision.py`` has no intra-package imports, so
    it loads cleanly on its own.  It IS registered in ``sys.modules`` first
    because ``PreTrainedConfig.__init_subclass__`` calls ``dataclass()``, which
    looks the defining module up by name.
    """
    import importlib.util

    name = "_univi_hybrid_vision_cpu"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "univi" / "hybrid" / "vision.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_processor(max_soft_tokens: int):
    return load_vision_module().Gemma4UnifiedImageProcessor(max_soft_tokens=int(max_soft_tokens))


# ---------------------------------------------------------------------------
# the measurement primitive
# ---------------------------------------------------------------------------
def patch_cells(processor, image):
    """``(cells, positions, n_soft, resized_hw)`` for ONE image.

    ``cells[m]`` is the ``(48, 48)`` grayscale (channel-0) float patch of soft
    token ``m``; ``positions[m] == (x, y)`` is its cell coordinate.  Both come
    straight out of the processor — the patch pixels ARE the merged patch vector
    reshaped, so the ink count needs no reimplementation of the patchifier.  The
    (m -> x, y) correspondence is what :func:`self_test` proves.
    """
    import torch

    out = processor(images=[image], return_tensors="pt")
    n_soft = int(out["num_soft_tokens_per_image"][0])
    pv = out["pixel_values"][0, :n_soft]                     # (n_soft, 48*48*3)
    pos = out["image_position_ids"][0, :n_soft]              # (n_soft, 2) = (x, y)
    side = int(math.isqrt(pv.shape[-1] // 3))
    grid = pv.reshape(n_soft, side, side, 3)
    if not (torch.equal(grid[..., 0], grid[..., 1]) and torch.equal(grid[..., 0], grid[..., 2])):
        # every lane here renders grayscale; an RGB source would need a
        # luminance conversion rather than channel 0.
        grid = grid.mean(-1, keepdim=True).squeeze(-1)
    else:
        grid = grid[..., 0]
    resized_h = int(pos[:, 1].max()) + 1, int(pos[:, 0].max()) + 1
    return grid, pos, n_soft, (resized_h[0] * side, resized_h[1] * side)


def cell_ink(cells, darkness_threshold: float):
    """``(ink_px_per_cell, ink_mass_per_cell)``.

    ``ink_px`` counts pixels darker than the threshold; ``ink_mass`` is the
    integral of darkness over the cell (threshold-free, reported so the binary
    count can be sanity-checked against a continuous quantity).
    """
    dark = 1.0 - cells
    ink_px = (dark > darkness_threshold).flatten(1).sum(1)
    ink_mass = dark.clamp(min=0).flatten(1).sum(1)
    return ink_px, ink_mass


# ---------------------------------------------------------------------------
# self-test of the pixel -> patch mapping
# ---------------------------------------------------------------------------
def _white(size, mode="L"):
    from PIL import Image

    return Image.new(mode, size, 255)


def self_test(max_soft_tokens: int = 280, verbose: bool = True) -> dict:
    """Assert the (soft token m) <-> (cell x, y) <-> (pixel block) mapping.

    Four checks, the first three on a **768x768** input, which the processor
    passes through with NO resize (``get_aspect_ratio_preserving_size(768, 768,
    16, 2520, 3) == (768, 768)``) — so the assertion compares against the
    synthetic source pixels directly and does not depend on reimplementing the
    resize:

    T1 exact raster identity: ``cells[m] == source[y*48:(y+1)*48, x*48:(x+1)*48]``
       for EVERY m (float-exact).  This alone pins the mapping.

    The no-resize side length is budget-dependent (``floor(sqrt(budget*2304)/48)*48``
    — 768 at 280, 1104 at 560, 1584 at 1120), so it is derived, not hard-coded.
    T2 index order: ``m == y * grid_w + x`` (row-major over the cell grid), which
       is what any "token index" reasoning downstream assumes.
    T3 asymmetric ink lands in exactly the expected cells — an x/y transposition
       survives T1 only if the grid were square AND the image symmetric, so the
       probe squares sit at (x != y) positions including two corners.
    T4 MUTATION: the T1 comparison with the mapping shifted one cell right must
       FAIL.  Without this, T1 passing proves nothing about T1 having teeth.
    T5 resize path (1024x1024, the real lane geometry): the ink centroid measured
       in resized pixel coordinates must equal 0.75x the ink centroid in original
       coordinates.  Independent of T1-T4 and catches a transposition or a whole-
       cell offset introduced by the resize.
    """
    import numpy as np
    import torch
    from PIL import ImageDraw

    proc = build_processor(max_soft_tokens)
    res = {"checks": {}, "failures": []}

    def _check(name, ok, detail=""):
        res["checks"][name] = bool(ok)
        if not ok:
            res["failures"].append(f"{name}: {detail}")
        if verbose:
            logger.info("  %-38s %s %s", name, "PASS" if ok else "FAIL", detail)

    # ---- T1-T3: no-resize path -------------------------------------------
    # The side the processor maps to ITSELF at this budget (so the raster
    # comparison is against the synthetic source, not a reimplemented resize).
    side = int(math.isqrt(max_soft_tokens * CELL_AREA) // 48) * 48
    g = side // 48
    img = _white((side, side))
    d = ImageDraw.Draw(img)
    # (cell_x, cell_y) -> a 10x10 black square well inside the cell; asymmetric
    # (x != y) positions plus both corners, so a transposition cannot survive.
    probes = sorted({(0, 0), (2, g - 5), (g - 5, 2), (g - 1, g - 1), (7 % g, 4 % g)})
    for cx, cy in probes:
        x0, y0 = cx * 48 + 19, cy * 48 + 19
        d.rectangle([x0, y0, x0 + 9, y0 + 9], fill=0)
    src = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)

    cells, pos, n_soft, resized = patch_cells(proc, img)
    grid_w = int(pos[:, 0].max()) + 1
    grid_h = int(pos[:, 1].max()) + 1

    _check(f"T0 geometry {side} -> {g * g} tokens, {g}x{g}",
           n_soft == g * g and (grid_w, grid_h) == (g, g) and resized == (side, side),
           f"n_soft={n_soft} grid={grid_w}x{grid_h} resized={resized}")

    bad = []
    for m in range(n_soft):
        x, y = int(pos[m, 0]), int(pos[m, 1])
        blk = src[y * 48:(y + 1) * 48, x * 48:(x + 1) * 48]
        if not torch.equal(cells[m], blk):
            bad.append(m)
    _check(f"T1 exact raster identity (all {n_soft})", not bad, f"mismatched tokens: {bad[:8]}")

    order_bad = [m for m in range(n_soft)
                 if m != int(pos[m, 1]) * grid_w + int(pos[m, 0])]
    _check("T2 index order m == y*grid_w + x", not order_bad, f"first bad: {order_bad[:8]}")

    ink_px, _ = cell_ink(cells, 8 / 255)
    got = {(int(pos[m, 0]), int(pos[m, 1])) for m in range(n_soft) if ink_px[m] > 0}
    _check("T3 asymmetric ink -> expected cells", got == set(probes),
           f"got {sorted(got)} expected {sorted(probes)}")

    # ---- T4: the mutation test -------------------------------------------
    shifted_bad = []
    for m in range(n_soft):
        x, y = int(pos[m, 0]), int(pos[m, 1])
        xs = (x + 1) % grid_w
        blk = src[y * 48:(y + 1) * 48, xs * 48:(xs + 1) * 48]
        if not torch.equal(cells[m], blk):
            shifted_bad.append(m)
    _check("T4 mutation (shift 1 cell) must FAIL", len(shifted_bad) > 0,
           f"{len(shifted_bad)} tokens differ under the shifted mapping (want > 0)")

    # ---- T5: resize path, the real lane geometry -------------------------
    img2 = _white((1024, 1024))
    d2 = ImageDraw.Draw(img2)
    # asymmetric, off-centre, and away from cell boundaries after x0.75
    d2.rectangle([300, 120, 359, 179], fill=0)
    a = np.asarray(img2).astype(np.float32)
    ys, xs = np.nonzero(255.0 - a > 8.0)
    cen_orig = (float(xs.mean()), float(ys.mean()))

    cells2, pos2, n2, resized2 = patch_cells(proc, img2)
    ink2, _ = cell_ink(cells2, 8 / 255)
    dark2 = (1.0 - cells2 > 8 / 255).numpy()
    tot = dark2.sum()
    sx = sy = 0.0
    for m in range(n2):
        if not dark2[m].any():
            continue
        yy, xx = np.nonzero(dark2[m])
        sx += (xx + int(pos2[m, 0]) * 48).sum()
        sy += (yy + int(pos2[m, 1]) * 48).sum()
    cen_res = (sx / tot, sy / tot)
    scale = resized2[1] / 1024.0
    err = (abs(cen_res[0] - cen_orig[0] * scale), abs(cen_res[1] - cen_orig[1] * scale))
    _check("T5 resize path: ink centroid x scale", max(err) < 1.5,
           f"orig{tuple(round(v, 1) for v in cen_orig)} x{scale} -> expected "
           f"{tuple(round(v * scale, 1) for v in cen_orig)}, measured "
           f"{tuple(round(v, 1) for v in cen_res)}, err {tuple(round(v, 2) for v in err)}")

    res["all_passed"] = all(res["checks"].values())
    res["note"] = (
        "T1 compares against the synthetic source directly on a 768x768 input, "
        "which the processor does NOT resize, so the mapping proof does not "
        "depend on reimplementing aspect_ratio_preserving_resize. T5 covers the "
        "resized (1024) path separately."
    )
    return res


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def summarize(values):
    import numpy as np

    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    out = {
        "n": int(len(v)), "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
        "min": float(v.min()), "max": float(v.max()),
        "median": float(np.median(v)),
    }
    out["se"] = (out["sd"] / math.sqrt(len(v))) if out["sd"] is not None else None
    out["ci95"] = ([out["mean"] - Z95 * out["se"], out["mean"] + Z95 * out["se"]]
                   if out["se"] is not None else [None, None])
    return out


def paired_stats(a, b, label_a="a", label_b="b", n_boot=5000, seed=0):
    """Paired ``b`` vs ``a``: mean difference and mean ratio, both with CIs."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 0:
        return {"n_pairs": 0}
    diff = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a > 0, b / a, np.nan)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n, n)
        sa, sb = a[pick].sum(), b[pick].sum()
        reps[i] = sb / sa if sa > 0 else np.nan
    reps = reps[np.isfinite(reps)]
    out = {
        "n_pairs": int(n),
        f"{label_a}_mean": float(a.mean()), f"{label_b}_mean": float(b.mean()),
        "diff": summarize(diff),
        "ratio_per_row": summarize(ratio[np.isfinite(ratio)]),
        "ratio_of_totals": float(b.sum() / a.sum()) if a.sum() > 0 else None,
        "ratio_of_totals_ci95_bootstrap": (
            [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
            if len(reps) else [None, None]
        ),
        "n_boot": int(n_boot),
        "n_rows_b_gt_a": int((b > a).sum()),
        "n_rows_b_eq_a": int((b == a).sum()),
        "n_rows_b_lt_a": int((b < a).sum()),
    }
    return out


# ---------------------------------------------------------------------------
# per-rung measurement
# ---------------------------------------------------------------------------
def assistant_target(row) -> str:
    for msg in row["messages"]:
        if msg["role"] != "assistant":
            continue
        c = msg["content"]
        if isinstance(c, list):
            return "".join(p.get("text") or "" for p in c if p.get("type") != "image")
        return str(c)
    return ""


def layout_geometry(text: str, font_size: int, canvas: int = 1024):
    """``(n_text_lines, line_height_px, lines_per_page)`` from the renderer's OWN
    layout pass, so lines and pixels come from one source of truth."""
    from data.preprocessing.render_utils import wrap_text_pages

    lines, lines_per_page, line_height = wrap_text_pages(
        text, canvas_width=canvas, canvas_height=canvas, font_size=font_size
    )
    return len(lines), int(line_height), int(lines_per_page)


def word_cell_stats(text: str, font_size: int, scale: float, grid_w: int, grid_h: int,
                    canvas: int = 1024):
    """Distinct words and distinct text lines overlapping each cell.

    Uses the renderer's per-word **advance** boxes (``word_boxes_for_pages``)
    scaled by ``scale`` into resized-image coordinates.  The box spans the whole
    ``line_height`` band, not the inked glyph extent, so a word can be credited to
    a cell its glyphs only graze; the inked-cell counts elsewhere in this report
    come from PIXELS, not from these boxes.
    """
    from data.preprocessing.render_utils import word_boxes_for_pages

    boxes = [b for b in word_boxes_for_pages(
        text, canvas_width=canvas, canvas_height=canvas, font_size=font_size) if b.page == 0]
    words_per_cell = {}
    lines_per_cell = {}
    for b in boxes:
        cx0 = max(0, int(math.floor(b.x0 * scale / 48)))
        cx1 = min(grid_w - 1, int(math.floor((b.x1 * scale - 1e-6) / 48)))
        cy0 = max(0, int(math.floor(b.y0 * scale / 48)))
        cy1 = min(grid_h - 1, int(math.floor((b.y1 * scale - 1e-6) / 48)))
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                words_per_cell.setdefault((cx, cy), set()).add(b.index)
                lines_per_cell.setdefault((cx, cy), set()).add(b.line)
    return len(boxes), words_per_cell, lines_per_cell


def measure_rows(rows, font_size, processor, ink_threshold, check_render):
    """Measure a list of ``(target_text, PIL image)`` pairs. Returns per-row dicts."""
    import numpy as np

    per_row = []
    for target, image in rows:
        cells, pos, n_soft, resized = patch_cells(processor, image)
        grid_w = int(pos[:, 0].max()) + 1
        grid_h = int(pos[:, 1].max()) + 1
        ink_px, ink_mass = cell_ink(cells, ink_threshold)
        ink_px = ink_px.numpy().astype(int)
        ink_mass = ink_mass.numpy().astype(float)

        inked = {lbl: int((ink_px >= c).sum())
                 for lbl, c in zip(INKED_CUTOFF_LABELS, INKED_CUTOFFS)}
        rows_with_ink = sorted({int(pos[m, 1]) for m in range(n_soft) if ink_px[m] >= 1})
        cols_with_ink = sorted({int(pos[m, 0]) for m in range(n_soft) if ink_px[m] >= 1})

        n_lines, line_h, lines_per_page = layout_geometry(target, font_size)
        scale = resized[1] / 1024.0
        n_words, wpc, lpc = word_cell_stats(target, font_size, scale, grid_w, grid_h)

        inked_cells = {(int(pos[m, 0]), int(pos[m, 1])) for m in range(n_soft) if ink_px[m] >= 1}
        w_counts = [len(wpc.get(c, ())) for c in inked_cells]
        l_counts = [len(lpc.get(c, ())) for c in inked_cells]

        chars_all = len(target)
        letters = sum(1 for ch in target if not ch.isspace())

        per_row.append({
            "n_soft": n_soft, "grid_w": grid_w, "grid_h": grid_h,
            "resized_h": resized[0], "resized_w": resized[1],
            "max_pos_x": int(pos[:, 0].max()), "max_pos_y": int(pos[:, 1].max()),
            "inked": inked,
            "ink_px_per_cell": ink_px.tolist(),
            "page_ink_px": int(ink_px.sum()),
            "page_ink_mass": float(ink_mass.sum()),
            "grid_rows_with_ink": len(rows_with_ink),
            "grid_cols_with_ink": len(cols_with_ink),
            "n_text_lines": n_lines, "line_height_px_orig": line_h,
            "lines_per_page_capacity": lines_per_page,
            "n_words": n_words,
            "words_per_inked_cell_mean": float(np.mean(w_counts)) if w_counts else None,
            "lines_per_inked_cell_mean": float(np.mean(l_counts)) if l_counts else None,
            "chars_all": chars_all, "letters": letters,
            "scale": scale,
        })
    return per_row


def aggregate(per_row, name, lane_spec, render_check, ink_threshold):
    import numpy as np

    g = lambda k: [r[k] for r in per_row]                                    # noqa: E731
    inked_ge1 = [r["inked"]["ge1"] for r in per_row]
    n_soft = g("n_soft")
    chars_all = g("chars_all")
    letters = g("letters")

    pooled = np.concatenate([np.asarray(r["ink_px_per_cell"]) for r in per_row])
    nz = pooled[pooled >= 1]
    buckets = [(1, 9), (10, 49), (50, 99), (100, 249), (250, 499), (500, 999), (1000, CELL_AREA)]
    hist = {f"{lo}-{hi}": int(((pooled >= lo) & (pooled <= hi)).sum()) for lo, hi in buckets}
    hist["0"] = int((pooled == 0).sum())

    def per_row_ratio(num, den):
        return summarize([a / b if b else None for a, b in zip(num, den)])

    line_h = per_row[0]["line_height_px_orig"]
    scale = per_row[0]["scale"]

    out = {
        "name": name,
        "lane_spec": lane_spec,
        "n_rows": len(per_row),
        "render_check": render_check,
        "ink_threshold_darkness": ink_threshold,
        "geometry": {
            "canvas_px": 1024,
            "resized_px": [per_row[0]["resized_h"], per_row[0]["resized_w"]],
            "resize_scale": scale,
            "grid": [per_row[0]["grid_w"], per_row[0]["grid_h"]],
            "cell_px_resized": 48,
            "cell_px_original": 48 / scale,
            "soft_tokens_per_page": summarize(n_soft),
            "max_image_position_id": {"x": max(g("max_pos_x")), "y": max(g("max_pos_y"))},
            "resize_identical_across_rows": len(set(g("resized_w"))) == 1
                                            and len(set(g("resized_h"))) == 1,
        },
        "inked_tokens": {
            lbl: summarize([r["inked"][lbl] for r in per_row]) for lbl in INKED_CUTOFF_LABELS
        },
        "inked_fraction_ge1": summarize([a / b for a, b in zip(inked_ge1, n_soft)]),
        "per_cell_ink_px": {
            "cutoffs": dict(zip(INKED_CUTOFF_LABELS, INKED_CUTOFFS)),
            "cell_area_px": CELL_AREA,
            "pooled_histogram_cells": hist,
            "nonzero_cells": summarize(nz.tolist()) if len(nz) else None,
            "nonzero_quantiles": (
                {q: float(np.percentile(nz, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
                if len(nz) else None
            ),
            "n_cells_pooled": int(len(pooled)),
        },
        "page_ink": {
            "ink_px": summarize(g("page_ink_px")),
            "ink_mass": summarize(g("page_ink_mass")),
            "page_ink_fraction": summarize(
                [r["page_ink_px"] / (r["resized_h"] * r["resized_w"]) for r in per_row]),
        },
        "chars": {
            "chars_all_incl_spaces": summarize(chars_all),
            "letters_only": summarize(letters),
            "chars_per_budget_280": summarize([c / 280 for c in chars_all]),
            "chars_per_soft_token": per_row_ratio(chars_all, n_soft),
            "chars_per_inked_token_ge1": per_row_ratio(chars_all, inked_ge1),
            "chars_per_inked_token_ge1pct": per_row_ratio(
                chars_all, [r["inked"]["ge1pct"] for r in per_row]),
            "letters_per_inked_token_ge1": per_row_ratio(letters, inked_ge1),
        },
        "lines": {
            "n_text_lines": summarize(g("n_text_lines")),
            "line_height_px_original_canvas": line_h,
            "line_pitch_px_resized": line_h * scale,
            "lines_per_cell_geometric_original_canvas": 48 / line_h,
            "lines_per_cell_geometric_resized": 48 / (line_h * scale),
            "grid_rows_with_ink": summarize(g("grid_rows_with_ink")),
            "grid_cols_with_ink": summarize(g("grid_cols_with_ink")),
            "lines_per_inked_grid_row": per_row_ratio(
                g("n_text_lines"), g("grid_rows_with_ink")),
            "lines_per_inked_cell_boxes": summarize(g("lines_per_inked_cell_mean")),
            "note": ("lines_per_cell_geometric_ORIGINAL_CANVAS is the figure H13's "
                     "deviation note quotes (48px patch on the 1024 page). The patch "
                     "grid actually lives on the RESIZED image, where the same text "
                     "line is scale x thinner, so ..._resized is the number the model "
                     "sees; it is 1/scale times larger."),
        },
        "words": {
            "n_words": summarize(g("n_words")),
            "words_per_inked_token_ge1": per_row_ratio(g("n_words"), inked_ge1),
            "words_per_inked_cell_overlap": summarize(g("words_per_inked_cell_mean")),
        },
    }
    return out


def run_rung(rung, root, n, processor, ink_threshold, seed, check_render):
    from datasets import load_from_disk

    subset = f"randstr-{rung}"
    path = Path(root) / subset / "validation"
    ds = load_from_disk(str(path))
    ds = ds.select(range(min(n, len(ds))))          # NOT shuffled: d3/d5 pair by index
    spec = json.loads((Path(root) / "manifest.json").read_text()).get("lane_specs", {}).get(subset, {})
    font_size = int(spec.get("font_size", 14))

    rows, targets, n_mismatch = [], [], 0
    for i in range(len(ds)):
        r = ds[i]
        t = assistant_target(r)
        img = r["images"][0]
        rows.append((t, img))
        targets.append(t)
        if check_render:
            from data.preprocessing.render_utils import render_text_pages
            pages = render_text_pages(t, canvas_width=1024, canvas_height=1024,
                                      font_size=font_size)
            if len(pages) != 1 or pages[0].convert("L").tobytes() != img.convert("L").tobytes():
                n_mismatch += 1

    logger.info("%s: %d rows, font %d", subset, len(rows), font_size)
    per_row = measure_rows(rows, font_size, processor, ink_threshold, check_render)
    agg = aggregate(per_row, subset, spec,
                    {"n_checked": len(rows) if check_render else 0, "n_mismatch": n_mismatch,
                     "note": "re-render of the stored target vs the stored PNG, byte-compared"},
                    ink_threshold)
    agg["_per_row_inked_ge1"] = [r["inked"]["ge1"] for r in per_row]
    agg["_per_row_inked_ge1pct"] = [r["inked"]["ge1pct"] for r in per_row]
    agg["_per_row_page_ink_px"] = [r["page_ink_px"] for r in per_row]
    agg["_targets"] = targets
    return agg


FINEWEB_VAL = "data/materialized/univi-3M-v0-split/fineweb-edu/validation"


def run_fineweb(n, processor, ink_threshold, path=FINEWEB_VAL, font_size=14):
    """Cross-reference: the real ``fineweb-edu`` lane H17's chars/soft-token table is about.

    Rows can span several pages, so ONLY page 0 is measured and the character
    count is the text that actually lands on page 0 (the first ``lines_per_page``
    wrapped lines), re-joined with newlines so the layout pass reproduces the
    same breaks.  Comparing the whole document's characters against one page's
    tokens would inflate chars/token by the page count.
    """
    from datasets import load_from_disk

    from data.preprocessing.render_utils import render_text_pages, wrap_text_pages

    ds = load_from_disk(str(path)).select(range(min(n, len(load_from_disk(str(path))))))
    rows, n_mismatch, n_pages = [], 0, []
    for i in range(len(ds)):
        r = ds[i]
        full = assistant_target(r)
        lines, lines_per_page, _lh = wrap_text_pages(
            full, canvas_width=1024, canvas_height=1024, font_size=font_size)
        page0 = "\n".join(lines[:lines_per_page])
        img = r["images"][0].convert("L")
        n_pages.append(len(r["images"]))
        pages = render_text_pages(full, canvas_width=1024, canvas_height=1024,
                                  font_size=font_size)
        if not pages or pages[0].convert("L").tobytes() != img.tobytes():
            n_mismatch += 1
        rows.append((page0, img))
    per_row = measure_rows(rows, font_size, processor, ink_threshold, False)
    agg = aggregate(per_row, "fineweb-edu (page 0 only)",
                    {"font_size": font_size, "source": str(path)},
                    {"n_checked": len(rows), "n_mismatch": n_mismatch,
                     "note": "re-render of the full target, page 0, vs the stored PNG"},
                    ink_threshold)
    agg["pages_per_row"] = summarize(n_pages)
    agg["caveat"] = ("characters counted are PAGE-0 characters only; rows averaging "
                     f"{summarize(n_pages)['mean']:.2f} pages here.")
    agg["_per_row_inked_ge1"] = [r["inked"]["ge1"] for r in per_row]
    return agg


def run_masked(n, processor, ink_threshold, seed):
    """Cross-reference: the H14 ``masked-randstr`` lane geometry, rendered in memory.

    Nothing is materialized — ``masked_regions``'s own sampler + renderer are
    called directly, which is the same code path the lane would use.  Its
    defaults (80 words of 3-7 letters, font 14) are H13's d3 geometry PLUS black
    occlusion rectangles, so the ink figures here are an upper bound on d3's and
    are reported separately for exactly that reason.
    """
    import random

    from data.preprocessing.masked_regions import (
        LaneSpec, _make_words, render_masked_pages, sample_mask_runs,
    )

    spec = LaneSpec()
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        words = _make_words(rng, spec)
        runs = sample_mask_runs(rng, len(words), spec)
        pages, _boxes = render_masked_pages(words, runs, spec)
        rows.append((" ".join(words), pages[0].convert("L")))
    per_row = measure_rows(rows, spec.font_size, processor, ink_threshold, False)
    agg = aggregate(per_row, "masked-randstr (rendered in memory, NOT materialized)",
                    {"group_len_min": spec.group_len_min, "group_len_max": spec.group_len_max,
                     "n_groups": spec.n_groups, "font_size": spec.font_size},
                    {"n_checked": 0, "n_mismatch": 0, "note": "n/a — rendered here, not stored"},
                    ink_threshold)
    agg["caveat"] = ("black occlusion rectangles are pure ink and cover whole words, so "
                     "inked-token counts here are an UPPER bound relative to the clean "
                     "d3 page of the same geometry.")
    agg["_per_row_inked_ge1"] = [r["inked"]["ge1"] for r in per_row]
    return agg


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _f(x, spec=".2f", dash="--"):
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return format(v, spec) if math.isfinite(v) else dash


def print_report(res):
    print("\n" + "=" * 108)
    print("H13 / H17 — INK OCCUPANCY OF THE SOFT-TOKEN GRID")
    print("=" * 108)
    m = res["meta"]
    print(f"rows/rung {m['n']}   soft-token budget {m['max_soft_tokens']}   "
          f"ink threshold darkness > {m['ink_threshold']:.4f} ({m['ink_threshold'] * 255:.0f}/255)")
    st = res.get("self_test")
    if st:
        print(f"mapping self-test: {'ALL PASS' if st['all_passed'] else 'FAILURES: ' + str(st['failures'])}")

    hdr = (f"{'rung':<10}{'font':>5}{'chars':>7}{'soft':>6}{'grid':>8}"
           f"{'inked>=1':>10}{'frac':>8}{'inked>=1%':>11}"
           f"{'ch/soft':>9}{'ch/inked':>10}{'lines/cell':>12}{'rows w/ink':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in res["rungs"].items():
        gm = r["geometry"]
        ln = r["lines"]
        print(f"{name.replace('randstr-', ''):<10}"
              f"{r['lane_spec'].get('font_size', '--'):>5}"
              f"{_f(r['chars']['chars_all_incl_spaces']['mean'], '.0f'):>7}"
              f"{_f(gm['soft_tokens_per_page']['mean'], '.0f'):>6}"
              f"{gm['grid'][0]}x{gm['grid'][1]:<5}"
              f"{_f(r['inked_tokens']['ge1']['mean'], '.1f'):>10}"
              f"{_f(r['inked_fraction_ge1']['mean'] * 100, '.1f') + '%':>8}"
              f"{_f(r['inked_tokens']['ge1pct']['mean'], '.1f'):>11}"
              f"{_f(r['chars']['chars_per_soft_token']['mean'], '.2f'):>9}"
              f"{_f(r['chars']['chars_per_inked_token_ge1']['mean'], '.1f'):>10}"
              f"{_f(ln['lines_per_cell_geometric_resized'], '.2f'):>12}"
              f"{_f(ln['grid_rows_with_ink']['mean'], '.1f'):>11}")

    print("\n-- per-cell ink-pixel distribution (pooled cells, cell area 2304 px)")
    for name, r in res["rungs"].items():
        h = r["per_cell_ink_px"]["pooled_histogram_cells"]
        nz = r["per_cell_ink_px"]["nonzero_cells"]
        print(f"  {name:<26} 0:{h['0']:>6}  1-9:{h['1-9']:>4}  10-49:{h['10-49']:>4}  "
              f"50-99:{h['50-99']:>4}  100-249:{h['100-249']:>5}  250-499:{h['250-499']:>5}  "
              f"500-999:{h['500-999']:>5}  1000+:{h['1000-2304']:>5}   "
              f"median nonzero {_f(nz['median'] if nz else None, '.0f')}")

    p = res.get("d3_vs_d5_paired")
    if p:
        print("\n-- d3 vs d5, PAIRED BY ROW (byte-identical targets, font 14 vs 40)")
        print(f"  targets byte-identical on {p['n_targets_identical']}/{p['n_compared']} rows")
        for key in ("inked_ge1", "inked_ge1pct", "page_ink_px"):
            s = p[key]
            rt = s.get("ratio_of_totals")
            ci = s.get("ratio_of_totals_ci95_bootstrap") or [None, None]
            d = s.get("diff") or {}
            dci = d.get("ci95") or [None, None]
            print(f"  {key:<14} d3 {_f(s.get('d3_mean'), '.2f'):>8}  d5 {_f(s.get('d5_mean'), '.2f'):>8}"
                  f"   ratio d5/d3 {_f(rt, '.3f')} [{_f(ci[0], '.3f')}, {_f(ci[1], '.3f')}]"
                  f"   paired diff {_f(d.get('mean'), '+.2f')} [{_f(dci[0], '+.2f')}, {_f(dci[1], '+.2f')}]"
                  f"   (d5>d3 on {s.get('n_rows_b_gt_a')}, = on {s.get('n_rows_b_eq_a')}, "
                  f"< on {s.get('n_rows_b_lt_a')} rows)")
        print(f"  chars/inked-token  d3 {_f(p['chars_per_inked_d3'], '.1f')}  ->  "
              f"d5 {_f(p['chars_per_inked_d5'], '.1f')}   "
              f"(ratio {_f(p['chars_per_inked_ratio'], '.3f')})")
    print()


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", action="append", default=None,
                    help="d1..d5, 'all', 'masked' or 'fineweb' (repeatable). "
                         "Default: all + masked + fineweb.")
    ap.add_argument("-n", "--n", type=int, default=30, help="validation rows per rung")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--ink-threshold", type=float, default=8 / 255,
                    help="darkness (1 - pixel) above which a pixel counts as ink")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render-check", action="store_true",
                    help="skip re-rendering each target and byte-comparing to the stored PNG")
    ap.add_argument("--self-test", action="store_true",
                    help="run ONLY the pixel->patch mapping self-test and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        logger.info("pixel -> patch mapping self-test (budget %d)", args.max_soft_tokens)
        st = self_test(args.max_soft_tokens)
        print(json.dumps(st, indent=2))
        return 0 if st["all_passed"] else 1

    wanted = args.rung or ["all", "masked", "fineweb"]
    rungs = []
    for w in wanted:
        if w == "all":
            rungs += list(RUNGS)
        elif w in ("masked", "fineweb"):
            rungs.append(w)
        else:
            rungs.append(w.lstrip("-"))
    seen, ordered = set(), []
    for r in rungs:
        if r not in seen:
            seen.add(r)
            ordered.append(r)

    st = self_test(args.max_soft_tokens, verbose=True)
    if not st["all_passed"]:
        logger.error("MAPPING SELF-TEST FAILED — refusing to report ink numbers: %s",
                     st["failures"])
        return 1

    proc = build_processor(args.max_soft_tokens)
    res = {
        "meta": {
            "root": args.root, "n": args.n, "max_soft_tokens": args.max_soft_tokens,
            "ink_threshold": args.ink_threshold, "seed": args.seed,
            "rungs": ordered, "cell_area_px": CELL_AREA,
            "device": "cpu (no model weights loaded; a training run holds the GPU)",
        },
        "self_test": st,
        "rungs": {},
    }
    for r in ordered:
        if r == "masked":
            res["rungs"]["masked-randstr"] = run_masked(
                args.n, proc, args.ink_threshold, args.seed)
        elif r == "fineweb":
            res["rungs"]["fineweb-edu"] = run_fineweb(args.n, proc, args.ink_threshold)
        else:
            res["rungs"][f"randstr-{r}"] = run_rung(
                r, args.root, args.n, proc, args.ink_threshold, args.seed,
                not args.no_render_check)

    # ---- the paired d3-vs-d5 comparison ----------------------------------
    d3 = res["rungs"].get("randstr-d3")
    d5 = res["rungs"].get("randstr-d5")
    if d3 and d5:
        t3, t5 = d3["_targets"], d5["_targets"]
        k = min(len(t3), len(t5))
        same = [i for i in range(k) if t3[i] == t5[i]]
        pair = {
            "n_compared": k,
            "n_targets_identical": len(same),
            "note": ("rows are paired by validation-split INDEX and only index-matched "
                     "rows whose target strings are byte-identical are used; d3 and d5 "
                     "were materialized from the same RNG stream and differ only in font."),
        }
        for key, col in (("inked_ge1", "_per_row_inked_ge1"),
                         ("inked_ge1pct", "_per_row_inked_ge1pct"),
                         ("page_ink_px", "_per_row_page_ink_px")):
            a = [d3[col][i] for i in same]
            b = [d5[col][i] for i in same]
            pair[key] = paired_stats(a, b, "d3", "d5", seed=args.seed)
        ch3 = sum(len(t3[i]) for i in same)
        i3 = sum(d3["_per_row_inked_ge1"][i] for i in same) or 1
        i5 = sum(d5["_per_row_inked_ge1"][i] for i in same) or 1
        pair["chars_per_inked_d3"] = ch3 / i3
        pair["chars_per_inked_d5"] = ch3 / i5
        pair["chars_per_inked_ratio"] = (ch3 / i5) / (ch3 / i3)
        res["d3_vs_d5_paired"] = pair

    for r in res["rungs"].values():
        r.pop("_targets", None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    logger.info("wrote %s", out)
    print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
