"""
H17 — is the soft-token budget the only thing that changes when you raise it?

H17 proposes raising ``max_soft_tokens`` 280 -> 1120 and pre-registers "K must
scale >= 3x for 4x the tokens", with a *refute* branch that promotes a much
larger architectural bet (a real ViT/SigLIP front-end).  The fourth correction
appended to that doc found the intervention is two things bundled together, and
this script measures the bundle so a null can be attributed to the right cause.

THE CONFOUND, STATED PRECISELY
  ``Gemma4UnifiedImageProcessor`` resizes every image to the largest
  aspect-preserving size whose patch count fits ``max_soft_tokens * 9`` teacher
  patches and is divisible by 48.  For a SQUARE page that target is a function of
  the BUDGET ALONE -- 768 at 280, 1104 at 560, 1584 at 1120 -- and does not
  depend on the render canvas at all.  Univi renders text pages at 1024x1024, so:

    280  -> 768   downsample 0.75x   (optical detail is DESTROYED here)
    560  -> 1104  upsample   1.08x   (source resolution fully recovered)
    1120 -> 1584  upsample   1.55x   (subdivides INTERPOLATED pixels)

  A 280 -> 1120 ladder on a 1024 canvas therefore mixes "recover lost resolution"
  (real, and complete by 560) with "more grid cells over the same information"
  (everything past 560).  If K fails to scale, that is ambiguous between "budget
  is not the binding constraint" (H17's refute branch) and "the rungs past 560
  never supplied more information".

WHAT THIS SCRIPT MEASURES
  1. geometry   -- the resize/grid/position-id table, straight from the processor,
                   for text pages at canvas 1024/1536/2048 and for the audio page.
  2. glyph      -- font metrics in ORIGINAL px and in MODEL-VISIBLE px at each
                   budget (x-height, cap height, advance, line pitch, stem width).
  3. legibility -- whether letter IDENTITY survives the sampling.  Letters are
                   cropped from the model-visible image at their true sub-pixel
                   phases and classified 1-NN leave-one-out.  Confusions come only
                   from real aliasing/phase jitter -- no synthetic noise is needed
                   for the headline number (a noise sweep is reported separately,
                   because it is the honest steelman for upsampling: extra pixels
                   do help against per-pixel sensor noise even when they carry no
                   new optical detail).
  4. ink        -- inked soft tokens and chars-per-inked-token at every
                   (canvas, budget) cell, to test whether the ink figures the
                   fourth correction reports are canvas-invariant.

CPU ONLY.  No model weights, no GPU (a training run holds the card; this repo's
rule is that training runs SOLO).  ``univi.hybrid.vision`` is loaded BY FILE PATH
because ``univi/__init__.py`` imports unsloth, which raises without an accelerator.

Usage:
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h17_render_resolution.py --self-test
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h17_render_resolution.py
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
logger = logging.getLogger("h17_render_resolution")

DEFAULT_OUT = "data/eval/h17-render-resolution.json"
BUDGETS = (280, 560, 1120)
#: (canvas, font).  Fonts chosen so line_height tracks the canvas scale as
#: closely as integer font metrics allow -- see ``layout_capacity``.
CANVASES = ((1024, 14), (1536, 21), (2048, 28))
CELL_AREA = 48 * 48
INK_DARKNESS = 8 / 255
FINEWEB_VAL = "data/materialized/univi-3M-v0-split/fineweb-edu/validation"
H13_ROOT = "data/materialized/h13-density-v0"


# ---------------------------------------------------------------------------
# module loading (deferred / path-based -- see the docstring)
# ---------------------------------------------------------------------------
def load_vision_module():
    import importlib.util

    name = "_univi_hybrid_vision_cpu"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "univi" / "hybrid" / "vision.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PROC_CACHE: dict[int, object] = {}


def processor(budget: int):
    if budget not in _PROC_CACHE:
        _PROC_CACHE[budget] = load_vision_module().Gemma4UnifiedImageProcessor(
            max_soft_tokens=int(budget)
        )
    return _PROC_CACHE[budget]


def resized_side(budget: int, height: int, width: int) -> tuple[int, int]:
    """``(target_h, target_w)`` the processor will resize a ``height x width`` page to."""
    return load_vision_module().get_aspect_ratio_preserving_size(
        height=height, width=width, patch_size=16, max_patches=budget * 9, pooling_kernel_size=3
    )


# ---------------------------------------------------------------------------
# 1. geometry, straight from the processor
# ---------------------------------------------------------------------------
def probe_geometry(image, budget: int) -> dict:
    """Grid dims / token count / max position id / resized px, MEASURED not derived."""
    out = processor(budget)(images=[image], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pos = out["image_position_ids"][0, :n]
    gx, gy = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
    w, h = image.size
    analytic = resized_side(budget, h, w)
    measured = (gy * 48, gx * 48)
    return {
        "budget": budget,
        "source_hw": [h, w],
        "resized_hw_analytic": list(analytic),
        "resized_hw_measured": list(measured),
        "analytic_matches_measured": tuple(analytic) == measured,
        "n_soft_tokens": n,
        "n_soft_tokens_padded_to": budget,
        "grid_wh": [gx, gy],
        "max_image_position_id": [int(pos[:, 0].max()), int(pos[:, 1].max())],
        "scale_h": measured[0] / h,
        "scale_w": measured[1] / w,
        "cell_px_in_source": 48 * w / measured[1],
        "is_downsample": measured[1] < w,
    }


def geometry_table() -> dict:
    from PIL import Image

    res = {"text_pages": [], "audio_page": [], "note": (
        "For a SQUARE source the resized side depends ONLY on the budget "
        "(768/1104/1584 at 280/560/1120) -- the canvas never enters. The canvas "
        "decides only whether that resize is a down- or an up-sample.")}
    for canvas, _font in CANVASES:
        for b in BUDGETS:
            res["text_pages"].append(probe_geometry(Image.new("RGB", (canvas, canvas), 255), b))
    # the production audio page: 1000 mel frames wide (10 ms/px), 80 mels
    # stretched to 160 px tall.
    for b in BUDGETS:
        res["audio_page"].append(probe_geometry(Image.new("RGB", (1000, 160), 255), b))
    res["audio_note"] = (
        "The audio render is 1000x160 for 1000 mel frames x 80 mel bins, i.e. the "
        "horizontal axis is ALREADY at native frame resolution (1 px = 1 frame) and "
        "the vertical axis is already a 2x stretch of 80 bins. Every budget "
        "UPSAMPLES it, so for audio even 280 supplies no lost detail to recover -- "
        "the whole budget ladder is grid-subdivision of interpolated pixels unless "
        "the spectrogram is re-rendered with more frames (smaller hop) / more mels.")
    return res


# ---------------------------------------------------------------------------
# 2. glyph metrics
# ---------------------------------------------------------------------------
def stem_width_px(font) -> dict:
    """Ink-mass width of the 'l' stem: threshold-free (sum of darkness / 255)."""
    import numpy as np
    from PIL import Image, ImageDraw

    size = max(64, font.size * 4)
    img = Image.new("L", (size, size), 255)
    ImageDraw.Draw(img).text((size // 4, size // 8), "l", font=font, fill=0)
    a = 255.0 - np.asarray(img).astype(float)
    row = a[int(np.argmax(a.sum(1)))]
    nz = np.nonzero(row > 8)[0]
    return {
        "ink_mass_width_px": float(row.sum() / 255.0),
        "width_ge_thresh_px": int(len(nz)),
        "profile": row[nz[0]: nz[-1] + 1].astype(int).tolist() if len(nz) else [],
    }


def layout_capacity(canvas: int, font_size: int) -> dict:
    from PIL import ImageFont

    from data.preprocessing.render_utils import _find_font, wrap_text_pages

    font = ImageFont.truetype(_find_font(), font_size)
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    adv = font.getlength("a")
    _lines, lines_per_page, lh2 = wrap_text_pages(
        "x", canvas_width=canvas, canvas_height=canvas, font_size=font_size)
    assert lh2 == lh
    chars_per_line = int((canvas - 40) // adv)
    bx = font.getbbox("x")
    bH = font.getbbox("H")
    return {
        "canvas": canvas, "font_size": font_size,
        "line_height_px": lh, "advance_px": adv,
        "x_height_px": bx[3] - bx[1], "cap_height_px": bH[3] - bH[1],
        "stem": stem_width_px(font),
        "lines_per_page": lines_per_page, "chars_per_line": chars_per_line,
        "chars_per_page": lines_per_page * chars_per_line,
    }


def glyph_table() -> dict:
    base = layout_capacity(*CANVASES[0])
    rows = []
    for canvas, fs in CANVASES:
        cap = layout_capacity(canvas, fs)
        cap["canvas_scale_vs_1024"] = canvas / CANVASES[0][0]
        cap["chars_per_page_rel_to_1024"] = cap["chars_per_page"] / base["chars_per_page"]
        cap["model_visible"] = {}
        for b in BUDGETS:
            s = resized_side(b, canvas, canvas)[0] / canvas
            cap["model_visible"][str(b)] = {
                "resize_scale": s,
                "x_height_px": cap["x_height_px"] * s,
                "cap_height_px": cap["cap_height_px"] * s,
                "advance_px": cap["advance_px"] * s,
                "line_pitch_px": cap["line_height_px"] * s,
                "stem_ink_mass_px": cap["stem"]["ink_mass_width_px"] * s,
                "px_per_char_cell": cap["advance_px"] * s * cap["line_height_px"] * s,
                # real optical samples the SOURCE can supply for one char cell,
                # capped by the source render (an upsample adds none)
                "real_px_per_char_cell": cap["advance_px"] * min(s, 1.0)
                                         * cap["line_height_px"] * min(s, 1.0),
            }
        rows.append(cap)
    return {"rows": rows, "note": (
        "'model_visible' multiplies the ORIGINAL-canvas metric by the processor's "
        "resize scale for that budget. 'real_px_per_char_cell' caps the scale at "
        "1.0: an upsample manufactures pixels, not samples.")}


# ---------------------------------------------------------------------------
# 3. legibility: does letter identity survive the sampling?
# ---------------------------------------------------------------------------
def render_letter_page(n_lines: int, canvas: int, font_size: int, seed: int):
    """A page of random a-z letters plus the (line, col, letter) index of each."""
    import random

    from PIL import ImageFont

    from data.preprocessing.render_utils import _find_font, render_text_pages

    font = ImageFont.truetype(_find_font(), font_size)
    adv = font.getlength("a")
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    per_line = int((canvas - 40) // adv)
    rng = random.Random(seed)
    lines = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(per_line))
             for _ in range(n_lines)]
    pages = render_text_pages("\n".join(lines), canvas_width=canvas, canvas_height=canvas,
                              font_size=font_size)
    assert len(pages) == 1, f"{n_lines} lines overflow a {canvas}px canvas at font {font_size}"
    idx = [(li, ci, ch) for li, ln in enumerate(lines) for ci, ch in enumerate(ln)]
    return pages[0].convert("L"), idx, adv, lh


def model_visible(image, budget: int):
    """The image EXACTLY as the processor hands it to the embedder, as a float array.

    Reconstructed from ``pixel_values`` (patch rasters) rather than by calling
    ``tvF.resize`` again, so this is the processor's own output, not a
    reimplementation of it.
    """
    import numpy as np
    import torch

    out = processor(budget)(images=[image], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pv = out["pixel_values"][0, :n]
    pos = out["image_position_ids"][0, :n]
    side = int(math.isqrt(pv.shape[-1] // 3))
    cells = pv.reshape(n, side, side, 3)
    cells = cells[..., 0] if torch.equal(cells[..., 0], cells[..., 2]) else cells.mean(-1)
    gx, gy = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
    canvas = np.zeros((gy * side, gx * side), dtype=np.float32)
    c = cells.numpy()
    for m in range(n):
        x, y = int(pos[m, 0]), int(pos[m, 1])
        canvas[y * side:(y + 1) * side, x * side:(x + 1) * side] = c[m]
    return canvas, (gx, gy), side


def letter_crops(vis, idx, adv, lh, scale, margin=20, pad=1):
    """Fixed-shape crops of each character cell at its TRUE sub-pixel phase."""
    import numpy as np

    cw = int(math.ceil(adv * scale)) + 2 * pad
    ch = int(math.ceil(lh * scale)) + 2 * pad
    H, W = vis.shape
    crops, labels = [], []
    for li, ci, letter in idx:
        x0 = int(round((margin + ci * adv) * scale)) - pad
        y0 = int(round((margin + li * lh) * scale)) - pad
        if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + ch > H:
            continue
        crops.append(vis[y0:y0 + ch, x0:x0 + cw])
        labels.append(letter)
    return np.stack(crops), np.asarray(labels), (ch, cw)


def one_nn_loo(X, y, sigma=0.0, seed=0, max_n=2600):
    """Leave-one-out 1-NN accuracy over flattened crops (optionally + AWGN)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        pick = rng.choice(len(X), max_n, replace=False)
        X, y = X[pick], y[pick]
    F = X.reshape(len(X), -1).astype(np.float32)
    if sigma > 0:
        F = F + rng.normal(0, sigma, F.shape).astype(np.float32)
    sq = (F * F).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (F @ F.T)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(1)
    acc = float((y[nn] == y).mean())
    return acc, int(len(X))


def class_separation(X, y):
    """RMS-per-pixel distances between class means; scale-comparable across
    resolutions because it divides by sqrt(#pixels)."""
    import numpy as np

    F = X.reshape(len(X), -1).astype(np.float64)
    npix = F.shape[1]
    classes = sorted(set(y.tolist()))
    means = np.stack([F[y == c].mean(0) for c in classes])
    within = float(np.mean([np.sqrt(((F[y == c] - means[i]) ** 2).mean())
                            for i, c in enumerate(classes)]))
    d = np.sqrt(((means[:, None, :] - means[None, :, :]) ** 2).mean(-1))
    iu = np.triu_indices(len(classes), 1)
    return {
        "n_pixels_per_crop": int(npix),
        "between_class_rms_min": float(d[iu].min()),
        "between_class_rms_mean": float(d[iu].mean()),
        "within_class_rms": within,
        "fisher_ratio_min": float(d[iu].min() / within) if within > 0 else None,
    }


def legibility_table(n_lines: int, sigmas: tuple[float, ...], seed: int) -> dict:
    rows = []
    for canvas, fs in CANVASES:
        page, idx, adv, lh = render_letter_page(n_lines, canvas, fs, seed)
        for b in BUDGETS:
            vis, grid, side = model_visible(page, b)
            scale = vis.shape[1] / canvas
            X, y, shape = letter_crops(vis, idx, adv, lh, scale)
            acc, n = one_nn_loo(X, y, 0.0, seed)
            sep = class_separation(X, y)
            row = {
                "canvas": canvas, "font_size": fs, "budget": b,
                "resize_scale": scale, "crop_hw": list(shape),
                "n_letters": n, "n_letters_total": len(X),
                "acc_1nn_clean": acc,
                "acc_1nn_noise": {str(s): one_nn_loo(X, y, s, seed)[0] for s in sigmas},
                **sep,
            }
            rows.append(row)
            logger.info("legibility canvas %d font %d budget %4d: scale %.3f crop %s "
                        "acc %.4f  min-between-RMS %.4f", canvas, fs, b, scale, shape,
                        acc, sep["between_class_rms_min"])
    return {"rows": rows, "sigmas": list(sigmas), "note": (
        "acc_1nn_clean is DETERMINISTIC: the only source of confusion is real "
        "aliasing / sub-pixel phase jitter of the same letter at different page "
        "positions. acc_1nn_noise adds per-pixel Gaussian sensor noise -- it is "
        "the steelman for upsampling, which cannot add optical detail but does "
        "average independent pixel noise.")}


# ---------------------------------------------------------------------------
# 4. ink occupancy across canvases
# ---------------------------------------------------------------------------
def ink_counts(image, budget: int) -> dict:
    import torch

    out = processor(budget)(images=[image], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pv = out["pixel_values"][0, :n]
    side = int(math.isqrt(pv.shape[-1] // 3))
    cells = pv.reshape(n, side, side, 3)
    cells = cells[..., 0] if torch.equal(cells[..., 0], cells[..., 2]) else cells.mean(-1)
    dark = 1.0 - cells
    ink_px = (dark > INK_DARKNESS).flatten(1).sum(1)
    return {
        "n_soft": n,
        "inked_ge1": int((ink_px >= 1).sum()),
        "inked_ge1pct": int((ink_px >= 23).sum()),
        "page_ink_px": int(ink_px.sum()),
    }


def _randstr_text(n_groups: int, group_len: int, seed: int) -> str:
    import random

    rng = random.Random(seed)
    return " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(group_len))
                    for _ in range(n_groups))


def ink_table(n_rows: int, seed: int) -> dict:
    """Inked tokens / chars-per-inked-token for each (lane, canvas, budget) cell.

    Lanes are the H13 rung geometries re-rendered at each canvas with the font
    scaled proportionally, plus real fineweb-edu page-0 documents.  The randstr
    text is IDENTICAL across canvases (same seed), so the only variable is the
    render scale.
    """
    import numpy as np

    from data.preprocessing.render_utils import render_text_pages, wrap_text_pages

    lanes = {"d1": (5, 5), "d3": (80, 5), "d4": (320, 5)}
    texts = {k: [_randstr_text(g, gl, seed + i) for i in range(n_rows)]
             for k, (g, gl) in lanes.items()}

    fineweb_docs = []
    fw_path = ROOT / FINEWEB_VAL
    if fw_path.exists():
        from datasets import load_from_disk

        ds = load_from_disk(str(fw_path))
        for i in range(min(n_rows, len(ds))):
            for msg in ds[i]["messages"]:
                if msg["role"] == "assistant":
                    c = msg["content"]
                    fineweb_docs.append("".join(p.get("text") or "" for p in c)
                                        if isinstance(c, list) else str(c))
                    break
    else:
        logger.warning("fineweb split missing at %s -- skipping that lane", fw_path)

    out = {}
    for lane in list(lanes) + (["fineweb"] if fineweb_docs else []):
        out[lane] = {}
        for canvas, fs in CANVASES:
            per_budget = {b: [] for b in BUDGETS}
            chars = []
            for i in range(n_rows):
                if lane == "fineweb":
                    full = fineweb_docs[i]
                    lines, lpp, _ = wrap_text_pages(full, canvas_width=canvas,
                                                    canvas_height=canvas, font_size=fs)
                    text = "\n".join(lines[:lpp])
                else:
                    text = texts[lane][i]
                pages = render_text_pages(text, canvas_width=canvas, canvas_height=canvas,
                                          font_size=fs)
                img = pages[0].convert("L")
                chars.append(len(text))
                for b in BUDGETS:
                    per_budget[b].append(ink_counts(img, b))
            cell = {}
            for b in BUDGETS:
                inked = np.array([r["inked_ge1"] for r in per_budget[b]], dtype=float)
                nsoft = np.array([r["n_soft"] for r in per_budget[b]], dtype=float)
                ch = np.array(chars, dtype=float)
                cell[str(b)] = {
                    "n_soft": float(nsoft.mean()),
                    "inked_ge1_mean": float(inked.mean()),
                    "inked_ge1_sd": float(inked.std(ddof=1)) if len(inked) > 1 else None,
                    "inked_fraction": float((inked / nsoft).mean()),
                    "chars_mean": float(ch.mean()),
                    # per-row mean of the ratio (the estimator the H17 doc uses)
                    "chars_per_inked_token": float(np.mean(ch / np.maximum(inked, 1))),
                    "inked_ge1pct_mean": float(
                        np.mean([r["inked_ge1pct"] for r in per_budget[b]])),
                }
            out[lane][str(canvas)] = {"font_size": fs, "n_rows": n_rows, "budgets": cell}
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test(verbose: bool = True) -> dict:
    """Prove the two reconstructions this script relies on.

    S1  ``model_visible`` really is the processor's output: reassembling the patch
        rasters and re-patchifying must return the identical tensor.
    S2  the letter-crop mapping lands on the right glyph: a page whose letters are
        all 'i' except ONE 'W' must have its brightest-ink crop at that index --
        and a deliberately shifted crop origin must NOT.
    S3  the geometry claim: for a square page the resized side depends only on the
        budget (checked over 1024/1536/2048/1584/1104/768).
    S4  ink counts are threshold-insensitive on these pages: >=1 and >=1%-of-cell
        inked counts must agree within 5%.
    """
    import numpy as np
    import torch
    from PIL import Image, ImageFont

    from data.preprocessing.render_utils import _find_font, render_text_pages

    res = {"checks": {}, "failures": []}

    def _check(name, ok, detail=""):
        res["checks"][name] = bool(ok)
        if not ok:
            res["failures"].append(f"{name}: {detail}")
        if verbose:
            logger.info("  %-52s %s %s", name, "PASS" if ok else "FAIL", detail)

    page, idx, adv, lh = render_letter_page(6, 1024, 14, 0)

    # S1 -------------------------------------------------------------------
    ok_all = []
    for b in BUDGETS:
        vis, (gx, gy), side = model_visible(page, b)
        out = processor(b)(images=[page], return_tensors="pt")
        n = int(out["num_soft_tokens_per_image"][0])
        pos = out["image_position_ids"][0, :n]
        pv = out["pixel_values"][0, :n]
        re_pv = torch.stack([torch.from_numpy(
            vis[int(pos[m, 1]) * side:(int(pos[m, 1]) + 1) * side,
                int(pos[m, 0]) * side:(int(pos[m, 0]) + 1) * side]) for m in range(n)])
        ok_all.append(torch.equal(re_pv, pv.reshape(n, side, side, 3)[..., 0]))
    _check("S1 model_visible round-trips pixel_values", all(ok_all), str(ok_all))

    # S2 -------------------------------------------------------------------
    font = ImageFont.truetype(_find_font(), 14)
    per_line = int((1024 - 40) // font.getlength("a"))
    target = (2, 7)
    lines = ["i" * per_line for _ in range(4)]
    lines[target[0]] = "i" * target[1] + "W" + "i" * (per_line - target[1] - 1)
    pg = render_text_pages("\n".join(lines), canvas_width=1024, canvas_height=1024,
                           font_size=14)[0].convert("L")
    ix = [(li, ci, ("W" if (li, ci) == target else "i")) for li in range(4)
          for ci in range(per_line)]
    vis, _g, _s = model_visible(pg, 280)
    X, y, _shape = letter_crops(vis, ix, adv, lh, vis.shape[1] / 1024.0)
    mass = (1.0 - X).reshape(len(X), -1).sum(1)
    _check("S2 crop mapping finds the one W", y[int(np.argmax(mass))] == "W",
           f"argmax label {y[int(np.argmax(mass))]}")
    Xs, ys, _ = letter_crops(vis, ix, adv, lh, vis.shape[1] / 1024.0, margin=20 + 9)
    mass_s = (1.0 - Xs).reshape(len(Xs), -1).sum(1)
    _check("S2b MUTATION: shifted origin must miss it",
           ys[int(np.argmax(mass_s))] != "W" or float(mass_s.max() / mass.max()) < 0.9,
           f"shifted argmax label {ys[int(np.argmax(mass_s))]}, "
           f"mass ratio {float(mass_s.max() / mass.max()):.3f}")

    # S3 -------------------------------------------------------------------
    sides = {b: {c: resized_side(b, c, c)[0] for c in (768, 1024, 1104, 1536, 1584, 2048)}
             for b in BUDGETS}
    _check("S3 square resize depends only on budget",
           all(len(set(v.values())) == 1 for v in sides.values()),
           str({b: sorted(set(v.values())) for b, v in sides.items()}))
    _check("S3b the three targets are 768/1104/1584",
           [sorted(set(sides[b].values()))[0] for b in BUDGETS] == [768, 1104, 1584],
           str([sorted(set(sides[b].values()))[0] for b in BUDGETS]))

    # S4 -------------------------------------------------------------------
    txt = _randstr_text(80, 5, 0)
    img = render_text_pages(txt, canvas_width=1024, canvas_height=1024,
                            font_size=14)[0].convert("L")
    rel = []
    for b in BUDGETS:
        c = ink_counts(img, b)
        rel.append(abs(c["inked_ge1"] - c["inked_ge1pct"]) / max(c["inked_ge1"], 1))
    _check("S4 ink threshold does no hidden work (<5%)", max(rel) < 0.05,
           str([round(r, 4) for r in rel]))

    # sanity: a blank page must have zero ink
    _check("S4b blank page has zero inked cells",
           ink_counts(Image.new("L", (1024, 1024), 255), 280)["inked_ge1"] == 0)

    res["all_passed"] = all(res["checks"].values())
    return res


# ---------------------------------------------------------------------------
def print_report(res):
    g = res["geometry"]
    print("\n" + "=" * 104)
    print("H17 — RENDER RESOLUTION vs SOFT-TOKEN BUDGET")
    print("=" * 104)

    print("\n1. PROCESSOR GEOMETRY (measured)")
    hdr = (f"{'source':>12}{'budget':>8}{'resized':>12}{'scale':>8}{'grid':>9}"
           f"{'n_soft':>8}{'maxpos':>9}{'cell=src px':>13}{'':>7}")
    print(hdr); print("-" * len(hdr))
    for r in g["text_pages"] + g["audio_page"]:
        h, w = r["source_hw"]
        rh, rw = r["resized_hw_measured"]
        gx, gy = r["grid_wh"]
        mx, my = r["max_image_position_id"]
        print(f"{f'{w}x{h}':>12}{r['budget']:>8}{f'{rw}x{rh}':>12}{r['scale_w']:>8.3f}"
              f"{f'{gx}x{gy}':>9}{r['n_soft_tokens']:>8}{f'{mx}/{my}':>9}"
              f"{r['cell_px_in_source']:>13.1f}"
              f"{('DOWN' if r['is_downsample'] else 'up'):>7}")

    print("\n2. GLYPH SCALE (original px -> model-visible px)")
    hdr = (f"{'canvas':>8}{'font':>6}{'lh':>5}{'adv':>7}{'x-ht':>6}{'cap':>5}{'stem':>6}"
           f"{'ch/page':>9}   | model-visible x-height @280/560/1120 | real px/char-cell")
    print(hdr); print("-" * len(hdr))
    for r in res["glyph"]["rows"]:
        mv = r["model_visible"]
        print(f"{r['canvas']:>8}{r['font_size']:>6}{r['line_height_px']:>5}"
              f"{r['advance_px']:>7.2f}{r['x_height_px']:>6}{r['cap_height_px']:>5}"
              f"{r['stem']['ink_mass_width_px']:>6.2f}{r['chars_per_page']:>9}   | "
              + "  ".join(f"{mv[str(b)]['x_height_px']:>5.2f}" for b in BUDGETS)
              + "        | "
              + "  ".join(f"{mv[str(b)]['real_px_per_char_cell']:>6.1f}" for b in BUDGETS))

    print("\n3. LEGIBILITY — 1-NN letter identity (deterministic; confusions = aliasing only)")
    hdr = (f"{'canvas':>8}{'font':>6}{'budget':>8}{'scale':>7}{'crop':>9}{'acc':>9}"
           f"{'minRMS':>9}{'fisher':>8}   acc @ noise sigma")
    print(hdr); print("-" * len(hdr))
    for r in res["legibility"]["rows"]:
        noise = "  ".join(f"{s}:{v:.3f}" for s, v in r["acc_1nn_noise"].items())
        ch, cw = r["crop_hw"]
        print(f"{r['canvas']:>8}{r['font_size']:>6}{r['budget']:>8}{r['resize_scale']:>7.3f}"
              f"{f'{cw}x{ch}':>9}"
              f"{r['acc_1nn_clean']:>9.4f}{r['between_class_rms_min']:>9.4f}"
              f"{(r['fisher_ratio_min'] or 0):>8.3f}   {noise}")

    print("\n4. INK OCCUPANCY across canvases (inked>=1 / fraction / chars-per-inked-token)")
    for lane, per_canvas in res["ink"].items():
        print(f"\n  lane {lane}")
        hdr = (f"    {'canvas':>7}{'font':>5}{'chars':>7}" +
               "".join(f"{'inked@' + str(b):>12}" for b in BUDGETS) +
               "".join(f"{'frac@' + str(b):>11}" for b in BUDGETS) +
               "".join(f"{'ch/ink@' + str(b):>12}" for b in BUDGETS))
        print(hdr); print("    " + "-" * (len(hdr) - 4))
        for canvas, cell in per_canvas.items():
            b0 = cell["budgets"]
            print(f"    {canvas:>7}{cell['font_size']:>5}{b0['280']['chars_mean']:>7.0f}"
                  + "".join(f"{b0[str(b)]['inked_ge1_mean']:>12.1f}" for b in BUDGETS)
                  + "".join(f"{b0[str(b)]['inked_fraction'] * 100:>10.1f}%" for b in BUDGETS)
                  + "".join(f"{b0[str(b)]['chars_per_inked_token']:>12.2f}" for b in BUDGETS))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-rows", type=int, default=12, help="rows per (lane, canvas) ink cell")
    ap.add_argument("--n-lines", type=int, default=22, help="letter lines for the legibility page")
    ap.add_argument("--sigma", type=float, action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    sigmas = tuple(args.sigma or (0.05, 0.15, 0.30))

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        return 0 if st["all_passed"] else 1

    st = self_test()
    if not st["all_passed"]:
        logger.error("SELF-TEST FAILED — refusing to report numbers: %s", st["failures"])
        return 1

    res = {
        "meta": {"budgets": list(BUDGETS), "canvases": [list(c) for c in CANVASES],
                 "n_rows": args.n_rows, "n_lines": args.n_lines, "seed": args.seed,
                 "device": "cpu (no model weights; a training run holds the GPU)"},
        "self_test": st,
        "geometry": geometry_table(),
        "glyph": glyph_table(),
        "legibility": legibility_table(args.n_lines, sigmas, args.seed),
        "ink": ink_table(args.n_rows, args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    logger.info("wrote %s", out)
    print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
