"""
H20 — what does raising ``max_soft_tokens`` actually do to an AUDIO page?

H20 proposes 280 -> 1120 to get "~120 ms per token-column" instead of 244, and
pre-registers a criterion whose refute branch is expensive ("the vision path
cannot represent speech at any budget we can afford").  H17's fifth correction
reports, from one measurement, that the 1000x160 spectrogram page is *upsampled
at every budget* (1.97x even at 280) -- which, if true, means the whole ladder
is grid subdivision over pixels that were already fully resolved.  That claim
materially changes H20's rationale and a ~9 h GPU run rests on it, so it is
re-derived here independently, on REAL materialized pages, by a different code
path, with the non-square resize rule worked out rather than assumed.

WHAT THIS MEASURES
  1. inventory   -- the stored page dimensions of both audio lanes (librispeech,
                    spoken-digits), pages per row, and how much of the time axis
                    is zero-padded silence (exact, from ``render_config``).
  2. rule        -- the processor's resize rule for NON-SQUARE inputs, as a
                    closed form, checked against the processor over a sweep.
  3. geometry    -- per budget, on real pages: resized px, grid, token count,
                    max image_position_ids, PER-AXIS scale, ms of audio per
                    patch cell, mel frames per cell, mel bins per cell.
  4. occupancy   -- a spectrogram has no "blank paper", so three complementary
                    definitions are reported (see ``occupancy``), plus the
                    threshold sweep that shows no threshold is doing the work.
  5. levers      -- what a wider / shorter render does to the same quantities,
                    and the critical render width at which the processor starts
                    THROWING AWAY real mel frames.
  6. redundancy  -- whether halving the hop supplies information a LANCZOS
                    stretch of the 10 ms render does not (real audio, real mels).
  7. capacity    -- real source pixels vs emitted embedding dimensions, i.e. the
                    "optical detail available" vs "capacity to encode it" split.

CPU ONLY.  No model weights, no GPU (a training run holds the card; this repo's
rule is that training runs SOLO).  ``univi/hybrid/vision.py`` is loaded BY FILE
PATH because ``univi/__init__.py`` imports unsloth, which raises without an
accelerator.

Usage:
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h20_audio_resolution.py --self-test
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h20_audio_resolution.py -n 24
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
logger = logging.getLogger("h20_audio_resolution")

DEFAULT_OUT = "data/eval/h20-audio-resolution.json"
BUDGETS = (280, 560, 1120)
CELL = 48                       # merged model patch side = patch_size * pooling_kernel_size
CELL_AREA = CELL * CELL
LANES = {
    "librispeech": "data/materialized/univi-3M-v0-split/librispeech",
    "spoken-digits": "data/materialized/spoken-digits-v0/spoken-digits",
}
LIBRISPEECH_FLAC = "data/eval/decodability/LibriSpeech/test-clean"


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


def load_trainer_module():
    """``univi/trainer.py`` with ``unsloth`` stubbed out.

    Only ``image_token_budget`` / ``_filter_training_tokens`` are wanted and
    neither touches unsloth, but the module imports it at top level for
    monkey-patch ordering, and unsloth raises without an accelerator.  The stub
    is registered ONLY if the real import would fail.
    """
    import importlib.util
    import types

    name = "_univi_trainer_cpu"
    if name in sys.modules:
        return sys.modules[name]
    if "unsloth" not in sys.modules:
        try:
            import unsloth  # noqa: F401
        except Exception:
            import importlib.machinery

            stub = types.ModuleType("unsloth")
            stub.FastVisionModel = type("FastVisionModel", (), {})
            # subclassed at import time by trainer.CheckedUnslothCollator
            stub.UnslothVisionDataCollator = type("UnslothVisionDataCollator", (), {})
            stub.is_bfloat16_supported = lambda: False
            stub.__univi_stub__ = True
            # trl calls importlib.util.find_spec("unsloth"), which raises if a
            # module object in sys.modules has __spec__ = None.
            stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
            sys.modules["unsloth"] = stub
    spec = importlib.util.spec_from_file_location(name, ROOT / "univi" / "trainer.py")
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


def resized_hw(budget: int, height: int, width: int) -> tuple[int, int]:
    return load_vision_module().get_aspect_ratio_preserving_size(
        height=height, width=width, patch_size=16, max_patches=budget * 9, pooling_kernel_size=3
    )


# ---------------------------------------------------------------------------
# 2. the resize rule, as a closed form
# ---------------------------------------------------------------------------
def closed_form_hw(budget: int, height: int, width: int) -> tuple[int, int]:
    """The processor's target size, re-derived from first principles.

    ``get_aspect_ratio_preserving_size`` picks the largest aspect-preserving
    size whose 16px-patch count fits ``budget * 9`` and whose sides are
    divisible by 48.  Substituting ``max_patches = 9B`` and ``patch_size = 16``:

        target_area = 9B * 16^2 = B * 48^2          (i.e. exactly B cells)
        s           = sqrt(target_area / (H * W))   (aspect-preserving scale)
        target_h    = floor(s * H / 48) * 48
        target_w    = floor(s * W / 48) * 48

    So the processor targets a constant *area* of B cells, NOT a constant side.
    For a SQUARE page that collapses to "the side depends only on B" (H17's
    rule).  For a non-square page the sides depend on the source aspect ratio,
    and the up/down-sample direction is decided by source AREA vs ``B * 2304``.

    The degenerate branch (an aspect ratio so extreme that one side floors to 0
    cells) is replicated too, so ``self_test`` can assert equality over a sweep
    that includes it.  No page in this repo comes near it.
    """
    s = math.sqrt(budget * CELL_AREA / float(height * width))
    th = int(math.floor(s * height / CELL)) * CELL
    tw = int(math.floor(s * width / CELL)) * CELL
    max_side = budget * CELL
    if th == 0 and tw == 0:
        raise ValueError("0 x 0")
    if th == 0:
        th = CELL
        tw = min(int(math.floor(width / height)) * CELL, max_side)
    elif tw == 0:
        tw = CELL
        th = min(int(math.floor(height / width)) * CELL, max_side)
    return th, tw


def resize_fixed_point(budget: int, height: int, width: int, iters: int = 40):
    """A size the processor maps to ITSELF at this budget, reached by iteration.

    The floor-to-48 quantization leaves slack, so the processor's own output is
    generally NOT a fixed point (feeding a 288x1968 page back in grows it to
    288x2160).  The raster self-test needs a genuine fixed point so the patch
    pixels can be compared against the synthetic source directly, without
    reimplementing the resize.
    """
    h, w = height, width
    for _ in range(iters):
        nh, nw = resized_hw(budget, h, w)
        if (nh, nw) == (h, w):
            return h, w
        h, w = nh, nw
    raise RuntimeError(f"no fixed point for budget {budget} from {height}x{width}")


def grid_for(budget: int, height: int, width: int) -> tuple[int, int]:
    """``(grid_w, grid_h)`` = (time columns, frequency rows) of 48px cells."""
    th, tw = resized_hw(budget, height, width)
    return tw // CELL, th // CELL


# ---------------------------------------------------------------------------
# 3. geometry, straight from the processor, on a real page
# ---------------------------------------------------------------------------
def probe_geometry(image, budget: int, *, page_ms: float, n_mels: int) -> dict:
    out = processor(budget)(images=[image], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pos = out["image_position_ids"][0, :n]
    gx, gy = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
    w, h = image.size
    analytic = resized_hw(budget, h, w)
    closed = closed_form_hw(budget, h, w)
    measured = (gy * CELL, gx * CELL)
    scale_w, scale_h = measured[1] / w, measured[0] / h
    ms_per_source_px = page_ms / w
    mels_per_source_px = n_mels / h
    return {
        "budget": budget,
        "source_hw": [h, w],
        "resized_hw_measured": list(measured),
        "resized_hw_analytic": list(analytic),
        "resized_hw_closed_form": list(closed),
        "all_three_agree": tuple(analytic) == measured == tuple(closed),
        "n_soft_tokens": n,
        "budget_padded_to": budget,
        "grid_time_cols": gx,
        "grid_freq_rows": gy,
        "max_image_position_id_xy": [int(pos[:, 0].max()), int(pos[:, 1].max())],
        "scale_time_axis": scale_w,
        "scale_freq_axis": scale_h,
        "time_axis_upsampled": scale_w > 1.0,
        "freq_axis_upsampled": scale_h > 1.0,
        # --- what one patch cell covers, in units that mean something for audio
        "cell_source_px_time": CELL / scale_w,
        "cell_source_px_freq": CELL / scale_h,
        "ms_per_cell": (CELL / scale_w) * ms_per_source_px,
        "ms_per_token_column": page_ms / gx,
        "mel_frames_per_cell": (CELL / scale_w) * ms_per_source_px / 10.0,
        "mel_bins_per_cell": (CELL / scale_h) * mels_per_source_px,
        "source_px": h * w,
        "resized_px": measured[0] * measured[1],
        "area_ratio_resized_over_source": (measured[0] * measured[1]) / float(h * w),
    }


# ---------------------------------------------------------------------------
# 4. occupancy
# ---------------------------------------------------------------------------
def patch_cells(image, budget: int):
    """``(cells, positions, n_soft)``; ``cells[m]`` is the 48x48 grayscale patch."""
    import torch

    out = processor(budget)(images=[image], return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    pv = out["pixel_values"][0, :n]
    pos = out["image_position_ids"][0, :n]
    side = int(math.isqrt(pv.shape[-1] // 3))
    g = pv.reshape(n, side, side, 3)
    g = g[..., 0] if torch.equal(g[..., 0], g[..., 2]) else g.mean(-1)
    return g, pos, n


def occupancy(image, budget: int, *, page_ms: float, audio_ms: float) -> dict:
    """Occupied-token fraction for a page with no blank paper.

    A spectrogram has energy everywhere, so "inked" in H13's sense (any pixel
    darker than white) is vacuously ~100%.  Three definitions are reported and
    none is privileged:

    ``active_*``   cell mean darkness exceeds the page's own silence floor (the
                   1st percentile of pixel darkness, which on a padded page IS
                   the padded silence) by a margin.  Reported at three margins
                   so the threshold can be seen not to be doing the work.
    ``structured`` cell pixel standard deviation > 1/255: a cell that is
                   CONSTANT carries nothing beyond its mean, whatever its level.
                   Threshold-light and the closest analogue of "contains ink".
    ``over_audio`` the exact, threshold-free one: the fraction of grid cells
                   whose time span overlaps the un-padded part of the page,
                   computed from ``render_config``'s source duration.  On
                   librispeech this is the number that matters -- the last page
                   of a 14.1 s utterance is 59% zero-padding by construction.
    """
    import numpy as np

    cells, pos, n = patch_cells(image, budget)
    dark = (1.0 - cells).numpy()
    floor = float(np.percentile(dark, 1))
    cell_mean = dark.reshape(n, -1).mean(1)
    cell_std = dark.reshape(n, -1).std(1)
    gx = int(pos[:, 0].max()) + 1
    col_ms = page_ms / gx
    cols_with_audio = min(gx, int(math.ceil(audio_ms / col_ms))) if audio_ms > 0 else 0
    over_audio = int((pos[:n, 0].numpy() < cols_with_audio).sum())
    return {
        "n_soft": n,
        "silence_floor_darkness": floor,
        "active_frac_m0.02": float((cell_mean > floor + 0.02).mean()),
        "active_frac_m0.05": float((cell_mean > floor + 0.05).mean()),
        "active_frac_m0.10": float((cell_mean > floor + 0.10).mean()),
        "structured_frac": float((cell_std > 1.0 / 255.0).mean()),
        "cell_std_mean": float(cell_std.mean()),
        "cell_mean_darkness": float(cell_mean.mean()),
        "cols_with_audio": cols_with_audio,
        "grid_time_cols": gx,
        "over_audio_frac": over_audio / n,
    }


# ---------------------------------------------------------------------------
# 1. inventory of the real materialized pages
# ---------------------------------------------------------------------------
def inventory(lane: str, path: str, split: str, n: int) -> dict:
    from collections import Counter

    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / path / split))
    total = len(ds)
    take = min(n, total)
    sizes, pages, dur, cfgs = Counter(), Counter(), [], Counter()
    for i in range(take):
        r = ds[i]
        pages[len(r["images"])] += 1
        for im in r["images"]:
            sizes[f"{im.size[0]}x{im.size[1]}"] += 1
        rc = json.loads(r["render_config"])
        cfgs[json.dumps({k: rc.get(k) for k in
                         ("output_width", "output_height", "hop_length", "n_mels",
                          "page_duration_sec", "sample_rate", "n_fft")}, sort_keys=True)] += 1
        if "source_duration_ms" in rc:
            dur.append((rc["source_duration_ms"], len(r["images"])))
    out = {
        "lane": lane, "split": split, "rows_total": total, "rows_sampled": take,
        "page_sizes": dict(sizes), "pages_per_row": {str(k): v for k, v in sorted(pages.items())},
        "render_config_variants": {k: v for k, v in cfgs.items()},
    }
    if dur:
        import numpy as np

        cover = np.array([d / (p * 10000.0) for d, p in dur])
        out["audio_coverage_of_time_axis"] = {
            "mean": float(cover.mean()), "min": float(cover.min()),
            "max": float(cover.max()), "median": float(np.median(cover)),
            "note": "source_duration_ms / (n_pages * page_duration_ms); the rest is "
                    "right-padded silence that still costs soft tokens.",
        }
    return out


def stretch_equivalence(n_clips: int = 6, seed: int = 0) -> dict:
    """Is 'render at output_width 2000' reachable from the STORED 1000px pages?

    Re-rendering the lane from audio needs the source corpus; stretching the
    stored PNG needs nothing.  Both start from the SAME 80x1000 mel array when
    the hop is unchanged, so they should differ only by one extra resampling.
    Measured on real clips: render the mel at 2000x160 directly, versus render
    it at 1000x160 (the production page) and LANCZOS-stretch that to 2000x160.
    """
    import numpy as np
    import soundfile as sf
    from PIL import Image

    from data.preprocessing.render_utils import _render_log_mel_image, _whisper_log_mel

    flacs = sorted(Path(ROOT / LIBRISPEECH_FLAC).rglob("*.flac"))
    if not flacs:
        return {"skipped": "no LibriSpeech flac found"}
    rng = np.random.default_rng(seed)
    pick = [flacs[i] for i in rng.choice(len(flacs), min(n_clips, len(flacs)), replace=False)]
    rows = []
    for f in pick:
        y, sr = sf.read(str(f), dtype="float32")
        y = np.pad(y[: 10 * sr], (0, max(0, 10 * sr - len(y[: 10 * sr]))))
        mel = _whisper_log_mel(y, sample_rate=sr, n_mels=80, n_fft=400,
                               hop_length=160, window="hann", power=2.0)
        direct = np.asarray(_render_log_mel_image(mel, output_width=2000, output_height=160)
                            .convert("L")).astype(np.float32) / 255.0
        prod = _render_log_mel_image(mel, output_width=1000, output_height=160).convert("L")
        stretched = np.asarray(prod.resize((2000, 160), Image.LANCZOS)).astype(np.float32) / 255.0
        rows.append({"clip": f.name,
                     "rmse": float(np.sqrt(((direct - stretched) ** 2).mean())),
                     "sd": float(direct.std())})
    return {"n_clips": len(rows), "per_clip": rows,
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "sd_mean": float(np.mean([r["sd"] for r in rows])),
            "rmse_over_sd": float(np.mean([r["rmse"] for r in rows])
                                  / np.mean([r["sd"] for r in rows])),
            "note": "small => a wider page can be materialized from the EXISTING PNGs "
                    "with no audio corpus, because at a fixed hop the extra width is "
                    "interpolation either way."}


def upsample_roundtrip(n_pages: int = 6) -> dict:
    """Is the model-visible page an invertible view of the stored page?

    "Upsampled at every budget" is a geometric statement; this turns it into a
    measurement.  The processor's own output is reassembled from
    ``pixel_values``, resampled back down to the stored 1000x160, and compared
    to the stored page.  A small residual means the resize kept the source
    recoverable -- i.e. no optical detail was destroyed on the way in.

    CAVEAT: the round trip includes the *return* resize's own error, so the
    residual is an UPPER bound on what the processor's upsample lost, not an
    estimate of it.  A downsampling budget would be expected to show a much
    larger residual; none exists for these pages, so there is no negative
    control here beyond the text page, which is included for contrast.
    """
    import numpy as np
    from datasets import load_from_disk
    from PIL import Image

    ds = load_from_disk(str(ROOT / LANES["librispeech"] / "train"))
    rows = []
    for i in range(n_pages):
        img = ds[i]["images"][0].convert("L")
        src = np.asarray(img).astype(np.float32) / 255.0
        for b in BUDGETS:
            cells, pos, n = patch_cells(img, b)
            gx, gy = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
            vis = np.zeros((gy * CELL, gx * CELL), dtype=np.float32)
            c = cells.numpy()
            for m in range(n):
                x, y = int(pos[m, 0]), int(pos[m, 1])
                vis[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = c[m]
            back = np.asarray(Image.fromarray((vis * 255).astype(np.uint8), "L")
                              .resize(img.size, Image.LANCZOS)).astype(np.float32) / 255.0
            rows.append({"budget": b, "rmse": float(np.sqrt(((back - src) ** 2).mean())),
                         "src_sd": float(src.std())})
    agg = {}
    for b in BUDGETS:
        r = [x for x in rows if x["budget"] == b]
        agg[str(b)] = {
            "rmse_mean": float(np.mean([x["rmse"] for x in r])),
            "src_sd_mean": float(np.mean([x["src_sd"] for x in r])),
            "rmse_over_sd": float(np.mean([x["rmse"] for x in r])
                                  / np.mean([x["src_sd"] for x in r])),
        }
    return {"n_pages": n_pages, "per_budget": agg}


def padding_waste(split: str = "train") -> dict:
    """How much of librispeech's rendered time axis is right-padded silence.

    Whole-split, not a sample: only the ``render_config`` column is read, so no
    image is decoded.  ``source_duration_ms`` is written by the materializer, so
    this is exact rather than thresholded.  spoken-digits does not record a
    source duration and is therefore excluded (its occupancy figure is the
    ``structured_frac`` above, not this).
    """
    import numpy as np
    from datasets import load_from_disk

    ds = load_from_disk(str(ROOT / LANES["librispeech"] / split)).select_columns(
        ["render_config"])
    dur, npages = [], []
    for rc in ds["render_config"]:
        c = json.loads(rc)
        dur.append(float(c["source_duration_ms"]))
        npages.append(int(c["n_pages"]))
    dur = np.asarray(dur)
    npages = np.asarray(npages)
    cover = dur / (npages * 10000.0)
    hist = {int(k): int(v) for k, v in zip(*np.unique(npages, return_counts=True))}
    return {
        "split": split, "rows": len(dur), "pages_per_row_histogram": hist,
        "audio_coverage_mean": float(cover.mean()),
        "audio_coverage_median": float(np.median(cover)),
        "audio_coverage_min": float(cover.min()),
        "audio_coverage_max": float(cover.max()),
        "padded_silence_fraction_of_time_axis": float(1.0 - cover.mean()),
        "duration_sec_mean": float(dur.mean() / 1000.0),
        "total_pages": int(npages.sum()),
    }


# ---------------------------------------------------------------------------
# 5. render levers, analytic (geometry only -- no pixels needed)
# ---------------------------------------------------------------------------
def lever_row(budget: int, width: int, height: int, page_sec: float, n_mels: int,
              hop_ms: float) -> dict:
    th, tw = resized_hw(budget, height, width)
    gx, gy = tw // CELL, th // CELL
    scale_w, scale_h = tw / width, th / height
    real_frames = page_sec * 1000.0 / hop_ms
    # A source pixel column carries a real mel frame only if the render is not
    # itself stretching the mel array; ``px_per_real_frame`` <1 would mean the
    # render DROPPED frames before the processor ever saw them.
    px_per_real_frame = width / real_frames
    return {
        "budget": budget, "render_w": width, "render_h": height,
        "page_sec": page_sec, "hop_ms": hop_ms, "n_mels": n_mels,
        "real_mel_frames_per_page": real_frames,
        "render_px_per_real_frame": px_per_real_frame,
        "resized_hw": [th, tw], "grid_time_cols": gx, "grid_freq_rows": gy,
        "n_soft_tokens": gx * gy,
        "scale_time": scale_w, "scale_freq": scale_h,
        "time_axis_downsampled_by_processor": scale_w < 1.0,
        "freq_axis_downsampled_by_processor": scale_h < 1.0,
        "ms_per_token_column": page_sec * 1000.0 / gx,
        "real_frames_per_cell": real_frames / gx,
        "mel_bins_per_cell_row": n_mels / gy,
        "max_pos_id_x": gx - 1,
        # linear-algebra accounting: real mel samples vs emitted embedding dims
        "real_mel_samples": real_frames * n_mels,
        "emitted_dims_2048": gx * gy * 2048,
        "dims_per_real_mel_sample": gx * gy * 2048 / (real_frames * n_mels),
    }


def critical_width(budget: int, height: int) -> int:
    """Largest render width whose time axis the processor does NOT downsample.

    The processor downsamples once source area exceeds ``budget * 2304``; at
    fixed height that is a width threshold.  Returned exactly by search, not by
    the continuous formula, because of the floor-to-48 quantization.
    """
    lo, hi = 48, 200_000
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _th, tw = resized_hw(budget, height, mid)
        if tw >= mid:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def levers_table() -> dict:
    """The alternative-render sweep H20's design question turns on."""
    rows = []
    # (width, height, hop_ms, label)
    configs = [
        (1000, 160, 10.0, "production (hop 10 ms, 2x vertical stretch)"),
        (2000, 160, 10.0, "H20 doc's 'output_width 2000' -- LANCZOS stretch, hop UNCHANGED"),
        (2000, 160, 5.0, "width 2000 AND hop 5 ms -- 2x real frames"),
        (1000, 80, 10.0, "no vertical stretch (80 px = 80 mels)"),
        (2000, 80, 5.0, "hop 5 ms AND no vertical stretch"),
        (4000, 160, 2.5, "hop 2.5 ms, width 4000"),
        (4000, 80, 2.5, "hop 2.5 ms, width 4000, no vertical stretch"),
    ]
    for w, h, hop, label in configs:
        for b in BUDGETS:
            r = lever_row(b, w, h, 10.0, 80, hop)
            r["label"] = label
            rows.append(r)
    crit = {str(b): {str(h): critical_width(b, h) for h in (80, 160)} for b in BUDGETS}
    return {"rows": rows, "critical_render_width_no_time_downsample": crit,
            "note": "critical width = the largest render width at which the processor's "
                    "resize still keeps every source pixel column on the time axis. "
                    "Above it the extra mel frames are resized away before the model."}


# ---------------------------------------------------------------------------
# 6. does a smaller hop carry information a stretch does not?
# ---------------------------------------------------------------------------
def hop_redundancy(n_clips: int, seed: int) -> dict:
    """Real audio: is the hop-5ms mel predictable from the hop-10ms mel?

    If a LANCZOS/linear upsample of the 10 ms mel reproduces the 5 ms mel, then
    "re-render wider" adds no information and only reallocates grid cells.  If
    it does not, the finer hop supplies real detail.  Reported as normalized RMS
    residual against the same statistic for a shift-control (how much the mel
    changes over one 5 ms step at all), so a small residual cannot be read as
    "the signal is flat".
    """
    import numpy as np

    flacs = sorted(Path(ROOT / LIBRISPEECH_FLAC).rglob("*.flac"))
    if not flacs:
        return {"skipped": "no LibriSpeech flac found at " + LIBRISPEECH_FLAC}
    rng = np.random.default_rng(seed)
    pick = [flacs[i] for i in rng.choice(len(flacs), min(n_clips, len(flacs)), replace=False)]

    import soundfile as sf

    from data.preprocessing.render_utils import _whisper_log_mel

    rows = []
    for f in pick:
        y, sr = sf.read(str(f), dtype="float32")
        y = y[: 10 * sr]
        kw = dict(sample_rate=sr, n_mels=80, n_fft=400, window="hann", power=2.0)
        m10 = _whisper_log_mel(y, hop_length=160, **kw)
        m5 = _whisper_log_mel(y, hop_length=80, **kw)
        T = min(m5.shape[1], 2 * m10.shape[1])
        # linear interpolation of the 10 ms mel onto the 5 ms grid
        x10 = np.arange(m10.shape[1]) * 2.0
        x5 = np.arange(T)
        pred = np.stack([np.interp(x5, x10, m10[k]) for k in range(m10.shape[0])])
        res = m5[:, :T] - pred
        # control: how much does the 5 ms mel change between adjacent 5 ms frames?
        step = np.diff(m5[:, :T], axis=1)
        rows.append({
            "clip": f.name,
            "frames_10ms": int(m10.shape[1]), "frames_5ms": int(m5.shape[1]),
            "rms_residual_interp": float(np.sqrt((res ** 2).mean())),
            "rms_adjacent_frame_step": float(np.sqrt((step ** 2).mean())),
            "signal_sd": float(m5[:, :T].std()),
        })
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("rms_residual_interp", "rms_adjacent_frame_step", "signal_sd")}
    agg["residual_over_signal_sd"] = agg["rms_residual_interp"] / agg["signal_sd"]
    agg["residual_over_adjacent_step"] = (agg["rms_residual_interp"]
                                          / agg["rms_adjacent_frame_step"])
    return {"n_clips": len(rows), "per_clip": rows, "mean": agg,
            "note": "mel values are in the renderer's normalized [0,1] whisper scale. "
                    "residual_over_signal_sd near 0 => the 5 ms frames are predictable "
                    "from the 10 ms ones and a finer hop adds little; near 1 => it adds "
                    "a lot. The n_fft=400 (25 ms) analysis window bounds how much a "
                    "finer hop CAN add."}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test(verbose: bool = True) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    res = {"checks": {}, "failures": []}

    def _check(name, ok, detail=""):
        res["checks"][name] = bool(ok)
        if not ok:
            res["failures"].append(f"{name}: {detail}")
        if verbose:
            logger.info("  %-56s %s %s", name, "PASS" if ok else "FAIL", detail)

    # S1 closed form == the processor's own helper, over a wide sweep ---------
    bad = []
    for b in (70, 140, 280, 560, 1120):
        for h in (48, 80, 160, 240, 320, 512, 1024, 1536):
            for w in (96, 160, 500, 1000, 1024, 2000, 4000, 8000):
                if resized_hw(b, h, w) != closed_form_hw(b, h, w):
                    bad.append((b, h, w, resized_hw(b, h, w), closed_form_hw(b, h, w)))
    _check("S1 closed form matches processor helper", not bad, f"{len(bad)} mismatches {bad[:3]}")

    # S2 helper == what the processor really emits (grid * 48) ---------------
    bad2 = []
    for b in BUDGETS:
        for (w, h) in ((1000, 160), (2000, 160), (1000, 80), (4000, 160), (1024, 1024)):
            img = Image.new("RGB", (w, h), 255)
            out = processor(b)(images=[img], return_tensors="pt")
            n = int(out["num_soft_tokens_per_image"][0])
            pos = out["image_position_ids"][0, :n]
            gx, gy = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
            if (gy * CELL, gx * CELL) != resized_hw(b, h, w) or gx * gy != n:
                bad2.append((b, w, h, (gx, gy), n, resized_hw(b, h, w)))
    _check("S2 emitted grid == analytic resize / 48", not bad2, str(bad2[:3]))

    # S3 area rule: resized area <= budget cells, and > (budget - one row) ----
    viol = []
    for b in BUDGETS:
        for (w, h) in ((1000, 160), (2000, 160), (1000, 80), (1024, 1024)):
            th, tw = resized_hw(b, h, w)
            if (th // CELL) * (tw // CELL) > b:
                viol.append((b, w, h, th, tw))
    _check("S3 emitted tokens never exceed the budget", not viol, str(viol[:3]))

    # S4 pixel -> cell mapping on a NON-SQUARE page, with a mutation test -----
    #    Uses a genuine fixed point of the resize at this budget (see
    #    ``resize_fixed_point``) so the patch pixels are compared against the
    #    synthetic source directly, with no resize in between.
    th, tw = resize_fixed_point(280, 160, 1000)
    _check("S4pre fixed point is unresized and non-square",
           resized_hw(280, th, tw) == (th, tw) and th != tw, f"{tw}x{th}")
    img = Image.new("L", (tw, th), 255)
    a = np.asarray(img).copy()
    probes = {(0, 0), (7, 2), (tw // CELL - 1, th // CELL - 1), (3, th // CELL - 1)}
    for cx, cy in probes:
        a[cy * CELL + 19: cy * CELL + 29, cx * CELL + 19: cx * CELL + 29] = 0
    img = Image.fromarray(a, mode="L")
    src = torch.from_numpy(a.astype(np.float32) / 255.0)
    cells, pos, n = patch_cells(img, 280)
    mism = [m for m in range(n)
            if not torch.equal(cells[m], src[int(pos[m, 1]) * CELL:(int(pos[m, 1]) + 1) * CELL,
                                             int(pos[m, 0]) * CELL:(int(pos[m, 0]) + 1) * CELL])]
    _check("S4 non-square pixel->cell mapping exact", not mism, f"{len(mism)} bad")
    inked = {(int(pos[m, 0]), int(pos[m, 1])) for m in range(n)
             if (1.0 - cells[m] > 8 / 255).any()}
    _check("S4b probes land in the expected cells", inked == probes,
           f"got {sorted(inked)} want {sorted(probes)}")
    shifted = [m for m in range(n)
               if not torch.equal(cells[m],
                                  src[int(pos[m, 1]) * CELL:(int(pos[m, 1]) + 1) * CELL,
                                      ((int(pos[m, 0]) + 1) % (tw // CELL)) * CELL:
                                      ((int(pos[m, 0]) + 1) % (tw // CELL)) * CELL + CELL])]
    _check("S4c MUTATION: shifted mapping must FAIL", len(shifted) > 0, f"{len(shifted)} differ")

    # S5 the ms-per-cell arithmetic, checked against the column count --------
    g = probe_geometry(Image.new("RGB", (1000, 160), 255), 280, page_ms=10000.0, n_mels=80)
    _check("S5 ms_per_cell == page_ms / time columns",
           abs(g["ms_per_cell"] - g["ms_per_token_column"]) < 1e-6,
           f"{g['ms_per_cell']:.3f} vs {g['ms_per_token_column']:.3f}")

    # S6 occupancy has teeth: a CONSTANT page must read 0 structured/active --
    o = occupancy(Image.new("L", (1000, 160), 128), 280, page_ms=10000.0, audio_ms=0.0)
    _check("S6 constant page -> 0 structured, 0 active",
           o["structured_frac"] == 0.0 and o["active_frac_m0.02"] == 0.0,
           f"structured {o['structured_frac']}, active {o['active_frac_m0.02']}")

    res["all_passed"] = all(res["checks"].values())
    return res


# ---------------------------------------------------------------------------
# retention through the trainer's filter
# ---------------------------------------------------------------------------
def retention(max_length: int, budgets=(280, 1120)) -> dict:
    from datasets import load_from_disk

    tr = load_trainer_module()
    out = {"max_length": max_length,
           "image_token_budget": {str(b): tr.image_token_budget(b) for b in budgets},
           "legacy_constant": tr._IMAGE_TOKEN_BUDGET,
           "template_overhead": tr._TEMPLATE_TOKEN_OVERHEAD,
           "splits": {}}
    splits = [
        ("librispeech/train", "data/materialized/h20-audio-v0/librispeech/train"),
        ("librispeech/validation", "data/materialized/h20-audio-v0/librispeech/validation"),
        ("spoken-digits-15pct/train", "data/materialized/h20-audio-v0/spoken-digits-15pct/train"),
        ("spoken-digits/validation", "data/materialized/h20-audio-v0/spoken-digits/validation"),
    ]
    for name, p in splits:
        path = ROOT / p
        if not path.exists():
            out["splits"][name] = {"missing": str(path)}
            continue
        ds = load_from_disk(str(path))
        cell = {"rows": len(ds)}
        for b in budgets:
            kept = tr._filter_training_tokens(ds, name, max_length,
                                              per_image_tokens=tr.image_token_budget(b))
            cell[f"retained_at_{b}"] = len(kept)
            cell[f"retained_frac_at_{b}"] = len(kept) / len(ds)
        out["splits"][name] = cell
    return out


# ---------------------------------------------------------------------------
def measure_lanes(n: int, budgets) -> dict:
    from datasets import load_from_disk

    res = {}
    for lane, path in LANES.items():
        res[lane] = {"inventory": {}, "geometry": {}, "occupancy": {}}
        for split in ("train", "validation"):
            res[lane]["inventory"][split] = inventory(lane, path, split, n)
        ds = load_from_disk(str(ROOT / path / "train"))
        take = min(n, len(ds))
        geo_acc, occ_acc = {b: [] for b in budgets}, {b: [] for b in budgets}
        for i in range(take):
            r = ds[i]
            rc = json.loads(r["render_config"])
            page_ms = float(rc.get("page_duration_sec", 10.0)) * 1000.0
            n_mels = int(rc.get("n_mels", 80))
            audio_ms_known = "source_duration_ms" in rc
            audio_ms = float(rc.get("source_duration_ms", page_ms * len(r["images"])))
            for pi, im in enumerate(r["images"]):
                remaining = max(0.0, audio_ms - pi * page_ms)
                for b in budgets:
                    geo_acc[b].append(probe_geometry(im, b, page_ms=page_ms, n_mels=n_mels))
                    occ_acc[b].append(occupancy(im, b, page_ms=page_ms,
                                                audio_ms=min(remaining, page_ms)))
        for b in budgets:
            g0 = geo_acc[b][0]
            same = all(g["resized_hw_measured"] == g0["resized_hw_measured"]
                       and g["n_soft_tokens"] == g0["n_soft_tokens"] for g in geo_acc[b])
            g0 = dict(g0)
            g0["identical_across_all_pages"] = same
            g0["n_pages_probed"] = len(geo_acc[b])
            res[lane]["geometry"][str(b)] = g0
            keys = ("active_frac_m0.02", "active_frac_m0.05", "active_frac_m0.10",
                    "structured_frac", "over_audio_frac", "cell_std_mean",
                    "cell_mean_darkness", "silence_floor_darkness")
            import numpy as np

            res[lane]["occupancy"][str(b)] = {
                k: float(np.mean([o[k] for o in occ_acc[b]])) for k in keys}
            res[lane]["occupancy"][str(b)]["n_pages"] = len(occ_acc[b])
            res[lane]["occupancy"][str(b)]["over_audio_frac_is_meaningful"] = bool(audio_ms_known)
            if not audio_ms_known:
                res[lane]["occupancy"][str(b)]["over_audio_frac_note"] = (
                    "VACUOUS for this lane: its render_config records no "
                    "source_duration_ms, so the whole page is assumed to be audio and "
                    "the figure is 100% by construction. Use structured_frac instead.")
    return res


def print_report(res):
    print("\n" + "=" * 112)
    print("H20 — AUDIO PAGE RESOLUTION vs SOFT-TOKEN BUDGET")
    print("=" * 112)

    for lane, d in res["lanes"].items():
        inv = d["inventory"]["train"]
        print(f"\n{lane}: train {inv['rows_total']} rows (sampled {inv['rows_sampled']}), "
              f"page sizes {inv['page_sizes']}, pages/row {inv['pages_per_row']}")
        cov = inv.get("audio_coverage_of_time_axis")
        if cov:
            print(f"   audio covers {cov['mean'] * 100:.1f}% of the rendered time axis "
                  f"(min {cov['min'] * 100:.1f}%, max {cov['max'] * 100:.1f}%) — rest is padding")
        hdr = (f"   {'budget':>7}{'resized':>12}{'scale t/f':>13}{'grid t x f':>12}{'tok':>6}"
               f"{'maxpos':>9}{'ms/col':>9}{'frames/cell':>13}{'mels/cell':>11}{'':>7}")
        print(hdr); print("   " + "-" * (len(hdr) - 3))
        for b, g in d["geometry"].items():
            rh, rw = g["resized_hw_measured"]
            mx, my = g["max_image_position_id_xy"]
            scales = "%.2f/%.2f" % (g["scale_time_axis"], g["scale_freq_axis"])
            gridxy = "%dx%d" % (g["grid_time_cols"], g["grid_freq_rows"])
            print(f"   {b:>7}{f'{rw}x{rh}':>12}{scales:>13}{gridxy:>12}"
                  f"{g['n_soft_tokens']:>6}{f'{mx}/{my}':>9}{g['ms_per_token_column']:>9.1f}"
                  f"{g['mel_frames_per_cell']:>13.1f}{g['mel_bins_per_cell']:>11.1f}"
                  f"{'UP' if g['time_axis_upsampled'] else 'DOWN':>7}")
        hdr = (f"   {'budget':>7}{'structured':>12}{'active .02':>12}{'active .05':>12}"
               f"{'active .10':>12}{'over-audio':>12}")
        print(hdr); print("   " + "-" * (len(hdr) - 3))
        for b, o in d["occupancy"].items():
            print(f"   {b:>7}{o['structured_frac'] * 100:>11.1f}%{o['active_frac_m0.02'] * 100:>11.1f}%"
                  f"{o['active_frac_m0.05'] * 100:>11.1f}%{o['active_frac_m0.10'] * 100:>11.1f}%"
                  f"{o['over_audio_frac'] * 100:>11.1f}%")

    for split, p in res.get("padding_waste", {}).items():
        print(f"\nlibrispeech {split}: {p['rows']} rows, {p['total_pages']} pages, "
              f"pages/row {p['pages_per_row_histogram']}, mean duration "
              f"{p['duration_sec_mean']:.1f} s; audio covers "
              f"{p['audio_coverage_mean'] * 100:.1f}% of the rendered time axis "
              f"=> {p['padded_silence_fraction_of_time_axis'] * 100:.1f}% of soft tokens "
              f"are padded silence")

    print("\nRENDER LEVERS (analytic; 10 s page, 80 mels)")
    hdr = (f"   {'render':>12}{'hop':>6}{'budget':>8}{'grid t x f':>12}{'tok':>6}"
           f"{'ms/col':>9}{'real fr/cell':>14}{'mels/cell':>11}{'proc t-down':>12}   label")
    print(hdr); print("   " + "-" * (len(hdr) - 3))
    for r in res["levers"]["rows"]:
        render = "%dx%d" % (r["render_w"], r["render_h"])
        gridxy = "%dx%d" % (r["grid_time_cols"], r["grid_freq_rows"])
        print(f"   {render:>12}{r['hop_ms']:>6.1f}{r['budget']:>8}{gridxy:>12}"
              f"{r['n_soft_tokens']:>6}{r['ms_per_token_column']:>9.1f}"
              f"{r['real_frames_per_cell']:>14.1f}{r['mel_bins_per_cell_row']:>11.1f}"
              f"{str(r['time_axis_downsampled_by_processor']):>12}   {r['label']}")
    print(f"\n   critical render width (no time downsampling): "
          f"{res['levers']['critical_render_width_no_time_downsample']}")

    hr = res.get("hop_redundancy", {})
    if "mean" in hr:
        m = hr["mean"]
        print(f"\nHOP REDUNDANCY ({hr['n_clips']} real clips): interp residual "
              f"{m['rms_residual_interp']:.4f} vs signal sd {m['signal_sd']:.4f} "
              f"(ratio {m['residual_over_signal_sd']:.3f}), vs adjacent-frame step "
              f"{m['rms_adjacent_frame_step']:.4f} (ratio {m['residual_over_adjacent_step']:.3f})")

    ret = res.get("retention")
    if ret:
        print(f"\nRETENTION through _filter_training_tokens at max_length {ret['max_length']}")
        for name, cell in ret["splits"].items():
            if "missing" in cell:
                print(f"   {name:<30} MISSING {cell['missing']}")
                continue
            print(f"   {name:<30} rows {cell['rows']:>7}   "
                  + "   ".join(f"@{b}: {cell[f'retained_at_{b}']} "
                               f"({cell[f'retained_frac_at_{b}'] * 100:.2f}%)"
                               for b in (280, 1120)))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("-n", "--n-rows", type=int, default=24)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-retention", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        st = self_test()
        print(json.dumps(st, indent=2))
        return 0 if st["all_passed"] else 1

    st = self_test()
    if not st["all_passed"]:
        logger.error("SELF-TEST FAILED — refusing to report numbers: %s", st["failures"])
        return 1

    res = {
        "meta": {"budgets": list(BUDGETS), "n_rows": args.n_rows, "seed": args.seed,
                 "device": "cpu (no model weights; the GPU is reserved for a training run)"},
        "self_test": st,
        "lanes": measure_lanes(args.n_rows, BUDGETS),
        "padding_waste": {s: padding_waste(s) for s in ("train", "validation")},
        "upsample_roundtrip": upsample_roundtrip(),
        "stretch_equivalence": stretch_equivalence(args.clips, args.seed),
        "levers": levers_table(),
        "hop_redundancy": hop_redundancy(args.clips, args.seed),
    }
    if not args.no_retention:
        res["retention"] = retention(args.max_length)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    logger.info("wrote %s", out)
    print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
