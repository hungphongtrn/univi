"""H20 — re-materialize an audio lane at a WIDER render, in image space.

Why this exists
---------------
The Gemma4 unified image processor targets a constant *area*::

    s = sqrt(B * 48**2 / (H * W));  target = (floor(s*H/48)*48, floor(s*W/48)*48)

so the soft-token grid is ``columns = sqrt(B * W/H)`` and the token count is ~B
whatever the aspect ratio.  Widening the render therefore buys time resolution
**at no token cost**, and moves ms/column exactly as hard as the budget does
(see ``data/eval/h20-audio-resolution.json``).  A 2000x160 page at budget 1120
is 118x9 = 1062 soft tokens at 84.7 ms/column -- *fewer* tokens than the 1079
the production 1000x160 page costs at the same budget, and 1.42x finer in time.

Why image space rather than re-rendering from audio
---------------------------------------------------
At a fixed hop (10 ms) the production page is already 1 render pixel per mel
frame, so the extra width is pure interpolation **either way**: LANCZOS-widening
the stored PNG and re-rendering the mel at ``output_width 2000`` differ by
<1% of the page's own sd (``--self-test`` measures this on real LibriSpeech
audio; H20 reports 0.72% by an independent path).  Re-rendering would need the
169 GB audio corpus and hours of mel recomputation for a result that is
numerically the same page.

The renderer's own resize (``render_utils._render_log_mel_image``) is
``Image.LANCZOS``; this script uses the same filter, so a widened page is the
page the renderer would have produced up to that <1%.

Usage
-----
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
        -m data.preprocessing.widen_audio_pages --self-test

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
        -m data.preprocessing.widen_audio_pages \
        --source data/materialized/univi-3M-v0-split/librispeech \
        --out    data/materialized/h20-audio-wide-v0/librispeech \
        --splits train validation --output-width 2000 --num-proc 4

CPU only.  No GPU, no model weights, no audio corpus (except ``--self-test``).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_WIDEN_VERSION = "0.1.0"
# The renderer's own resampling filter -- keep these in lockstep.
_RESAMPLE = Image.LANCZOS


def widen_page(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize one rendered spectrogram page to ``(width, height)``.

    Identity when the page is already that size, so re-running is free and a
    mixed-geometry lane converges rather than degrading.
    """
    if image.size == (width, height):
        return image
    return image.resize((width, height), _RESAMPLE)


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _rewrite_render_config(raw: str, width: int, height: int) -> str:
    """Keep the provenance honest: the stored geometry must match the pixels."""
    try:
        cfg = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        cfg = {}
    old_w = cfg.get("output_width", cfg.get("image_width"))
    old_h = cfg.get("output_height", cfg.get("image_height"))
    cfg["output_width"] = width
    cfg["output_height"] = height
    cfg["image_width"] = width
    cfg["image_height"] = height
    page_ms = float(cfg.get("page_duration_sec", 10.0)) * 1000.0
    cfg["ms_per_horizontal_pixel"] = round(page_ms / width, 6)
    cfg["widened_from"] = [old_w, old_h]
    cfg["widen_method"] = f"PIL.Image.LANCZOS (widen_audio_pages {_WIDEN_VERSION})"
    return json.dumps(cfg, sort_keys=True)


def _row_fn(width: int, height: int):
    def _fn(batch: dict) -> dict:
        out_images = []
        for images in batch["images"]:
            out_images.append(
                [
                    {"bytes": _encode_png(widen_page(im, width, height)), "path": None}
                    for im in images
                ]
            )
        out = {"images": out_images}
        if "render_config" in batch:
            out["render_config"] = [
                _rewrite_render_config(rc, width, height)
                for rc in batch["render_config"]
            ]
        return out

    return _fn


def widen_split(
    source: Path,
    out: Path,
    width: int,
    height: int,
    num_proc: int,
    writer_batch_size: int,
    limit: int | None = None,
    select_n: int | None = None,
    select_seed: int = 42,
):
    import shutil

    from datasets import load_from_disk

    ds = load_from_disk(str(source))
    if select_n is not None and select_n < len(ds):
        ds = ds.shuffle(seed=select_seed).select(range(select_n)).flatten_indices()
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    features = ds.features
    out.parent.mkdir(parents=True, exist_ok=True)
    # `map`'s default cache location is the SOURCE dataset's own directory. On
    # librispeech that is ~39 GB of shard files dumped into the shared
    # univi-3M-v0-split artifact, which then survives the run. Pin the cache
    # next to the OUTPUT instead and delete it once the result is saved.
    cache_dir = out.parent / f".{out.name}.mapcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        widened = ds.map(
            _row_fn(width, height),
            batched=True,
            batch_size=16,
            num_proc=num_proc if num_proc and num_proc > 1 else None,
            features=features,
            writer_batch_size=writer_batch_size,
            cache_file_name=str(cache_dir / "widen.arrow"),
            desc=f"Widening {source.name} -> {width}x{height}",
        )
        widened.save_to_disk(str(out))
        rows = len(widened)
        del widened
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return rows


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def _flac_paths(root: Path, k: int) -> list[Path]:
    return sorted(root.rglob("*.flac"))[:k]


def self_test(n_clips: int = 6, verbose: bool = True) -> dict:
    """Gate the two claims this script rests on.

    S1-S4 are pure-function properties (no corpus).  S5 is the one that
    matters: on REAL audio, does widening the stored 1000-wide page land on the
    page the renderer would have drawn at ``output_width 2000``?
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.preprocessing.render_utils import render_log_mel_spectrogram

    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    # S1 identity when the size already matches (idempotent re-runs).
    rng = np.random.default_rng(0)
    src = Image.fromarray(
        rng.integers(0, 256, size=(160, 1000, 3), dtype=np.uint8), mode="RGB"
    )
    checks["S1 widen is identity at the source size"] = (
        widen_page(src, 1000, 160).tobytes() == src.tobytes()
    )

    # S2 output geometry is exactly what was asked for.
    wide = widen_page(src, 2000, 160)
    checks["S2 widened page is exactly WxH"] = wide.size == (2000, 160)

    # S3 determinism (the materializer must be re-runnable).
    checks["S3 widen is deterministic"] = (
        widen_page(src, 2000, 160).tobytes() == wide.tobytes()
    )

    # S4 render_config is rewritten consistently.
    cfg = json.loads(
        _rewrite_render_config(
            json.dumps({"output_width": 1000, "output_height": 160,
                        "page_duration_sec": 10.0, "source_duration_ms": 4321.0}),
            2000, 160,
        )
    )
    checks["S4 render_config geometry rewritten"] = (
        cfg["output_width"] == 2000
        and cfg["image_width"] == 2000
        and cfg["ms_per_horizontal_pixel"] == 5.0
        and cfg["widened_from"] == [1000, 160]
        and cfg["source_duration_ms"] == 4321.0  # provenance preserved
    )

    # S5 the load-bearing one: widen(render@1000) vs render@2000 on real audio.
    corpus = Path("data/eval/decodability/LibriSpeech")
    clips = _flac_paths(corpus, n_clips) if corpus.exists() else []
    if clips:
        import soundfile as sf

        per_clip = []
        for path in clips:
            samples, sr = sf.read(str(path), dtype="float32")
            kw = dict(
                sample_rate=16000, source_sample_rate=int(sr), n_mels=80,
                n_fft=400, hop_length=160, window="hann", power=2.0,
                page_duration_sec=10.0, max_pages=4, output_height=160,
            )
            pages_1000 = render_log_mel_spectrogram(samples, output_width=1000, **kw)
            pages_2000 = render_log_mel_spectrogram(samples, output_width=2000, **kw)
            pages_1000 = pages_1000 if isinstance(pages_1000, list) else [pages_1000]
            pages_2000 = pages_2000 if isinstance(pages_2000, list) else [pages_2000]
            for p1, p2 in zip(pages_1000, pages_2000):
                a = np.asarray(widen_page(p1, 2000, 160).convert("L"), np.float64) / 255
                b = np.asarray(p2.convert("L"), np.float64) / 255
                per_clip.append(
                    {
                        "clip": path.name,
                        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                        "sd": float(b.std()),
                    }
                )
        rmse = float(np.mean([c["rmse"] for c in per_clip]))
        sd = float(np.mean([c["sd"] for c in per_clip]))
        detail["S5 stretch_equivalence"] = {
            "n_pages": len(per_clip),
            "rmse_mean": rmse,
            "sd_mean": sd,
            "rmse_over_sd": rmse / sd,
            "per_clip": per_clip,
        }
        checks["S5 widen(render@1000) ~= render@2000 on real audio (<3% of sd)"] = (
            rmse / sd < 0.03
        )
    else:
        detail["S5 stretch_equivalence"] = "SKIPPED: no LibriSpeech corpus on disk"

    # S6 negative control on synthetic pages: widening a different page must
    # NOT reproduce the target (guards against a vacuous S5).
    other = Image.fromarray(
        np.random.default_rng(1).integers(0, 256, (160, 1000, 3), dtype=np.uint8),
        mode="RGB",
    )
    a = np.asarray(widen_page(src, 2000, 160).convert("L"), np.float64)
    b = np.asarray(widen_page(other, 2000, 160).convert("L"), np.float64)
    checks["S6 MUTATION: a different page must NOT match"] = (
        np.sqrt(np.mean((a - b) ** 2)) > 1.0
    )

    failures = [k for k, v in checks.items() if not v]
    result = {"checks": checks, "failures": failures, "all_passed": not failures,
              "detail": detail}
    if verbose:
        for key, value in checks.items():
            print(f"  [{'PASS' if value else 'FAIL'}] {key}")
        eq = detail.get("S5 stretch_equivalence")
        if isinstance(eq, dict):
            print(
                f"  S5: {eq['n_pages']} real pages, rmse/sd = "
                f"{100 * eq['rmse_over_sd']:.3f}% of page sd"
            )
        else:
            print(f"  S5: {eq}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="materialized lane dir holding the splits")
    ap.add_argument("--out", help="destination lane dir")
    ap.add_argument("--splits", nargs="*", default=["train", "validation"])
    ap.add_argument("--output-width", type=int, default=2000)
    ap.add_argument("--output-height", type=int, default=160)
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--writer-batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows")
    ap.add_argument(
        "--select-n", type=int, default=None,
        help="deterministic shuffle(seed)+select(N) BEFORE widening (subsample a lane)",
    )
    ap.add_argument("--select-seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-clips", type=int, default=6)
    args = ap.parse_args(argv)

    if args.self_test:
        result = self_test(args.self_test_clips)
        print("SELF-TEST", "PASSED" if result["all_passed"] else "FAILED")
        if not result["all_passed"]:
            raise SystemExit(1)
        if not (args.source and args.out):
            return

    if not (args.source and args.out):
        ap.error("--source and --out are required unless only --self-test is given")

    for split in args.splits:
        src = Path(args.source) / split
        dst = Path(args.out) / split
        if not src.exists():
            print(f"[skip] {src} does not exist")
            continue
        rows = widen_split(
            src, dst, args.output_width, args.output_height,
            args.num_proc, args.writer_batch_size, args.limit,
            args.select_n if split == "train" else None, args.select_seed,
        )
        print(f"[done] {src} -> {dst}  rows={rows}  "
              f"geometry={args.output_width}x{args.output_height}")


if __name__ == "__main__":
    main()
