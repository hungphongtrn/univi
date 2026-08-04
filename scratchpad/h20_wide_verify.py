"""H20 — verify the WIDE (2000x160) audio render end-to-end, on real pages.

Everything here is MEASURED through the code the training run will actually use:

  1. geometry   -- the real ``Gemma4UnifiedImageProcessor`` on real materialized
                   2000x160 pages: emitted soft tokens, grid, max image position
                   ids, ms/token-column, mel bins per cell.  Compared against the
                   1000x160 production render at the same budgets.
  2. retention  -- the real ``univi.trainer._filter_training_tokens`` with
                   ``per_image_tokens = univi.trainer.image_token_budget(b)``,
                   over every split, at a ladder of ``max_length`` values.
                   Reports the smallest max_length giving 100.00% everywhere.
  3. collator   -- the real ``univi.hybrid.data.HybridCollator`` on real rows:
                   the ASSEMBLED sequence length (image soft tokens + target +
                   chat scaffold), min/mean/max, per budget.  This is what
                   ``max_length`` must clear; the filter's estimate is only a
                   proxy and uses a DIFFERENT tokenizer's token count.
  4. padding    -- what trimming the right-padded silence would buy (exact, from
                   ``render_config``; no thresholds).

CPU ONLY.  No model weights, no GPU.  ``univi/hybrid/{vision,data}.py`` and
``univi/trainer.py`` are loaded BY FILE PATH because ``univi/__init__.py``
imports unsloth, which raises without an accelerator.

  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
      scratchpad/h20_wide_verify.py --self-test
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
      scratchpad/h20_wide_verify.py -n 48
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import statistics
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CELL = 48
BUDGETS = (280, 560, 1120)
DEFAULT_OUT = "data/eval/h20-wide-verify.json"
TOKENIZER = "data/checkpoints/hybrid-pretrained-spokendigits-trainvis-v0/final"

WIDE_ROOT = "data/materialized/h20-audio-wide-v0"
PROD = {
    "librispeech": "data/materialized/univi-3M-v0-split/librispeech",
    "spoken-digits": "data/materialized/spoken-digits-v0/spoken-digits",
}


# ---------------------------------------------------------------------------
# module loading (by path; univi/__init__.py imports unsloth)
# ---------------------------------------------------------------------------
def _load_by_path(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_vision():
    return _load_by_path("_univi_hybrid_vision_cpu", "univi/hybrid/vision.py")


def load_hybrid_data():
    return _load_by_path("_univi_hybrid_data_cpu", "univi/hybrid/data.py")


def load_trainer():
    """``univi/trainer.py`` with ``unsloth`` stubbed (it raises without a GPU)."""
    if "_univi_trainer_cpu" in sys.modules:
        return sys.modules["_univi_trainer_cpu"]
    if "unsloth" not in sys.modules:
        try:
            import unsloth  # noqa: F401
        except Exception:
            stub = types.ModuleType("unsloth")
            stub.FastVisionModel = type("FastVisionModel", (), {})
            stub.UnslothVisionDataCollator = type("UnslothVisionDataCollator", (), {})
            stub.is_bfloat16_supported = lambda: False
            stub.__univi_stub__ = True
            stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
            sys.modules["unsloth"] = stub
    return _load_by_path("_univi_trainer_cpu", "univi/trainer.py")


_PROC: dict[int, object] = {}


def processor(budget: int):
    if budget not in _PROC:
        _PROC[budget] = load_vision().Gemma4UnifiedImageProcessor(
            max_soft_tokens=int(budget)
        )
    return _PROC[budget]


def closed_form_hw(budget: int, height: int, width: int) -> tuple[int, int]:
    """The processor's target size: constant AREA of ``budget`` 48px cells."""
    s = math.sqrt(budget * CELL * CELL / float(height * width))
    return (
        int(math.floor(s * height / CELL)) * CELL,
        int(math.floor(s * width / CELL)) * CELL,
    )


# ---------------------------------------------------------------------------
# 1. geometry, measured on real pages
# ---------------------------------------------------------------------------
def measure_geometry(pages: list, budget: int, page_ms: float, n_mels: int) -> dict:
    proc = processor(budget)
    out = proc(images=pages, return_tensors="pt")
    counts = out["num_soft_tokens_per_image"].tolist()
    pos = out["image_position_ids"]
    valid = pos[..., 0] != -1
    max_x = int(pos[..., 0][valid].max())
    max_y = int(pos[..., 1][valid].max())
    h, w = pages[0].size[1], pages[0].size[0]
    th, tw = closed_form_hw(budget, h, w)
    cols, rows = tw // CELL, th // CELL
    return {
        "budget": budget,
        "source_hw": [h, w],
        "resized_hw_closed_form": [th, tw],
        "grid_time_cols": cols,
        "grid_freq_rows": rows,
        "n_soft_tokens_measured": counts[0],
        "n_soft_tokens_all_equal": len(set(counts)) == 1,
        "n_pages_probed": len(pages),
        "grid_product_equals_tokens": cols * rows == counts[0],
        "max_image_position_id_xy": [max_x, max_y],
        "max_pos_id_matches_grid": (max_x == cols - 1 and max_y == rows - 1),
        "ms_per_token_column": page_ms / cols,
        "mel_frames_per_cell": (page_ms / 10.0) / cols,  # 10 ms hop
        "mel_bins_per_cell": n_mels / rows,
        "scale_time": tw / w,
        "scale_freq": th / h,
    }


# ---------------------------------------------------------------------------
# 3. assembled sequence lengths through the REAL collator
# ---------------------------------------------------------------------------
def measure_collator(rows: list, budget: int, tokenizer, image_token_id: int) -> dict:
    data = load_hybrid_data()
    coll = data.HybridCollator(
        tokenizer=tokenizer,
        image_processor=processor(budget),
        image_token_id=image_token_id,
        max_length=100_000,  # deliberately huge: we want the UNTRUNCATED length
        response_only=True,
        emit_blank_images=False,
    )
    lengths: list[int] = []
    supervised: list[int] = []
    for row in rows:  # one row at a time: per-row length, not batch-padded
        batch = coll([row])
        lengths.append(int(batch["attention_mask"][0].sum()))
        supervised.append(int((batch["labels"][0] != data.IGNORE_INDEX).sum()))
    return {
        "budget": budget,
        "n_rows": len(rows),
        "seq_len_min": min(lengths),
        "seq_len_mean": statistics.fmean(lengths),
        "seq_len_max": max(lengths),
        "supervised_tokens_mean": statistics.fmean(supervised),
        "supervised_tokens_max": max(supervised),
    }


def _row_costs(ds, tokenizer) -> tuple[list[int], list[int]]:
    """``(pages_per_row, qwen3_target_tokens)`` for EVERY row of a split.

    ``original_token_length`` cannot be used here: on the audio lanes it was
    written by a *Gemma* tokenizer and under-counts the Qwen3 length the
    collator actually produces (measured: up to 26 tokens on librispeech).
    """
    cols = ds.select_columns(["render_config", "messages"])
    pages, targets = [], []
    for rc, messages in zip(cols["render_config"], cols["messages"]):
        pages.append(int(json.loads(rc).get("n_pages", 1)))
        for m in messages:
            if m["role"] == "assistant":
                targets.append(
                    "".join(
                        p.get("text") or ""
                        for p in m["content"]
                        if p.get("type") == "text"
                    )
                )
                break
    lens = [len(x) for x in tokenizer(targets, add_special_tokens=False)["input_ids"]]
    return pages, lens


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test() -> dict:
    from PIL import Image

    checks: dict[str, bool] = {}
    vision = load_vision()

    # S1 closed form == the processor's own helper, over a sweep incl. 2000x160.
    ok = True
    for b in BUDGETS:
        for h, w in [(160, 1000), (160, 2000), (80, 2000), (1024, 1024), (160, 4000)]:
            ref = vision.get_aspect_ratio_preserving_size(
                height=h, width=w, patch_size=16, max_patches=b * 9,
                pooling_kernel_size=3,
            )
            ok &= tuple(ref) == closed_form_hw(b, h, w)
    checks["S1 closed form == processor resize helper"] = ok

    # S2 a real 2000x160 page emits grid_w*grid_h tokens, <= budget.
    img = Image.new("RGB", (2000, 160), (200, 200, 200))
    ok = True
    for b in BUDGETS:
        g = measure_geometry([img], b, 10_000.0, 80)
        ok &= g["grid_product_equals_tokens"] and g["max_pos_id_matches_grid"]
        ok &= g["n_soft_tokens_measured"] <= b
    checks["S2 emitted tokens == grid product, <= budget, pos ids match"] = ok

    # S3 MUTATION: a wrong grid must FAIL the same assertion.
    g = measure_geometry([img], 1120, 10_000.0, 80)
    checks["S3 MUTATION: shifted grid must not match"] = (
        (g["grid_time_cols"] + 1) * g["grid_freq_rows"] != g["n_soft_tokens_measured"]
    )

    # S4 ms/column is page_ms / time columns, and halves when the page is 2x wide
    #    at the same budget only if the columns double -- check the actual ratio.
    narrow = measure_geometry(
        [Image.new("RGB", (1000, 160), (200, 200, 200))], 1120, 10_000.0, 80
    )
    wide = measure_geometry([img], 1120, 10_000.0, 80)
    checks["S4 wide render is strictly finer in time at the same budget"] = (
        wide["ms_per_token_column"] < narrow["ms_per_token_column"]
    )
    checks["S4b wide render costs no more tokens"] = (
        wide["n_soft_tokens_measured"] <= narrow["n_soft_tokens_measured"]
    )

    # S5 the trainer's filter is the real one and its budget tracks the config.
    tr = load_trainer()
    checks["S5 image_token_budget tracks max_soft_tokens"] = (
        tr.image_token_budget(280) == 282 and tr.image_token_budget(1120) == 1122
    )

    failures = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return {"checks": checks, "failures": failures, "all_passed": not failures}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=48, help="rows per split for geometry/collator")
    ap.add_argument("--wide-root", default=WIDE_ROOT)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--skip-retention", action="store_true")
    ap.add_argument("--configs", nargs="*", default=[
        "configs/h20_leg1_wide_560.yaml", "configs/h20_leg2_wide_1120.yaml",
    ])
    ap.add_argument("--eval-samples", type=int, default=64,
                    help="training.max_eval_samples_per_subset the config will use")
    ap.add_argument(
        "--max-lengths", type=int, nargs="*", default=[2048, 4096, 6144, 8192],
    )
    a = ap.parse_args()

    if a.self_test:
        r = self_test()
        print("SELF-TEST", "PASSED" if r["all_passed"] else "FAILED")
        raise SystemExit(0 if r["all_passed"] else 1)

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tr = load_trainer()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    # univi.hybrid.pretrained.IMAGE_PLACEHOLDER, inlined so this stays importable
    # without pulling univi/__init__.py (which imports unsloth).
    placeholder = "<|univi_image|>"
    if placeholder not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [placeholder]})
    image_token_id = tokenizer.convert_tokens_to_ids(placeholder)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise SystemExit("image placeholder token not found in tokenizer")

    result: dict = {
        "meta": {
            "wide_root": a.wide_root,
            "tokenizer": a.tokenizer,
            "image_token_id": image_token_id,
            "budgets": list(BUDGETS),
            "n_rows_probed": a.n,
            "device": "cpu (no model weights; a GPU training queue is live)",
        },
        "self_test": self_test(),
        "geometry": {},
        "collator": {},
        "retention": {},
        "padding_waste": {},
    }

    splits = {
        "librispeech/train": f"{a.wide_root}/librispeech/train",
        "librispeech/validation": f"{a.wide_root}/librispeech/validation",
        "spoken-digits/train": f"{a.wide_root}/spoken-digits-15pct/train",
        "spoken-digits/validation": f"{a.wide_root}/spoken-digits/validation",
    }

    # --- 1 geometry + 3 collator -------------------------------------------
    for name, path in splits.items():
        if not Path(path).exists():
            print(f"[skip] {path} missing")
            continue
        ds = load_from_disk(path)
        n = min(a.n, len(ds))
        sub = ds.select(range(n))
        pages, page_sizes = [], {}
        for row in sub:
            for im in row["images"]:
                pages.append(im.convert("RGB"))
                page_sizes[f"{im.size[0]}x{im.size[1]}"] = (
                    page_sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
                )
        rc = json.loads(sub[0]["render_config"])
        page_ms = float(rc.get("page_duration_sec", 10.0)) * 1000.0
        n_mels = int(rc.get("n_mels", 80))
        result["geometry"][name] = {
            "rows_total": len(ds),
            "rows_probed": n,
            "page_sizes": page_sizes,
            # cap what goes through the processor: at 1120 one page is a
            # (1062, 6912) fp32 patch tensor (~29 MB), so a large batch is
            # gigabytes of host RAM for a number that is identical on every page.
            "per_budget": {
                str(b): measure_geometry(pages[:16], b, page_ms, n_mels)
                for b in BUDGETS
            },
        }
        # The first-n sample cannot contain the LONGEST rows (3-page librispeech
        # utterances are 17 of 104,014), and the longest row is what decides
        # max_length and peak VRAM. So: score EVERY row of the split by
        # n_pages * soft_tokens_per_page + target tokens, take the argmax, and
        # push that exact row through the real collator to confirm the analytic
        # worst case. A sampled max here would have been a guess.
        pages_per_row, tgt_tokens = _row_costs(ds, tokenizer)
        result["collator"][name] = {}
        for b in BUDGETS:
            soft = result["geometry"][name]["per_budget"][str(b)][
                "n_soft_tokens_measured"
            ]
            costs = [p * soft + t for p, t in zip(pages_per_row, tgt_tokens)]
            worst = max(range(len(costs)), key=costs.__getitem__)
            rows = [sub[i] for i in range(min(n, 48))] + [ds[worst]]
            m = measure_collator(rows[:-1], b, tokenizer, image_token_id)
            worst_m = measure_collator([rows[-1]], b, tokenizer, image_token_id)
            scaffold = worst_m["seq_len_max"] - costs[worst]
            m["worst_row_index"] = worst
            m["worst_row_pages"] = pages_per_row[worst]
            m["worst_row_target_tokens"] = tgt_tokens[worst]
            m["worst_row_seq_len_measured"] = worst_m["seq_len_max"]
            m["scaffold_tokens"] = scaffold
            m["seq_len_max_whole_split"] = worst_m["seq_len_max"]
            result["collator"][name][str(b)] = m
        g = result["geometry"][name]["per_budget"]
        print(f"\n{name}  rows={len(ds)}  pages probed={len(pages)}  sizes={page_sizes}")
        for b in BUDGETS:
            gg, cc = g[str(b)], result["collator"][name][str(b)]
            print(
                f"  budget {b:5d}: grid {gg['grid_time_cols']:3d}x{gg['grid_freq_rows']:<2d} "
                f"= {gg['n_soft_tokens_measured']:5d} tok  "
                f"{gg['ms_per_token_column']:7.1f} ms/col  "
                f"{gg['mel_bins_per_cell']:5.1f} mel bins/cell  |  "
                f"seq min/mean {cc['seq_len_min']:5d}/{cc['seq_len_mean']:7.1f}  "
                f"WORST ROW (whole split) {cc['seq_len_max_whole_split']:5d} "
                f"({cc['worst_row_pages']}p + {cc['worst_row_target_tokens']}tgt "
                f"+ {cc['scaffold_tokens']} scaffold)"
            )

    # --- 2 retention through the REAL filter --------------------------------
    if not a.skip_retention:
        print("\nRETENTION (real univi.trainer._filter_training_tokens)")
        for name, path in splits.items():
            if not Path(path).exists():
                continue
            ds = load_from_disk(path)
            rec = {"rows": len(ds), "per_budget": {}}
            for b in BUDGETS:
                per_image = tr.image_token_budget(b)
                rec["per_budget"][str(b)] = {
                    "per_image_tokens": per_image,
                    "retained": {
                        str(ml): len(
                            tr._filter_training_tokens(
                                ds, name, ml, num_proc=None, per_image_tokens=per_image
                            )
                        )
                        for ml in a.max_lengths
                    },
                }
            result["retention"][name] = rec
            for b in BUDGETS:
                r = rec["per_budget"][str(b)]["retained"]
                line = "  ".join(
                    f"{ml}:{r[str(ml)]}/{len(ds)}"
                    f"({100.0 * r[str(ml)] / len(ds):.2f}%)"
                    for ml in a.max_lengths
                )
                print(f"  {name:26s} b={b:5d}  {line}")

    # --- 3b the EVAL rows the trainer will actually pick ---------------------
    # `_cap_eval_dataset` does shuffle(seed=data_seed).select(range(N)), so the
    # LONGEST rows (3- and 4-page librispeech utterances) can land in eval even
    # though they are rare. Eval runs without gradient checkpointing AND with a
    # second blank forward, so its peak sequence is what sizes
    # per_device_eval_batch_size — not the training mean.
    print("\nEVAL SELECTION (exact: filters + shuffle(seed=42).select(N))")
    result["eval_selection"] = {}
    for name, split_key in (
        ("librispeech", "librispeech/validation"),
        ("spoken-digits", "spoken-digits/validation"),
    ):
        path = splits[split_key]
        if not Path(path).exists():
            continue
        base = load_from_disk(path)
        rec = {"rows_total": len(base), "per_budget": {}}
        for b in BUDGETS:
            ds = tr._filter_training_images(base, name, 4, num_proc=None)
            ds = tr._filter_training_tokens(
                ds, name, 8192, num_proc=None,
                per_image_tokens=tr.image_token_budget(b),
            )
            picked = tr._cap_eval_dataset(ds, max_samples=a.eval_samples, seed=42)
            rows = [picked[i] for i in range(len(picked))]
            m = measure_collator(rows, b, tokenizer, image_token_id)
            m["images_max"] = max(len(r["images"]) for r in rows)
            m["images_histogram"] = {
                str(k): sum(1 for r in rows if len(r["images"]) == k)
                for k in sorted({len(r["images"]) for r in rows})
            }
            rec["per_budget"][str(b)] = m
            print(
                f"  {name:16s} b={b:5d}  n={m['n_rows']:3d}  "
                f"seq max {m['seq_len_max']:5d} mean {m['seq_len_mean']:7.1f}  "
                f"pages/row {m['images_histogram']}"
            )
        result["eval_selection"][name] = rec

    # --- 4 padding waste -----------------------------------------------------
    for name in ("librispeech/train", "librispeech/validation"):
        path = splits[name]
        if not Path(path).exists():
            continue
        ds = load_from_disk(path).select_columns(["render_config"])
        cov, npages = [], []
        for rc in ds["render_config"]:
            cfg = json.loads(rc)
            dur = float(cfg.get("source_duration_ms") or 0.0)
            npg = int(cfg.get("n_pages", 1))
            page_ms = float(cfg.get("page_duration_sec", 10.0)) * 1000.0
            npages.append(npg)
            if dur:
                cov.append(min(1.0, dur / (npg * page_ms)))
        result["padding_waste"][name] = {
            "rows": len(ds),
            "total_pages": sum(npages),
            "pages_per_row": {str(k): npages.count(k) for k in sorted(set(npages))},
            "audio_coverage_mean": statistics.fmean(cov),
            "padded_silence_fraction": 1.0 - statistics.fmean(cov),
        }
        w = result["padding_waste"][name]
        print(
            f"  {name:26s} pages={w['total_pages']}  "
            f"padded silence {100 * w['padded_silence_fraction']:.1f}%  "
            f"=> trimming would recover {1 / w['audio_coverage_mean']:.2f}x effective budget"
        )

    # --- 4b what TRIMMING the right-padded silence would actually buy -------
    # The doc calls the 31% right-padded silence "~1.45x of the effective budget
    # for free". That is the TOKEN-count reading. The RESOLUTION reading is
    # different and weaker, because tokens are allocated by AREA: cropping the
    # page to a fraction f of its width gives columns' = sqrt(f) * columns over
    # a time span of f * T, so ms/column scales as **sqrt(f)**, not f.
    # Quantified here on the real per-row coverage distribution, through the
    # exact resize rule (no thresholds, no sampling).
    if "librispeech/train" in result["padding_waste"]:
        trim = {}
        ds = load_from_disk(splits["librispeech/train"]).select_columns(
            ["render_config"]
        )
        covs = []
        for rc in ds["render_config"]:
            cfg = json.loads(rc)
            dur = float(cfg.get("source_duration_ms") or 0.0)
            npg = int(cfg.get("n_pages", 1))
            pms = float(cfg.get("page_duration_sec", 10.0)) * 1000.0
            if dur:
                covs.append(min(1.0, dur / (npg * pms)))
        w_full, h = 2000, 160
        for b in BUDGETS:
            # per-row ms/column, tokens and frequency granularity under trimming
            ms, toks, freq = [], [], []
            for f in covs:
                w = max(CELL, int(round(f * w_full)))
                th, tw = closed_form_hw(b, h, w)
                cols, frows = max(1, tw // CELL), max(1, th // CELL)
                toks.append(cols * frows)
                ms.append((f * 10_000.0) / cols)
                freq.append(80.0 / frows)
            base = result["geometry"]["librispeech/train"]["per_budget"][str(b)]
            trim[str(b)] = {
                "untrimmed": {
                    "tokens_per_page": base["n_soft_tokens_measured"],
                    "ms_per_column": base["ms_per_token_column"],
                    "mel_bins_per_cell": base["mel_bins_per_cell"],
                },
                "trimmed_mean": {
                    "tokens_per_page": statistics.fmean(toks),
                    "ms_per_column": statistics.fmean(ms),
                    "mel_bins_per_cell": statistics.fmean(freq),
                },
                "ms_per_column_gain": base["ms_per_token_column"]
                / statistics.fmean(ms),
                "aspect_ratio_spread_note": (
                    "trimming makes the page aspect ratio ROW-DEPENDENT, so the "
                    "time/frequency split of the token grid varies per row -- "
                    "which is the very quantity this hypothesis manipulates"
                ),
                "mel_bins_per_cell_min_max": [min(freq), max(freq)],
            }
        result["trim_analysis"] = trim
        print("\nTRIM ANALYSIS (crop pages to the utterance)")
        for b in BUDGETS:
            t = trim[str(b)]
            print(
                f"  budget {b:5d}: {t['untrimmed']['ms_per_column']:6.1f} -> "
                f"{t['trimmed_mean']['ms_per_column']:6.1f} ms/col "
                f"({t['ms_per_column_gain']:.2f}x), tokens "
                f"{t['untrimmed']['tokens_per_page']:5.0f} -> "
                f"{t['trimmed_mean']['tokens_per_page']:6.1f}, mel bins/cell "
                f"{t['untrimmed']['mel_bins_per_cell']:4.1f} -> "
                f"{t['trimmed_mean']['mel_bins_per_cell']:4.1f}"
            )

    # --- 5 VRAM, on H17's calibrated model, at H20's measured seq lengths ----
    # Model reproduced EXACTLY from data/eval/h17-config-verification.json
    # (params 1.778B bf16 + bf16 grads + AdamW8bit; logits at 12 B/element,
    # GiB units; decoder activations 1.833e-4 GiB per micro-batch token).
    # Calibration anchors in that file: H07 (bs16 @280, ran OK) 20.16 GB,
    # H13 (bs4 @280, d4 seq~1220, ran OK) 20.05 GB.
    # The H16 blind branch does NOT raise peak: it runs forward-only under
    # no_grad BEFORE the aligned pass and frees its logits (train_pretrained.py
    # compute_loss), so peak stays the aligned graph. Eval's blank pass is also
    # no_grad and sequential with the aligned one.
    print("\nVRAM (H17's calibrated model, H20's measured sequence lengths)")
    result["vram"] = {"model": {
        "static_GiB": 9.93, "cuda_context_GiB": 0.9, "vision_act_GiB": 0.11,
        "bytes_per_logit": 12, "vocab": 151670,
        "decoder_act_GiB_per_token": 1.833e-4,
        "source": "data/eval/h17-config-verification.json",
    }, "legs": {}}

    def vram(batch: int, seq: int) -> float:
        logits = batch * seq * 151670 * 12 / 1024 ** 3
        return 9.93 + 0.9 + 0.11 + logits + batch * seq * 1.833e-4

    for leg, b in (("leg1", 560), ("leg2", 1120)):
        train_max = max(
            result["collator"][k][str(b)]["seq_len_max_whole_split"]
            for k in result["collator"]
            if k.endswith("/train")
        ) if result["collator"] else None
        eval_max = max(
            result["eval_selection"][k]["per_budget"][str(b)]["seq_len_max"]
            for k in result.get("eval_selection", {})
        ) if result.get("eval_selection") else None
        if train_max is None:
            continue
        rec = {
            "budget": b,
            "train_seq_max_measured": train_max,
            "eval_seq_max_measured": eval_max,
            "train": {str(x): round(vram(x, train_max), 2) for x in (1, 2, 4, 8)},
            "eval": {str(x): round(vram(x, eval_max), 2) for x in (1, 2, 4)}
            if eval_max else None,
        }
        result["vram"]["legs"][leg] = rec
        print(f"  {leg} (budget {b}): train seq max {train_max}, eval seq max {eval_max}")
        print(f"    train GiB by per-device batch: {rec['train']}")
        print(f"    eval  GiB by per-device batch: {rec['eval']}")

    # --- 6 do the configs say what the code reads? ---------------------------
    print("\nCONFIG CHECK")
    result["config_check"] = {}
    for cfg_path in a.configs:
        result["config_check"][cfg_path] = _check_config(cfg_path, tr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {a.out}")


# Keys `train_pretrained.build_training_args` / `main` / `univi.trainer` read.
_MODEL_KEYS = {"vision_source", "text_source", "vision_weights", "max_soft_tokens",
               "init_from"}
_DATASET_KEYS = {"num_proc", "source", "path", "max_train_images", "subsets",
                 "train_splits", "validation_splits", "shuffle_seed",
                 "max_train_rows_per_subset", "revision", "hf_hub_repo_id"}
_TRAINING_KEYS = {
    "output_dir", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "lr_adapter", "lr_vision", "lr_decoder",
    "warmup_ratio", "lr_scheduler_type", "weight_decay", "max_grad_norm",
    "num_train_epochs", "max_steps", "logging_steps", "do_eval", "eval_steps",
    "save_steps", "save_total_limit", "gradient_checkpointing",
    "dataloader_num_workers", "report_to", "seed", "freeze_vision",
    "gap_weighted_loss", "gap_beta", "log_blank_ce", "max_length",
    "loss_masking", "validation_subsets", "max_eval_samples_per_subset",
    "data_seed",
}


def _check_config(path: str, tr) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    problems: list[str] = []
    for block, known in (
        ("model", _MODEL_KEYS), ("dataset", _DATASET_KEYS),
        ("training", _TRAINING_KEYS),
    ):
        for key in cfg.get(block, {}):
            if key not in known:
                problems.append(f"{block}.{key} is read by NO code path")
    tc, dc, mc = cfg["training"], cfg["dataset"], cfg["model"]

    eff = tc["per_device_train_batch_size"] * tc["gradient_accumulation_steps"]
    if eff != 64:
        problems.append(f"effective batch {eff} != 64 (the cross-run invariant)")
    if tc.get("freeze_vision", False):
        problems.append("freeze_vision is set — H08: audio needs a TRAINABLE tower")
    bad = set(dc["subsets"]) - tr.VALID_SUBSETS
    if bad:
        problems.append(f"subsets not in VALID_SUBSETS: {bad}")
    bad = set(tc.get("validation_subsets", [])) - tr.VALID_SUBSETS
    if bad:
        problems.append(f"validation_subsets not in VALID_SUBSETS: {bad}")
    for p in (dc["path"], mc.get("init_from"), mc["vision_weights"]):
        if p and not Path(p).exists():
            problems.append(f"path does not exist: {p}")
    mpath = Path(dc["path"]) / "manifest.json"
    if mpath.exists():
        man = json.load(mpath.open())
        names = {e["config_name"] for e in man["subsets"]}
        missing = set(dc["subsets"]) - names
        if missing:
            problems.append(f"manifest has no entry for {missing}")
        for e in man["subsets"]:
            if not (Path(dc["path"]) / e["path"]).exists():
                problems.append(f"manifest path missing on disk: {e['path']}")
    else:
        problems.append(f"no manifest at {mpath}")
    print(f"  {path}: {'OK' if not problems else 'PROBLEMS'}")
    for p in problems:
        print(f"    ! {p}")
    return {"problems": problems, "ok": not problems}


if __name__ == "__main__":
    main()
