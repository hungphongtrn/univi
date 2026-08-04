#!/usr/bin/env python
"""H17 config verification — CPU ONLY, no GPU, no model weights on device.

Verifies, for the REPAIRED H17 design (d3-only lane, warm-start from the H07
checkpoint, three legs differing only in ``max_soft_tokens``):

 1. **Token-length retention.** For each budget, the assembled sequence length
    distribution of ``randstr-d3`` train+validation, computed with the REAL Qwen3
    tokenizer through the REAL ``HybridCollator._build_row``, plus the length the
    dataset filter (``univi.trainer._filter_training_tokens``) actually estimates.
    Picks the smallest power-of-two ``max_length`` giving 100.00% retention AND
    zero collator truncation.
 2. **Emitted soft tokens per budget** on real d3 pages via the real image
    processor (the *budget* and the *emitted* count differ: 280 -> 256).
 3. **Readable fraction** — token-weighted share of supervised tokens inside
    scan depth K ~= 48.
 4. **Warm-start compatibility** — whether ANY tensor in the H07 checkpoint is
    shaped by ``max_soft_tokens``. If one is, the whole design is blocked.

Run:
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python scratchpad/h17_config_verify.py
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python scratchpad/h17_config_verify.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ``univi/__init__`` imports ``univi.trainer``, which imports unsloth at module
# scope, and unsloth hard-fails without a CUDA device. This whole script is
# CPU-only by policy (a GPU job may be running), so stub the three names
# ``univi.trainer`` pulls in. None of them is touched by anything used here
# (``_filter_training_tokens`` / ``image_token_budget`` are pure pyarrow).
try:  # pragma: no cover - depends on the box having a GPU
    import unsloth  # noqa: F401
except Exception:
    import types

    _stub = types.ModuleType("unsloth")
    class _StubBase:  # subclassed at module scope by univi.trainer
        pass

    _stub.FastVisionModel = _StubBase
    _stub.UnslothVisionDataCollator = _StubBase
    _stub.is_bfloat16_supported = lambda: False
    _stub.__UNIVI_CPU_STUB__ = True
    # trl calls importlib.util.find_spec("unsloth"), which raises unless the
    # module carries a __spec__.
    import importlib.machinery

    _stub.__spec__ = importlib.machinery.ModuleSpec("unsloth", loader=None)
    sys.modules["unsloth"] = _stub

BUDGETS = (280, 560, 1120)
LANE = "randstr-d3"
DATA_ROOT = REPO / "data/materialized/h13-density-v0"
CKPT = REPO / "data/checkpoints/hybrid-pretrained-randstr-v0/final"
TEXT_SOURCE = "unsloth/Qwen3-1.7B"
SCAN_DEPTH = 48
OUT = REPO / "data/eval/h17-config-verification.json"

# max_length candidates, smallest first (HybridCollator pads to the BATCH max, so
# a generous max_length costs nothing at train time).
CANDIDATES = (1024, 2048, 3072, 4096, 6144, 8192)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def safetensors_header(path: Path) -> dict:
    """Read a .safetensors header (names -> {dtype, shape}) WITHOUT loading data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head


def build_tokenizer():
    from transformers import AutoTokenizer

    from univi.hybrid.pretrained import IMAGE_PLACEHOLDER

    # Same construction path as build_pretrained_hybrid: the base Qwen3 tokenizer
    # plus the image placeholder special token.
    tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
    if IMAGE_PLACEHOLDER not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    return tok, tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER)


def percentiles(xs: list[int]) -> dict:
    s = sorted(xs)
    n = len(s)

    def q(p):
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "min": s[0],
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": s[-1],
        "mean": round(sum(s) / n, 3),
    }


# ---------------------------------------------------------------------------
# 1. soft-token grid, on REAL d3 pages
# ---------------------------------------------------------------------------


def measure_soft_tokens(n_pages: int = 6) -> dict:
    import torch
    from datasets import load_from_disk

    from univi.hybrid.pretrained import build_image_processor
    from univi.hybrid.vision import get_aspect_ratio_preserving_size

    out = {}
    pages = []
    for split in ("train", "validation"):
        ds = load_from_disk(str(DATA_ROOT / LANE / split))
        for i in range(n_pages):
            im = ds[i]["images"][0]
            pages.append((split, i, im.convert("RGB"), im.size))

    for b in BUDGETS:
        proc = build_image_processor(b)
        inp = proc(images=[p[2] for p in pages], return_tensors="pt")
        soft = inp["num_soft_tokens_per_image"].tolist()
        pos = inp["image_position_ids"]  # (n_img, n_patch, 2), -1 = padding
        valid = pos[..., 0] != -1
        rows = int(pos[..., 0][valid].max()) + 1
        cols = int(pos[..., 1][valid].max()) + 1
        h, w = get_aspect_ratio_preserving_size(
            1024, 1024, patch_size=16, max_patches=b * 9, pooling_kernel_size=3
        )
        out[str(b)] = {
            "budget": b,
            "emitted_soft_tokens": sorted(set(soft)),
            "grid_rows_x_cols": [rows, cols],
            "grid_product": rows * cols,
            "max_position_id": int(pos[valid].max()),
            "resize_target_px": [h, w],
            "resize_factor_vs_1024": round(h / 1024.0, 4),
            "pixel_values_shape": list(inp["pixel_values"].shape),
            "padded_patch_slots": int(inp["pixel_values"].shape[1]),
            "source_page_size": list(pages[0][3]),
            "image_token_budget_filter": b + 2,
        }
        del inp, proc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


# ---------------------------------------------------------------------------
# 2/3. assembled lengths + filter retention + readable fraction
# ---------------------------------------------------------------------------


def measure_lengths(soft_by_budget: dict) -> dict:
    import pyarrow.compute as pc
    from datasets import load_from_disk

    from univi.hybrid.data import IGNORE_INDEX, HybridCollator
    from univi.trainer import _filter_training_tokens, image_token_budget

    tok, image_token_id = build_tokenizer()
    coll = HybridCollator(
        tokenizer=tok,
        image_processor=None,
        image_token_id=image_token_id,
        max_length=10**9,
        response_only=True,
    )

    res = {}
    for split in ("train", "validation"):
        ds = load_from_disk(str(DATA_ROOT / LANE / split))
        n_img = pc.list_value_length(ds.data.column("images")).to_pylist()
        otl = ds.data.column("original_token_length").to_pylist()
        messages = ds.data.column("messages").to_pylist()

        per_budget = {}
        for b in BUDGETS:
            soft = soft_by_budget[str(b)]["emitted_soft_tokens"]
            assert len(soft) == 1, f"non-uniform soft counts at {b}: {soft}"
            soft = soft[0]

            assembled, supervised = [], []
            for msgs, k in zip(messages, n_img):
                ids, labels = coll._build_row(msgs, [soft] * k)
                assembled.append(len(ids))
                supervised.append(sum(1 for x in labels if x != IGNORE_INDEX))

            budget_tok = image_token_budget(b)
            est = [o + k * budget_tok + 256 for o, k in zip(otl, n_img)]

            # retention under the REAL filter, per candidate max_length
            retention = {}
            for ml in CANDIDATES:
                keep = sum(1 for o, e in zip(otl, est) if o <= 0 or e <= ml)
                trunc = sum(1 for a in assembled if a > ml)
                retention[str(ml)] = {
                    "filter_retained": keep,
                    "filter_retained_pct": round(100.0 * keep / len(est), 4),
                    "collator_truncated_rows": trunc,
                }
            chosen = next(
                (
                    ml
                    for ml in CANDIDATES
                    if retention[str(ml)]["filter_retained"] == len(est)
                    and retention[str(ml)]["collator_truncated_rows"] == 0
                ),
                None,
            )
            per_budget[str(b)] = {
                "emitted_soft_tokens": soft,
                "image_token_budget_used_by_filter": budget_tok,
                "assembled_seq_len": percentiles(assembled),
                "filter_estimate": percentiles(est),
                "supervised_tokens": percentiles(supervised),
                "retention_by_max_length": retention,
                "min_max_length_for_100pct": chosen,
            }

            # confirm with the production filter itself at the chosen length
            if chosen is not None:
                filt = _filter_training_tokens(
                    ds, LANE, chosen, per_image_tokens=budget_tok
                )
                per_budget[str(b)]["production_filter_check"] = {
                    "max_length": chosen,
                    "rows_in": len(ds),
                    "rows_out": len(filt),
                    "retention_pct": round(100.0 * len(filt) / len(ds), 4),
                }
                # and confirm 2048 really is a trap / not, for the record
                filt2k = _filter_training_tokens(
                    ds, LANE, 2048, per_image_tokens=budget_tok
                )
                per_budget[str(b)]["production_filter_at_2048"] = {
                    "rows_out": len(filt2k),
                    "retention_pct": round(100.0 * len(filt2k) / len(ds), 4),
                }

        # readable fraction is budget-independent (it is about target tokens)
        sup = [
            sum(
                1
                for x in coll._build_row(msgs, [1] * k)[1]
                if x != IGNORE_INDEX
            )
            for msgs, k in zip(messages[:2000], n_img[:2000])
        ]
        tgt = otl[:2000]
        res[split] = {
            "rows": len(ds),
            "images_per_row": sorted(set(n_img)),
            "original_token_length": percentiles(otl),
            "by_budget": per_budget,
            "readable_fraction_scan_depth_48": {
                "n_rows_sampled": len(sup),
                "vs_supervised_tokens_incl_im_end": round(
                    sum(min(SCAN_DEPTH, t) for t in sup) / sum(sup), 5
                ),
                "vs_original_token_length": round(
                    sum(min(SCAN_DEPTH, t) for t in tgt) / sum(tgt), 5
                ),
                "mean_supervised_tokens": round(sum(sup) / len(sup), 3),
                "mean_original_token_length": round(sum(tgt) / len(tgt), 3),
            },
        }
    return res


# ---------------------------------------------------------------------------
# 4. warm-start compatibility
# ---------------------------------------------------------------------------


def check_warm_start() -> dict:
    """Is ANY checkpoint tensor shaped by ``max_soft_tokens``?

    Builds the model config at every budget and instantiates on the META device
    (no memory), then compares full state_dict shapes against each other AND
    against the H07 checkpoint's safetensors header.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3Config

    from univi.hybrid.build import load_vision_config
    from univi.hybrid.pretrained import (
        IMAGE_PLACEHOLDER,
        UniViHybridPretrained,
        UniViHybridPretrainedConfig,
    )

    tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
    if IMAGE_PLACEHOLDER not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    image_token_id = tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    vision_config = load_vision_config("google/gemma-4-12B-it")

    shapes_by_budget = {}
    for b in BUDGETS:
        text_config = Qwen3Config.from_pretrained(TEXT_SOURCE)
        text_config.vocab_size = len(tok)
        cfg = UniViHybridPretrainedConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
            max_soft_tokens=b,
        )
        with torch.device("meta"):
            m = UniViHybridPretrained(cfg)
        shapes_by_budget[b] = {k: list(v.shape) for k, v in m.state_dict().items()}
        del m

    ref = shapes_by_budget[280]
    budget_shaped = []
    for b in BUDGETS[1:]:
        s = shapes_by_budget[b]
        assert set(s) == set(ref), "parameter NAMES differ across budgets"
        for k in ref:
            if s[k] != ref[k]:
                budget_shaped.append({"tensor": k, "280": ref[k], str(b): s[k]})
        # a tensor whose shape merely *contains* the budget is worth flagging too
    contains_budget = sorted(
        k for k, v in ref.items() if any(d in BUDGETS for d in v)
    )

    ck = safetensors_header(CKPT / "model.safetensors")
    ck_shapes = {k: list(v["shape"]) for k, v in ck.items()}
    tied = {"language_model.lm_head.weight"}
    missing = sorted(set(ref) - set(ck_shapes) - tied)
    unexpected = sorted(set(ck_shapes) - set(ref))
    mismatched = sorted(
        {"tensor": k, "model": ref[k], "checkpoint": ck_shapes[k]}.__repr__()
        for k in set(ref) & set(ck_shapes)
        if ref[k] != ck_shapes[k]
    )

    ck_cfg = json.loads((CKPT / "config.json").read_text())
    return {
        "checkpoint": str(CKPT),
        "checkpoint_recorded_max_soft_tokens": ck_cfg.get(
            "max_soft_tokens", "ABSENT -> resolves to 280"
        ),
        "n_tensors_checkpoint": len(ck_shapes),
        "n_tensors_model": len(ref),
        "tensors_whose_shape_depends_on_max_soft_tokens": budget_shaped,
        "tensors_whose_shape_merely_contains_a_budget_number": [
            {"tensor": k, "shape": ref[k]} for k in contains_budget
        ],
        "missing_beyond_tied_lm_head": missing,
        "unexpected": unexpected,
        "shape_mismatches_vs_checkpoint": mismatched,
        "verdict": (
            "COMPATIBLE — no tensor is shaped by max_soft_tokens; init_from loads "
            "identically at 280/560/1120"
            if not budget_shaped and not missing and not unexpected and not mismatched
            else "BLOCKING DEFECT"
        ),
    }


# ---------------------------------------------------------------------------


def estimate_vram(lengths: dict) -> dict:
    """Analytic peak-VRAM estimate per leg. CPU only — nothing is allocated on a GPU.

    Terms (all measured or read from the installed source, none guessed):
      static   = weights(bf16) + grads(bf16) + AdamW8bit state(2 x 1 byte)
      logits   = B*S*V * BYTES_PER_LOGIT.  ``transformers.loss.loss_utils
                 .ForCausalLMLoss`` does an unconditional ``logits.float()``, so
                 a bf16 (2B) logit tensor, an fp32 (4B) copy, the fp32
                 log_softmax saved for backward (4B) and the fp32 grad (4B) can
                 coexist. 12 B/elem is the working figure; 14 is the pessimistic
                 momentary peak.
      decoder  = gradient checkpointing keeps ONE saved input per layer:
                 L * B*S*H * 2 bytes, plus one layer's recompute.
      vision   = NOT checkpointed, but tiny: B * padded_slots * (patch_dim +
                 k*mm_embed_dim) * 2 bytes.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3Config

    from univi.hybrid.build import load_vision_config
    from univi.hybrid.pretrained import (
        IMAGE_PLACEHOLDER,
        UniViHybridPretrained,
        UniViHybridPretrainedConfig,
    )

    tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
    if IMAGE_PLACEHOLDER not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    text_config = Qwen3Config.from_pretrained(TEXT_SOURCE)
    text_config.vocab_size = len(tok)
    cfg = UniViHybridPretrainedConfig(
        vision_config=load_vision_config("google/gemma-4-12B-it"),
        text_config=text_config,
        image_token_id=tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        max_soft_tokens=280,
    )
    with torch.device("meta"):
        m = UniViHybridPretrained(cfg)
    n_par = sum(p.numel() for p in m.parameters())
    n_vis = sum(p.numel() for p in m.vision_tower.parameters())
    n_ad = sum(p.numel() for p in m.adapter.parameters())
    del m

    V = text_config.vocab_size
    H = text_config.hidden_size
    L = text_config.num_hidden_layers
    GB = 1024.0**3
    static = n_par * (2 + 2 + 2) / GB  # bf16 weights + bf16 grads + 2x int8 state
    ctx = 0.9  # CUDA context + allocator fragmentation, empirical

    def leg(b, s, bs, slots, bytes_per_logit=12):
        logits = bs * s * V * bytes_per_logit / GB
        dec = L * bs * s * H * 2 / GB + 20 * bs * s * H * 2 / GB
        vis = bs * slots * (6912 + 5 * 3840) * 2 / GB
        return {
            "budget": b,
            "per_device_train_batch_size": bs,
            "max_seq_len": s,
            "tokens_per_microbatch": bs * s,
            "static_GB": round(static, 2),
            "logits_GB": round(logits, 2),
            "decoder_activations_GB": round(dec, 2),
            "vision_activations_GB": round(vis, 2),
            "cuda_context_GB": ctx,
            "total_GB": round(static + logits + dec + vis + ctx, 2),
            "total_GB_pessimistic_14B_per_logit": round(
                static + logits * 14 / 12 + dec + vis + ctx, 2
            ),
        }

    smax = {
        b: max(
            lengths["train"]["by_budget"][str(b)]["assembled_seq_len"]["max"],
            lengths["validation"]["by_budget"][str(b)]["assembled_seq_len"]["max"],
        )
        for b in BUDGETS
    }
    recommended = {280: 8, 560: 4, 1120: 2}
    out = {
        "params_total": n_par,
        "params_vision": n_vis,
        "params_adapter": n_ad,
        "vocab_size": V,
        "bytes_per_logit_assumed": 12,
        "ceiling_GB": 26,
        "recommended": {
            str(b): leg(b, smax[b], recommended[b], b) for b in BUDGETS
        },
        "sensitivity": {
            f"{b}@bs{bs}": leg(b, smax[b], bs, b)["total_GB"]
            for b in BUDGETS
            for bs in (1, 2, 4, 8, 16)
        },
        "calibration_anchors": {
            "H07 (ran OK): budget 280, bs 16, seq~303": leg(280, 303, 16, 280)["total_GB"],
            "H13 (ran OK): budget 280, bs 4, d4 seq~1220": leg(280, 1220, 4, 280)["total_GB"],
        },
    }
    return out


def measure_probe_rungs(soft_by_budget: dict, n: int = 500) -> dict:
    """d2/d4 are NOT trained on — they are the post-hoc density cross-check. The
    probe scripts (h13_analyze.py, hybrid_4lane_position_decay.py,
    hybrid_pretrained_ablation.py) all default to ``--max-length 2048`` and go
    straight through ``HybridCollator``, which TRUNCATES (``ids[:max_length]``).
    Truncation removes the DEEP target positions — exactly where K is measured —
    so this quantifies how many rows would be silently cut at each budget.
    """
    from datasets import load_from_disk

    from univi.hybrid.data import HybridCollator

    tok, iid = build_tokenizer()
    coll = HybridCollator(
        tokenizer=tok, image_processor=None, image_token_id=iid,
        max_length=10**9, response_only=True,
    )
    out = {}
    for rung in ("randstr-d2", "randstr-d4"):
        ds = load_from_disk(str(DATA_ROOT / rung / "validation"))
        msgs = ds.data.column("messages").to_pylist()[:n]
        per = {}
        for b in BUDGETS:
            s = soft_by_budget[str(b)]["emitted_soft_tokens"][0]
            L = [len(coll._build_row(m, [s])[0]) for m in msgs]
            per[str(b)] = {
                **percentiles(L),
                "rows_truncated_at_max_length_2048": sum(1 for x in L if x > 2048),
                "rows_truncated_at_max_length_4096": sum(1 for x in L if x > 4096),
            }
        out[rung] = per
    return out


CONFIGS = {b: REPO / f"configs/h17_leg_{b}.yaml" for b in BUDGETS}
ALLOWED_DIFFS = {
    ("model", "max_soft_tokens"),
    ("training", "output_dir"),
    ("training", "per_device_train_batch_size"),
    ("training", "per_device_eval_batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("wandb", "run_name"),
}


def check_configs() -> dict:
    """End-to-end: the three legs parse, differ only where allowed, and drive the
    PRODUCTION dataset loaders to identical row counts at every budget."""
    import yaml

    from univi.trainer import _load_eval_datasets, load_dataset

    cfgs = {b: yaml.safe_load(p.read_text()) for b, p in CONFIGS.items()}

    def flat(d, prefix=()):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, prefix + (k,)))
            else:
                out[prefix + (k,)] = v
        return out

    ref = flat(cfgs[280])
    diffs, illegal = {}, []
    for b in BUDGETS[1:]:
        f = flat(cfgs[b])
        assert set(f) == set(ref), f"key set differs at {b}"
        for k in ref:
            if f[k] != ref[k]:
                diffs.setdefault(".".join(k), {})[str(b)] = f[k]
                diffs[".".join(k)]["280"] = ref[k]
                if k[-2:] not in ALLOWED_DIFFS and (k[0], k[-1]) not in ALLOWED_DIFFS:
                    illegal.append(".".join(k))

    eff = {
        b: c["training"]["per_device_train_batch_size"]
        * c["training"]["gradient_accumulation_steps"]
        for b, c in cfgs.items()
    }

    rows = {}
    for b, c in cfgs.items():
        tr = load_dataset(c, source="auto")
        ev = _load_eval_datasets(c)
        rows[str(b)] = {
            "train_rows": len(tr),
            "eval": {k: len(v) for k, v in ev.items()},
        }

    identical = len({json.dumps(v, sort_keys=True) for v in rows.values()}) == 1
    return {
        "configs": {str(b): str(p) for b, p in CONFIGS.items()},
        "differing_keys": diffs,
        "illegal_differences": sorted(set(illegal)),
        "effective_batch": {str(k): v for k, v in eff.items()},
        "loaded_rows": rows,
        "row_counts_identical_across_legs": identical,
        "verdict": (
            "OK — legs differ only in max_soft_tokens + bookkeeping + per-device "
            "batch; effective batch 64 everywhere; identical data at every budget"
            if not illegal and set(eff.values()) == {64} and identical
            else "DEFECT"
        ),
    }


def self_test() -> int:
    """Cheap invariants that must hold for the measurements to mean anything."""
    from univi.hybrid.data import IGNORE_INDEX, HybridCollator
    from univi.hybrid.vision import (
        _SUPPORTED_SOFT_TOKENS,
        get_aspect_ratio_preserving_size,
    )
    from univi.trainer import _IMAGE_TOKEN_BUDGET, image_token_budget

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    check("all three budgets supported by the processor",
          all(b in _SUPPORTED_SOFT_TOKENS for b in BUDGETS))
    check("image_token_budget(280) == legacy constant 282",
          image_token_budget(280) == _IMAGE_TOKEN_BUDGET == 282)
    check("image_token_budget scales with the budget",
          (image_token_budget(560), image_token_budget(1120)) == (562, 1122))
    check("resize target is budget-only for a square page",
          get_aspect_ratio_preserving_size(1024, 1024, 16, 280 * 9, 3)
          == get_aspect_ratio_preserving_size(1536, 1536, 16, 280 * 9, 3)
          == (768, 768))

    tok, iid = build_tokenizer()
    coll = HybridCollator(
        tokenizer=tok, image_processor=None, image_token_id=iid,
        max_length=10**9, response_only=True,
    )
    msgs = [
        {"role": "user", "content": [
            {"type": "image", "text": None},
            {"type": "text", "text": "Transcribe the text shown in the image."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "abc def"}]},
    ]
    ids256, lbl256 = coll._build_row(msgs, [256])
    ids529, _ = coll._build_row(msgs, [529])
    check("assembled length grows exactly by the soft-token delta",
          len(ids529) - len(ids256) == 529 - 256)
    check("placeholder count == soft-token count",
          sum(1 for i in ids256 if i == iid) == 256)
    check("image placeholders are never supervised",
          all(lbl256[j] == IGNORE_INDEX for j, i in enumerate(ids256) if i == iid))
    check("supervised tokens = answer + <|im_end|>\\n trailer",
          sum(1 for x in lbl256 if x != IGNORE_INDEX)
          == len(tok.encode("abc def", add_special_tokens=False))
          + len(tok.encode("<|im_end|>\n", add_special_tokens=False)))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--vram-only",
        action="store_true",
        help="recompute only the VRAM section and merge it into the existing JSON",
    )
    ap.add_argument(
        "--check-configs",
        action="store_true",
        help="parse the three legs, prove they differ only where allowed, and "
             "drive the production loaders to identical row counts",
    )
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.check_configs:
        blob = json.loads(OUT.read_text())
        blob["config_check"] = check_configs()
        OUT.write_text(json.dumps(blob, indent=2))
        print(json.dumps(blob["config_check"], indent=2))
        return 0
    if args.vram_only:
        blob = json.loads(OUT.read_text())
        blob["vram"] = estimate_vram(blob["lengths"])
        OUT.write_text(json.dumps(blob, indent=2))
        print(json.dumps(blob["vram"], indent=2))
        return 0

    print("[1/3] soft-token grid on real d3 pages ...")
    soft = measure_soft_tokens()
    for b, v in soft.items():
        print(f"   budget {b:>4}: emitted={v['emitted_soft_tokens']} "
              f"grid={v['grid_rows_x_cols']} maxpos={v['max_position_id']} "
              f"resize={v['resize_target_px']}")

    print("[2/3] assembled lengths + filter retention ...")
    lens = measure_lengths(soft)
    for split, v in lens.items():
        for b, pb in v["by_budget"].items():
            print(f"   {split:<10} budget {b:>4}: assembled max="
                  f"{pb['assembled_seq_len']['max']} est max="
                  f"{pb['filter_estimate']['max']} -> max_length="
                  f"{pb['min_max_length_for_100pct']}")

    print("[3/3] warm-start compatibility ...")
    ws = check_warm_start()
    print("   " + ws["verdict"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {
            "lane": LANE,
            "data_root": str(DATA_ROOT),
            "budgets": list(BUDGETS),
            "scan_depth": SCAN_DEPTH,
            "soft_token_grid": soft,
            "lengths": lens,
            "warm_start": ws,
            "vram": estimate_vram(lens),
        },
        indent=2,
    ))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
