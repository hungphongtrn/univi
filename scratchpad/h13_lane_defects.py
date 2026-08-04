#!/usr/bin/env python
"""H13 lane SECONDARY defect hunt — everything except image/target pairing.

Assumes ``h13_data_integrity.py`` has already established that each row's image
encodes that row's target.  This script asks the follow-up question: given
correct data, is anything ELSE about the lane's instrumentation broken?

Checks (all CPU, tokenizer only — NO model weights are loaded):
  A. token accounting  — real Qwen3 token counts vs the stored
     ``original_token_length``; the collator's assembled sequence length vs
     ``training.max_length``; whether anything is truncated.
  B. loss mask         — replay ``HybridCollator._build_row`` and report exactly
     which tokens are supervised.
  C. corrected floor   — ``floor.json`` measures nats/token over the TARGET
     tokens only, but the trainer supervises ``target + "<|im_end|>\\n"``.  Those
     trailer tokens are deterministic (~0 nats), so the CE a NON-READING model
     achieves is lower than the published floor.  Recompute and compare with the
     observed eval losses.
  D. rung independence — the rungs share one RNG stream (same seed, same
     ``random.Random`` draw order), so rung targets are re-segmentations of the
     SAME letter sequence.  Test it directly.
  E. leakage/dupes     — duplicate targets within a split and train/val overlap.

Usage:
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python scratchpad/h13_lane_defects.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("data/materialized/h13-density-v0")
RUNGS = ["randstr-d1", "randstr-d2", "randstr-d3", "randstr-d4", "randstr-d5"]
TOKENIZER_DIR = "data/checkpoints/encoder-free-v0/best"  # the Qwen3 tokenizer

# Observed final per-rung eval loss from data/checkpoints/h13-density-v0-run.log
OBSERVED_EVAL = {
    "randstr-d1": 4.994,
    "randstr-d2": 5.513,
    "randstr-d3": 5.634,
    "randstr-d4": 5.658,
    "randstr-d5": 5.634,
}
MAX_LENGTH = 2048
SOFT_TOKENS = 280


def row_target(row) -> str:
    for m in row["messages"]:
        if m["role"] == "assistant":
            return "".join(c["text"] or "" for c in m["content"])
    raise KeyError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/eval/h13-lane-defects.json")
    p.add_argument("--n", type=int, default=64)
    a = p.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    # ``univi/__init__.py`` imports unsloth (GPU-only), so load the collator
    # module straight from its file instead of via the package.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_hybrid_data", Path(__file__).resolve().parent.parent / "univi/hybrid/data.py"
    )
    hd = importlib.util.module_from_spec(spec)
    sys.modules["_hybrid_data"] = hd  # @dataclass needs the module registered
    spec.loader.exec_module(hd)
    IGNORE_INDEX, HybridCollator = hd.IGNORE_INDEX, hd.HybridCollator

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    coll = HybridCollator(
        tokenizer=tok, image_processor=None, image_token_id=0,
        max_length=MAX_LENGTH, response_only=True,
    )

    rep: dict = {"tokenizer_dir": TOKENIZER_DIR, "max_length": MAX_LENGTH,
                 "soft_tokens_assumed": SOFT_TOKENS, "rungs": {}}

    trailer_ids = tok.encode("<|im_end|>\n", add_special_tokens=False)
    rep["trailer"] = {"text": "<|im_end|>\\n", "ids": trailer_ids,
                      "n_tokens": len(trailer_ids)}

    targets_by_rung: dict[str, list[str]] = {}

    for rung in RUNGS:
        val = load_from_disk(str(ROOT / rung / "validation"))
        k = min(a.n, len(val))
        idxs = [int(round(i * (len(val) - 1) / max(k - 1, 1))) for i in range(k)]
        rows = [val[i] for i in idxs]
        targets = [row_target(r) for r in rows]
        targets_by_rung[rung] = targets

        otl_stored = [r["original_token_length"] for r in rows]
        otl_real = [len(tok.encode(t, add_special_tokens=False)) for t in targets]

        # --- B: replay the collator's row builder (with SOFT_TOKENS placeholders)
        seq_lens, sup_counts, sup_texts = [], [], []
        for r in rows:
            ids, labels = coll._build_row(r["messages"], [SOFT_TOKENS])
            seq_lens.append(len(ids))
            sup = [i for i, l in enumerate(labels) if l != IGNORE_INDEX]
            sup_counts.append(len(sup))
            sup_texts.append(tok.decode([ids[i] for i in sup]))

        n_letters = len(targets[0].replace(" ", ""))
        h_char = math.log(26)
        mean_tgt_tok = statistics.fmean(otl_real)
        mean_sup = statistics.fmean(sup_counts)
        floor_published = json.loads((ROOT / rung / "floor.json").read_text())[
            "no_reading_floor_nats_per_token"
        ]
        floor_corrected = n_letters * h_char / mean_sup
        obs = OBSERVED_EVAL[rung]

        rep["rungs"][rung] = {
            "n_sampled": len(rows),
            "target_chars": len(targets[0]),
            "target_letters": n_letters,
            "original_token_length_stored": {
                "min": min(otl_stored), "max": max(otl_stored),
                "mean": statistics.fmean(otl_stored),
            },
            "original_token_length_recomputed": {
                "min": min(otl_real), "max": max(otl_real),
                "mean": mean_tgt_tok,
            },
            "stored_equals_recomputed": otl_stored == otl_real,
            "assembled_seq_len": {
                "min": min(seq_lens), "max": max(seq_lens),
                "mean": statistics.fmean(seq_lens),
            },
            "n_truncated_at_max_length": sum(s > MAX_LENGTH for s in seq_lens),
            "supervised_tokens": {
                "min": min(sup_counts), "max": max(sup_counts),
                "mean": mean_sup,
            },
            "supervised_text_equals_target_plus_trailer": all(
                s == t + "<|im_end|>\n" for s, t in zip(sup_texts, targets)
            ),
            "supervised_text_sample": sup_texts[0][:80],
            "floor_published_nats_per_tok": floor_published,
            "floor_corrected_for_trailer_nats_per_tok": floor_corrected,
            "observed_final_eval_loss": obs,
            "obs_minus_published_floor": obs - floor_published,
            "obs_minus_corrected_floor": obs - floor_corrected,
        }

    # --- D: shared RNG stream across rungs --------------------------------
    d = {}
    tr = {r: load_from_disk(str(ROOT / r / "train")) for r in RUNGS}
    t1 = [row_target(tr["randstr-d1"][i]) for i in range(64)]
    for rung, mult in [("randstr-d2", 4), ("randstr-d3", 16), ("randstr-d4", 64)]:
        t = row_target(tr[rung][0])
        joined = " ".join(t1[:mult])
        d[f"{rung}_row0_equals_d1_rows_0_{mult-1}_joined"] = (t == joined)
        d[f"{rung}_row0_prefix25_equals_d1_row0"] = t[:29] == t1[0]
    d["d5_row0_equals_d3_row0"] = (
        row_target(tr["randstr-d5"][0]) == row_target(tr["randstr-d3"][0])
    )
    rep["shared_rng_stream"] = d

    # --- E: duplicates / leakage ------------------------------------------
    leak = {}
    for rung in RUNGS:
        trs = tr[rung]
        n = min(2000, len(trs))
        tt = [row_target(trs[i]) for i in range(n)]
        val = load_from_disk(str(ROOT / rung / "validation"))
        vt = [row_target(val[i]) for i in range(len(val))]
        leak[rung] = {
            "n_train_checked": n,
            "n_duplicate_targets_in_train_sample": n - len(set(tt)),
            "n_val": len(vt),
            "n_duplicate_targets_in_val": len(vt) - len(set(vt)),
            "n_val_targets_seen_in_train_sample": len(set(vt) & set(tt)),
        }
    rep["duplicates_and_leakage"] = leak

    # --- eval-subsample identity between d3 and d5 -------------------------
    # _cap_eval_dataset uses shuffle(seed=42).select(range(128)); d3/d5 have the
    # same row count so the SAME indices are picked, and their targets are equal
    # ⇒ d3 and d5 eval losses are identical by construction if the image is
    # ignored. Verify the eval subsample really does coincide.
    v3 = load_from_disk(str(ROOT / "randstr-d3" / "validation")).shuffle(seed=42).select(range(128))
    v5 = load_from_disk(str(ROOT / "randstr-d5" / "validation")).shuffle(seed=42).select(range(128))
    same = sum(row_target(v3[i]) == row_target(v5[i]) for i in range(128))
    rep["d3_d5_eval_subsample"] = {
        "n": 128, "n_identical_targets": same,
        "note": "Trainer caps eval at 128 rows via shuffle(seed=42).select(range(128)); "
                "identical targets ⇒ identical eval loss whenever the image is ignored.",
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)[:6000])
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
