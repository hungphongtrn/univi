# Phase 3: Full Training & Evaluation

## Phase Goal
Run the full training schedule on an A100 40 GB (or equivalent), save checkpoints, reload, and evaluate both the image-only lane and native upper-bound lane for each modality label.

## Stub Note
This phase will be detailed after Phase 2 confirms the smoke run is stable. The sections below are high-level placeholders.

## Files to Touch (rough sketch)
- `train_full.py` — Full training script (inherits from smoke config)
- `eval_image_only.py` — Image-only lane evaluation
- `eval_native_upper_bound.py` — Native upper-bound evaluation
- `configs/full.yaml` — Full training hyperparameters
- `configs/eval.yaml` — Evaluation configuration

## Tasks (to be detailed)
1. Run full training on A100 with longer schedule, higher LoRA rank, and longer context.
2. Save final checkpoint and upload to HF Hub.
3. Implement image-only evaluation harness: load checkpoint, run each eval split, collect per-modality accuracy.
4. Implement native upper-bound evaluation harness: same examples with native text/audio paths.
5. Report: accuracy by modality, category breakdown, retention metric.

## Completion Criteria (rough)
- Checkpoint save and reload produces identical eval results.
- Image-only lane metrics reported for each modality label.
- Native upper-bound metrics reported for comparison.
- Results logged to a structured file (JSON/CSV) for Phase 4 analysis.
