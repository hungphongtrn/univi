# Phase 2: Smoke Training

## Phase Goal
Run Gemma 4 E2B on the RTX 3060 with QLoRA/LoRA, batch size 1, short max length, and small max steps to validate dataset loading, image collation, loss masking, checkpoint save/load, and that loss trends down without divergence.

## Stub Note
This phase will be detailed after Phase 1 completes and the materialized dataset is available. The sections below are high-level placeholders.

## Files to Touch (rough sketch)
- `train_smoke.py` — Training script
- `configs/smoke.yaml` — Hyperparameters
- `tests/test_training.py` — Test harness validation

## Tasks (to be detailed)
1. Write training script with Unsloth `FastVisionModel`, `UnslothVisionDataCollator`, TRL `SFTTrainer`, Gemma 4 non-thinking template.
2. Configure QLoRA/LoRA: rank, alpha, target modules (vision + language + attention + MLP).
3. Run smoke: batch size 1, max steps 50-100, max seq length that fits 12 GB.
4. Verify loss trends downward (not diverging), no crashes.
5. Verify checkpoint save and reload produces same loss on a held-out batch.
6. Push checkpoints to HF Hub for Phase 3 continuation.

## Completion Criteria (rough)
- Smoke run completes without CUDA OOM or trainer crash.
- Loss decreases over 50+ steps.
- Checkpoint save and reload works.
- Phase 3 can continue from the smoke checkpoint.

## Handoff Notes (rough)
- Phase 3 will use the same materialized dataset with longer schedule.
- Note the max context length that fit RTX 3060 — this constrains text rendering density for Phase 1 adjustments if needed.
