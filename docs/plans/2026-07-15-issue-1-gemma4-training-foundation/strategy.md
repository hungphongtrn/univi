# Training Foundation — Strategy

## Goal

Refactor the training implementation for `unsloth/gemma-4-E2B-it` LoRA fine-tuning to follow the official Unsloth VLM path with all latest 2026-07-15 decisions. Replace the current incomplete `univi/trainer.py`, stale configs, and outdated test expectations with a production-grade foundation that validates, fingerprints, loads, trains, and reports correctly — without adding a full tokenizer-length preflight or deterministic overlength exclusion. This foundation covers every non-model-loading component so that Phase 2 (GPU smoke) runs the exact same stack on real CUDA hardware.

## Architecture

The training pipeline is decomposed into layered modules with explicit CLI phases:

```
CLI (argparse + modes)
  |
  ├── Config (normalize, merge overrides, compute hash)
  ├── Dataset Loader (local manifest / Hub pinned revision)
  ├── Validation (structural schema, 32-row processor review, manifest)
  ├── Fingerprint (processor/tokenizer before and after model load)
  ├── W&B Lifecycle (init, config, artifacts, teardown)
  ├── Model Setup (FastVisionModel, LoRA, Unsloth gradient checkpointing)
  ├── Trainer Construction (SFTConfig, SFTTrainer, response-only masking)
  └── Phase Gates (preflight→smoke→train→resume→evaluate)
```

Each module is independently testable on CPU (except the model+GPU path). The CLI wraps these into a single `train` entry point whose mode argument selects which gates to run.

## Tech Stack

- **Runtime:** Python 3.12 via `uv`
- **Model:** `unsloth/gemma-4-E2B-it` pinned to revision `4abfca14e6c6bfb5888b80288185b1243fb8d539`
- **Framework:** Unsloth (`FastVisionModel`, `UnslothVisionDataCollator`, `train_on_responses_only`) + TRL `SFTTrainer` + Transformers 5.5.0
- **Config:** YAML with CLI overrides, configuration hash fingerprints
- **Data:** Hugging Face `datasets` — local `load_from_disk` or Hub non-streaming `load_dataset` with pinned revision `5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375`
- **Monitoring:** W&B (online required), `report_to: ["wandb"]`
- **Compute:** RTX 3060 12 GB (smoke) → A100 40 GB (full epoch)
- **Tests:** pytest with CPU-safe defaults and GPU-gated `@requires_gpu` markers

## Constraints & Assumptions

1. **Standard Unsloth VLM flow** is authoritative: `remove_unused_columns=false`, `dataset_text_field=""`, `dataset_kwargs.skip_prepare_dataset=true`, `max_length=2048`, no packing.
2. **No full tokenizer preflight.** The accepted limitation is that the first run does not prove all-row tokenizer-level placeholder equality and accepts standard processor truncation under the 2,048-token limit. Reports must state this explicitly.
3. **Response-only masking** via `unsloth.chat_templates.train_on_responses_only` with auto-detected Gemma 4 markers. Hard fail on zero active labels in any batch.
4. **Explicit LoRA projections** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with `finetune_audio_layers=false`. No `all-linear`.
5. **W&B online required.** No silent offline fallback. Preflight initializes W&B before model loading.
6. **Pinned revisions** for both dataset (`5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375`) and model (`4abfca14e6c6bfb5888b80288185b1243fb8d539`). Changes create distinct run identities.
7. **Resume is explicit.** `--resume` selects the latest checkpoint and continues the same W&B run. Overwriting an existing output directory is refused by default.
8. **Active subsets:** `fineweb-edu`, `densefusion`, `smoltalk`, `librispeech` (matching manifest/config names). Valor32k remains deferred behind issue #2.
9. **Local default:** `data/materialized/univi-3M-v0-split`; **Hub default:** `hungphongtrn/univi-3M-v0` pinned revision.
10. **GPU smoke** validates the full production stack before the epoch run. It is a hard gate.
11. **`max_length=2048`** (not `max_seq_length`) in SFTConfig. `include_num_input_tokens_seen="non_padding"`.
12. **One epoch**, batch 1, grad accumulation 4, rank 8/alpha 16, cosine LR with 0.03 warmup ratio, `max_grad_norm=1.0`.

## Phases (High-Level)

### Phase 1: Training Foundation
**Outcome:** Complete CPU-testable training stack: config normalization, dataset loader with pinned revisions, structural validation, 32-row processor review, processor/template fingerprint, phase artifact manifests, official Unsloth model/processor setup, response-only masking, trainer construction, and W&B lifecycle. All tests pass on CPU; GPU tests skip cleanly.
**Rough scope:** 7 tasks, 14 files created/modified, no GPU required.

### Phase 2: GPU Smoke Gate
**Outcome:** 8-row-per-subset smoke on RTX 3060 verifies actual collation, response masking, finite loss/gradients, W&B telemetry, checkpoint save/reload, and processor fingerprint match. Reports peak VRAM, grad norm, and per-subset NLL.
**Depends on:** Phase 1

### Phase 3: Full Epoch + Evaluation
**Outcome:** One-epoch training completes. Best and last checkpoints evaluated across all retained validation subsets. Final JSON `results.json` and W&B artifact published.
**Depends on:** Phase 2

## Open Questions

1. **CPU preflight model load path.** Can `FastVisionModel.from_pretrained` run on CPU / off-GPU with `load_in_4bit=false` for fingerprinting, or must it load on GPU? The preflight path attempts official processor load on CPU; if the installed Unsloth version requires CUDA even for processor extraction, the status is recorded as "unavailable_skip" (no fallback to a different processor API). Official processor loading and fingerprint matching are gated to GPU smoke (Phase 2). The dry-run mode explicitly avoids model loading, so this risk only affects `--preflight-only`.
3. **`train_on_responses_only` compatibility with the specific TRL/Unsloth versions.** The installed stack is TRL 0.24.0, Transformers 5.5.0, Unsloth 2026.7-era. The auto-detection path must be verified in the collator smoke test.
4. **Trainer `num_input_tokens_seen` support.** Verify that `include_num_input_tokens_seen="non_padding"` works with the TRL 0.24.0 SFTTrainer on this Transformers version.
