# Decisions — issue #6 hybrid

## Architecture

- **Random initialization, architecture-only replication** (explicit user
  instruction). No pretrained weights for either tower. Rationale: remove the
  pretrained-language-prior confound behind the "learnability-limited" verdict.
- **Vision tower = Gemma 4 `vision_config` (shared across sizes).** The Gemma 4
  vision embedder is size-independent, so the cached `gemma-4-E2B-it`
  `vision_config` faithfully *is* the "Gemma 4 12B image embedder" from the issue.
  Instantiated via `Gemma4VisionModel(vision_config)`.
- **Projector = `Gemma4MultimodalEmbedder(vision_config, text_config)`, reused
  verbatim.** It is already parameterized by the text hidden size, so it maps
  768 → 2048 (Qwen) with zero code change. No custom projector was written.
- **LLM = `Qwen3ForCausalLM` from the 1.7B config**, random init, `vocab_size`
  bumped by one for the added `<|univi_image|>` placeholder token.
- **Merge = Gemma's exact recipe:** embed ids → `masked_scatter` projected soft
  tokens onto `input_ids == image_token_id`. The vision tower already returns the
  valid (non-padding) soft tokens flattened `[total_soft, hidden]`, so no
  per-image slicing is needed — verified empirically for 1 and 2 images.

## Training path

- **Plain `transformers.Trainer`, not Unsloth/TRL.** The model is a bespoke
  `PreTrainedModel`, not an Unsloth `FastVisionModel`, so the repo's `univi.train`
  path (which hard-depends on `FastVisionModel`/`UnslothVisionDataCollator`/Gemma-4
  chat markers) does not apply. Only `univi.trainer.load_dataset` is reused.
- **Gradient checkpointing delegated** to the two sub-towers via an overridden
  `gradient_checkpointing_enable` (the base `PreTrainedModel` raised "does not
  support gradient checkpointing" because the composite class declares no blocks).

## Setup

- Qwen3-1.7B config + tokenizer curled to `data/hybrid/qwen3-1.7b/` (Xet hub hang
  → curl tiny files, never weights). Gitignored; curl block in README.

## GPU smoke (A100-40GB, full-size random-init model, real univi-3M-v0-split)

- Config `configs/hybrid_smoke.yaml`, 20 steps, bs=1, grad-accum 8,
  `max_length=4096`, gradient checkpointing on, adamw_torch, bf16.
- Peak memory observed ~37.5 GiB / 40 GiB during stepping (≈2.1B params trained in
  full precision + Adam states). For a longer run, drop `max_length`, switch to an
  8-bit optimizer, or shard — headroom is thin at this config.
- **Result (exit 0):** loss falls monotonically-ish **12.51 → ~8.0** over 20 steps
  (cosine LR 3e-4 → 0; grad_norm 0.24–4.6, healthy). `train_loss` avg 9.561,
  `train_runtime` 278 s (effective batch 8 → 160 examples). A random-init model
  showing a clear downward slope on real `univi-3M` rows confirms the full
  pipeline — collation, soft-token merge, masking, loss, backward, save — works.
  Final model (config + tied Qwen weights + Gemma vision tower + projector +
  tokenizer + image processor) saved to `data/checkpoints/hybrid-smoke-v0/final`
  (~7.5 GB), then deleted as a throwaway smoke artifact.
- **Not a trained model:** 20 steps is a plumbing check, not convergence. The
  bounded full run (E4) is the next step.

## Correction: vision tower is encoder-free, not the E2B ViT

- The "Architecture" section above (`Gemma4VisionModel` / 16-layer ViT / 768-dim)
  described the **first cut**, which used the E2B `gemma4_vision` config. That is
  architecturally wrong for what the issue asks to replicate: `google/gemma-4-12B-it`'s
  `vision_config` (`model_type: gemma4_unified_vision`) has **no**
  `num_hidden_layers`/`num_attention_heads` at all — the 12B "unified" line's
  vision path is **encoder-free**: patchify → merge k×k teacher patches → Dense +
  LayerNorm → factorized row/col positional embeddings → LayerNorm → RMSNorm →
  Linear projector. No self-attention between patches, ever.
- Replaced with `Gemma4UnifiedVisionEmbedder` / `Gemma4UnifiedVisionConfig`
  (`univi/hybrid/vision.py`), vendored from `transformers.models.gemma4_unified`
  (not yet in this repo's pinned `transformers==5.5.0`). `vision_source` in
  configs changed from `unsloth/gemma-4-E2B-it` to `google/gemma-4-12B-it`.
  CPU tests (4/4) and a re-run GPU smoke both pass against the corrected tower.

## Reorg: `univi/hybrid_*.py` → `univi/hybrid/` subpackage

- Flat `univi/hybrid_{model,vision,build,data,train}.py` moved to
  `univi/hybrid/{model,vision,build,data,train}.py` for a clearer package
  boundary now that there are 5 files. Entry point is now
  `python -m univi.hybrid.train`. Test file renamed
  `tests/test_hybrid_model.py` → `tests/test_hybrid.py` (it covers the whole
  package). No behavior change — verified by re-running the CPU test suite
  and confirming the in-flight training process (already holding the old
  module objects in memory) was unaffected.

## E4 — bounded training run (in progress)

- `configs/hybrid_bounded.yaml`: 3000 steps, same bs=1/grad-accum=8/max_length=4096
  as the smoke, full `univi-3M-v0-split` mixture (all 4 subsets), `report_to: []`.
  Launched via `data/eval/hybrid_bounded_run.sh`, detached with `setsid` +
  `nohup` (PPID=1) so it survives harness/session restarts — a plain
  `run_in_background` Bash task does **not** survive those (observed first-hand:
  the first launch attempt was silently killed by a session restart at step 125
  with no error, only a gap in the log and an idle GPU).
  - **Loss trajectory so far:** step 250 checkpoint saved, loss 11.5 → 6.2 over
    the first 250 steps (~4.5s/step, ETA ~3.5h total).
- `univi/hybrid/ablation.py` + `configs/hybrid_ablation.yaml` written ahead of
  completion: a thin wrapper that monkeypatches `eval_ablation.py`'s
  model-dependent seams (`load_model`, `compute_loss`) for
  `UniViHybridForConditionalGeneration` + `HybridCollator`, reusing its
  model-independent machinery (donor-image permutation, blank images, paired
  Δperm/Δblank, grounding flag, report). "`base_model`" here is a **freshly
  random-initialized** model (`random:<vision_source>,<text_source>`), not a
  pretrained donor — there isn't one for this architecture, so the meaningful
  comparison is trained-Δperm vs. random-init-Δperm.
  - Run once `checkpoint-3000` exists:
    `uv run python -m univi.hybrid.ablation --config configs/hybrid_ablation.yaml`
