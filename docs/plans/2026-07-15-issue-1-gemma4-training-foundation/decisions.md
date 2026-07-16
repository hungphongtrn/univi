# Decision Log — Training Foundation

This log records decisions made during the training foundation planning. Entries from the superseded `2026-07-08-issue-1-gemma4-phase0-mixture/decisions.md` are referenced and reconciled. **Entries dated 2026-07-15 are authoritative.** Where a 2026-07-15 decision contradicts an earlier entry, the 2026-07-15 entry governs.

---

## 2026-07-15: Training foundation as a separate plan from preprocessing

**Context:** The original plan (`2026-07-08-issue-1-gemma4-phase0-mixture`) merged preprocessing and training into one issue lifecycle. Preprocessing tasks are substantially complete (four-source materialized dataset published as `hungphongtrn/univi-3M-v0`), but the training-layer codebase still implements an outdated version of the Unsloth path with stale configs and missing components.

**Decision:** Create a new plan directory focused exclusively on the training implementation. This plan inherits the materialized dataset contract and the decisions.md from the earlier plan, but re-records the reconciled 2026-07-15 decisions that supersede older entries.

**Rationale:** Separating concerns prevents confusion between the preprocessing pipeline (finished) and the training refactor (current work). The plan directory date tracks the latest authoritative decisions.

**Consequences:** The old plan directory remains as a historical record. All new work references this plan.

---

## 2026-07-15: Standard Unsloth VLM flow (authoritative — supersedes deterministic overlength filtering)

**Context:** The earlier plan contained multiple entries about deterministic overlength exclusion, retained manifests per subset, and a full processor-length preflight before model loading. The operator chose the standard Unsloth VLM training path with `skip_prepare_dataset: true`.

**Decision:** Use the official VLM SFT flow:
- `remove_unused_columns: false`
- `dataset_text_field: ""`
- `dataset_kwargs: { skip_prepare_dataset: true }`
- `max_length: 2048` (not `max_seq_length`)
- No packing for the first run (Unsloth VLM packing support is rejected by the installed trainer)

Do NOT perform a separate untruncated token-length scan or retained-row filtering. Let the Unsloth collator process all raw rows under the 2,048-token limit.

**This explicitly supersedes:**
- 2026-07-15 "Exclude overlength rows deterministically" — no exclusion filter
- 2026-07-15 "Full processor-level preflight before GPU training" — no all-row preflight
- 2026-07-15 "Apply deterministic overlength filtering to train and validation" — no filter
- 2026-07-15 "Store dataset indices in retained manifests" — no retained manifest by index
- 2026-07-15 "Let tokenizer budget determine subset retention" — no pre-computed retention

**Rationale:** Minimizes custom infrastructure for the first run and keeps the implementation close to the documented Unsloth path.

**Consequences:** The first run does not prove tokenizer-level placeholder equality for every row and accepts standard processor truncation under the 2,048-token limit. Reports must state this limitation explicitly. Any batch with zero active response labels remains a runtime hard failure. `validation_retained_manifest.jsonl` and overlength exclusion counts are removed from the required artifacts.

---

## 2026-07-15: Remove inference-only flags from production training config

**Context:** The existing 12 GB config (`3060_full.yaml`) includes `gpu_memory_utilization`, `float8_kv_cache`, and `unsloth_tiled_mlp` — flags documented near vLLM/inference loading or experimental MLP behavior.

**Decision:** Remove these flags from all production configs. Keep only supported training loader options:
- 4-bit loading (`load_in_4bit: true`)
- Unsloth gradient checkpointing (`use_gradient_checkpointing: "unsloth"`)
- Pinned model revision
- `max_lora_rank: 8`

**Rationale:** Inference/vLLM settings do not improve standard VLM SFT and experimental tiled MLP behavior has not passed a Gemma 4 training smoke.

**Consequences:** `3060_full.yaml` and `3060_1epoch.yaml` must be rewritten to remove inference flags. `gpu_memory_utilization` is not a training loader argument.

---

## 2026-07-15: Use explicit LoRA projections with native audio frozen (authoritative — supersedes `all-linear`)

**Context:** The operator preferred the official Unsloth Gemma 4 guide's `all-linear` target, but the installed Unsloth implementation forces `finetune_audio_layers: true` when `target_modules: "all-linear"`, conflicting with the image-only training contract.

**Decision:** Use the explicit projection targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` with rank 8/alpha 16, `finetune_audio_layers: false`. Keep vision, language, attention, and MLP components enabled.

**This explicitly supersedes:**
- 2026-07-15 "Use Unsloth all-linear LoRA targets" — all-linear is not used for production
- 2026-07-15 "Align loader LoRA capacity with rank 8" — retained as `max_lora_rank: 8`

**Rationale:** The explicit list preserves the frozen native audio tower and avoids custom Unsloth adapter-selection code.

**Consequences:** The resolved production config must not contain `target_modules: "all-linear"`. The smoke gate must inspect trainable module names and assert no native audio module is trainable.

---

## 2026-07-15: Use ratio-based warmup with one-point gradient clipping

**Context:** The one-epoch run is approximately 720,354 optimizer steps. A fixed 20-step warmup would be negligible.

**Decision:** Use:
- `learning_rate: 2.0e-4`
- `warmup_ratio: 0.03`
- `lr_scheduler_type: "cosine"`
- `weight_decay: 0.001`
- `max_grad_norm: 1.0`
- `warmup_steps: 0` (ratio is authoritative)

**Rationale:** A ratio scales warmup with the actual one-epoch step count.

**Consequences:** Effective warmup is approximately 21,600 optimizer steps and must be printed in the resolved config. W&B must report `grad_norm`.

---

## 2026-07-15: Configurable loss masking with response-only default

**Context:** The first run should optimize assistant targets, but future experiments may benefit from training on user/input tokens.

**Decision:** Add `loss_masking: response_only` as the production default with `full_sequence` as an alternative. Use `unsloth.chat_templates.train_on_responses_only` with auto-detected Gemma 4 markers. Assert markers are `<|turn>user\n` and `<|turn>model\n`.

**Rationale:** The helper's VLM path masks labels at collator time, avoiding a text-only dataset rewrite that would lose top-level image handling.

**Consequences:** The masking mode is part of the resolved config, W&B run identity, checkpoint metadata, and processor/collator validation. Llama marker strings are forbidden in the Gemma 4 configuration.

---

## 2026-07-15: Initialize W&B before preflight

**Context:** Dataset validation is part of the training experiment and can fail before model loading.

**Decision:** Initialize the online W&B run before processor-level preflight. Log validation progress, fingerprints, aggregate counts, artifacts, and failures. Close failed preflight runs with a failed status.

**Rationale:** Makes data-quality failures observable and preserves the exact preflight evidence in the same experiment record.

**Consequences:** W&B credentials/network checks happen before GPU model initialization. Preflight failures must not leave ambiguous active runs.

---

## 2026-07-15: Preserve materialized chat roles without synthetic system prompts

**Context:** The materialized rows contain exactly `[user, assistant]`. No system message is needed.

**Decision:** Do not add a synthetic system message. Apply the Original Gemma E2B Template to the messages exactly as materialized. The startup review renders the system section as `none` when absent.

**Rationale:** Preserving the materialized conversation avoids changing the training distribution.

**Consequences:** Validation must assert role order and forbid unexpected system/native answer-bearing content.

---

## 2026-07-15: Pin both dataset and model Hub revisions

**Context:** Both the dataset and model currently follow mutable `main` branches.

**Decision:**
- Dataset pinned to `5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375`
- Model pinned to `4abfca14e6c6bfb5888b80288185b1243fb8d539`

**Rationale:** Exact data and model reproducibility are required for one-epoch accounting, validation coverage, and W&B comparison across machines.

**Consequences:** Any update requires a new explicit revision and distinct run/checkpoint identity. Resume with mismatched revisions is rejected.

---

## 2026-07-15: Validate image sentinels separately from expanded visual tokens

**Context:** Gemma 4 expands each image into multiple visual token IDs, so comparing image count directly with total visual-token count would reject valid rows.

**Decision:** Require equality between:
1. Count of `type: image` content entries in messages
2. Count of per-image sentinel occurrences in the Gemma 4 chat-template tokenization
3. Processor's decoded image/image-grid count

Report expanded visual-token length separately for the 2,048-token budget. Do NOT require tokenizer-level equality for every row (per the standard Unsloth VLM flow decision above).

**Rationale:** The three-layer invariant checks message/template alignment and actual image preprocessing without confusing one image with its expanded visual representation.

**Consequences:** The validator must discover model-specific sentinel IDs from the loaded processor rather than hard-code token strings. The 32-row review provides the exact evidence; aggregate counts are not computed across all 3M rows.

---

## 2026-07-15: Fingerprint processor and tokenizer before and after model load

**Context:** Preflight should avoid loading model weights, but training must use the exact processor/tokenizer behavior that Unsloth returns with Gemma 4.
**Decision:** Compute a canonical processor fingerprint before preflight and again after `FastVisionModel.from_pretrained` plus `get_chat_template(processor.tokenizer, "gemma-4")`. Because `FastVisionModel.from_pretrained` returns a `Gemma4Processor` wrapper (whose `.tokenizer` attribute holds the underlying tokenizer), the compatible official path extracts the inner tokenizer via `processor.tokenizer`, applies the template, and preserves the resulting template reference through the processor wrapper. Include tokenizer vocabulary size/hash, tokenizer class, special-token map, chat-template text/hash, image sentinel IDs, processor class, processor configuration, and image-processing configuration. Abort if fingerprints differ.

**Rationale:** Matching only the tokenizer name is insufficient; a changed chat template or special-token mapping changes image placeholders and sequence lengths.

**Consequences:** The fingerprint and both load-stage summaries are written to the validation report, resolved config, W&B run metadata, and checkpoint manifest.

---

## 2026-07-15: Use batch-one gradient accumulation on 12 GB

**Context:** The production run uses a 2,048-token multimodal sequence budget without packing.

**Decision:** Set `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, yielding effective batch size 4.

**Rationale:** Batch one is the safest memory boundary for variable-length visual examples on 12 GB VRAM.

**Consequences:** A complete retained-data epoch is approximately `retained_rows / 4` optimizer steps. Evaluation/checkpoint intervals are measured in optimizer steps.

---

## 2026-07-15: Explicit checkpoint and W&B resume semantics

**Context:** One epoch may run for hundreds of thousands of optimizer steps and can be interrupted.

**Decision:** Refuse to overwrite a non-empty output directory by default. Add `--resume` to select the latest valid checkpoint and continue the same W&B run. Persist the W&B run ID, dataset revision/hash, resolved config hash, global step, and checkpoint path.

**Rationale:** Silent restart would invalidate token/epoch accounting and create ambiguous W&B curves.

**Consequences:** Checkpoint discovery and W&B run identity become pre-training validation requirements.

---

## 2026-07-15: Use NLL loss as the sole validation score

**Decision:** Evaluate negative log-likelihood loss only. Report one full-validation NLL loss for each subset and the unweighted four-subset mean at every 1,000 optimizer steps. No generation metrics (WER, CER, ROUGE-L, exact-match) during this training run.

**Rationale:** NLL is directly aligned with the SFT objective and avoids introducing generation-decoding choices into checkpoint selection.

**Consequences:** The final report must clearly label NLL as the validation score. Generation metrics are deferred to later evaluation work.

---

## 2026-07-15: Use batch-one full validation

**Decision:** Set `per_device_eval_batch_size: 1`, `eval_accumulation_steps: 1`. Evaluation is un-packed with the same 2,048-token sequence budget.

**Rationale:** Batch one is the safe VRAM boundary for variable-length visual rows.

**Consequences:** Full validation cadence will be slow and must be timed in the smoke run.

---

## 2026-07-15: Supply eval_dataset as a mapping of four named subsets

**Decision:** Supply `eval_dataset` as a dict of four named subsets. Metrics from Trainer each produce `eval_<subset>_loss`. A custom `UniViSFTTrainer(SFTTrainer)` overrides `evaluate()` to call `super().evaluate()`, inspect the final merged mapping, compute the unweighted mean of all `eval_<subset>_loss` keys, add `eval_mean_total_loss`, and return metrics — the prefixed key the Trainer resolves from `metric_for_best_model="mean_total_loss"`. Trainer construction uses this subclass whenever a dict eval mapping is configured. Mirror to W&B as `eval/<subset>_loss` and `eval/mean_total_loss`. Select best checkpoint against `metric_for_best_model: "mean_total_loss"`.

**Rationale:** An unweighted mean prevents the million-row subsets from overwhelming smaller ones. A callback-based approach is insufficient because Transformers 5.5.0 `Trainer.evaluate(eval_dataset=dict)` recursively evaluates each named subset and invokes callbacks per subset before merging metrics — a callback cannot see all four subset losses at once.

**Consequences:** A custom `UniViSFTTrainer` subclass is required because Trainer's native dict-evaluation metric names do not automatically produce the requested aggregate key, and callbacks cannot access the final merged metric dict. Tests must exercise the subclass with a stubbed parent/merged metrics path.
---

## 2026-07-15: Use Unsloth minimum image resize

**Decision:** Construct `UnslothVisionDataCollator` with `resize: "min"`, `resize_dimension: 0`, `snap_to_patch_size: false`.

**Rationale:** Unsloth's minimum resize reduces visual tokens and VRAM while leaving source images unchanged.

**Consequences:** Resize policy is part of the processor/collator fingerprint and W&B config.

---

## 2026-07-15: Trust materialization contract for native-content leakage

**Decision:** Do not add a subset-specific source-text leakage checker. Retain structural message/schema validation, image-placeholder validation, target validation, and processor/collator checks.

**Rationale:** Avoid duplicating source-specific preprocessing logic when the materialized artifact is already governed by the documented contract.

**Consequences:** Native-content leakage is an accepted unverified assumption for this run.

---

## 2026-07-15: Use W&B for human divergence monitoring (no auto-abort)

**Decision:** Do not auto-abort on loss spikes, moving-average increases, or subset-specific degradation. Keep hard safeguards for NaN/Inf, invalid labels, CUDA OOM, checkpoint failures, and unrecoverable W&B failure.

**Rationale:** Mixed-subset NLL curves are not guaranteed monotonic, and human inspection is the intended monitoring path.

**Consequences:** The final run report must distinguish completed training from qualitative W&B monitoring.

---

## 2026-07-15: Use the official Unsloth processor path for preflight

**Decision:** Acquire the processor through `FastVisionModel.from_pretrained` and apply `get_chat_template(processor.tokenizer, "gemma-4")` — extracting the wrapper's inner tokenizer — then preserve the resolved template reference on the processor wrapper. If CPU/off-GPU official load is unsupported, fail rather than silently substitute a different processor API.
**Rationale:** Exact Unsloth processor/template behavior is more important than avoiding one additional model-load cycle.

**Consequences:** Preflight records the official processor/template fingerprint and the GPU phase must match it.

---

## 2026-07-15: Propagate Unsloth gradient-checkpointing mode through full pipeline

**Decision:** Set `model.use_gradient_checkpointing: "unsloth"` and `training.gradient_checkpointing: true`. Propagate the effective mode through model loading, PEFT setup, `FastVisionModel.for_training`, Trainer transitions, evaluation reloads, and checkpoint reloads.

**Rationale:** A default-boolean transition could silently disable Unsloth's specialized memory-saving path.

**Consequences:** The smoke gate must report the effective checkpointing mode and peak VRAM.

---

## 2026-07-15: Gate production behind explicit CLI phases

**Context:** A full 3M-row run should not start accidentally before CPU validation and the GPU smoke pass.

**Decision:** Expose `--preflight-only`, `--smoke`, `--resume`, and `--evaluate` modes. Default mode runs production only after matching preflight and smoke artifacts exist. Evaluation mode never trains; resume mode never overwrites.

**Rationale:** Named phases make the safety gates operational and make handoff to another machine reproducible.

**Consequences:** Each phase records its mode, config hash, dataset/model revisions, W&B run ID, and artifact paths.

---

## 2026-07-15: Expose explicit dataset/preflight CLI overrides

**Decision:** Add `--dataset-source auto|local|hub`, `--dataset-cache-dir`, `--dataset-revision`, and `--preflight-workers` CLI overrides. Config values remain defaults; every override is written to the resolved config and W&B metadata.

**Rationale:** Explicit overrides make cross-machine operation clear while preserving a reproducible base configuration.

**Consequences:** Resume rejects dataset revision/source changes unless starting a new run.

---

## 2026-07-15: Keep pinned Torch 2.10 stack

**Decision:** Do not upgrade to Torch 2.11. Keep `torch==2.10.0+cu130`, `torchvision==0.25.0+cu130`, `torchcodec==0.10`.

**Rationale:** Preserves the pinned CUDA/Torch/Torchcodec combination. Warnings about optional extensions are not blocking.

**Consequences:** GPU smoke must pass on the pinned stack.

---

## 2026-07-15: Non-streaming cached Hub loading

**Decision:** Remote mode uses normal non-streaming `datasets.load_dataset` with an explicit cache directory and disk-space preflight. Streaming is never a silent fallback.

**Rationale:** Non-streaming preserves deterministic shuffling, full validation passes, and reliable checkpoint resumption.

**Consequences:** Startup checks must report required/free cache disk space and the resolved Hub revision.

---

## 2026-07-15: Require a full-stack GPU smoke gate (Phase 2)

**Context:** Processor preflight cannot verify CUDA memory, Unsloth collation, response masking, Trainer token accounting, W&B telemetry, or checkpoint restoration.

**Decision:** Before the one-epoch run, execute a GPU smoke with eight retained rows per subset and the full production stack. Required checks: finite train/eval NLL, active assistant labels, finite grad_norm, increasing num_input_tokens_seen, no CUDA OOM, peak VRAM, W&B metrics/table, checkpoint weights, optimizer/scheduler/global-step restoration, processor fingerprint match.

**Note:** This decision is implemented in Phase 2, not Phase 1. Phase 1 builds the infrastructure; Phase 2 executes the smoke.

**Rationale:** This is the smallest gate that exercises the actual production contract.

**Consequences:** A failed smoke blocks the 3M-row run. Smoke and production runs use distinct W&B run IDs and checkpoint directories.

---

## 2026-07-15: Render each LibriSpeech utterance as one spectrogram image

**Decision:** Rebuild LibriSpeech so each complete admitted utterance is rendered directly from its 80-bin log-mel matrix into one 512×512 image and one image placeholder. Remove the 4× raster upscaling, fixed ten-second padding, central cropping, and chronological square tiling. Exclude source utterances longer than ten seconds rather than pairing a cropped input with the complete transcript.

**Rationale:** Gemma 4 represents each input image with 256 visual tokens. The existing 13-tile rendering costs 3,328 visual tokens per row solely because a 4×-upscaled strip is cut into squares. A single frame preserves the full admitted utterance at roughly 20 ms per horizontal pixel for ten seconds while keeping the input/target pair complete.

**Consequences:** Rebuild and republish both LibriSpeech splits, record source duration and temporal scale in `render_config`, and require exactly one image and one placeholder for every retained row.

---

## 2026-07-15: Admit fewer than four images per training example

**Decision:** Filter the training mixture before shuffling so only rows with zero to three input images are admitted. Validation remains unfiltered so subset reporting exposes the original distribution. Set production SFT `max_length` to 2,048 and reject processor-level overflows rather than truncating image placeholders or assistant targets.

**Rationale:** Four images consume 1,024 visual tokens before chat-template and assistant tokens. On the 12 GB training target, retaining at most three images leaves a safer text and response budget while avoiding a 4,096-token context increase.

**Consequences:** Training startup must report retained and excluded row counts by subset and image count. The smoke dataset must be sampled after this filter.

---

## 2026-07-16: Use four fixed rectangular LibriSpeech pages

**Context:** The one-square-image policy discarded most `train.clean.100` rows and forced a naturally rectangular time-frequency matrix into a square. The Gemma 4 image processor preserves aspect ratio: a 1000 × 160 page produces about 246 visual tokens with substantially more horizontal patch positions than a 512 × 512 square. Four such pages consume about 984 language-side visual tokens, which fits the 2,048-token budget for LibriSpeech's short transcript targets.

**Decision:** Supersede the one-image/ten-second and three-image-ceiling decisions for LibriSpeech. Train on `openslr/librispeech_asr` config `clean`, split `train.360`; evaluate on config `clean`, split `validation`. Render complete clips as one to four chronological ten-second 1000 × 160 log-mel pages, right-pad only the final page, and filter clips longer than 40 seconds. Admit up to four images in every training configuration.

**Rationale:** Fixed-duration pages preserve a stable time scale, rectangular rendering lets Gemma allocate its visual grid along time, and four pages retain the selected public data without exceeding the 2,048-token language-side sequence budget.

**Verification:** After rebuilding, validation row `5338-24615-0002` contains four 1000 × 160 pages. The installed Gemma 4 E2B processor produces 984 image-token positions and 1,159 total input tokens including the instruction and complete transcript, below the 2,048-token ceiling.

**Consequences:** Rebuild and republish both LibriSpeech splits from source; remove the legacy tiling converter; report source duration, page count, dimensions, normalization, and fixed time scale in `render_config`. Training startup now retains rows with at most four images.

---

## 2026-07-16: Block the RTX 3060 launch at the VRAM gate

**Context:** The production-stack smoke run completed ten optimizer steps, response-only masking, all four validation subsets, and checkpoint reload. With the four-image ceiling and 2,048-token sequence limit, it reached 13.912 GiB peak allocated VRAM before checkpoint export. Export itself peaked at 7.049 GiB. The configured RTX 3060 safety gate is 11.5 GiB.

**Decision:** Do not launch `configs/3060_full.yaml` on a 12 GB GPU. Keep the four-image data filter and the 2,048-token contract unchanged until choosing a different hardware target or a representation/training change that passes the same worst-case smoke gate.

**Rationale:** Raising or bypassing the memory gate would relabel an observed incompatibility as readiness. Freezing vision-layer LoRA did not materially change the measured peak (13.879 GiB), so it is not an adequate fix.

**Verification:** `data/smoke/dgx76af5/results.json` records finite per-step loss and gradient norms, increasing input-token counts, finite losses for all four validation subsets, a matching processor fingerprint, global step 10, `peak_before_save_gib: 13.912`, and failure check `peak_vram_exceeded`.

**Consequences:** The full run remains blocked. The next experiment must test one explicit lever—larger-VRAM hardware, fewer visual pages, or a lower-memory but gradient-correct training path—and rerun the unchanged smoke gate before launch.
