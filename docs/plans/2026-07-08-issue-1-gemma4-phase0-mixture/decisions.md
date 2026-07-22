# Decision Log

## 2026-07-08: Phase breakdown into 4 phases
**Context:** Issue #1 has 8 acceptance criteria spanning preprocessing, training, evaluation, and publication. A single phase would be too large for progressive implementation.
**Decision:** Split into Phase 1 (preprocessing), Phase 2 (smoke training), Phase 3 (full training + evaluation), Phase 4 (diagnostics + publication).
**Rationale:** Each phase has a clear gate (dataset materialized, smoke passes, full run completes, results published). Later phases depend on earlier ones, so detailed planning for Phase 3+ is deferred.
**Consequences:** Phase 2-4 are stubbed until their predecessors complete.

## 2026-07-08: Preprocessing scripts as standalone modules
**Context:** The five source datasets have different schemas, preprocessing steps, and dependencies. A monolithic script would be hard to test and maintain.
**Decision:** Each source gets its own file in `data/preprocessing/` with a consistent function signature, plus a shared `render_utils.py` for common rendering operations.
**Rationale:** Independent testability, parallel development, clear ownership of each source's quirks.
**Consequences:** The merge script (`merge_mixture.py`) handles concatenation and shuffle; source-specific scripts focus only on their own schema conversion.

## 2026-07-08: Smoke and full runs use same materialized dataset
**Context:** Issue #1 specifies both a local RTX 3060 smoke and an A100 full run.
**Decision:** Preprocessing produces one materialized dataset. Smoke and full training differ only in hyperparameters (batch size, max steps, LoRA rank, max seq length).
**Rationale:** Eliminates the risk of dataset differences between runs. Smoke validates the dataset, full run uses the validated data.
**Consequences:** Phase 1 must produce a dataset that works for both compute profiles (fitting in 12 GB via small batch/context while still being useful for A100 scale-up).

## 2026-07-08: First-exact-sizes deferred to Phase 1 or Phase 2
**Context:** Open questions about exact subset sizes for DenseFusion, LibriSpeech configs, FineWeb-Edu chunks, and SmolTalk configs.
**Decision:** Choose small pragmatic defaults during Phase 1 implementation (e.g., 100-500 samples per source for smoke materialization). Document chosen subsets and sizes in metadata. Tune after Phase 2 smoke results.
**Rationale:** Locking exact sizes now would be speculative. Small first cuts validate the pipeline; scaling up is mechanical.
**Consequences:** Phase 1 will document the chosen subset sizes. Phase 2 smoke will validate whether the mixture proportions are reasonable (no single source dominates loss). Adjustments happen between Phase 2 and Phase 3.

## 2026-07-08: Keep structure flat under `data/preprocessing/`
**Context:** The project is greenfield with no existing Python package structure.
**Decision:** Place preprocessing modules directly under `data/preprocessing/` with a `__init__.py` rather than nesting deeper or creating a separate `univi/` package.
**Rationale:** Flat structure is simpler for a single-purpose research codebase. Refactoring into a proper package can happen later if the project grows.
**Consequences:** Import paths will be `from data.preprocessing.librispeech_asr import ...` or similar. Tests mirror this structure.

## 2026-07-08: `render_config` stored as JSON string instead of nested dict
**Context:** `datasets.concatenate_datasets` requires compatible Arrow feature schemas across all sources being concatenated. Each source preprocessor defines a different set of `render_config` keys (e.g. LibriSpeech has `sample_rate/n_mels/n_fft/hop_length` etc., while DenseFusion has `render_method`). If stored as a nested dict, Arrow would assign a struct type per source that varies across the group, causing concatenation to fail.
**Decision:** Serialise `render_config` as a JSON string via `json.dumps(config, sort_keys=True)`. Downstream consumers parse with `json.loads`. This gives every source the same `Value(string)` Arrow type, making concatenation safe.
**Rationale:** JSON strings are the simplest way to maintain heterogeneous config schemas while satisfying Arrow's type uniformity requirement. The penalty of `json.loads` per row is negligible at dataset-iteration scale.
**Consequences:** All preprocessors must use `json.dumps` with `sort_keys=True`. All test assertions on config contents must go through `json.loads`.

## 2026-07-08: Valor32k preprocessor requires attached media columns

**Context:** `inesriahi/valor32k-avqa-v2` HF Dataset rows expose only QA metadata columns (`question`, `options`, `correct_answer_idx`, `modality`, `video_id`, etc.) and do **not** include decoded audio arrays or video frames. The `video_id` column references the source YouTube video but no media is bundled in the dataset.

**Decision:** The Phase 1 `preprocess_valor32k` function requires attached media columns as input — an `audio` dict with `array`/`sampling_rate` for audio/audio-visual rows, and one of `frames`/`images`/`video_frames`/`image` for visual/audio-visual rows. If required media is missing, the function raises a `RuntimeError` with the `video_id` and a pointer to this decision entry.

**Rationale:** Media retrieval (downloading YouTube videos, extracting audio, sampling frames) is a separate, heavyweight concern outside the scope of Phase 1 schema conversion. Requiring attached media keeps the preprocessor testable with synthetic data and avoids leaking I/O concerns into schema transformation.

**Consequences:**
- Tests that supply synthetic audio/frames pass; tests against raw HF rows without media columns fail loudly.
- Future work (Phase 1a or Phase 2) must implement a media-retrieval pipeline that produces the required columns from `video_id`, or find a Valor32k variant with media already included.
- The failure message explicitly references `decisions.md` so operators know this is a known gap, not a bug.
- Downstream materialization scripts (Task 7 merge) must ensure media columns are present before calling `preprocess_valor32k`.

## 2026-07-08: Defer Valor32k from current Phase 1 materialization

**Context:** The Hugging Face dataset card documents bundled videos through `load_dataset(..., data_dir="videos")`, but those videos are test split only. Train/validation source videos are not redistributed and require a separate retrieval policy from `video_id`. `CONTEXT.md` currently says Valor32k train is for training while validation/test are held out for evaluation.

**Decision:** Strip Valor32k from the current active Phase 1 materialization and proceed with a four-source mixture: LibriSpeech, DenseFusion, FineWeb-Edu, and SmolTalk. Keep the Valor32k preprocessor code as deferred media-enriched support, but do not call it from the merge CLI until GitHub issue #2 resolves media retrieval and split policy.

**Rationale:** Training on bundled test videos would violate the documented split policy, while implementing YouTube retrieval is a larger research/design decision with availability and legal caveats.

**Consequences:** Task 7/8 should produce a four-source smoke dataset. Valor32k returns to the Phase 0 Training Mixture only after issue #2 is grilled and resolved.

## 2026-07-09: Native audio decode is preprocessing-only; training lane stays image-only

**Context:** `datasets==5.0.0` uses `torchcodec`-backed audio decoding for datasets like `openslr/librispeech_asr`. The LibriSpeech preprocessor reads raw audio arrays from the HF dataset and renders them into Log-Mel Spectrogram Images for the multimodal messages format. The downstream training lane operates exclusively on images.

**Constraint:** torchcodec versions must match the pinned PyTorch version. Current pin is `torch==2.10.0+cu130` → `torchcodec==0.10`. Upgrading requires coordinated torch + torchcodec bump.

**Decision:** Allow native audio decode (via datasets/torchcodec) at preprocessing time only. The training lane remains image-only because audio is rendered into Log-Mel Spectrogram Images before messages are materialized. This means the training pipeline never sees raw audio tensors.

**Rationale:** Preprocessing is a one-time materialization step that converts source-native formats into the unified image-text message format. Letting datasets handle audio decoding natively (instead of requiring pre-decoded WAV files) keeps the preprocessing pipeline simpler and avoids an extra I/O step.

**Consequences:**
- `torch` and `torchcodec` are now project dependencies (added in Task 0 of phase-01-preprocessing-pipeline.md).
- The training lane (`FastVisionModel`) receives only images and text — no audio or video tensors.
- Future audio sources can follow the same pattern: decode at preprocessing time, render to spectrogram image, downstream sees images only.

## 2026-07-08: `row_id` stored as string across all sources
**Context:** Source datasets use different identifier types. Some provide string ids, some provide numeric ids, and some have no stable id column. Mixed Arrow column types would make source concatenation fragile.
**Decision:** Store every `row_id` as a string. Use the source id when present and non-empty; if it is missing, `None`, or `""`, use the `datasets.map(..., with_indices=True)` index, coerced with `str(...)`.
**Rationale:** A uniform `Value(string)` column keeps the Materialized Render Dataset merge-safe while preserving source provenance.
**Consequences:** All preprocessors must coerce `row_id` with `str(...)`, and tests should include numeric-id coverage.

## 2026-07-09: Phase 3 scales from smoke-v0 to full-v0 with non-overlap controls

**Context:** The earlier smoke/full decision said the same materialized dataset serves both smoke and full runs. Phase 2 validated the dataset schema, Unsloth collation, training loop, checkpoint save, and checkpoint reload on `data/materialized/smoke-v0`. Phase 3 now needs enough rows to make A100 training and held-out evaluation meaningful.

**Decision:** Keep the same materialized dataset schema and rendering methods, but materialize a larger `data/materialized/full-v0` training dataset and separate `data/materialized/eval-v0/*` evaluation datasets. Phase 3 must add deterministic slicing/offset metadata and tests that assert source/split/row identifiers do not overlap between train and eval rows.

**Rationale:** Reusing only the 40-row smoke dataset would validate mechanics but not the larger training/evaluation behavior. Scaling the sample count while preserving schema keeps the Phase 2 validation relevant and makes the full run reproducible.

**Consequences:** Phase 3 implementation must not silently use default first-N train rows for both train and eval. Eval materialization must preserve native input fields (`native_user_content`, `target_text`, native availability flags) because the existing rendered-only training rows cannot reconstruct Native Upper Bound inputs reliably.

## 2026-07-13: Rebuild the 3M artifact as four source configurations

**Context:** The published `univi-3M-v0` artifact rendered FineWeb-Edu and SmolTalk as one unbounded-height image after silently truncating their input to 2,000 characters. LibriSpeech ASR and DenseFusion rows are correctly materialized.

**Decision:** Publish `fineweb-edu`, `librispeech_asr`, `densefusion`, and `smoltalk` as distinct Hugging Face configurations. Retain the existing LibriSpeech ASR and DenseFusion rows. Rebuild one million FineWeb-Edu examples and one million SmolTalk examples. Render the complete answer-bearing input on fixed 1024 px pages at a readable font size and use multiple ordered images when content overflows. Keep one output row per selected source row. Limit every assistant target to the first 1,024 Gemma tokenizer tokens; this cap applies to the output only and does not truncate the rendered input.

**Rationale:** Fixed pages preserve optical readability and source-specific configurations make composition auditable. Token-aware targets avoid silently exceeding the training context, while complete visual inputs retain the source content for later output-budget ablations.

**Consequences:** Materialization requires the `unsloth/gemma-4-E2B-it` tokenizer. Render metadata must record page geometry, image count policy, tokenizer, and output-token cap. The old `full-v0` directory remains immutable while the replacement is built and validated.

## 2026-07-13: Keep the 3060 pilot and A100 scale-up sequence-equivalent

**Context:** The first full-data training pass will run on an RTX 3060 with 12 GB VRAM before scaling to an A100 40 GB.

**Decision:** Use a 2,048-token total sequence contract for both runs. The RTX 3060 pilot uses 4-bit QLoRA, batch size 1, gradient accumulation, gradient checkpointing, and LoRA rank 8. The A100 run may increase batch size and LoRA rank, but must not change the materialized examples or sequence contract.

**Rationale:** Holding data and context length fixed makes stability and quality differences attributable to optimization scale rather than a changed task.

**Consequences:** The 1,024-token target cap is not an independent context budget: visual tokens, generic instructions, and target tokens must jointly fit within 2,048 tokens. Worst-case multi-page rows require a processor-level preflight before the 3060 run.

## 2026-07-15: Separate output cap from multimodal sequence budget; defer vision packing

**Context:** The materialized artifact caps assistant targets at 1,024 Gemma tokenizer tokens. The 12 GB training machine needs a bounded total context, and the current environment resolves to TRL 0.24.0 with Transformers 5.5.0.

**Decision:** Keep a 1,024-token **Assistant Output Cap** and a 2,048-token **Multimodal Sequence Budget**. Set generation/evaluation `max_new_tokens=1024`, but use `max_length=2048` for SFT. Enable no sequence packing for the first visual run. Use batch-size-one accumulation initially; investigate post-processor vision length bucketing only after a collator smoke test.

**Rationale:** A 2,048-token total budget is the minimum documented compromise that can retain long assistant targets while leaving room for image tokens and the task instruction. TRL's `packing=True` concatenates multiple examples into fixed-length blocks; it is not the same as length grouping and may invalidate multimodal image-placeholder alignment or response-only loss masks. The installed TRL/Transformers versions do not expose a `group_by_length` SFTConfig argument.

**Consequences:** Training configuration must use `max_length`, not the removed `max_seq_length` spelling. Validation must measure post-processor sequence lengths and report rows that cannot preserve their target under 2,048 tokens. `packing` remains explicitly false until a dedicated visual-collator test proves placeholder counts, image boundaries, and assistant loss masks.

## 2026-07-15: One-epoch training and periodic validation checkpoints

**Context:** The four training configurations contain 2,881,414 train rows. The requested run should consume the complete mixture once rather than stop after a small fixed step count.

**Decision:** Train for exactly one epoch. Evaluate and save every 1,000 optimizer steps. Select and retain the best checkpoint by lowest validation loss. Log total tokens seen using the available Trainer token-counting argument and include examples seen, optimizer steps, effective epochs, and per-subset validation losses in the run summary.

**Rationale:** One epoch gives a reproducible data-coverage contract. Fixed 1,000-step intervals make progress and checkpoint recovery observable while avoiding an arbitrary 500-step cap.

**Consequences:** With batch size 1 and gradient accumulation 4, the run is approximately 720,354 optimizer steps before accounting for dropped/rejected rows. Validation cost and the definition of “best” must be specified before implementation; a full validation pass over all four subsets every 1,000 steps may dominate training time. `save_total_limit: 3` is required, and the final checkpoint must be preserved even when it is not the best checkpoint.

## 2026-07-15: Count non-padding multimodal tokens in Trainer state

**Context:** The run must report total tokens seen without counting batch padding. Transformers 5.5.0 exposes `include_num_input_tokens_seen` and stores the counter as `TrainerState.num_input_tokens_seen`.

**Decision:** Set `include_num_input_tokens_seen: "non_padding"` and log `num_input_tokens_seen` to W&B at the normal logging interval and final summary. Treat this as the number of non-padding token IDs in the collated multimodal sequence, including visual placeholders and assistant tokens; report assistant-target tokens separately only if the validation/export pipeline computes them.

**Rationale:** Non-padding counting reflects actual tokenized work more closely than padded tensor capacity. The Trainer's field is named input tokens but counts the model's `input_ids`, not only the user turn.

**Consequences:** The run report must distinguish `num_input_tokens_seen` from any separately computed assistant-target count. Token tracking must be checked in a one-step smoke run before the one-epoch job.

## 2026-07-15: Enable training-only multimodal packing after a smoke gate

**Context:** The operator wants to maximize GPU utilization with TRL/Unsloth packing, while evaluation must preserve one independent visual conversation per batch item.

**Decision:** Set `packing: true` for training and `eval_packing: false` for evaluation. Do not start the one-epoch run until a dedicated packed-training smoke test passes image-placeholder alignment, image-to-example association, response-only label masking, finite loss/gradients, and checkpoint reload. Evaluation uses the un-packed `eval_dataset` mapping with one complete validation dataset per source.

**Rationale:** Packing can improve utilization, but its correctness is a hard prerequisite for a visual training run. Keeping evaluation un-packed makes per-subset losses and sample accounting auditable and avoids conflating evaluation results with packed-example behavior.

**Consequences:** The training run uses TRL's fixed-block packing at `max_length: 2048`; the smoke test must measure actual throughput and peak VRAM before the one-epoch job. Evaluation must never silently inherit training packing.

## 2026-07-15: Select checkpoints by macro-average validation loss

**Context:** The requested evaluation uses the complete validation split for each of the four source configurations at every 1,000 optimizer steps.

**Decision:** Supply `eval_dataset` as a mapping of four named datasets. Emit Trainer-compatible keys `eval_<subset>_loss` and `eval_mean_total_loss`, where the latter is the unweighted arithmetic mean of the four subset losses. Also mirror these metrics to W&B as `eval/<subset>_loss` and `eval/mean_total_loss`. Configure best-checkpoint selection against `metric_for_best_model: "mean_total_loss"` with lower-is-better semantics; Trainer resolves this to `eval_mean_total_loss`.

**Rationale:** An unweighted mean prevents the million-row FineWeb-Edu and SmolTalk subsets from overwhelming the smaller LibriSpeech and DenseFusion losses. Full validation preserves the requested score for every complete held-out subset.

**Consequences:** A custom evaluation callback/aggregation layer is required because Trainer's native dict-evaluation metric names do not automatically produce the requested aggregate key or W&B namespace. Every validation event may process approximately 266,391 rows, so runtime and checkpoint cadence must be measured in the packed smoke run.

## 2026-07-15: Installed Unsloth rejects VLM packing

**Context:** The requested training configuration prefers `packing: true`. Inspection of the installed generated `UnslothSFTTrainer` (Unsloth 2026.7-era environment) shows an explicit VLM guard that raises `ValueError` when `self._is_vlm and args.packing`, with the message that packing is not supported for vision-language models. It also rejects `padding_free=True` for VLMs.

**Decision:** Treat `packing: true` as a deliberate unsupported-path change, not a configuration toggle. Do not launch the 3M-row run with it until either the stack officially supports VLM packing or a custom trainer/collator implementation passes the required visual correctness gate.

**Rationale:** A smoke test that only reaches the current `SFTTrainer` constructor cannot validate packing because the constructor rejects it before collation. Bypassing the guard would make the run depend on unverified image-boundary, placeholder, position, and loss-mask behavior.

**Consequences:** The implementation must choose between the supported un-packed Unsloth VLM path and a separately isolated custom packed trainer. The earlier training-only packing decision is superseded by this environment constraint; evaluation remains un-packed in either path.

## 2026-07-15: Use supported un-packed VLM memory optimizations

**Context:** The operator accepted the installed Unsloth VLM restriction and chose `packing: false`. The goal is to maximize memory efficiency without enabling flags that target inference or an unverified model path.

**Decision:** Use 4-bit QLoRA, `use_gradient_checkpointing: "unsloth"`, 8-bit AdamW, mixed precision selected by `is_bfloat16_supported()`, gradient accumulation, and explicit CUDA peak-memory logging. Keep `fast_inference`, `float8_kv_cache`, `offload_embedding`, and `padding_free` disabled for training unless a Gemma 4-specific smoke test proves a measurable benefit without changing the training contract. Treat `unsloth_tiled_mlp` and `load_in_fp8` as experimental flags requiring isolated compatibility tests, not “enable everything” defaults.

**Rationale:** Unsloth exposes flags for different execution paths; `gpu_memory_utilization` and `float8_kv_cache` are documented near vLLM/inference loading, while VLM packing and padding-free execution are explicitly rejected in the installed trainer. Combining every flag would make failures and memory changes un-attributable.

**Consequences:** Configuration must distinguish supported training optimizations from optional experiments. Each experimental flag gets its own short smoke run with loss finiteness, grad norm, peak VRAM, and checkpoint reload checks before it can enter the 3M-row run.

## 2026-07-15: Use ratio-based warmup with one-point gradient clipping

**Context:** The one-epoch run is approximately 720,354 optimizer steps. A fixed 20-step warmup would be negligible, while the operator prefers the existing `max_grad_norm` default.

**Decision:** Use `learning_rate: 2.0e-4`, `warmup_ratio: 0.03`, `lr_scheduler_type: cosine`, `weight_decay: 0.001`, and `max_grad_norm: 1.0`. Set `warmup_steps: 0` so the ratio is authoritative. Log every 10 steps and send gradient norm metrics to W&B.

**Rationale:** A ratio scales warmup with the actual one-epoch step count. Clipping at 1.0 preserves the requested less-aggressive clipping while still bounding exploding gradients.

**Consequences:** The effective warmup is approximately 21,600 optimizer steps and must be printed in the resolved training configuration. The W&B smoke run must confirm `grad_norm` is present and finite before the full job.

## 2026-07-15: Exclude overlength rows deterministically

**Context:** The materialized targets are capped at 1,024 tokens, but total post-processor length also includes image tokens and the task instruction. TRL truncates sequences at `max_length` from the right, which could remove assistant targets.

**Decision:** Use `max_length: 2048` and `overlength_policy: "exclude"`. Run a processor-level preflight before training, exclude rows whose complete multimodal sequence exceeds 2,048 tokens, and write counts/reasons by subset and source row ID to the training manifest. Log excluded counts and retained counts to W&B. Never rely on silent Trainer truncation.

**Rationale:** Deterministic exclusion preserves target integrity while allowing the 3M-row job to proceed. Auditable counts reveal whether a render configuration or subset is systematically incompatible with the 2,048-token budget.

**Consequences:** The effective one-epoch dataset size is the retained-row count, not the raw manifest count. The resolved config and final report must include raw rows, excluded rows, retained rows, and effective epochs.

## 2026-07-15: Stratified startup data review

**Context:** Image counts vary substantially across the four materialized subsets, and the training run must expose the exact Gemma 4 multimodal prompt before optimization begins.

**Decision:** Generate a deterministic 32-row startup review: eight rows per subset, with selection stratified to include rare image-count buckets where available. Export `data/validation/<run>/training_data_review.md` and a matching W&B table.

**Required fields:** The review must show the full `processor.apply_chat_template(..., add_generation_prompt=True)` output, image previews/references, actual image count, image-placeholder count, placeholder equality, image dimensions, target-token count, total post-processor token count, source/split/row ID, render configuration, and validation/exclusion status.

**Rationale:** A small, reproducible artifact is inspectable before training while still covering source and image-count variation. The full dataset validator supplies aggregate confidence; the review artifact supplies human-readable evidence.

**Consequences:** Startup review generation is a hard pre-training gate. Any placeholder mismatch, missing image, malformed message, or non-finite tokenization result blocks the one-epoch run.

## 2026-07-15: Full processor-level preflight before GPU training

**Context:** The 32-row startup review is human-readable but cannot establish artifact-wide correctness. The approved overlength policy also requires measuring complete post-processor sequence lengths.

**Decision:** Validate every train and validation row across all four configurations with the actual Gemma 4 processor and original chat template before loading the training model for the production run. Cache the report and make reruns resumable by subset/configuration boundaries.

**Required aggregate counts:** `raw_rows`, `valid_rows`, `excluded_overlength_rows`, `placeholder_mismatch_rows`, `missing_image_rows`, `malformed_message_rows`, and `nonfinite_tokenization_rows`.

**Rationale:** Full validation is the only way to guarantee placeholder alignment and overlength accounting for the complete 3M-row artifact rather than a sample.

**Consequences:** Overlength rows are the only permitted exclusions. Any placeholder mismatch, missing/malformed image, malformed message, or non-finite processor result blocks training. The cached report and 32-row review are retained as run artifacts.

## 2026-07-15: Preserve manifest-container local artifact with Hub portability

**Context:** The local 3M artifact is a manifest container whose child train/validation directories are valid `save_to_disk` datasets; the root itself is intentionally not a `Dataset` or `DatasetDict`.

**Decision:** Keep `data/materialized/univi-3M-v0-split` unchanged. The local contract is manifest resolution plus child `load_from_disk` calls. On another machine, use the published `hungphongtrn/univi-3M-v0` configurations and splits through `datasets.load_dataset`.

**Rationale:** This preserves per-subset validation boundaries and avoids duplicating or destructively rebuilding the 3M artifact. Remote configurations provide the portable equivalent of the local child datasets.

**Consequences:** Training configuration must distinguish `local_path` and `hub_repo_id`; the loader validates the manifest and child datasets locally, and validates remote configuration names/splits when using Hub. W&B logs the resolved source and dataset revision.

## 2026-07-15: Use non-streaming cached Hub loading

**Context:** The published four-configuration Hub artifact is approximately 298 GB at dataset size and is intended to be portable to another training machine.

**Decision:** Remote mode uses normal non-streaming `datasets.load_dataset` with an explicit cache directory and disk-space preflight. It must load the same four configuration names and `train`/`validation` splits as local mode. Streaming is not a silent fallback.

**Rationale:** Non-streaming loading preserves deterministic concatenation/shuffling, full processor preflight, complete validation passes, and reliable checkpoint resumption. Streaming would introduce buffer-based shuffle and network-dependent failure modes.

**Consequences:** Startup checks must report required/free cache disk space and the resolved Hub revision. Remote runs may need roughly 300 GB or more of local storage including cache overhead.

## 2026-07-15: Use batch-one gradient accumulation on 12 GB

**Context:** The production run uses a 2,048-token multimodal sequence budget without packing, with substantial image-count variation across subsets.

**Decision:** Set `per_device_train_batch_size: 1` and `gradient_accumulation_steps: 4`, yielding effective batch size 4. Do not increase physical batch size unless a separate memory smoke test proves it safe.

**Rationale:** Batch one is the safest memory boundary for variable-length visual examples on 12 GB VRAM. Accumulation provides a modest effective batch without multiplying per-step visual memory.

**Consequences:** A complete retained-data epoch is approximately `retained_rows / 4` optimizer steps. Evaluation/checkpoint intervals are measured in optimizer steps, and the resolved config must report the resulting step count.

## 2026-07-15: Explicit checkpoint and W&B resume semantics

**Context:** One epoch over the retained 3M mixture may run for hundreds of thousands of optimizer steps and can be interrupted by machine or CUDA failures.

**Decision:** Refuse to overwrite a non-empty output directory by default. Add explicit `--resume` to select the latest valid checkpoint and continue the same W&B run; add an explicit destructive restart path rather than silently replacing checkpoints. Persist the W&B run ID, dataset revision/hash, resolved config, global step, and checkpoint path.

**Rationale:** Silent restart would invalidate token/epoch accounting and create ambiguous W&B curves. Explicit resume preserves reproducibility and makes interruptions recoverable.

**Consequences:** Checkpoint discovery and W&B run identity become pre-training validation requirements. A resume smoke test must verify optimizer/scheduler/global-step restoration, not merely model-weight loading.

## 2026-07-15: Preserve materialized chat roles without synthetic system prompts

**Context:** The materialized rows currently contain exactly `[user, assistant]`; the requested visualization described a possible system section but did not require one in training.

**Decision:** Do not add a synthetic system message. Apply the Original Gemma E2B Template to the messages exactly as materialized. The startup review renders the system section as `none` when absent and must not invent a system prompt.

**Rationale:** Preserving the materialized conversation avoids changing the training distribution, avoids a new preprocessing version, and keeps the review faithful to the actual training input.

**Consequences:** Validation must assert role order and forbid unexpected system/native answer-bearing content. Any future system prompt requires a separate dataset version and decision.

## 2026-07-15: Use rank-8 LoRA for the 12 GB epoch run

**Context:** The repository's A100 configuration uses rank 32, while the existing 12 GB configuration uses rank 8. The complete production run must fit a 2,048-token multimodal budget on 12 GB VRAM.

**Decision:** Use LoRA `r: 8`, `alpha: 16`, `dropout: 0.0`, `bias: none`, and the existing attention/MLP projection targets. Enable vision, language, attention, and MLP adapter modules. Reserve rank 16/32 for later ablations or larger GPUs.

**Rationale:** Rank 8 reduces trainable and optimizer memory while preserving the agreed multimodal adapter coverage. The 12 GB run is a constrained training measurement, not the final rank-scaling study.

**Consequences:** The resolved W&B config must include rank, alpha, target modules, and all finetuned component flags. A rank change creates a distinct run/checkpoint identity.

## 2026-07-15: Use NLL loss as the sole validation score

**Context:** The requested training objective is text generation from visual inputs, and the operator wants validation scores comparable to the training loss.

**Decision:** Evaluate negative log-likelihood loss only. Report one full-validation NLL loss for each subset and the unweighted four-subset mean at every 1,000 optimizer steps. Do not run generation-based WER, CER, ROUGE-L, or exact-match metrics during this training run.

**Rationale:** NLL is directly aligned with the SFT objective and avoids introducing generation-decoding choices into checkpoint selection.

**Consequences:** The final report must clearly label NLL as the validation score and state that task-specific generation metrics were intentionally deferred. Existing issue acceptance criteria mentioning task metrics remain a later evaluation concern unless separately re-scoped.

## 2026-07-15: Use batch-one full validation

**Context:** Validation uses the complete four-subset held-out data and has the same variable image counts as training.

**Decision:** Set `per_device_eval_batch_size: 1` and `eval_accumulation_steps: 1`. Keep evaluation un-packed and use the same 2,048-token sequence budget.

**Rationale:** Batch one is the safe VRAM boundary for 13-image audio rows and multi-page text rows. Evaluation throughput is secondary to completing every validation row without OOM.

**Consequences:** Full validation cadence will be slow and must be timed in the smoke run. The validation report must include rows processed and wall time per subset.

## 2026-07-15: Validate image sentinels separately from expanded visual tokens

**Context:** Gemma 4 expands each image into multiple visual token IDs, so comparing image count directly with total visual-token count would reject valid rows.

**Decision:** For every row, require equality between (1) the count of `type: image` content entries, (2) the count of per-image sentinel occurrences in the original Gemma 4 chat-template tokenization, and (3) the processor's decoded image/image-grid count. Report expanded visual-token length separately for the 2,048-token budget.

**Rationale:** The three-layer invariant checks both message/template alignment and actual image preprocessing without confusing one image with its expanded visual representation.

**Consequences:** The validator must discover the model-specific sentinel IDs from the loaded processor rather than hard-code a generic token string. A mismatch in any count blocks training and is included in the markdown/W&B diagnostics.

## 2026-07-15: Fingerprint processor and tokenizer before and after model load

**Context:** Preflight should avoid loading model weights, but training must use the exact processor/tokenizer behavior that Unsloth returns with Gemma 4.

**Decision:** Compute a canonical processor fingerprint before preflight and again after `FastVisionModel.from_pretrained` plus `get_chat_template(processor, "gemma-4")`. Include tokenizer vocabulary size/hash, tokenizer class, special-token map, chat-template text/hash, image sentinel IDs, processor class, processor configuration, and image-processing configuration. Abort if fingerprints differ.

**Rationale:** Matching only the tokenizer name is insufficient; a changed chat template or special-token mapping changes image placeholders and sequence lengths.

**Consequences:** The fingerprint and both load-stage summaries are written to the validation report, resolved config, W&B run metadata, and checkpoint manifest. The same fingerprint is required when reloading the final/best checkpoint for evaluation.

## 2026-07-15: Separate data-order and model seeds without strict CUDA determinism

**Context:** Reproducibility requires stable dataset order and adapter initialization, while strict deterministic CUDA kernels can disable optimized execution or reduce throughput.

**Decision:** Use `dataset.shuffle_seed: 42`, `training.seed: 3407`, `training.data_seed: 42`, and model/LoRA `random_state: 3407`. Do not require strict CUDA determinism. Log all seeds, library versions, CUDA device/driver, dataset revision, and processor fingerprint.

**Rationale:** Separating data order from adapter initialization makes reruns diagnosable while retaining Unsloth's optimized kernels.

**Consequences:** A resume must restore the original seed/config metadata. Any intentional seed change creates a distinct W&B run and checkpoint identity.

## 2026-07-15: Require online W&B monitoring

**Context:** W&B is a hard monitoring requirement, including gradient norm, per-subset NLL, token counts, validation artifacts, and checkpoint metadata.

**Decision:** Set `report_to: ["wandb"]` and require online W&B initialization before model training. Default project is `univi-gemma4`; entity is supplied by `WANDB_ENTITY` when needed. Persist the run ID and use explicit resume semantics. Do not silently fall back to `none` or offline mode.

**Rationale:** A long one-epoch run without live loss/gradient/checkpoint telemetry is not auditable or safely recoverable.

**Consequences:** Missing credentials, failed network initialization, or invalid project configuration blocks the run before GPU training. Run names include dataset revision, resolved-config hash, and seed.

## 2026-07-15: Keep compact but auditable preflight artifacts

**Context:** Full validation scans every row, but duplicating all multimodal payloads in a JSON report would be unnecessarily large.

**Decision:** Write `validation_summary.json`, `validation_failures.jsonl`, `validation_retained_manifest.jsonl`, and the 32-row `training_data_review.md`. The summary stores aggregate counts, fingerprints, revisions, hashes, timing, and software metadata. Failures store row identity, reason, and measured values. The retained manifest stores row IDs and resolved sequence-length metadata.

**Rationale:** The artifacts preserve all decisions and failures needed to reproduce the retained training set without copying images or messages.

**Consequences:** The training loader must consume the retained manifest or equivalent deterministic row filter, and W&B must upload all four validation artifacts.

## 2026-07-15: Apply deterministic overlength filtering to train and validation

**Context:** The 2,048-token multimodal budget can be exceeded by image tokens plus instruction and target. Keeping overlength validation rows would require silent truncation or a larger evaluation context.

**Decision:** Apply the same deterministic overlength filter to both train and validation. Evaluate NLL over every retained validation row in each subset, and report raw, excluded, and retained validation counts alongside each loss.

**Rationale:** Symmetric filtering keeps the processor contract identical between training and evaluation and avoids silently changing validation inputs. “Full validation” now means the complete retained/eligible validation split, not rows that cannot satisfy the declared sequence budget.

**Consequences:** Per-subset validation scores must include coverage metadata. The report must never present filtered validation NLL as if it covered the raw split; excluded counts and exclusion reasons are mandatory.

## 2026-07-15: Preserve top-level images with placeholder-only message content

**Context:** The local and Hub datasets store decoded images in a top-level `images` column and represent each image in `messages` as `{"type": "image"}`/`{"type": "image", "text": null}`. The installed Unsloth vision collator explicitly supports this two-column format.

**Decision:** Keep the published schema unchanged. The validator counts message image placeholders against `len(row["images"])`, then passes `messages` plus the top-level images column through the actual processor/collator path. It must not require nested `content[*]["image"]` objects.

**Rationale:** Rewriting 3M rows into nested image payloads would duplicate image storage and diverge from the format consumed by the installed Unsloth collator.

**Consequences:** Schema validation must test both message placeholders and top-level image payloads. The startup markdown must show image references/previews from `images`, while the rendered chat template shows placeholder positions separately.

## 2026-07-15: Configurable loss masking with response-only default

**Context:** The first run should optimize assistant targets, but future experiments may benefit from training on user/input tokens as well.

**Decision:** Add `loss_masking: response_only` as the production default, with an explicit configurable alternative `full_sequence`. In `response_only` mode, user/image/control labels must be `-100` and assistant target labels must be active. The validator and collator smoke test assert the selected mode rather than hard-coding one behavior.

**Rationale:** Response-only loss isolates the requested output-learning objective while keeping input-token training available as a deliberate ablation.

**Consequences:** The masking mode is part of the resolved config, W&B run identity, checkpoint metadata, and processor/collator validation report. Changing it creates a distinct training experiment.

## 2026-07-15: Use Unsloth response-only masking helper for Gemma 4 VLM

**Context:** The installed `unsloth_zoo.dataset_utils.train_on_responses_only` explicitly detects a VLM collator and installs a masking callable on the collator while preserving image processing. It also supports dict-valued evaluation datasets. The provided marker example is Llama-specific and is not correct for Gemma 4.

**Decision:** After constructing `SFTTrainer`, call `unsloth.chat_templates.train_on_responses_only` with the Gemma 4 processor. Prefer auto-detection after applying `get_chat_template(processor, "gemma-4")`; if explicit markers are needed, use the resolved Gemma markers `<|turn>user\\n` and `<|turn>model\\n`, with `force_match: true` and `last_response_only: false`.

**Rationale:** The helper's VLM path masks labels at collator time, avoiding a text-only dataset rewrite that would lose top-level image handling. Using Gemma's actual markers prevents silently masking every target.

**Consequences:** The collator smoke test must confirm active labels occur only in assistant spans for every subset, and must verify the helper does not alter image counts or processor outputs. Llama marker strings are forbidden in the Gemma 4 configuration.

## 2026-07-15: Auto-detect and assert Gemma 4 response markers

**Context:** Response-only masking depends on chat-template markers, and hard-coding markers from another model family could silently mask all targets.

**Decision:** Let `train_on_responses_only` auto-detect markers from the post-`get_chat_template(processor, "gemma-4")` processor, then assert the resolved markers are `<|turn>user\\n` and `<|turn>model\\n` for this run.

**Rationale:** Auto-detection follows the actual tokenizer while the assertion prevents an unexpected template or tokenizer revision from silently changing the loss mask.

**Consequences:** Marker resolution is included in the processor fingerprint and collator validation report. Any mismatch blocks training until the template/configuration is explicitly reviewed.

## 2026-07-15: Initialize W&B before preflight

**Context:** Dataset validation is part of the training experiment and can fail before model loading.

**Decision:** Initialize the online W&B run before processor-level preflight. Log validation progress, fingerprints, aggregate counts, artifacts, and failures. Close failed preflight runs with a failed status; only load model weights after validation passes.

**Rationale:** This makes data-quality failures observable and preserves the exact preflight evidence in the same experiment record.

**Consequences:** W&B credentials/network checks happen before GPU model initialization. Preflight failures must not leave ambiguous active runs.

## 2026-07-15: Pin the Hub dataset revision

**Context:** The remote dataset's `main` branch currently resolves to commit `5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375`. Unpinned Hub loading could silently change the 3M artifact between machines or resumes.

**Decision:** Pin remote loading to `5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375`. Log the revision and include it in the dataset/config hash. A checkpoint resume with a different dataset revision is rejected.

**Rationale:** Exact data reproducibility is required for one-epoch accounting, validation coverage, and W&B comparison across machines.

**Consequences:** Updating the Hub artifact requires a new explicit revision and distinct run/checkpoint identity. Local runs must record an equivalent manifest hash.

## 2026-07-15: Retain full periodic checkpoint state

**Context:** Reliable resume requires optimizer/scheduler/RNG state, not only LoRA weights.

**Decision:** Save full Trainer state at every 1,000-step checkpoint, apply `save_total_limit: 3` while preserving the best checkpoint, and save a separate final model/adapter plus processor export. The final export is not the resume source.

**Rationale:** Full state restores optimizer moments, cosine schedule, warmup progress, gradient accumulation, global step, and RNG state. A separate final export remains convenient for inference/evaluation.

**Consequences:** Disk-space preflight must include full checkpoint overhead. Resume smoke testing must compare restored global step, optimizer/scheduler state, and RNG metadata.

## 2026-07-15: Pin the Gemma 4 base-model revision

**Context:** The dataset revision is pinned, but `unsloth/gemma-4-E2B-it` currently follows a mutable Hub `main` branch at commit `4abfca14e6c6bfb5888b80288185b1243fb8d539`.

**Decision:** Pin model loading to revision `4abfca14e6c6bfb5888b80288185b1243fb8d539`. Include it in the resolved config, processor fingerprint, W&B run identity, and checkpoint manifest.

**Rationale:** Model/tokenizer/template changes can invalidate image sentinels, sequence lengths, adapter compatibility, and resume reproducibility.

**Consequences:** A model update requires a new explicit revision and a new run/checkpoint identity. Best/final checkpoints may only be reloaded with the matching base-model revision.

## 2026-07-15: Keep the pinned Torch 2.10 stack

**Context:** The current environment warns that some cpp extensions prefer Torch >=2.11, but the project intentionally pins `torch==2.10.0+cu130`, `torchvision==0.25.0+cu130`, and `torchcodec==0.10`.

**Decision:** Do not upgrade to Torch 2.11. Keep the locked Torch 2.10 stack and treat the cpp-extension warning as a known environment condition. Replace the warning-only gate with a functional smoke gate covering Gemma 4 processor/collator forward, response masking, finite loss/gradients, peak VRAM, W&B logging, checkpoint save, and reload.

**Rationale:** Preserving the pinned CUDA/Torch/Torchcodec combination avoids an unplanned dependency migration. Functional behavior is more relevant than a warning about optional extensions.

**Consequences:** The run report must record the warning and any disabled extension path. If the functional smoke fails, fix the Torch 2.10-compatible configuration rather than silently upgrading dependencies.

## 2026-07-15: Manage W&B through the uv lockfile

**Context:** W&B is mandatory, but it was absent from the project dependencies and the current environment.

**Decision:** Add W&B with `uv add wandb` and use the locked version for all training machines. The current lock resolves `wandb==0.28.0` and its runtime dependencies.

**Rationale:** Lockfile-managed dependencies prevent machine-specific monitoring behavior and avoid undocumented ad-hoc installs.

**Consequences:** W&B availability is now part of environment preflight. Future package changes must use `uv add`/`uv lock`, not manual installation.

## 2026-07-15: Require a full-stack GPU smoke gate

**Context:** Processor preflight cannot verify CUDA memory, Unsloth collation, response masking, Trainer token accounting, W&B telemetry, or checkpoint restoration.

**Decision:** Before the one-epoch run, execute a GPU smoke with eight retained rows per subset and the production stack: batch one, accumulation four, 2,048-token budget, QLoRA rank 8, response-only masking, un-packed VLM training, and online W&B. Run approximately 10 optimizer steps with at least one per-subset NLL evaluation and checkpoint save/reload.

**Required checks:** finite train/eval NLL, active assistant labels, finite `grad_norm`, increasing `num_input_tokens_seen`, no CUDA OOM, peak VRAM, W&B metrics/table, checkpoint weights, optimizer/scheduler/global-step restoration, and processor fingerprint match.

**Rationale:** This is the smallest gate that exercises the actual production contract rather than only imports or CPU schema parsing.

**Consequences:** A failed smoke blocks the 3M-row run. Smoke and production runs use distinct W&B run IDs and checkpoint directories.

## 2026-07-15: Evaluate best and last checkpoints separately

**Context:** The checkpoint selected by lowest macro-average validation NLL may differ from the final one-epoch checkpoint.

**Decision:** Reload and evaluate both the best checkpoint and the last checkpoint across every retained validation subset. Treat the best checkpoint as the primary result, the last checkpoint as a secondary result, and verify that the separate final export matches the last checkpoint's processor/fingerprint and model outputs.

**Rationale:** This exposes late-training degradation while preserving the requested best-model selection.

**Consequences:** The final JSON/W&B report must store checkpoint paths, dataset/model revisions, per-subset NLL, macro-average NLL, coverage counts, and processor fingerprints for both checkpoints and the final export.

## 2026-07-15: Persist final evaluation as JSON and W&B artifact

**Context:** W&B provides monitoring, but later comparisons and issue updates need a stable local result artifact.

**Decision:** Write `data/eval/univi-3M-1epoch-v0/results.json` after best/last/final-export evaluation and upload the same JSON to W&B as an artifact. Include per-subset retained/raw/excluded counts, NLL losses, macro-average loss, checkpoint identities, revisions, processor fingerprint, and resolved config.

**Rationale:** A local JSON contract is easy to diff, archive, and consume in later evaluation phases while W&B retains the live run context.

**Consequences:** Missing or incomplete JSON/W&B result artifacts fail the completion gate even if training itself finishes.

## 2026-07-15: Abort safely on persistent W&B failure

**Context:** W&B is mandatory, but transient network interruptions should not be confused with a deliberately offline run. The training job can resume from a full-state checkpoint.

**Decision:** Require online W&B initialization. Allow the SDK's transient retry/local queue behavior, but if telemetry enters a persistent unrecoverable failure, stop the run safely after the latest valid checkpoint and mark the run failed. Resume later with the explicit checkpoint/W&B resume policy.

**Rationale:** Continuing indefinitely without auditable telemetry violates the monitoring contract, while safe abort plus resume avoids losing a long run to a network outage.

**Consequences:** The trainer must expose W&B failure status, preserve the latest checkpoint, and avoid reporting a run complete until telemetry and artifacts synchronize successfully.

## 2026-07-15: Use Unsloth all-linear LoRA targets

**Context:** The operator prefers the official Gemma 4 vision guide's broader adapter coverage over the repository's explicit projection list.

**Decision:** Set `target_modules: "all-linear"` with rank 8 and alpha 16. Keep vision, language, attention, and MLP fine-tuning enabled. Treat peak VRAM and trainable-parameter count as mandatory smoke metrics.

**Rationale:** `all-linear` follows the official Unsloth Gemma 4 path and avoids missing model-specific linear modules. Rank 8 limits the adapter memory increase on the 12 GB machine.

**Consequences:** The GPU smoke gate must verify that all-linear fits at 2,048 tokens without OOM. The resolved config and W&B run identity must include the target policy; changing it creates a distinct experiment.

## 2026-07-15: Freeze the native audio tower

**Context:** Audio examples are represented as Log-Mel Spectrogram Images, and the image-only training lane never supplies native audio tensors.

**Decision:** Keep `finetune_audio_layers: false` while enabling vision, language, attention, and MLP adapter components with `target_modules: "all-linear"`.

**Rationale:** Updating the unused native audio pathway would violate the image-only training contract and spend memory on a pathway not exercised by this dataset.

**Consequences:** The resolved adapter configuration must record the frozen audio tower. Native audio fine-tuning remains a separate experiment.

## 2026-07-15: Use Unsloth minimum image resize

**Context:** Materialized pages and spectrogram tiles can be 1,024 px, but native resolution may exceed the 12 GB visual-token budget.

**Decision:** Construct `UnslothVisionDataCollator` with `resize: "min"`, `resize_dimension: 0`, and `snap_to_patch_size: false`. Use the same policy during processor preflight, training, validation, and checkpoint reload.

**Rationale:** Unsloth's minimum resize reduces visual tokens and VRAM while leaving the materialized source images unchanged for reproducibility and later resolution ablations.

**Consequences:** Resize policy is part of the processor/collator fingerprint and W&B config. Any resolution change creates a distinct run and requires a new preflight.

## 2026-07-15: Align loader LoRA capacity with rank 8

**Context:** The loader defaults to `max_lora_rank: 64`, while the approved production adapter rank is 8.

**Decision:** Set `model.max_lora_rank: 8` explicitly.

**Rationale:** Matching loader planning/compilation capacity to the actual adapter rank avoids unnecessary memory and compilation overhead.

**Consequences:** Any rank increase requires a new loader configuration and distinct smoke/run identity.

## 2026-07-15: Remove inference-only flags from production training config

**Context:** The existing 12 GB config includes `gpu_memory_utilization`, `float8_kv_cache`, and `unsloth_tiled_mlp`, but the verified training path does not require them.

**Decision:** Remove these flags from the production config. Keep only supported training loader options: 4-bit loading, Unsloth gradient checkpointing, pinned model revision, and `max_lora_rank: 8`.

**Rationale:** Inference/vLLM settings do not improve standard VLM SFT and experimental tiled MLP behavior has not passed a Gemma 4 training smoke.

**Consequences:** Any future use of these options requires an isolated experiment and separate W&B/checkpoint identity.

## 2026-07-15: Store dataset indices in retained manifests

**Context:** Row IDs provide provenance but may not be unique or efficient for selecting rows from local/Hub Arrow datasets after preflight.

**Decision:** Store `dataset_index` alongside subset, split, row ID, and sequence metadata. Consume retained manifests by index, then verify row ID/source fields before training or evaluation.

**Rationale:** Index-based selection avoids an additional full string filter and makes the retained set deterministic across local and pinned Hub sources.

**Consequences:** Any source reorder or revision mismatch is detected by the row-ID verification and blocks the run.

## 2026-07-15: Gate production behind explicit CLI phases

**Context:** A full 3M-row run should not start accidentally before CPU validation and the 12 GB GPU smoke pass.

**Decision:** Expose explicit `--preflight-only`, `--smoke`, `--resume`, and `--evaluate` modes. Default mode runs production only after matching preflight and smoke artifacts exist. Evaluation mode never trains; resume mode never overwrites.

**Rationale:** Named phases make the safety gates operational and make handoff to another machine reproducible.

**Consequences:** Each phase records its mode, config hash, dataset/model revisions, W&B run ID, and artifact paths. Production mode rejects stale or mismatched phase artifacts.

## 2026-07-15: Expose explicit dataset/preflight CLI overrides

**Context:** Moving between local and remote machines should not require editing the committed experiment configuration.

**Decision:** Add `--dataset-source auto|local|hub`, `--dataset-cache-dir`, `--dataset-revision`, and `--preflight-workers` CLI overrides. Config values remain defaults; every override is written to the resolved config and W&B metadata.

**Rationale:** Explicit overrides make cross-machine operation clear while preserving a reproducible base configuration.

**Consequences:** Resume rejects dataset revision/source changes unless the operator starts a new run. Hub overrides remain non-streaming.

## 2026-07-15: Use explicit LoRA projections to keep native audio frozen

**Context:** The installed Unsloth implementation forces `finetune_audio_layers: true` when `target_modules: "all-linear"`, conflicting with the image-only training contract and the approved frozen audio tower.

**Decision:** Supersede the all-linear target decision for production. Use the explicit projection targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, with rank 8/alpha 16 and `finetune_audio_layers: false`. Keep vision, language, attention, and MLP components enabled.

**Rationale:** The explicit list preserves the approved frozen native audio tower and avoids custom Unsloth adapter-selection code. `all-linear` remains a separate experiment requiring an explicit decision to fine-tune audio or a tested exclusion patch.

**Consequences:** The resolved production config must not contain `target_modules: "all-linear"`. The smoke gate must inspect trainable module names and assert no native audio module is trainable.

## 2026-07-15: Trust materialization contract for native-content leakage

**Context:** The preprocessing contract already defines image-only answer-bearing inputs and generic user instructions. The operator does not want an additional lexical leakage validator in this training run.

**Decision:** Do not add a subset-specific source-text leakage check. Retain structural message/schema validation, image-placeholder validation, target validation, and processor/collator checks.

**Rationale:** Avoid duplicating source-specific preprocessing logic in the training gate when the materialized artifact is already governed by the documented contract.

**Consequences:** Native-content leakage is an accepted unverified assumption for this run and must not be described as independently tested in the final report.

## 2026-07-15: Use W&B for human divergence monitoring

**Context:** The operator will monitor loss and gradient behavior directly in W&B rather than applying an automated loss-trend policy.

**Decision:** Do not auto-abort on loss spikes, moving-average increases, or subset-specific degradation. Keep hard runtime/data safeguards for NaN/Inf tensors, invalid labels, CUDA OOM, checkpoint failures, and unrecoverable W&B failure.

**Rationale:** Mixed-subset NLL curves are not guaranteed to be monotonic, and human inspection of the logged per-subset curves is the intended monitoring path.

**Consequences:** The final run report must distinguish completed training from qualitative W&B monitoring; no automated claim of “loss trending down” is made.

## 2026-07-15: Use the official Unsloth processor path for preflight

**Context:** Direct `AutoProcessor` loading does not expose the same wrapper behavior as the official Unsloth Gemma 4 notebook. The operator chose the official path.

**Decision:** Acquire the processor through `FastVisionModel.from_pretrained` and apply `get_chat_template(processor, "gemma-4")` exactly as in the official guide. Run the processor-level preflight without allocating the model to the training GPU; if the CPU/off-GPU official load is unsupported, fail rather than silently substitute a different processor API. Reload the pinned model through the same official path after preflight for the GPU smoke/training phase.

**Rationale:** Exact Unsloth processor/template behavior is more important than avoiding one additional model-load cycle.

**Consequences:** Preflight records the official processor/template fingerprint and the GPU phase must match it. The environment smoke must verify that the CPU/off-GPU official load works under the pinned Torch 2.10 stack.

## 2026-07-15: Propagate Unsloth gradient-checkpointing mode

**Context:** The model loader is configured for `use_gradient_checkpointing: "unsloth"`, but the existing `FastVisionModel.for_training(model)` call defaults to ordinary `True`.

**Decision:** Pass the configured checkpointing mode explicitly through model loading, PEFT setup, `FastVisionModel.for_training`, Trainer transitions, evaluation reloads, and checkpoint reloads.

**Rationale:** A default-boolean transition could silently disable Unsloth's specialized memory-saving path after model initialization.

**Consequences:** The smoke gate must report the effective checkpointing mode and peak VRAM. Any checkpoint reload that does not restore the configured mode blocks evaluation.

## 2026-07-15: Enable Trainer-level gradient checkpointing explicitly

**Context:** Unsloth model loading selects the specialized `"unsloth"` checkpointing mode, while Trainer arguments have an independent `gradient_checkpointing` flag.

**Decision:** Set `model.use_gradient_checkpointing: "unsloth"` and `training.gradient_checkpointing: true`. Propagate the effective Unsloth mode through model/Trainer transitions.

**Rationale:** Both layers must remain enabled to preserve the memory-saving path during training and evaluation reloads.

**Consequences:** Smoke reports both the Trainer flag and effective model mode. A mismatch blocks production.

## 2026-07-15: Let tokenizer budget determine subset retention

**Context:** Overlength eligibility is determined by the actual Gemma 4 processor/tokenizer under the 2,048-token budget; arbitrary per-subset retention thresholds are not desired.

**Decision:** Apply the tokenizer-based filter without a manual percentage threshold. Report raw, excluded, retained, and retention-rate counts for every subset. A subset with no eligible rows remains a structural failure because it cannot contribute training or validation data.

**Rationale:** The processor is the authority on multimodal fit; imposing an arbitrary retention cutoff before observing actual lengths would add an unsupported data policy.

**Consequences:** The preflight report must make any low-retention subset visible, and the operator can review the exact excluded row IDs/reasons without silently changing the budget.

## 2026-07-15: Use standard Unsloth VLM truncation without length preflight

**Context:** The operator chose the standard Unsloth VLM training path with `skip_prepare_dataset: true` and `max_length: 2048`, without a separate processor-length pass.

**Decision:** Supersede the deterministic overlength-filter decisions for this run. Keep the full structural/image/template preflight, but do not perform a separate untruncated token-length scan or retained-row filtering. Let the standard Unsloth VLM collator process all raw train/validation rows under the 2,048-token limit.

**Rationale:** This avoids a separate dataset tokenization pass and follows the official VLM SFT data path.

**Consequences:** The 1,024-token assistant cap remains a materialization target cap, not a guarantee that every multimodal sequence preserves its complete target under 2,048 total tokens. The run accepts processor-level truncation for overlength rows. Reports must not claim zero target truncation or exact retained-row coverage. `validation_retained_manifest.jsonl` and overlength exclusion counts are removed from the required artifacts for this run.

## 2026-07-15: Prefer official Unsloth VLM flow over duplicate tokenization

**Context:** The operator wants the first implementation close to the official Gemma 4 Unsloth guide and does not want a separate full tokenizer pass before training.

**Decision:** Use the official VLM SFT flow with `remove_unused_columns: false`, empty `dataset_text_field`, `skip_prepare_dataset: true`, and `max_length: 2048`. Keep all-row structural/image/schema validation and the 32-row exact processor review, but do not add a full tokenizer-level preflight or precomputed token cache. Let the Unsloth collator tokenize each training/evaluation batch once.

**Rationale:** This minimizes custom infrastructure for the first run and keeps the implementation close to the documented Unsloth path. Later work can add exact all-row token audits or cached multimodal features if needed.

**Consequences:** The first run does not prove tokenizer-level placeholder equality for every row and accepts standard processor truncation under the 2,048-token limit. Reports must state this limitation explicitly. Any batch with zero active response labels remains a runtime hard failure.
