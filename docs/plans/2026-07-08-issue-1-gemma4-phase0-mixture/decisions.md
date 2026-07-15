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
