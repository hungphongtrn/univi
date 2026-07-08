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
