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

## 2026-07-08: `row_id` stored as string across all sources
**Context:** Source datasets use different identifier types. Some provide string ids, some provide numeric ids, and some have no stable id column. Mixed Arrow column types would make source concatenation fragile.
**Decision:** Store every `row_id` as a string. Use the source id when present, otherwise the `datasets.map(..., with_indices=True)` index, coerced with `str(...)`.
**Rationale:** A uniform `Value(string)` column keeps the Materialized Render Dataset merge-safe while preserving source provenance.
**Consequences:** All preprocessors must coerce `row_id` with `str(...)`, and tests should include numeric-id coverage.
