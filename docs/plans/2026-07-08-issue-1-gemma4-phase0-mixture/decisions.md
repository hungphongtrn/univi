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
