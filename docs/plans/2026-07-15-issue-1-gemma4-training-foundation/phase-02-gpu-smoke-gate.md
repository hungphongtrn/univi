# Phase 2: GPU Smoke Gate

## Phase Goal

Execute the Phase 1 production training stack on the available 12 GB GPU with the retained Phase 0 rows and prove that collation, response-only masking, finite loss, W&B telemetry, checkpoint restoration, and processor fingerprints work on real CUDA tensors.

This phase is a gate, not a model-quality experiment. A failed gate blocks the full 3M-row training run.

## Depends On

- Phase 1 CPU foundation and focused tests pass.
- `data/materialized/univi-3M-v0-split` exists locally, or the pinned Hub revision is reachable.
- Pinned Torch 2.10 + CUDA 13 stack and a CUDA-capable GPU are available.
- W&B credentials are available for the smoke run; disabled W&B is allowed only in CPU tests.

## Resolved Smoke Contract

- Dataset source: local by default; Hub only when explicitly selected.
- Rows: exactly eight retained rows per active subset (32 rows for the four Phase 0 subsets), selected by the Phase 1 review/retention policy and recorded by row ID.
- Optimizer budget: exactly 10 optimizer steps; `training.max_steps=10` is authoritative.
- Sequence budget: `training.max_length=2048`; assistant target cap remains the materialized 1024-token limit.
- Batch: use the production batch size and accumulation from `configs/smoke.yaml`; no packing or cross-example concatenation.
- Loss: response-only masking with `force_match=True`; any batch with zero active assistant labels fails the gate.
- Evaluation: evaluate every configured validation subset at the smoke endpoint; all reported NLL values must be finite.
- Telemetry: W&B online run with resolved config, per-subset eval metrics, finite `grad_norm`, increasing `num_input_tokens_seen`, validation artifacts, checkpoint metadata, and a smoke summary table.
- Memory: no CUDA OOM; report peak allocated and reserved VRAM; peak allocated must remain below 11.5 GiB.
- Reproducibility: smoke uses a distinct output directory and W&B run ID from production; all artifacts record config hash, dataset revision, model revision, processor fingerprint, and run ID.

## Files to Touch

- `configs/smoke.yaml` — retain the 10-step, 8-row-per-subset smoke contract and dedicated output/W&B identity.
- `univi/smoke.py` — GPU dependency checks, retained-row loading, collator assertions, smoke execution, checkpoint reload, and JSON report.
- `univi/cli.py` / `univi/trainer.py` — only if the existing Phase 1 mode cannot expose a required smoke check without weakening production gates.
- `tests/test_gpu_smoke.py` — GPU-gated integration tests; CPU-safe tests for report schema and selection logic may be included.
- `tests/test_trainer.py` — focused regression tests only when a smoke integration bug is found.

## Tasks

### Task 1: Freeze and verify smoke configuration

1. Add or update focused config tests asserting:
   - `max_steps == 10`;
   - `max_length == 2048`, `packing is false`, and response-only masking is enabled;
   - all four active subsets have eight retained rows and validation splits;
   - output directory and W&B run identity are smoke-specific;
   - model and dataset revisions are pinned 40-character commits.
2. Resolve the config through `resolve_config()` and write its hash into the smoke report before model loading.
3. Reject a smoke configuration that requests a different step budget, missing subset, unpinned revision, or production output directory.

### Task 2: Add GPU smoke harness and dependency gate

1. Implement one callable smoke entry point used by `python -m univi.train --config configs/smoke.yaml --smoke`.
2. Check `torch.cuda.is_available()`, CUDA device name/driver, compute capability, Torch/CUDA versions, and available VRAM before model loading.
3. Fail with a structured report when CUDA is unavailable or the device has less than 12 GiB capacity; do not fall back to CPU or silently run a partial smoke.
4. Create a dedicated smoke output directory and W&B run identity. Never reuse production checkpoints or run IDs.

### Task 3: Validate retained rows and production collation

1. Load only the retained rows per subset using the pinned local manifest or Hub revision; preserve source subset and row IDs.
2. Load the official Unsloth processor/model path, apply the Gemma 4 chat template, and compare the post-load fingerprint with the Phase 1 preflight fingerprint.
3. Collate one batch per subset using the exact production collator.
4. Assert for every batch:
   - `input_ids`, labels, and image tensors have valid shapes and finite values;
   - the number of actual image payloads equals the image placeholder count expected by the processor/tokenizer;
   - at least one assistant label is active;
   - active labels occur only in assistant spans;
   - no row exceeds the resolved 2048-token multimodal budget without an explicit failure.
5. Store representative rendered conversations and label-position diagnostics in the smoke report and W&B table.

### Task 4: Execute ten-step training and endpoint evaluation

1. Start the production trainer with `max_steps=10`, no packing, response-only masking, and the retained smoke dataset.
2. Assert after each optimizer step that loss is finite and that `grad_norm` is present and finite in the trainer/W&B log history.
3. At the endpoint, evaluate each configured validation subset separately and record raw loss, NLL, coverage, and macro-average. Any missing subset, NaN/Inf metric, zero active labels, or invalid label mask fails the smoke.
4. Assert `num_input_tokens_seen` increases across log entries and that no hidden full-sequence fallback occurred.
5. Record peak allocated/reserved VRAM and fail if allocated memory reaches 11.5 GiB or a CUDA OOM occurs.

### Task 5: Save, reload, and verify artifacts

1. Save a checkpoint at step 10 plus optimizer, scheduler, trainer state, tokenizer, processor fingerprint, config hash, dataset/model revisions, and W&B run ID.
2. Reload the checkpoint in a fresh model/trainer instance and assert global step 10, optimizer/scheduler state presence, and matching processor fingerprint.
3. Verify the smoke W&B run contains resolved config, grad norm, per-subset eval metrics, validation artifacts, checkpoint metadata, and the 32-row review table.
4. Write `data/smoke/<run-id>/results.json` with status, failure reason (if any), environment, revisions, fingerprints, metrics, memory, checkpoint paths, and W&B URL.
5. Write a concise human-readable summary next to the JSON report.

### Task 6: GPU-gated test and phase gate

1. Add `tests/test_gpu_smoke.py` with `@pytest.mark.requires_gpu` tests for dependency checks, one-batch collation, response masking, finite metrics, checkpoint restoration, and fingerprint matching.
2. Keep report-schema and retained-row selection tests CPU-safe and deterministic.
3. Run focused CPU tests for changed modules, then run:

```bash
uv run pytest tests/test_gpu_smoke.py -m requires_gpu -v
uv run python -m univi.train --config configs/smoke.yaml --smoke
```

4. Phase completion requires a successful `results.json`, a W&B URL, all GPU-gated checks passing, peak allocated VRAM below 11.5 GiB, and no unresolved failure criterion. A missing GPU is an environment blocker, not a passing smoke.

## Failure Criteria

The smoke is failed and production remains blocked on any of:

- CUDA unavailable, insufficient VRAM, or CUDA OOM;
- model/processor or dataset revision mismatch;
- image payload and tokenizer placeholder mismatch;
- zero active assistant labels or labels outside assistant spans;
- non-finite train/eval loss, grad norm, or token counter;
- absent per-subset validation metric;
- checkpoint reload does not restore step/optimizer/scheduler state;
- processor fingerprint changes after reload;
- missing W&B config, telemetry, artifacts, or run ID;
- peak allocated VRAM at or above 11.5 GiB;
- malformed or incomplete `data/smoke/<run-id>/results.json`.

## Phase Completion Criteria

- [ ] Focused CPU tests for smoke helpers pass.
- [ ] All GPU smoke tests pass on the target machine.
- [ ] The unified `--smoke` command exits 0.
- [ ] A structured smoke report and W&B URL are available.
- [ ] The full training gate can consume the smoke manifest without manual edits.

## Handoff Notes

Phase 3 may begin only after this smoke report is successful. Phase 3 must use a distinct production output directory and W&B run ID, preserve the same processor/model/dataset revisions, and consume the smoke report as evidence rather than rerunning smoke checks implicitly.
