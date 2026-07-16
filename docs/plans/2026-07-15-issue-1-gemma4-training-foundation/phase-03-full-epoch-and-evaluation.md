# Phase 3: Full Epoch + Evaluation (Stub)

## Phase Goal

Complete one epoch of training across the full 3M-row materialized dataset (approximately 720,000 optimizer steps at batch 1 / accumulation 4), evaluate the best and last checkpoints, and produce a structured JSON report plus W&B artifact with per-subset NLL loss, macro-average loss, and coverage metadata.

## High-Level Outcome

- One epoch of training completes on A100 40 GB (or scaled RTX 3060)
- Checkpoints saved every 1,000 steps, best selected by `mean_total_loss`
- Best and last checkpoints evaluated against all validation subsets
- Final JSON written to `data/eval/univi-3M-1epoch-v0/results.json`
- W&B artifact contains results JSON, config hash, revisions, fingerprints
- Report states accepted limitations: no full tokenizer preflight, standard processor truncation, no generation metrics

## Depends On

- Phase 1: Training Foundation (all CPU-testable components)
- Phase 2: GPU Smoke Gate (production stack verified on real hardware)
- A100 40 GB (or confirmed RTX 3060 can complete one epoch)
- Materialized dataset at `data/materialized/univi-3M-v0-split` (or Hub)
- `wandb` project `univi-gemma4` accessible

## Rough Task Outline

This phase will be detailed after Phase 2 completes. Expected tasks:

1. **Production config finalization** — Based on Phase 2 smoke learnings, finalize `configs/3060_1epoch.yaml` and (if A100 available) `configs/full.yaml` with confirmed batch size, gradient accumulation, and save/eval intervals.
2. **Training execution** — Launch one-epoch run via `univi.train --config configs/3060_1epoch.yaml`. Monitor via W&B.
3. **Resume testing** — Verify that interrupting and resuming preserves global step, optimizer state, LR schedule, and W&B continuity.
4. **Best checkpoint evaluation** — Select checkpoint with lowest `mean_total_loss`, reload, compute per-subset NLL on validation rows.
5. **Last checkpoint evaluation** — Reload the final checkpoint (separate from the final export) and compute per-subset NLL.
6. **Final export verification** — Confirm the `final/` export matches last checkpoint's processor fingerprint and produces equivalent NLL.
7. **Results JSON** — Write structured `results.json` with all per-subset losses, coverage counts, checkpoint identities, revisions, fingerprints.
8. **W&B artifact upload** — Upload results JSON and ensure W&B run summary contains all key metrics.
9. **Limitations report** — Document accepted limitations (no full preflight, standard processor truncation, no generation metrics, native-content leakage unchecked).
10. **Run report** — Summary suitable for issue #1 update.

## Files to Touch (Rough)

- `configs/3060_1epoch.yaml` — finalize
- `configs/full.yaml` — finalize (A100 variant)
- `univi/trainer.py` — evaluation reload and NLL computation (extend if needed)
- `univi/checkpoint_evaluation.py` — checkpoint reload + NLL evaluation helper (new; distinct from Phase 1's `univi/evaluation.py` which holds shared metric aggregation and UniViSFTTrainer)
- `tests/test_checkpoint_evaluation.py` — checkpoint evaluation tests (new)
- `data/eval/univi-3M-1epoch-v0/results.json` — output artifact

## Completion Criteria (Rough)

- [ ] One-epoch training completes with `train()` returning normally
- [ ] Best checkpoint restored: `mean_total_loss` confirmed as minimum
- [ ] `data/eval/univi-3M-1epoch-v0/results.json` contains:
  - Per-subset: `raw_rows` (note: rows are subject to standard processor truncation; no separate overlength pre-filter is applied)
  - Per-subset: `best_nll`, `last_nll`, `final_export_nll`
  - `mean_total_loss` for best and last checkpoints
  - `config_hash`, `dataset_revision`, `model_revision`, `processor_fingerprint`
  - Run metadata: `total_steps`, `total_tokens_seen`, `effective_epochs`
  - `limitations` section documenting accepted gaps
- [ ] W&B artifact `results-3060-1epoch-v0` uploaded
- [ ] Issue #1 updated with run summary and link to W&B

## Stub Note

This stub will be expanded into a full detailed plan after Phase 2 passes the GPU smoke gate. The exact checkpoint intervals, validation overhead, disk-space requirements, and expected wall time will be informed by Phase 2 measurements. The A100 path may differ significantly from the 3060 path; the detailed plan will address both with clear runtime estimates.

**Do not begin implementation of this phase until Phase 2 is complete and this document has been detailed.**
