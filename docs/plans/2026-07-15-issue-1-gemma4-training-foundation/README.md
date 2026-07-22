# Training Foundation — Gemma 4 E2B Visual-Modality Unification

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status

- **Current Phase:** Phase 2 — GPU Smoke Gate (detailed below)
- **Next Up:** Phase 3 — Full Epoch + Evaluation (pending Phase 2 completion)
- **Overall Progress:** Phase 1 complete; 1/3 phases complete
- **Latest Authoritative Decisions:** 2026-07-15 (reconcile older entries against these)
- **Phase 1 Evidence:** 56 trainer tests, 16 CLI/W&B tests, plus earlier focused config/evaluation/validation tests pass; GPU smoke remains pending.

## Start Here

New implementer? Read in this order:

1. [strategy.md](./strategy.md) — Understand the big picture, architecture, and constraints (5 min)
2. [phase-02-gpu-smoke-gate.md](./phase-02-gpu-smoke-gate.md) — Current phase contract and tasks (20 min)
3. [phase-01-training-foundation.md](./phase-01-training-foundation.md) — Foundation APIs (reference only)
4. [decisions.md](./decisions.md) — Reconciled decision log with authoritative entries highlighted (10 min)

**Do NOT implement Phase 3 yet. It remains stubbed until the GPU smoke produces evidence.**

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Training Foundation | Complete | Config/CLI framework, dataset loader, validation, fingerprint, trainer construction, W&B lifecycle, CPU-safe tests | [phase-01](./phase-01-training-foundation.md) |
| 2 — GPU Smoke Gate | Active | 12 GB GPU smoke verifies collation, masking, loss, checkpoint, W&B, and fingerprints | [phase-02](./phase-02-gpu-smoke-gate.md) |
| 3 — Full Epoch + Evaluation | Stub | One-epoch training, best/last checkpoint evaluation, JSON+W&B results | Stub only |

## Key Decisions

All decisions are recorded in [decisions.md](./decisions.md). The 2026-07-15 entries are authoritative and supersede earlier contradictory entries:
- **Standard Unsloth VLM flow** (no full tokenizer preflight, no deterministic overlength exclusion)
- **Explicit LoRA projections** with native audio frozen (not `all-linear`)
- **Response-only masking** via `train_on_responses_only`, with an active-label collator guard
- **W&B online required** with explicit resume semantics
- **`max_length=2048`** (not `max_seq_length`)

## Source Issue

[GitHub #1](https://github.com/hungphongtrn/univi/issues/1) — Train Gemma 4 E2B on Phase 0 visual-unification training mixture.
