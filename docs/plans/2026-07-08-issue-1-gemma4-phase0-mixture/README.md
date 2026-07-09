# Train Gemma 4 E2B on Phase 0 Visual-Unification Mixture

> **For agentic workers:** Use subagent-driven-development. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 3 — Full Training & Evaluation
- **Previous Phase:** Phase 2 — Smoke Training ✅
- **Overall Progress:** 2/4 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) — Understand the big picture (5 min)
2. [phase-03-full-training-and-evaluation.md](./phase-03-full-training-and-evaluation.md) — Only the phase you're implementing (15 min)
3. [decisions.md](./decisions.md) — Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 — Preprocessing Pipeline | ✅ Complete | Materialized HF dataset for 4 active sources; Valor32k deferred behind [#2](https://github.com/hungphongtrn/univi/issues/2) | [phase-01](./phase-01-preprocessing-pipeline.md) |
| 2 — Smoke Training | ✅ Complete | RTX 3060 smoke run validates loading, training, checkpoint | [phase-02](./phase-02-smoke-training.md) |
| 3 — Full Training & Evaluation | 🔲 Active | A100 full run completes with checkpoint reload; metrics reported | [phase-03](./phase-03-full-training-and-evaluation.md) |
| 4 — Diagnostics & Publication | 🔲 Pending | Retention, shuffled-option, invalid-rate diagnostics published | [phase-04](./phase-04-diagnostics-and-publication.md) stub |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.

## Source Issue
[GitHub #1](https://github.com/hungphongtrn/univi/issues/1) — Train Gemma 4 E2B on a Phase 0 visual-unification training mixture and evaluate text-as-image, audio-as-image, and mixed visual-input retention.
