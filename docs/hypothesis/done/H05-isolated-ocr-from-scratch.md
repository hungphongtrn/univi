# H05 — Isolating a prior-proof OCR lane rescues the from-scratch embedder

**Status:** DONE · **Verdict: REFUTED** (2026-07-26) · Related: [H03](H03-from-scratch-embedder-shallow-reading.md), [H06](H06-mlp-connector-escalation.md)

## Claim

Train on **one** lane with zero language shortcut — render random lowercase strings, transcribe them
— so a steering prefix cannot cheat. This separates gradient competition from embedder capacity and
should force the from-scratch embedder to read. (Encoder-free "H3".)

## Setup

New materializer `data/preprocessing/random_strings.py` → `data/materialized/h3-randstr-v0/`
(100k/2k, floor 5.614 nats/tok). Recipe = [H03](H03-from-scratch-embedder-shallow-reading.md) exactly
(single LR 3e-4, no freeze), warm-started from `encoder-free-v0/best`.
W&B `gshey7fa`, 1500 steps, ran SOLO.

## Result

**Grounding collapsed to exactly zero.**

| condition | CE | tok-acc |
|---|---|---|
| aligned | 4.023 | 32.07% |
| permuted | 4.023 | — |
| blank | 4.023 | 32.07% (**bit-identical**) |

Δperm +0.0005%, Δblank +0.001%. The image is **provably ignored**. Not a plumbing bug: the same code
and rows show H03's checkpoint responding (Δblank +1.61%, reading gain +0.26 pts).

## Learning

- H05 **destroyed** the weak reading H03 had — a regression, not a stall.
- CE fell 6.20 → 4.02 (below the 5.61 floor) entirely via **format modelling**: the decoder learned
  the deterministic scaffold (fixed 5-letter groups, spaces, `im_end`, uniform letter marginal,
  guessing at chance) while the embedder's signal decayed to nothing.
- This is [H04](H04-frozen-decoder-warmup.md)'s collapse basin reappearing **without any language
  shortcut** — a strong pretrained decoder plus a from-scratch embedder finds "model the output
  marginal and the format, ignore the pixels" as the lowest-loss basin.
- The mixture's other lanes had been keeping the embedder marginally alive; isolation removed them.
- Vindicates the pre-registered warning that **crossing the no-reading floor is not the clean
  signal** — H05 sits below floor while reading nothing.

## Artifacts

`data/eval/h3-ablation-h3.json`, `data/eval/h3-ablation-h1-baseline.json`, `scratchpad/h3_ablation.py`
