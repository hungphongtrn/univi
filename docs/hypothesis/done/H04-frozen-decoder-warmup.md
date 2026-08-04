# H04 — Freezing the decoder first lets the embedder catch up

**Status:** DONE · **Verdict: REFUTED** (2026-07-25) · Related: [H03](H03-from-scratch-embedder-shallow-reading.md)

## Claim

[H03](H03-from-scratch-embedder-shallow-reading.md) was shallow because the decoder won the gradient
competition. Freezing the decoder for a warmup period, then splitting learning rates, will let the
from-scratch embedder establish reading first. (Encoder-free "H2".)

## Setup

`configs/encoder_free_h2.yaml`, W&B `xc2vypxx`. Decoder frozen 800 steps, then embedder 3e-4 /
decoder 2e-5. Checkpoints `data/checkpoints/encoder-free-h2-v0/`.

## Result

**Worse than H03 on every axis.** Final macro **1.875** vs H03's 1.743. Warmup barely moved loss;
all descent came after unfreeze; hard plateau after step 1250.

Both grounding signals **reversed**:

| lane | Δperm H03 → H04 | Δblank H04 |
|---|---|---|
| fineweb-edu | 3.15% → **0.22%** | **+23%** |
| densefusion | 3.57% → **0.14%** | **+68%** |
| smoltalk | 3.43% → **1.38%** | **+95%** |
| librispeech | 1.87% → **0.57%** | — |

## Learning

- Mechanism = **content-blind soft-prompt collapse**. Against a frozen decoder, the only way a
  from-scratch embedder can lower loss is a constant, content-independent steering vector. Δperm
  collapses to the ignore-floor (content doesn't matter) while Δblank explodes (the prefix matters).
- fineweb aligned CE dropped below the no-image floor (2.64 → 2.47) — **and that was the steering
  prefix, not reading.** This is the origin of the standing warning that crossing a floor is not
  evidence of grounding.
- **Never freeze the decoder for a from-scratch embedder.** The warmup *caused* the collapse.
- This failure mode is why every later run pre-registers Δperm *and* Δblank: either alone is
  ambiguous.

## Artifacts

`data/checkpoints/encoder-free-h2-v0/`, `docs/encoder-free-thread-postmortem.md`
