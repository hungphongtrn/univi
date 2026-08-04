# H06 — The connector's expressivity is the bottleneck

**Status:** DONE · **Verdict: REFUTED** (2026-07-26) · Related: [H05](H05-isolated-ocr-from-scratch.md), [H07](H07-pretrained-vision-adapter-qwen.md)

## Claim

[H05](H05-isolated-ocr-from-scratch.md) stalled because a **linear** Fuyu-style connector cannot
express the pixel→embedding map. First rung of the pre-registered capacity escalation
(linear → MLP → conv stem).

## Setup

`configs/h3_randstr_mlp.yaml`. Connector replaced with **Linear→GELU→Linear** (mlp_ratio 4, +30M
params), config-gated via `model.connector` in `train_encoder_free.py`. **Only changed variable**
vs [H05](H05-isolated-ocr-from-scratch.md). Killed early at step ~300 per the pre-registered rule.

## Result

Eval replicated H05's plateau to the decimal (step 100 → 2.2245, step 200 → **2.2217**, step 300 →
2.2241; H05 was 2.231 → 2.224 → 2.225).

Ablation: **identical collapse** — aligned = permuted = blank = 4.031, Δperm −0.000%, Δblank +0.000%,
tok-acc 32.32% aligned == blank bit-identical, reading gain **+0.00 pts**.

## Learning

- +30M parameters of non-linearity changed **nothing**. Linear and MLP land in the identical
  ignore-pixels basin.
- **Rules out connector expressivity.** The bottleneck is the optimization dynamic — the decoder's
  format shortcut wins gradient from step 0 and the vision path is never recruited — which sits
  *upstream* of the embedder's function class.
- A conv stem (the next rung) would fail the same way. **Capacity escalation is exhausted as a cheap
  fix**; the escalation ladder was abandoned here.
- This closed the encoder-free thread. The resolution came from the opposite direction: swap in a
  *pretrained* vision front-end ([H07](H07-pretrained-vision-adapter-qwen.md)).

## Artifacts

`data/eval/h3-ablation-h3-mlp.json`, `data/checkpoints/h3-randstr-mlp-v0/`,
`docs/encoder-free-thread-postmortem.md`
