# H03 — A from-scratch pixel embedder + pretrained Qwen learns to read

**Status:** DONE · **Verdict: PARTIAL, superseded** (2026-07-24) · Related: [H04](H04-frozen-decoder-warmup.md), [H05](H05-isolated-ocr-from-scratch.md), [H07](H07-pretrained-vision-adapter-qwen.md)

## Claim

Reusing a pretrained Qwen3-1.7B decoder and teaching it to see through a **from-scratch tiny pixel
embedder** will produce reading across all lanes. (Encoder-free thread, "H1".)

## Setup

`train_encoder_free.py`, `configs/encoder_free.yaml`, W&B `9hb9aike`. Full fine-tune, single LR 3e-4.
Checkpoints `data/checkpoints/encoder-free-v0/`.

## Result

All four lanes read, but **uniformly shallow**:

| lane | Δperm |
|---|---|
| fineweb-edu | +3.15% |
| densefusion | +3.57% |
| smoltalk | +3.43% |
| librispeech | +1.87% |

(Reference: Gemma strong reading ≈ +575%; ignore-floor ≈ +0.1%.)

Key positive: **audio Δblank was POSITIVE (+1.75%)** — the spectrogram beat a blank image, reversing
Gemma's result. Sharing one pixel path made audio ground at all.

## Learning

- Weak, not broken: the pretrained LM prior out-competes the from-scratch vision path for gradient
  when both train at a uniform 3e-4.
- Two probes ruled out the obvious alternatives: rendered text was fully legible at the resolution
  the model sees (OCR gate), and all four lanes did respond to pixels.
- The run died at step 2890 from **host-RAM OOM** when a concurrent GPU probe piled on.
  **Standing rule since: run training SOLO.**
- This shallow-but-real state turned out to be the *best* the from-scratch design ever reached —
  [H04](H04-frozen-decoder-warmup.md) and [H05](H05-isolated-ocr-from-scratch.md) both made it worse.

## Artifacts

`scratchpad/h1_ablation.json`, `data/checkpoints/encoder-free-v0/`,
`docs/encoder-free-thread-postmortem.md`
