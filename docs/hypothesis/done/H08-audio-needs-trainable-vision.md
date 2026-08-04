# H08 — Audio reads only if the vision tower is trainable

**Status:** DONE · **Verdict: VALIDATED** (2026-07-26) · Related: [H07](H07-pretrained-vision-adapter-qwen.md), [H20](../todo/H20-audio-phoneme-resolution.md)

## Claim

[H07](H07-pretrained-vision-adapter-qwen.md) showed rendered text reads through frozen pretrained
Gemma vision features. Spectrograms are **out-of-distribution** for that tower, so audio will need
the vision weights to adapt.

## Setup

Built a prior-proof **spoken-digit** lane: random digit sequences from FSDD real speech →
production log-mel spectrogram → transcribe. Uniform i.i.d. digits ⇒ zero language-prior shortcut.
`data/preprocessing/spoken_digits.py` → `data/materialized/spoken-digits-v0/` (30k/1k, floor 1.228
nats/tok). Added a `freeze_vision` knob to `train_pretrained.py` / `pretrained.py`.

Two runs, identical but for `freeze_vision`.

## Result

**FROZEN vision** (`configs/hybrid_pretrained_spokendigits.yaml`) — **no reading.** Eval flat at
~1.08 (≈ the 1.228 floor), grad_norm small.

**TRAINABLE vision** (`..._trainvis.yaml`, lr_vision 1e-4) — **reads.** Eval descended 1.084 →
**0.726** (below floor), grad_norm rising ~2.4.

| checkpoint | Δperm | Δblank | reading gain | tok-acc |
|---|---|---|---|---|
| @200 | +7.4% | — | — | — |
| **final** | **+134.8%** | **+129.9%** | **+17.6 pts** | 75% vs 57% blank |

Permuted (1.67) worse than blank (1.63) → content-conditioning.

## Learning

- **Mechanistic split**: text/image reading transfers **frozen** (in-distribution for Gemma vision);
  **audio requires adapting the tower** (OOD). Frozen features simply do not encode spectrograms and
  a linear adapter cannot recover it.
- The reused-Qwen + pretrained-Gemma-vision path reads text, images, **and** audio — all prior-proof.
- Caveat: partial (tok-acc 75%, CE 0.71, still descending at the 600-step LR-out). The frozen-audio
  "no reading" conclusion rests on flat eval, not a separate ablation.
- Spoken **digits** resolved fine at 244 ms/token-column; continuous speech did not
  ([H09](H09-balanced-mixture-prevents-drowning.md), [H20](../todo/H20-audio-phoneme-resolution.md)). That
  contrast is the resolution story.

## Artifacts

`data/eval/hybrid-pretrained-ablation-spdigit-{200,final}.json`,
`data/checkpoints/hybrid-pretrained-spokendigits-trainvis-v0/`
