# H01 — A full-Gemma mixture reads all four lanes

**Status:** DONE · **Verdict: PARTIAL** (2026-07-21) · Related: [H02](H02-render-destroys-audio-signal.md), [H09](H09-balanced-mixture-prevents-drowning.md)

## Claim

Gemma-4-E2B (4-bit QLoRA) trained on the Phase-0 mixture will learn to read all four rendered
lanes: text, natural images, rendered instructions, and spectrograms.

## Setup

`configs/3060_full.yaml`, W&B `k3ceh56s`, ~19,600 steps. Best checkpoint
`data/checkpoints/full-v0/checkpoint-2800` (macro 1.361). Grounding ablation on 120 samples,
`data/eval/val-ablation/ablation.json`.

## Result

| lane | rel Δperm | note |
|---|---|---|
| fineweb-edu | **+575%** | reads strongly (aligned CE 0.457 vs blank 2.882) |
| densefusion | +63% | reads |
| smoltalk | +39% | reads |
| librispeech | **+0.16%** | **ignore-floor**, and Δblank *negative* |

Re-ablated at checkpoint-19600 after 7× more audio exposure: librispeech went 0.16% → 0.07%.
More audio did not help.

## Learning

- Grounding is **modality-specific**, not a global prior-exploitation problem. Three lanes read
  pixels strongly; only audio failed.
- A *negative* Δblank means a blank image beat the real spectrogram — the model learned to treat
  the spectrogram as an active distractor and lean on the text prior.
- librispeech was only **3.52%** of the mixture (104k/2.96M rows). This motivated
  [H09](H09-balanced-mixture-prevents-drowning.md).
- Untrained base Gemma already read fineweb at CE 0.905 — vision-side pretraining carries real
  OCR ability before any of our training.

## Artifacts

`data/eval/val-ablation/ablation.json`, `data/eval/val-ablation/ablation-ckpt19600.json`
