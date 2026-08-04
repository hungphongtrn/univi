# H07 — A pretrained vision front-end makes a reused Qwen decoder read

**Status:** DONE · **Verdict: VALIDATED** (2026-07-26) · Related: [H06](H06-mlp-connector-escalation.md), [H08](H08-audio-needs-trainable-vision.md)

## Claim

The from-scratch vision path — not the reused-decoder idea — was the bottleneck
([H03](H03-from-scratch-embedder-shallow-reading.md)–[H06](H06-mlp-connector-escalation.md)).
A **pretrained** vision front-end bridged to a *different* pretrained decoder will read.

## Setup

```
Gemma-4-12B unified vision embedder (PRETRAINED, 3840-d, 280 soft tokens)
  -> Linear(3840 -> 2048)  (from-scratch adapter)
  -> Qwen3-1.7B            (PRETRAINED)
```

3-group LR (adapter 3e-4 > vision 5e-5 > decoder 2e-5), no freeze. Prior-proof random-string OCR
lane, 600 steps, eff batch 64, SOLO. `univi/hybrid/pretrained.py`,
`configs/hybrid_pretrained_randstr.yaml`.

Weights trick: the 12B repo is a single 24GB safetensors. Only the **10 vision tensors (~100MB)**
were range-downloaded via curl byte-ranges (`scratchpad/fetch_gemma12b_vision.py`), loading 10/10
clean.

## Result

| checkpoint | Δperm | Δblank | reading gain |
|---|---|---|---|
| from-scratch ([H05](H05-isolated-ocr-from-scratch.md)/[H06](H06-mlp-connector-escalation.md)) | ~0% | ~0% | +0.00 pts |
| @200 | +27.4% | +39.5% | +4.6 pts |
| **final** | **+114.7%** | **+90.8%** | **+22.2 pts** |

Aligned CE 3.69; **permuted 7.92 is worse than blank 7.03** → genuine content-conditioning. Eval
descended 5.06 → 3.735, still descending when the cosine LR decayed out. grad_norm healthy ~4–5.

## Learning

- **Thesis supported**: a reused decoder does read pixels through a pretrained vision front-end.
- **Frozen transfer**: the vision embedder barely moved (rel-delta ~0 across patch_dense/pos/LN;
  only Gemma's final projector +4% and the from-scratch adapter changed). Pretrained Gemma vision
  *features* are reused essentially frozen, and a from-scratch **linear** adapter suffices to make
  them readable by a distinct decoder — LLaVA-style cross-decoder transfer.
- This is exactly what the from-scratch embedder lacked: **pretrained features to transfer**.
- **grad_norm dip-then-rise is the recruitment signature** (0.95 → 3.4/4.8/4.7). Its absence in
  [H09](H09-balanced-mixture-prevents-drowning.md) was the first sign of trouble there.
- Caveat: **partial** reading (tok-acc 32%, CE 3.69) — a bounded run whose LR ran out mid-descent.
- Architecture note: Gemma-4 **E2B** vision is a 16-layer ViT; **12B** vision is the unified
  patchify+linear embedder (3840-d, patch 48, 280 soft tokens). Different architectures — this used
  the 12B unified one.

## Artifacts

`data/eval/hybrid-pretrained-ablation-hybrid-pre-{200,final}.json`,
`data/checkpoints/hybrid-pretrained-randstr-v0/`
