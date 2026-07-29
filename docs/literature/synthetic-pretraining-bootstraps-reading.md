# Synthetic prior-proof pretraining bootstraps reading

**Confidence:** Established · **Used by:** [H13](../hypothesis/done/H13-density-ladder-long-targets.md), [H18](../hypothesis/todo/H18-no-learned-scan.md), [H19](../hypothesis/todo/H19-four-lane-rematch.md)

## Claim

Text-reading ability in vision models is reliably bootstrapped by a **synthetic, reading-first
stage** — rendered text with known ground truth, often at line or word granularity — before any
real-document or downstream training. Models trained this way transfer to real documents.

## Evidence

- **Pix2Struct** (Lee et al., ICML 2023, arXiv:2210.03347) — screenshot-parsing warmup; **+11.6 ANLS
  on DocVQA** attributable to the reading-first stage. Direct precedent for a curriculum.
- **Donut / SynthDoG** (ECCV 2022) — OCR-free document understanding trained on synthetic documents.
- **TrOCR** (arXiv:2109.10282) — **684M synthetic text lines** as the pretraining corpus.
- **MJSynth / SynthText** (Jaderberg 2014; Gupta et al., CVPR 2016) — synthetic-only training
  transfers to real scene text.

## Why it matters for univi

This validates the shape of our prior-proof lanes — randstr and spoken-digits are exactly this kind
of synthetic reading-first data, and they are the **only** lanes where our architecture has ever read
strongly ([H07](../hypothesis/done/H07-pretrained-vision-adapter-qwen.md),
[H08](../hypothesis/done/H08-audio-needs-trainable-vision.md)).

Two specific transfers:

1. **Keep them as anchors, don't discard them.**
   [H05](../hypothesis/done/H05-isolated-ocr-from-scratch.md) showed that removing prior-proof
   pressure lets grounding decay to zero. [H19](../hypothesis/todo/H19-four-lane-rematch.md) retains
   them at 15%.
2. **Line/word granularity is the norm.** TrOCR and the scene-text corpora train at *line* level, not
   full-page. That is suggestive for [H18](../hypothesis/todo/H18-no-learned-scan.md): a chunked
   curriculum where image extent and answer extent are aligned may be how the scan is learned at all.

## Caveat

These systems pair synthetic pretraining with encoders built for text (high or variable resolution).
None of them demonstrates that synthetic data alone overcomes a fixed low token budget — so this
claim supports our *curriculum*, not our *budget*. See
[full-page OCR token budgets](full-page-ocr-token-budget.md).
