# Spectrogram patch models classify but do not transcribe

**Confidence:** Established · **Used by:** [H20](../hypothesis/todo/H20-audio-phoneme-resolution.md), [H17](../hypothesis/todo/H17-raise-soft-token-budget.md)

## Claim

Vision-style patch models over spectrograms are demonstrated for **classification** (audio tagging,
keyword spotting) at coarse time resolution. **Transcription** systems use far finer time resolution,
and no published vision-path patch model transcribes continuous speech.

## Evidence

- **AST — Audio Spectrogram Transformer** (Gong et al., INTERSPEECH 2021): ViT-style patches over
  spectrograms, ~160 ms effective patches. Classification only.
- **Whisper** (Radford et al., 2022): the working ASR reference encodes at **20 ms per token** — an
  order of magnitude finer.
- Phoneme durations are typically **50–150 ms**.
- A survey for this project (2026-07-28) found **no counterexample**: no vision-patch model
  transcribing continuous speech at AST-like resolution.

## Why it matters for univi

Our current render gives **244 ms per token-column** at 280 soft tokens — coarser than AST, and 12×
coarser than Whisper. That predicts precisely the split we observe:

- **spoken digits** (prior-proof, ~1–2 token-columns per digit-word) read at **Δperm +134.8%**
  ([H08](../hypothesis/done/H08-audio-needs-trainable-vision.md)) — a *classification-like* task at
  this resolution.
- **continuous librispeech** reads at +12.9% but with **zero phonetic correspondence**; the signal is
  utterance duration, and position-0 reading gain is **−12.7 pts**
  ([H09](../hypothesis/done/H09-balanced-mixture-prevents-drowning.md),
  [H10](../hypothesis/done/H10-reading-concentrated-at-start.md)).

At 1120 soft tokens the same render gives ~120 ms/column, which is inside the phoneme range —
the basis for [H20](../hypothesis/todo/H20-audio-phoneme-resolution.md).

## Caveat

Note this is a claim about **time resolution per token**, not about render fidelity — the latter is
separately and decisively refuted ([H02](../hypothesis/done/H02-render-destroys-audio-signal.md): 0%
WER loss through the PNG round-trip). The pixels are faithful; there are just too few tokens across
the time axis.
