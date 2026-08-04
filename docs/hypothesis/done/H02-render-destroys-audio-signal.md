# H02 — Spectrogram rendering destroys the audio signal

**Status:** DONE · **Verdict: REFUTED** (2026-07-21) · Related: [H01](H01-gemma-e2b-mixture-audio-drowns.md)

## Claim

Audio fails to ground ([H01](H01-gemma-e2b-mixture-audio-drowns.md)) because the log-mel →
PNG render loses information a model would need to transcribe.

## Setup

Model-free **Visual Decodability Gate** (`data/eval/decodability/wer_direct.py`): 30 real
LibriSpeech test-clean clips fed to Whisper tiny.en three ways —

1. **NATIVE** — raw audio
2. **CONTROL** — librosa mel computed directly
3. **GATE** — mel recovered from the rendered PNG

## Result

**All three: WER 3.75%**, identical transcripts including identical errors. The PNG round-trip
loses **0%** for a strong reader.

## Learning

- Render fidelity is hard-falsified as an explanation. Do not revisit it.
- The production render is the paged path (`page_duration_sec=10`, `output_width=1000` → 1000 mel
  frames = 10 ms/px, no time-warp; 80→160 vertical stretch only).
- An earlier "GATE FAIL" result was a Griffin-Lim artifact, not a render defect.
- Combined with [H01](H01-gemma-e2b-mixture-audio-drowns.md): the pixels are readable and the model
  would not learn to read them ⇒ **learnability-limited**, which reopened curriculum/reweighting as
  the fix.

Note this refutes *render fidelity*, not *render resolution* — time resolution per soft token is a
separate live constraint, see [H20](../todo/H20-audio-phoneme-resolution.md).

## Artifacts

`data/eval/decodability/wer_gate.json`
