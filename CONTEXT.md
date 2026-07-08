# Univi Context

Univi investigates whether text, audio, and ordinary images can be normalized into visual inputs for a single multimodal model training path.

## Language

**Visual Modality Unification**:
A training strategy where every conversational input and prior conversational state is presented to the model through image-like representations.
_Avoid_: omni model, universal model, all-in-one model

**Text-as-Image**:
Textual input rendered into pixels before it is given to the model.
_Avoid_: OCR data, screenshot text, optical text unless the rendering/compression mechanism matters

**Audio-as-Image**:
Audio input converted into a log-mel spectrogram image before it is given to the model.
_Avoid_: audio tokens, waveform unless the exact representation is waveform-based

**Log-Mel Spectrogram Image**:
A 2D time-frequency rendering of audio using mel-scaled frequency bins and log-scaled magnitude.
_Avoid_: waveform image, raw spectrogram when mel/log scaling matters

**Natural Image**:
An ordinary visual input that is already an image before preprocessing.
_Avoid_: raw image, normal image

**Visual Turn**:
One conversational turn represented as one or more visual inputs plus the target assistant response.
_Avoid_: message, sample, example when the turn structure matters

**Composite Conversation Image**:
A multi-turn representation where previous user turns, assistant turns, and the current user turn are arranged into one visual transcript.
_Avoid_: chat screenshot unless the layout intentionally mimics a chat UI

**Fully Visual Transcript**:
A multi-turn conversation format where the model sees conversation history as pixels rather than native text tokens.
_Avoid_: visual inputs plus text history, hybrid transcript

**Compact Transcript Layout**:
A dense document-like rendering of a fully visual transcript with role labels and minimal whitespace.
_Avoid_: chat UI, chat bubbles, screenshot layout

**Oracle Text**:
The ground-truth text known during dataset creation but not directly available to the model when it only receives rendered pixels.
_Avoid_: label leakage, transcript unless discussing audio data specifically

**Baseline Model**:
The model used to test whether visual unification is competitive before custom architecture work.
_Avoid_: foundation model, backbone unless architecture is the focus

## Relationships

- **Text-as-Image**, **Audio-as-Image**, and **Natural Image** are input forms under **Visual Modality Unification**.
- **Audio-as-Image** uses a **Log-Mel Spectrogram Image** in the first proof.
- A **Visual Turn** may contain one or more **Text-as-Image**, **Audio-as-Image**, or **Natural Image** inputs.
- A **Fully Visual Transcript** preserves multi-turn history by rendering prior turns into a **Composite Conversation Image**.
- A **Visual Turn** in the fully visual scope depends on the transcript renderer to include enough prior context for follow-up reasoning.
- The first multi-turn proof uses a two-turn **Compact Transcript Layout**.
- **Oracle Text** exists for rendered text and transcribed audio datasets, but should not be assumed available to the model at inference time.
- `google/gemma-4-E2B` is the current candidate **Baseline Model**.

## Example Dialogue

> **Dev:** "At turn 2, does the model receive the previous assistant answer as text tokens or as pixels?"
> **Domain expert:** "As pixels. The MVP should test a fully visual transcript, even though that makes token budget and layout design first-order risks."

## Flagged Ambiguities

- "Audio-as-Image" resolved for the first proof: use **Log-Mel Spectrogram Image** rather than waveform plots or learned audio renderings.
- "Single visual modality" resolved for multi-turn: prior user turns, assistant turns, and the current user turn belong in a **Fully Visual Transcript** rather than native text history.
- "Multi-turn" partially resolved: the first proof uses a two-turn **Compact Transcript Layout**, but image/token budget is still unresolved.
- "Baseline" is unresolved: Gemma 4 E2B has native text, image, and audio support, so the experiment must define whether it is used as-is, constrained to image inputs, or fine-tuned against visualized inputs.
