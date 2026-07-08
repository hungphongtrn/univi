# Univi Context

Univi investigates whether text, audio, and ordinary images can be normalized into visual inputs for a single multimodal model training path.

## Language

**Visual Modality Unification**:
A training strategy where every user-side input modality is presented to the model through an image-like representation.
_Avoid_: omni model, universal model, all-in-one model

**Text-as-Image**:
Textual input rendered into pixels before it is given to the model.
_Avoid_: OCR data, screenshot text, optical text unless the rendering/compression mechanism matters

**Audio-as-Image**:
Audio input converted into a visual representation before it is given to the model.
_Avoid_: audio tokens, waveform unless the exact representation is waveform-based

**Natural Image**:
An ordinary visual input that is already an image before preprocessing.
_Avoid_: raw image, normal image

**Visual Turn**:
One conversational turn represented as one or more visual inputs plus the target assistant response.
_Avoid_: message, sample, example when the turn structure matters

**Composite Conversation Image**:
A candidate multi-turn representation where previous turns and the current turn are arranged into one image.
_Avoid_: chat screenshot unless the layout intentionally mimics a chat UI

**Oracle Text**:
The ground-truth text known during dataset creation but not directly available to the model when it only receives rendered pixels.
_Avoid_: label leakage, transcript unless discussing audio data specifically

**Baseline Model**:
The model used to test whether visual unification is competitive before custom architecture work.
_Avoid_: foundation model, backbone unless architecture is the focus

## Relationships

- **Text-as-Image**, **Audio-as-Image**, and **Natural Image** are input forms under **Visual Modality Unification**.
- A **Visual Turn** may contain one or more **Text-as-Image**, **Audio-as-Image**, or **Natural Image** inputs.
- A **Composite Conversation Image** is one candidate way to preserve multi-turn history across **Visual Turns**.
- **Oracle Text** exists for rendered text and transcribed audio datasets, but should not be assumed available to the model at inference time.
- `google/gemma-4-E2B` is the current candidate **Baseline Model**.

## Example Dialogue

> **Dev:** "At turn 2, does the model receive the previous assistant answer as text tokens or as pixels?"
> **Domain expert:** "Unresolved. If all conversation history must be visual, previous assistant answers may need to be rendered into a composite visual context. If only user inputs are visual, Gemma's native text history can carry assistant responses."

## Flagged Ambiguities

- "Single visual modality" is unresolved: it could mean all user inputs are visual, or both user inputs and assistant outputs are rendered visually for future turns.
- "Audio-as-Image" is unresolved: candidates include mel-spectrograms, log-mel spectrograms, waveform plots, or learned audio image/token renderings.
- "Multi-turn" is unresolved: candidates include a single composite conversation image, multiple interleaved images, native text chat history plus visual current input, or recurrent/KV-cache reuse.
- "Baseline" is unresolved: Gemma 4 E2B has native text, image, and audio support, so the experiment must define whether it is used as-is, constrained to image inputs, or fine-tuned against visualized inputs.
