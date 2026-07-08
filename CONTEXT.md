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

**Image-Only Lane**:
The experimental baseline lane where rendered text, rendered audio, and natural images are passed as image inputs while native text/audio content is withheld.
_Avoid_: visual baseline unless native modality paths are explicitly disabled

**Native Upper Bound**:
The comparison lane where the same model may use its native text, audio, and image pathways.
_Avoid_: baseline when the distinction from the image-only lane matters

**Retention Metric**:
The image-only lane score expressed as a percentage of the native upper-bound score for the same task.
_Avoid_: accuracy when comparing visual unification against native pathways

**No-History Control**:
A multi-turn evaluation variant that removes prior visual turns to measure whether the model is actually using visual conversation history.
_Avoid_: ablation unless the removed context is specified

**Linguistic Prior Dependence**:
The risk that a model appears to read rendered text while actually reconstructing likely language from priors.
_Avoid_: OCR success when visual evidence has not been isolated

**Public Benchmark First**:
A dataset strategy that starts from established public datasets before synthetic diagnostic data.
_Avoid_: synthetic-first, generated-only dataset

**Dataset of Record**:
The public dataset selected as the primary source for a phase of experiments.
_Avoid_: candidate dataset, benchmark option after selection is resolved

**Visual Bundle**:
The bounded set of images passed to the image-only lane for one example.
_Avoid_: prompt when referring specifically to image inputs

**Conservative Render Settings**:
The first Valor32k rendering profile that favors readability over compression.
_Avoid_: default settings unless the token budgets are specified

## Relationships

- **Text-as-Image**, **Audio-as-Image**, and **Natural Image** are input forms under **Visual Modality Unification**.
- **Audio-as-Image** uses a **Log-Mel Spectrogram Image** in the first proof.
- A **Visual Turn** may contain one or more **Text-as-Image**, **Audio-as-Image**, or **Natural Image** inputs.
- A **Fully Visual Transcript** preserves multi-turn history by rendering prior turns into a **Composite Conversation Image**.
- A **Visual Turn** in the fully visual scope depends on the transcript renderer to include enough prior context for follow-up reasoning.
- The first multi-turn proof uses a two-turn **Compact Transcript Layout**.
- **Oracle Text** exists for rendered text and transcribed audio datasets, but should not be assumed available to the model at inference time.
- `google/gemma-4-E2B` is the current candidate **Baseline Model**.
- The **Image-Only Lane** tests the visual-unification claim; the **Native Upper Bound** estimates the performance lost by forbidding native text/audio paths.
- A **Retention Metric** compares the **Image-Only Lane** to the **Native Upper Bound**.
- A **No-History Control** is required for fully visual multi-turn evaluation.
- **Linguistic Prior Dependence** requires shuffled or low-prior controls beyond the **Retention Metric**.
- The first dataset cut follows **Public Benchmark First** rather than synthetic-first.
- Valor32k-AVQA v2.0 is the **Dataset of Record** for Phases 1-3.
- The first Valor32k **Visual Bundle** contains one text image, one log-mel spectrogram image when audio is needed, and four sampled video frames when vision is needed.
- The first Valor32k **Visual Bundle** uses **Conservative Render Settings**.

## Example Dialogue

> **Dev:** "At turn 2, does the model receive the previous assistant answer as text tokens or as pixels?"
> **Domain expert:** "As pixels. The MVP should test a fully visual transcript, even though that makes token budget and layout design first-order risks."

## Flagged Ambiguities

- "Audio-as-Image" resolved for the first proof: use **Log-Mel Spectrogram Image** rather than waveform plots or learned audio renderings.
- "Single visual modality" resolved for multi-turn: prior user turns, assistant turns, and the current user turn belong in a **Fully Visual Transcript** rather than native text history.
- "Multi-turn" partially resolved: the first proof uses a two-turn **Compact Transcript Layout**, but image/token budget is still unresolved.
- "Baseline" resolved for the first proof: run both an **Image-Only Lane** and a **Native Upper Bound** with Gemma 4 E2B.
- "Dataset strategy" resolved for Phases 1-3: use Valor32k-AVQA v2.0 as the **Dataset of Record** with included test media first.
- "Visual budget" resolved for the first Valor32k run: use a bounded **Visual Bundle** of 1 text image, 1 spectrogram image, and 4 sampled frames where applicable.
- "Render settings" resolved for the first Valor32k run: use **Conservative Render Settings** with a 560-token text image budget and 280-token text ablation.
- "DeepSeek-OCR lesson" resolved: do not treat natural-text OCR success as proof of visual reading without shuffled or low-prior controls.
