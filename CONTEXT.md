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

**Smoke Run**:
The smallest run intended to validate data loading, rendering, prompting, and metric wiring before evaluation.
_Avoid_: pilot when exact sample counts matter

**Central Audio Window**:
The fixed middle segment of a clip used for the first audio-as-image spectrogram.
_Avoid_: full audio when only the bounded middle segment is used

**Deferred Multi-Turn**:
The sequencing decision to postpone fully visual multi-turn datasets until single-turn visual unification is measured.
_Avoid_: abandoned multi-turn, out of scope permanently

**Train-First Sequencing**:
The sequencing decision to fine-tune on the visual-unification training mixture before primary evaluation.
_Avoid_: zero-shot first, prompt-only project

**Phase 0 Training Mixture**:
The first training mixture combining LibriSpeech ASR as spectrogram images, DenseFusion text-image description, FineWeb-Edu text reconstruction, SmolTalk instruction following, and Valor32k examples.
_Avoid_: calibration gate, zero-shot gate

**Text-Compressed Raw Text**:
A training task where FineWeb-Edu raw text is packed into rendered images and the model outputs the same raw text.
_Avoid_: native text input, raw OCR unless discussing the source dataset

**Text-Compressed Instruction Following**:
A training task where a SmolTalk instruction-following prompt is packed into rendered images and the model outputs the response.
_Avoid_: native instruction input when the instruction carries the answer-bearing task content

**Audio Transcription Image**:
A training task where speech is rendered as a log-mel spectrogram image and the model outputs the spoken transcript.
_Avoid_: native audio input, audio tokens

**Text-Image Description Pair**:
A training task where a DenseFusion image/text example is given as image input and the model outputs the paired description.
_Avoid_: image tag, native metadata

**Generic A-D Instruction**:
The fixed native text prompt allowed in the image-only lane that describes only the answer format.
_Avoid_: native question, native choices

**Visual Bundle Order**:
The fixed sequence in which images are passed to the image-only lane.
_Avoid_: arbitrary image order

**Original Gemma E2B Template**:
The official Gemma 4 E2B chat/input template used by the model and processor without custom prompt wrapping.
_Avoid_: custom chat template, hand-rolled prompt format

**Strict First-Letter Parsing**:
An output parser that accepts only A, B, C, or D as the first non-whitespace response character.
_Avoid_: regex-anywhere parsing, judge-based parsing

**Unsloth Gemma 4 Training Path**:
The training implementation path using Unsloth's Gemma 4 multimodal fine-tuning guide, `FastVisionModel`, Gemma 4 chat template, and vision data collator.
_Avoid_: generic Transformers trainer when discussing the agreed first training implementation

**Materialized Render Dataset**:
A preprocessed Hugging Face dataset where rendered images are stored as dataset image objects or relative dataset files in Gemma 4 multimodal `messages` format.
_Avoid_: local absolute image paths, render-at-train-time dataset unless explicitly benchmarking preprocessing

**DeepSeek-OCR-Style Text Packing**:
A page-like text rendering policy that uses dense but readable optical compression while avoiding the high-compression regime where OCR accuracy collapses.
_Avoid_: chat screenshot, tiny unreadable text, native text tokens

**Whisper-Style Log-Mel Rendering**:
An audio rendering policy for speech transcription using mono 16 kHz audio, a 25 ms Hann window, a 10 ms hop, 80 log-mel bins, and Whisper's 80 dB (`max - 8` in log10 power) dynamic-range clipping before image rendering.
_Avoid_: waveform image, arbitrary spectrogram settings for the first ASR run

**Fixed Audio Page**:
A ten-second, 1000 × 160 rectangular log-mel image that preserves a fixed horizontal time scale; one LibriSpeech example contains one to four chronological pages, with only the final page right-padded.
_Avoid_: square spectrogram, duration-dependent stretching, central crop, more than 40 seconds

**Concatenate-And-Shuffle Mixture**:
The first Phase 0 mixing rule: concatenate all selected source train examples and shuffle, without hand-tuned modality ratios beyond the selected per-source subsets.
_Avoid_: balanced sampler, learned curriculum unless later results justify it

**Assistant Output Cap**:
The maximum number of assistant target tokens retained for training or requested during generation; it is distinct from the total multimodal sequence budget.

**Multimodal Sequence Budget**:
The total tokenized length available to image tokens, native task instruction, and assistant target tokens in one training example.

**Vision Length Bucketing**:
A batching strategy that groups examples with similar post-processor multimodal lengths to reduce padding; it does not concatenate independent visual conversations.

**Vision Packing**:
Concatenating multiple visual conversations into one fixed-length training sequence; disabled for the first Gemma 4 run until the Unsloth vision collator proves image-placeholder and loss-mask correctness.

**Visual Decodability Gate**:
A small held-out experiment that tests whether the image-only lane recovers task-relevant information from a rendered modality above non-informative controls before scaling that modality's data or making competitive-performance claims.
_Avoid_: vibe check, full-scale benchmark, competitive retention when only partial signal recovery is required

**Modality-Permutation Control**:
A held-out evaluation condition that deterministically reassigns rendered inputs across examples while leaving targets and non-answer-bearing instructions fixed, testing whether predictions depend on the presented modality content.
_Avoid_: shuffled training, random split, synthetic dataset when only the input-target alignment is changed

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
- Valor32k-AVQA v2.0 is the **Dataset of Record** for tri-modal training/evaluation, with the train split used only for training and validation/test held out for evaluation.
- The first Valor32k **Visual Bundle** contains one text image, one log-mel spectrogram image when audio is needed, and four sampled video frames when vision is needed.
- The first Valor32k **Visual Bundle** uses **Conservative Render Settings**.
- The first Valor32k **Smoke Run** uses 100 examples per modality label; the first evaluation run uses 1,000 examples per modality label.
- The first audio-as-image rendering uses a 10-second **Central Audio Window** with 80 log-mel bins.
- Fully visual multi-turn work is **Deferred Multi-Turn** until Valor32k Phases 1-3 produce interpretable results.
- The first milestone follows **Train-First Sequencing** rather than zero-shot-first evaluation.
- The image-only lane uses a **Generic A-D Instruction** and must not include native question or answer-choice content.
- The first **Visual Bundle Order** is rendered question/options, then spectrogram if present, then video frames in chronological order.
- All Gemma 4 E2B runs must use the **Original Gemma E2B Template**.
- Valor32k multiple-choice outputs use **Strict First-Letter Parsing**.
- **Phase 0 Training Mixture** trains on **Audio Transcription Image**, **Text-Image Description Pair**, **Text-Compressed Raw Text**, **Text-Compressed Instruction Following**, and Valor32k examples.
- **Phase 0 Training Mixture** uses the **Unsloth Gemma 4 Training Path**.
- Preprocessed Phase 0 sources are published as a **Materialized Render Dataset** before training.
- The first local smoke run uses E2B QLoRA/LoRA on the RTX 3060; the larger run targets the A100 40GB using the same materialized dataset.
- The first **Phase 0 Training Mixture** uses a **Concatenate-And-Shuffle Mixture**.
- Text rendering uses **DeepSeek-OCR-Style Text Packing** for both raw text and rendered instruction following.
- Audio rendering uses **Whisper-Style Log-Mel Rendering** for speech transcription and Valor32k spectrograms.
- LibriSpeech **Audio Transcription Image** training uses `clean/train.360`; held-out evaluation uses `clean/validation`.
:- LibriSpeech audio uses one to four **Fixed Audio Pages**, filters source audio longer than 40 seconds, and admits up to four training images under the 8,192-token **Multimodal Sequence Budget** (raised from 2,048 after the Gemma4 image-token expansion fix).
- Each rendered modality must pass a **Visual Decodability Gate** before its dataset is scaled or its performance is framed as competitive with a native pathway.
- A **Visual Decodability Gate** compares correctly aligned held-out inputs against a **Modality-Permutation Control** and reports whether performance depends on the rendered content.

## Example Dialogue

> **Dev:** "At turn 2, does the model receive the previous assistant answer as text tokens or as pixels?"
> **Domain expert:** "As pixels. The MVP should test a fully visual transcript, even though that makes token budget and layout design first-order risks."

## Flagged Ambiguities

- "Audio-as-Image" resolved for the first proof: use **Log-Mel Spectrogram Image** rather than waveform plots or learned audio renderings.
- "Single visual modality" resolved for multi-turn: prior user turns, assistant turns, and the current user turn belong in a **Fully Visual Transcript** rather than native text history.
- "Multi-turn" partially resolved: the first proof uses a two-turn **Compact Transcript Layout**, but image/token budget is still unresolved.
- "Baseline" resolved for the first proof: run both an **Image-Only Lane** and a **Native Upper Bound** with Gemma 4 E2B.
- "Dataset strategy" resolved for Phases 1-3: use Valor32k-AVQA v2.0 as the **Dataset of Record** with train for training only and validation/test held out for evaluation.
- "Visual budget" resolved for the first Valor32k run: use a bounded **Visual Bundle** of 1 text image, 1 spectrogram image, and 4 sampled frames where applicable.
- "Render settings" resolved for the first Valor32k run: use **Conservative Render Settings** with a 560-token text image budget and 280-token text ablation.
- "Subset sizes" resolved for the first Valor32k runs: 100 examples per modality label for smoke, then 1,000 per modality label for first evaluation.
- "Audio duration" resolved for the first Valor32k runs: use a 10-second **Central Audio Window** rather than full-clip spectrograms.
- "Multi-turn timing" resolved: defer OmniInteract until after Valor32k smoke and first evaluation.
- "Training timing" resolved: train first on the **Phase 0 Training Mixture**, then evaluate.
- "Native instruction" resolved: use a fixed **Generic A-D Instruction** in the image-only lane.
- "Image order" resolved: use fixed **Visual Bundle Order** with semantic labels rendered inside images, not native text.
- "Prompt template" resolved: use the **Original Gemma E2B Template** rather than custom prompt wrappers.
- "Output parsing" resolved: use **Strict First-Letter Parsing** and count invalid outputs as wrong.
- "Phase 0" resolved: use the specified **Phase 0 Training Mixture** rather than zero-shot calibration gates.
- "DeepSeek-OCR lesson" resolved: do not treat natural-text OCR success as proof of visual reading without shuffled or low-prior controls.
- "Preprocessing artifact" resolved: push a **Materialized Render Dataset** to Hugging Face rather than relying on local absolute image paths.
- "Mixture rule" resolved: concatenate and shuffle selected Phase 0 train examples without hand-tuned modality ratios in the first run.
- "Text render policy" resolved: use **DeepSeek-OCR-Style Text Packing** with conservative compression for Gemma E2B.
- "Audio render policy" resolved: use **Whisper-Style Log-Mel Rendering** as the first ASR spectrogram convention.
- "LibriSpeech splits" resolved: use `clean/train.360` for training and the 2,703-row `clean/validation` split for held-out evaluation.
- "LibriSpeech page policy" resolved: preserve complete audio in up to four chronological ten-second **Fixed Audio Pages** and filter clips longer than 40 seconds.
