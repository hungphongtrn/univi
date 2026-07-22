# Research Plan

## Brief

Train or fine-tune a model that consumes text, audio, and images through a single visual input pathway: text is rendered as an image, audio is rendered as an image, and images are used directly. Use Gemma 4 E2B as the initial baseline candidate.

## Current Hypothesis Draft

A Gemma-4-E2B-class vision-language model can learn useful reasoning over text, audio, and natural images when text and audio are transformed into visual representations, but the approach will fail unless the visual representation preserves enough detail and the multi-turn context format avoids unbounded image growth.

This is not final. The first grilling target is to make the hypothesis falsifiable.

## Core Questions

1. What visual token budget can support a two-turn compact fully visual transcript?
2. What does the model see at turn 2 when the answer depends on turn 1?
3. Is the goal competitive performance, a compression experiment, a unified data pipeline, or architectural simplicity?
4. Which public datasets should instantiate the first text, audio, image, and multi-turn tasks?
5. What audio duration, mel-bin count, and image dimensions fit the first token budget?

## Minimum Viable Experiment

The first milestone is a LoRA fine-tuning run on the Phase 0 image-input training mixture. Evaluation follows training.

### Phase 0: Training Data Mixture

Phase 0 trains Gemma 4 E2B with all task inputs presented as images. A small native text system prompt and instruction are allowed, but native text/audio content that carries the answer should not be provided as input.

1. **ASR from LibriSpeech**: render `openslr/librispeech_asr` audio as log-mel spectrogram images and train the model to output the spoken transcript.
2. **Text-image description**: use a `BAAI/DenseFusion-1M` subset where the image is the answer-bearing input and the target is the paired description.
3. **Text-compressed raw text**: pack raw text into rendered images and train the model to output the raw text.
4. **Text-compressed instruction following**: pack `HuggingFaceTB/smoltalk` instruction-following prompts into rendered images and train the model to output the response.
5. **Valor32k-AVQA v2.0**: use the official Valor train split only for training; present rendered question/options, spectrograms, and sampled video frames as images and train the model to answer A/B/C/D.

Training follows the Unsloth Gemma 4 multimodal fine-tuning guide at `https://unsloth.ai/docs/models/gemma-4/train`.

- Loader: use `FastVisionModel.from_pretrained` for Gemma 4 multimodal fine-tuning.
- Base model: `unsloth/gemma-4-E2B-it` unless a later decision changes the baseline.
- Template: use the original non-thinking `gemma-4` template through Unsloth `get_chat_template`; do not hand-roll chat formatting.
- Reference notebook: adapt `https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Vision.ipynb#scrollTo=QmUBVEnvCDJv`, switching the base model to `unsloth/gemma-4-E2B-it`.
- Data shape: Unsloth multimodal `messages` examples with image content before short text instruction content.
- Collator: use `UnslothVisionDataCollator` with TRL `SFTTrainer`.
- Loss masking: train on assistant responses only; do not train loss on the rendered-input instructions.
- LoRA: start with QLoRA/LoRA and do not freeze the visual path for the first smoke; train adapters over vision, language, attention, and MLP layers unless a measured memory issue forces a smaller target set.
- Model surgery: first establish a working baseline without stripping Gemma 4 components; removing `audio_tower` is an optimization experiment only after the unmodified `FastVisionModel` path loads, trains, saves, and reloads correctly.
- Failure signal: training diverges, one source dominates the mixture, rendered inputs become unreadable under the chosen token budget, or held-out evaluation does not improve over the base model.

### Execution Profiles

- Local smoke: run Gemma 4 E2B on the RTX 3060 with QLoRA/LoRA, batch size 1, short max length, and small max steps to validate dataset loading, image collation, loss masking, and checkpoint save/load.
- Full run: use the A100 40GB when available with the same materialized dataset, longer schedule, longer context if needed, higher LoRA rank if useful, and more complete evaluation.
- Preprocessing: materialize all rendered examples as Hugging Face datasets in Gemma 4 multimodal `messages` format before training.
- Split discipline: preserve each dataset's dedicated train/validation/test split during preprocessing; explore exact split mappings during implementation.

### Phase 1: Text-As-Image Evaluation

- Evaluate held-out rendered text examples.
- Compare the trained image-only lane against the base model and native-text upper bound.
- Success signal: image-only accuracy reaches at least 80% of native-text upper-bound accuracy.
- Failure signal: OCR or reasoning collapses under shuffled, random, or dense text where linguistic priors cannot fill gaps.

### Phase 2: Audio-As-Image Evaluation

- Evaluate held-out log-mel spectrogram transcription and audio-question examples.
- Ask transcript or content questions from the rendered audio image.
- Compare the image-only lane against Gemma 4 E2B native audio and an ASR plus LLM pipeline if available.
- Success signal: image-only accuracy reaches at least 50% of native-audio or ASR plus LLM upper-bound accuracy on short clean speech.
- Failure signal: the visual audio path cannot recover speech content above a weak baseline.

### Phase 3: Mixed Visual Inputs Evaluation

- Evaluate mixed text-as-image, audio-as-image, natural-image, and Valor32k examples.
- Measure whether unified training degrades any single source type.
- Failure signal: one representation dominates training or hurts the others.

### Phase 4: Fully Visual Multi-Turn

- Construct two-turn examples where turn 2 requires turn 1 context.
- Render prior user turns, assistant turns, and the current user turn into a compact document-like visual transcript.
- Defer chat-style screenshots until the compact layout works, so early failures are less likely to be caused by wasted pixels.
- Success signal: image-only accuracy beats a no-history visual control by at least 20 percentage points.
- Failure signal: context length grows faster than the model can process, or turn-2 accuracy drops to near single-turn/no-context baselines.
- Timing: defer Phase 4 implementation until Valor32k Phases 1-3 produce interpretable smoke and first-eval results.

## Success Metrics

- Phase 0 training: loss should trend down without divergence; Gemma 4 E2B/E4B multimodal losses around 13-15 may be normal per Unsloth guidance.
- Phase 0 held-out checks: each source type must improve over the base model or justify why it is retained in the mixture.
- Text-as-image: image-only lane reaches at least 80% retention against the native-text upper bound.
- Audio-as-image: image-only lane reaches at least 50% retention against native-audio or ASR plus LLM upper bound on short clean speech.
- Fully visual multi-turn: compact two-turn transcript beats a no-history visual control by at least 20 percentage points.
- Mixed visual inputs: no single modality drops by more than 10 percentage points compared with its single-modality image-only run.
- Output parsing: accept only A, B, C, or D as the first non-whitespace character after punctuation normalization.
- Invalid outputs: score as wrong and report invalid rate separately.

## Dataset Strategy

- Phase 0 training uses ASR, text-image description, text-compressed raw text, text-compressed instruction following, and Valor32k.
- ASR source: `openslr/librispeech_asr`, rendered as log-mel spectrogram images.
- Text-image description source: `BAAI/DenseFusion-1M` subset.
- Text-compressed raw text source: `HuggingFaceFW/fineweb-edu` config `sample-10BT` packed into images with raw text as the target.
- Text-compressed instruction-following source: `HuggingFaceTB/smoltalk` rendered instruction images with response text as the target.
- Valor32k-AVQA v2.0 remains the tri-modal QA source because it has video, audio, text QA, and per-question modality labels; use train for training only and reserve validation/test for evaluation.
- Current fallback: JointAVBench because it has released clips, Apache 2.0 repository license, and questions designed to require joint audio-visual reasoning.
- Current small evaluation candidate: Daily-Omni for temporal audio-visual alignment.
- Current multi-turn candidate: OmniInteract, deferred until the compact fully visual transcript pipeline is stable.
- See `docs/DATASETS.md` for dataset research notes.

## Feasibility Risks

- Text readability competes with image token budget.
- DeepSeek-OCR-style optical compression may rely on linguistic priors, so evaluations need adversarial or low-prior text.
- Spectrograms are image-like but not natural images; a generic VLM may need targeted adaptation.
- Gemma 4 E2B has native audio and text paths, so the baseline must be constrained or the experiment will not isolate visual unification.
- Stripping unused Gemma 4 components such as `audio_tower` can break Unsloth patches, processor assumptions, PEFT target discovery, checkpoint loading, or save/reload compatibility; measure this only after the unmodified path works.
- Fully visual multi-turn context requires repeatedly re-encoding conversation history instead of using cheap text KV-cache behavior.

## DeepSeek-OCR Lessons

- Avoid aggressive text compression in the first proof; DeepSeek-OCR-style results can overstate genuine reading when language priors fill missing visual evidence.
- Treat text rendering density, font size, and spatial layout as experiment hyperparameters.
- Add shuffled-answer or low-prior controls for text-as-image evaluation.
- Use a vertical linear layout for compact transcripts so raster-scan reading order matches conversational causality.
- Do not assume Gemma 4 E2B has DeepSeek-OCR's document-specialized encoder; success on Gemma is a stronger but riskier test of generic VLM visual reading.

## Text-Reading Controls

- Shuffled-options control: randomly permute answer choices and measure accuracy separately.
- Low-prior diagnostic: replace answer choices with random strings on a small diagnostic subset to estimate genuine visual character reading.
- Failure signal: if shuffled-options accuracy drops more than 10 percentage points below natural accuracy at the same token budget, flag likely **Linguistic Prior Dependence**.

## Baseline Protocol

- **Image-only lane**: pass rendered text, rendered audio, natural images, and fully visual transcripts as images; withhold native text/audio equivalents except fixed task instructions needed to query the model.
- **Native upper bound**: run the same examples through Gemma 4 E2B's native text, audio, and image pathways to estimate the cost of visual unification.
- Report both lanes together; do not claim visual unification works from native-lane results.
- Fixed image-only native instruction: `Answer the multiple-choice question shown in the images. Reply with only A, B, C, or D.`
- The fixed instruction must not include the actual question, answer choices, transcript, audio transcript, or modality-specific content.
- All Gemma 4 E2B calls must use the original Gemma E2B model/processor chat template; do not introduce a custom chat template or prompt wrapper.

For training examples, the instruction may vary by task but must stay short and must not include the answer-bearing native content. Example instructions: `Transcribe the text shown in the image.`, `Transcribe the speech represented by this spectrogram image.`, `Describe this image.`, and `Answer the multiple-choice question shown in the images. Reply with only A, B, C, or D.`

## First Valor32k Visual Bundle

- Text image: one rendered question/options image for every example.
- Spectrogram image: one log-mel spectrogram image for `audio` and `audio-visual` examples.
- Video frames: four sampled frames for `visual` and `audio-visual` examples.
- Omit unused modalities by modality label in the image-only lane to keep the first prompt bounded.
- Image order: rendered question/options first, spectrogram second if present, then video frames in chronological order.
- Semantic labels such as `Question`, `Audio spectrogram`, and `Frame 1` should be rendered inside the relevant images rather than supplied as native text.

## Conservative Render Settings

- Text image: 1024px-wide canvas, variable height, white background, black text, left-aligned question and A/B/C/D answer choices.
- Text font: legible sans-serif or monospace, at least 14pt, with the question preferably at 16pt.
- Text image token budget: 560 tokens for the main run, with a 280-token ablation in the smoke run.
- Text packing policy: use DeepSeek-OCR-style page packing, but keep compression conservative for Gemma E2B; avoid the tiny/high-compression regime in the first run.
- Audio image: one Whisper-style log-mel spectrogram image at a 280-token budget.
- Audio window: central 10 seconds of the clip for smoke and first eval.
- Spectrogram shape: mono 16 kHz audio, 25 ms Hann window, 10 ms hop, 80 log-mel bins over the central window.
- Video frames: four frames sampled uniformly over the clip, each at a 140-token budget.
- Compact transcript layout: vertical linear order with role labels and separators; no chat bubbles in the first proof.

## First Valor32k Subsets

- Smoke run: 100 `visual`, 100 `audio`, and 100 `audio-visual` examples from the Valor32k validation split.
- First evaluation run: 1,000 `visual`, 1,000 `audio`, and 1,000 `audio-visual` examples from held-out Valor32k validation/test splits.
- Scale-up: remaining held-out Valor32k examples after smoke and first eval pass without rendering, token-budget, or metric failures.

## Related Work Pointers

- DeepSeek-OCR and DeepSeek-OCR 2: optical context compression, variable visual token budgets, and visual causal flow.
- Critiques of DeepSeek-OCR: tests suggesting heavy reliance on linguistic priors under compression.
- ImageBind: aligns multiple modalities through a shared embedding space using image-paired data rather than rendering everything into images.
- Unified-IO 2: unifies text, image, audio, action, and structures through tokenization and a shared transformer; uses spectrogram/audio tokenization rather than generic image rendering only.
- AudioCLIP, CLAP, Whisper, Audio Spectrogram Transformer, EnCodec, SoundStream, and WavTokenizer: prior art for audio representations and audio-language alignment.
- Pix2Struct, Donut, Nougat, LayoutLM-family, DocVQA, and TextVQA: prior art around document/text understanding from pixels.

## First Decision

The project will pursue a fully visual transcript for multi-turn experiments: prior user turns, assistant turns, and the current user turn are rendered as pixels rather than preserved as native text history. This makes layout, compression, and visual token budget primary research constraints.

## Next Decision Needed

Choose the DenseFusion subset, exact subset sizes, LoRA hyperparameters, and whether `audio_tower` stripping is worth benchmarking after the unmodified smoke run.
