# Research Plan

## Brief

Train or fine-tune a model that consumes text, audio, and images through a single visual input pathway: text is rendered as an image, audio is rendered as an image, and images are used directly. Use Gemma 4 E2B as the initial baseline candidate.

## Current Hypothesis Draft

A Gemma-4-E2B-class vision-language model can learn useful reasoning over text, audio, and natural images when text and audio are transformed into visual representations, but the approach will fail unless the visual representation preserves enough detail and the multi-turn context format avoids unbounded image growth.

This is not final. The first grilling target is to make the hypothesis falsifiable.

## Core Questions

1. What exactly is visualized: only user inputs, or both user inputs and assistant outputs for future turns?
2. What does the model see at turn 2 when the answer depends on turn 1?
3. Is the goal competitive performance, a compression experiment, a unified data pipeline, or architectural simplicity?
4. What visual representation of audio is in scope for the first proof-of-concept?
5. How will the baseline be constrained so Gemma 4 E2B does not use native audio/text pathways when the experiment claims visual unification?

## Minimum Viable Experiment

### Phase 1: Text-as-Image Single-Turn

- Render QA prompts into images.
- Fine-tune or prompt the baseline to answer from rendered text only.
- Compare against the same prompts passed as native text.
- Failure signal: OCR or reasoning collapses under shuffled, random, or dense text where linguistic priors cannot fill gaps.

### Phase 2: Audio-as-Image Single-Turn

- Convert short speech clips to spectrogram-like images.
- Ask transcript or content questions from the rendered audio image.
- Compare against a native ASR plus LLM pipeline and Gemma 4 E2B native audio if available.
- Failure signal: the visual audio path cannot recover speech content above a weak baseline.

### Phase 3: Mixed Visual Inputs

- Mix text-as-image, audio-as-image, and natural-image examples.
- Measure whether unified training degrades any single modality.
- Failure signal: one representation dominates training or hurts the others.

### Phase 4: Multi-Turn

- Construct two-turn examples where turn 2 requires turn 1 context.
- Compare composite conversation images, interleaved image inputs, and native text-history variants.
- Failure signal: context length grows faster than the model can process, or turn-2 accuracy drops to near single-turn/no-context baselines.

## Dataset Notes

- Text-as-image can start from small QA sets rendered with controlled fonts, density, and layouts.
- Audio-as-image can start from short speech clips with transcripts, rendered as log-mel spectrograms.
- Natural-image examples can start from small VQA-style data.
- Multi-turn examples should be synthetic at first so the required dependency between turns is measurable.

## Feasibility Risks

- Text readability competes with image token budget.
- DeepSeek-OCR-style optical compression may rely on linguistic priors, so evaluations need adversarial or low-prior text.
- Spectrograms are image-like but not natural images; a generic VLM may need targeted adaptation.
- Gemma 4 E2B has native audio and text paths, so the baseline must be constrained or the experiment will not isolate visual unification.
- Multi-turn visual context may require repeatedly re-encoding conversation history instead of using cheap text KV-cache behavior.

## Related Work Pointers

- DeepSeek-OCR and DeepSeek-OCR 2: optical context compression, variable visual token budgets, and visual causal flow.
- Critiques of DeepSeek-OCR: tests suggesting heavy reliance on linguistic priors under compression.
- ImageBind: aligns multiple modalities through a shared embedding space using image-paired data rather than rendering everything into images.
- Unified-IO 2: unifies text, image, audio, action, and structures through tokenization and a shared transformer; uses spectrogram/audio tokenization rather than generic image rendering only.
- AudioCLIP, CLAP, Whisper, Audio Spectrogram Transformer, EnCodec, SoundStream, and WavTokenizer: prior art for audio representations and audio-language alignment.
- Pix2Struct, Donut, Nougat, LayoutLM-family, DocVQA, and TextVQA: prior art around document/text understanding from pixels.

## First Decision Needed

Decide the multi-turn representation before implementing data generation. It determines rendering, dataset schema, token budget, evaluation design, and whether the project is a visual-input experiment or a fully visual conversation experiment.
