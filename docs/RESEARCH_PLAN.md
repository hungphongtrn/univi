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
4. Which small datasets should instantiate the first text, audio, image, and multi-turn tasks?
5. What audio duration, mel-bin count, and image dimensions fit the first token budget?

## Minimum Viable Experiment

### Phase 1: Text-as-Image Single-Turn

- Render QA prompts into images.
- Fine-tune or prompt the baseline to answer from rendered text only.
- Compare the image-only lane against a native-text upper bound.
- Success signal: image-only accuracy reaches at least 80% of native-text upper-bound accuracy.
- Failure signal: OCR or reasoning collapses under shuffled, random, or dense text where linguistic priors cannot fill gaps.

### Phase 2: Audio-as-Image Single-Turn

- Convert short speech clips to log-mel spectrogram images.
- Ask transcript or content questions from the rendered audio image.
- Compare the image-only lane against Gemma 4 E2B native audio and an ASR plus LLM pipeline if available.
- Success signal: image-only accuracy reaches at least 50% of native-audio or ASR plus LLM upper-bound accuracy on short clean speech.
- Failure signal: the visual audio path cannot recover speech content above a weak baseline.

### Phase 3: Mixed Visual Inputs

- Mix text-as-image, audio-as-image, and natural-image examples.
- Measure whether unified training degrades any single modality.
- Failure signal: one representation dominates training or hurts the others.

### Phase 4: Fully Visual Multi-Turn

- Construct two-turn examples where turn 2 requires turn 1 context.
- Render prior user turns, assistant turns, and the current user turn into a compact document-like visual transcript.
- Defer chat-style screenshots until the compact layout works, so early failures are less likely to be caused by wasted pixels.
- Success signal: image-only accuracy beats a no-history visual control by at least 20 percentage points.
- Failure signal: context length grows faster than the model can process, or turn-2 accuracy drops to near single-turn/no-context baselines.

## Success Metrics

- Text-as-image: image-only lane reaches at least 80% retention against the native-text upper bound.
- Audio-as-image: image-only lane reaches at least 50% retention against native-audio or ASR plus LLM upper bound on short clean speech.
- Fully visual multi-turn: compact two-turn transcript beats a no-history visual control by at least 20 percentage points.
- Mixed visual inputs: no single modality drops by more than 10 percentage points compared with its single-modality image-only run.

## Dataset Notes

- Text-as-image can start from small QA sets rendered with controlled fonts, density, and layouts.
- Audio-as-image can start from short speech clips with transcripts, rendered as log-mel spectrogram images.
- Natural-image examples can start from small VQA-style data.
- Multi-turn examples should be synthetic at first so the required dependency between turns is measurable.

## Feasibility Risks

- Text readability competes with image token budget.
- DeepSeek-OCR-style optical compression may rely on linguistic priors, so evaluations need adversarial or low-prior text.
- Spectrograms are image-like but not natural images; a generic VLM may need targeted adaptation.
- Gemma 4 E2B has native audio and text paths, so the baseline must be constrained or the experiment will not isolate visual unification.
- Fully visual multi-turn context requires repeatedly re-encoding conversation history instead of using cheap text KV-cache behavior.

## Baseline Protocol

- **Image-only lane**: pass rendered text, rendered audio, natural images, and fully visual transcripts as images; withhold native text/audio equivalents except fixed task instructions needed to query the model.
- **Native upper bound**: run the same examples through Gemma 4 E2B's native text, audio, and image pathways to estimate the cost of visual unification.
- Report both lanes together; do not claim visual unification works from native-lane results.

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

Choose the first datasets for text-as-image, audio-as-image, natural-image, and fully visual multi-turn evaluation.
