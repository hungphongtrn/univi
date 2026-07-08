# Dataset Notes

This project prefers public datasets that contain visual, audio, and text together instead of stitching separate text-only, audio-only, and image-only benchmarks.

## Current Shortlist

### Valor32k-AVQA v2.0

- Source: `inesriahi/valor32k-avqa-v2` on Hugging Face and ACM MM 2025.
- Modalities: video frames, audio track, text questions/answers.
- Size: 28,861 videos and 225,487 question-answer pairs in the Hugging Face release.
- Useful labels: each question has a required modality label: `visual`, `audio`, or `audio-visual`; each also has a category such as description, action, count, temporal, location, or relative-position.
- Access note: test videos are included in the Hugging Face dataset; train/validation media are referenced by `video_id` and may require external retrieval.
- Fit for Univi: best first cut for single-turn text-as-image, audio-as-image, natural-image, and mixed visual-input evaluation.

### JointAVBench

- Source: `roverx12345/jointavbench` on Hugging Face and GitHub.
- Modalities: released video clips, audio, subtitles/metadata, text questions/answers.
- Size: 2,853 multiple-choice questions across 15 task types.
- Useful labels: cognitive dimensions, audio information types, and scene spans.
- Access note: Apache 2.0 repository license; released benchmark clips are provided for reproducible evaluation while original YouTube URLs are retained.
- Fit for Univi: strong fallback or evaluation set because every question is designed to require joint audio-visual reasoning.

### Daily-Omni

- Source: `liarliar/Daily-Omni` on Hugging Face and arXiv `2505.17862`.
- Modalities: real-world videos, audio, multiple-choice text questions/answers.
- Size: 684 videos and 1,197 questions.
- Useful labels: task families around audio-visual temporal alignment, comparison, context understanding, event sequence, inference, and reasoning.
- Fit for Univi: small evaluation set for temporal alignment and audio-visual reasoning after the first pipeline works.

### OmniInteract

- Source: `lucky-lance/OmniInteract` on Hugging Face and arXiv `2605.26485`.
- Modalities: continuous video, audio containing user queries and ambient sound, text annotations/targets.
- Size: 250 videos and 1,430 temporally grounded response slots.
- Useful labels: 1Q1A and 1QnA splits, real-time/proactive/nested interaction categories.
- Fit for Univi: best current candidate for fully visual multi-turn/streaming evaluation, but too complex for the first single-turn MVP.

## Dataset Of Record

Use Valor32k-AVQA v2.0 first for Phases 1-3 because it is large, public, tri-modal, and has per-question modality labels. Use the included test videos first to avoid YouTube availability issues.

Keep JointAVBench as the fallback if Valor32k media access or licensing blocks progress. Add Daily-Omni as a small temporal-alignment evaluation once the rendering pipeline works. Defer OmniInteract until Valor32k Phases 1-3 produce interpretable results.

## Valor32k Usage

Dataset of interest: `inesriahi/valor32k-avqa-v2`, specifically the included test videos first.

Each Valor32k example has a video, audio track, question, four answer choices, correct answer index, modality label, and question category. Univi would convert each example into an image-only prompt bundle:

- **Text-as-Image**: render the question and answer choices into a compact text image.
- **Audio-as-Image**: extract the video audio and render it as a log-mel spectrogram image.
- **Natural Image**: sample a small fixed number of video frames as ordinary image inputs.
- **Target**: predict the multiple-choice answer index or answer text.

Initial visual bundle:

- One rendered question/options image for every example.
- One log-mel spectrogram image for `audio` and `audio-visual` examples.
- Four sampled video frames for `visual` and `audio-visual` examples.
- No unused modality images in the first image-only lane.
- Image order: rendered question/options, spectrogram if present, then chronological video frames.
- Bundle role labels are rendered into the images, not supplied as native text.

Conservative render settings:

- Text image: 1024px-wide canvas, variable height, at least 14pt legible sans-serif or monospace font.
- Text image token budget: 560 tokens, with a 280-token ablation in the smoke run.
- Spectrogram image token budget: 280 tokens.
- Spectrogram content: central 10 seconds of audio rendered with 80 log-mel bins.
- Video frame token budget: 140 tokens per frame.
- Frame sampling: four frames uniformly sampled over the clip.

Use the modality label to control which visual inputs are present:

- `visual`: rendered question/options plus sampled frames; omit the spectrogram in the image-only lane.
- `audio`: rendered question/options plus log-mel spectrogram; omit sampled frames unless needed as a negative-control variant.
- `audio-visual`: rendered question/options plus sampled frames plus log-mel spectrogram.

Run two lanes for each subset:

- **Image-only lane**: model receives only images: rendered question/options, sampled frames, and/or spectrograms. The fixed native text instruction can say only something like "Answer the multiple-choice question shown in the images." It must not include the actual question or answer choices as native text.
- **Native upper bound**: model receives the question/options as native text and the video/audio through native supported pathways where available.
- First milestone: zero-shot/prompt-only evaluation; no fine-tuning until smoke and first-eval results are interpretable.

Fixed image-only instruction:

`Answer the multiple-choice question shown in the images. Reply with only A, B, C, or D.`

This instruction must be placed through the original Gemma 4 E2B template. Do not use a custom chat template or wrapper for either the image-only lane or native upper-bound lane.

Initial measurable slices:

- Smoke: 100 examples per modality label.
- First eval: 1,000 `visual`, 1,000 `audio`, and 1,000 `audio-visual` examples from the included test media.
- Scale-up: all included test examples if media processing and token budgets are stable.

Primary metrics:

- Multiple-choice accuracy by modality label.
- Retention metric: image-only accuracy divided by native upper-bound accuracy.
- Category breakdown across description, action, count, temporal, location, and relative-position.
- Shuffled-options accuracy to detect dependence on answer-position or language priors.
- Invalid output rate under strict first-letter A/B/C/D parsing.

Key failure criteria:

- Text-as-image question rendering is unreadable at the chosen visual token budget.
- Shuffled-options accuracy drops more than 10 percentage points below natural accuracy at the same token budget.
- Audio-as-image cannot reach 50% retention on `audio` examples.
- Mixed `audio-visual` examples perform worse than either single-modality slice by more than 10 percentage points.
- The fixed image bundle requires too many visual tokens to fit Gemma 4 E2B practical inference or fine-tuning limits.

## Open Decisions

- Exact subset sizes for the first run.
- Log-mel spectrogram settings for audio-as-image.
- Spectrogram hop length and color mapping.
- Whether OmniInteract is the Phase 4 dataset after Valor32k results are available.
