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

## Recommended First Cut

Use Valor32k-AVQA v2.0 first for Phases 1-3 because it is large, public, tri-modal, and has per-question modality labels. Use the included test videos first to avoid YouTube availability issues.

Keep JointAVBench as the fallback if Valor32k media access or licensing blocks progress. Add Daily-Omni as a small temporal-alignment evaluation once the rendering pipeline works. Defer OmniInteract until the compact two-turn transcript pipeline is stable.

## Open Decisions

- Whether to select Valor32k-AVQA v2.0 as the first dataset of record.
- Exact subset sizes for the first run.
- Frame sampling rate and maximum video duration.
- Log-mel spectrogram settings for audio-as-image.
- Whether multi-turn should use OmniInteract immediately or wait until after single-turn tri-modal results.
