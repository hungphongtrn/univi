# Dataset Notes

This project uses a Phase 0 training mixture to teach visualized audio transcription, natural image/text description, rendered text reading, rendered instruction following, and tri-modal QA before evaluating unified behavior.

## Phase 0 Training Data Mixture

Phase 0 fine-tunes Gemma 4 on the datasets below. All answer-bearing inputs are images. A short native system prompt or task instruction is allowed, but it must not include the text to transcribe, the audio transcript, the image caption, the rendered instruction, or the Valor32k question/choices.

### 1. ASR: LibriSpeech As Log-Mel Images

- Source: `openslr/librispeech_asr`, config `clean`; use `train.360` for training and `validation` for held-out evaluation.
- Preprocessing: convert complete clips of at most 40 seconds into one to four chronological ten-second 1000 × 160 log-mel images.
- Task: present the ordered spectrogram images and output the spoken transcript.
- Input: all spectrogram pages first, then `Transcribe the speech represented by this spectrogram image.`
- Target: complete transcript text.
- Selected size: up to 104,014 source training rows and 2,703 validation rows before the 40-second duration filter.
- Failure criterion: held-out spectrogram-to-transcript examples do not improve over the base model.

### 2. Text-Image Description

- Source: `HuggingFaceM4/FineVision`, config `densefusion_1m`.
- Task: present an image with text or visually grounded content; output the paired description.
- Input: image first, then a short instruction such as `Describe this image.`
- Target: description/caption text.
- Initial size: TBD.
- Failure criterion: descriptions remain generic, hallucinated, or worse than base-model descriptions on held-out examples.

### 3. Text-Compressed Raw Text

- Source: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`.
- Preprocessing: pack raw text into rendered images.
- Task: present rendered raw text; output the same raw text.
- Input: rendered text image first, then a short instruction such as `Transcribe the text shown in the image.`
- Target: raw text.
- Initial size: TBD.
- Failure criterion: held-out rendered text transcription does not improve over the base model or collapses under low-prior controls.

### 4. Text-Compressed Instruction Following

- Source: `HuggingFaceTB/smoltalk`.
- Preprocessing: pack the instruction into rendered images.
- Task: present rendered instruction text; output the response.
- Input: rendered instruction image first, then a short native wrapper such as `Follow the instruction shown in the image.`
- Target: response text.
- Initial size: TBD.
- Failure criterion: held-out rendered instruction-following does not improve over the base model or depends on native instruction leakage.

### 5. Valor32k-AVQA v2.0

- Source: `inesriahi/valor32k-avqa-v2`.
- Split discipline: use Valor32k train split for training only; reserve validation/test splits for evaluation.
- Task: tri-modal QA from rendered question/options, optional log-mel spectrogram, and optional sampled video frames.
- Input: rendered question/options image, spectrogram if present, sampled frames if present, then a short A/B/C/D instruction.
- Target: A/B/C/D answer.
- Initial size: smoke 300 examples (100 per modality label), first training/eval cut TBD.
- Failure criterion: mixed examples degrade individual source performance or invalid output rate remains high.

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

Valor32k-AVQA v2.0 is the tri-modal QA source in the Phase 0 training mixture and the primary post-training evaluation benchmark for mixed visual inputs. Use the official train split for training only; use validation/test splits for held-out evaluation.

Keep JointAVBench as the fallback if Valor32k media access or licensing blocks progress. Add Daily-Omni as a small temporal-alignment evaluation once the rendering pipeline works. Defer OmniInteract until Valor32k Phases 1-3 produce interpretable results.

## Valor32k Usage

Dataset of interest: `inesriahi/valor32k-avqa-v2`, using its train/validation/test split discipline.

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
- First milestone: LoRA fine-tuning on the Phase 0 training mixture; evaluation follows training on held-out splits.

Fixed image-only instruction:

`Answer the multiple-choice question shown in the images. Reply with only A, B, C, or D.`

This instruction must be placed through the original Gemma 4 E2B template. Do not use a custom chat template or wrapper for either the image-only lane or native upper-bound lane.

Initial measurable slices:

- Smoke: 100 examples per modality label from the Valor32k validation split.
- First eval: 1,000 `visual`, 1,000 `audio`, and 1,000 `audio-visual` examples from held-out Valor32k validation/test splits.
- Scale-up: remaining held-out examples if media processing and token budgets are stable.

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

## Preprocessed Hugging Face Dataset Contract

Preprocess each source into a materialized Hugging Face dataset before training. The training code should consume the same schema for every source:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "HF Image object or relative dataset image"},
        {"type": "text", "text": "Short task instruction with no answer-bearing native content."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "target text"}
      ]
    }
  ]
}
```

Rules:

- Store rendered images as materialized dataset images, not local absolute paths.
- Put image content before the text instruction for Gemma 4 multimodal examples.
- Do not include native answer-bearing text/audio in the user instruction.
- Include source metadata for traceability: source dataset ID, source split, source row ID, render config ID, image count, target type, modality label when applicable, and preprocessing version.
- Preserve each source dataset's dedicated split during preprocessing; train on train splits and evaluate on held-out validation/test splits.

## Phase 0 Preprocessing Summary

All Phase 0 sources are converted into the same Gemma 4 multimodal `messages` format, then concatenated and shuffled for training. The first 3060 smoke run and A100 full run use the same materialized dataset; they differ only in training schedule, LoRA settings, and available compute.

### Audio To Images

- Source: `openslr/librispeech_asr`, config `clean`.
- Training split: `train.360`; held-out evaluation split: `validation` (2,703 rows).
- Filter source clips longer than 40 seconds.
- Resample each retained clip to mono 16 kHz and compute one 80-bin power log-mel matrix using a 25 ms Hann window and 10 ms hop.
- Apply Whisper normalization: clamp log10 power to `max - 8`, apply `(x + 4) / 4`, then linearly map the resulting `[-1, 1]` range onto `[0, 255]` grayscale without discarding the lower half.
- Split the matrix into chronological ten-second pages. Render each page at 1000 × 160, right-padding only the final page; retain one to four images per example.
- Store every page as a materialized HF image in chronological order.
- User content: all spectrogram pages first, then `Transcribe the speech represented by this spectrogram image.`
- Assistant content: the complete transcript.
- Metadata: source dataset ID and split, source row ID, source duration, sample rates, mel bins, hop/window lengths, page dimensions and count, fixed time scale, normalization, render version, and preprocessing version.

### Raw Text To Images

- Source: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`.
- Select raw text rows from the source split.
- Pack bounded text chunks into rendered text images using DeepSeek-OCR-style optical compression practice: fixed page-like canvases, dense readable text, no chat chrome, and compression kept conservative enough to stay below the failure-prone high-compression regime.
- Render the complete selected source text across ordered fixed 1024 x 1024 pages at 14 px; do not shrink the font or truncate the visual input to fit one image.
- Store every rendered page as a materialized HF image.
- User content: rendered text image first, then `Transcribe the text shown in the image.`
- Assistant content: the source-text prefix capped at 1,024 `unsloth/gemma-4-E2B-it` tokenizer tokens.
- Metadata: source dataset ID, source split, source row ID, text byte/character count, token estimate, font, font size, canvas width/height, render config ID, preprocessing version.

### Instruction Following To Images

- Source: `HuggingFaceTB/smoltalk`.
- Extract instruction/user content and target assistant response from the source conversation format.
- Pack the instruction-bearing user content into rendered text images using the same DeepSeek-OCR-style optical page packing as raw text.
- Keep the complete instruction across ordered fixed 1024 x 1024 pages.
- Store rendered instruction images as materialized HF images.
- User content: rendered instruction image first, then `Follow the instruction shown in the image.`
- Assistant content: response text capped at 1,024 `unsloth/gemma-4-E2B-it` tokenizer tokens.
- Metadata: source dataset ID, source split, source row ID, original subset/config when available, image count, render config ID, preprocessing version.

### Text-Image Description

- Source: `BAAI/DenseFusion-1M` subset.
- Preserve the source image as the answer-bearing image input.
- User content: source image first, then a short instruction such as `Describe this image.`
- Assistant content: paired description text.
- Metadata: source dataset ID, source split, source row ID, selected subset, image dimensions, target type, preprocessing version.

### Valor32k Visual Bundle

- Source: `inesriahi/valor32k-avqa-v2`.
- Use train split for training only; reserve validation/test for held-out evaluation.
- Render question and A/B/C/D answer choices into one text image.
- For `audio` and `audio-visual` examples, extract audio and render a log-mel spectrogram image.
- For `visual` and `audio-visual` examples, sample four video frames chronologically.
- User content: rendered question/options image first, spectrogram if present, video frames if present, then the fixed A/B/C/D instruction.
- Assistant content: correct answer letter.
- Metadata: source split, video ID, question ID, modality label, category, frame timestamps, audio window, render config IDs, preprocessing version.

### 3M Artifact Configurations

- Publish `librispeech`, `densefusion`, `fineweb-edu`, and `smoltalk` as separate Hugging Face configurations in `hungphongtrn/univi-3M-v0`.
- Every configuration exposes exactly `train` and `validation`:
  - `librispeech`: freshly render source `clean/train.360` as `train` and source `clean/validation` as `validation`; filter clips over 40 seconds and retain one to four fixed ten-second rectangular pages.
  - `densefusion`: preserve source images and deterministically partition retained rows 90/10.
  - `fineweb-edu`: render ordered 1024 x 1024 pages, then deterministically partition the selected source rows 90/10.
  - `smoltalk`: map source `train` to `train` and source `test` to `validation`.
- Record `original_token_length` from each untruncated assistant target. This supports later filtering to targets of at most 1,024 tokens without losing the original length during preprocessing.
- Training consumes only `train`; `validation` must not be concatenated into training.
- Rebuild exactly 1,000,000 FineWeb-Edu and 1,000,000 SmolTalk rows.
- Rebuild LibriSpeech from the selected source splits; do not reuse or retile the obsolete square/strip artifact.
- Concatenate and deterministically shuffle all four configurations at training load time rather than storing a duplicate merged copy.
- Build validation from a dedicated source split where available; otherwise use the documented deterministic 90/10 partition.
- Do not apply hand-tuned modality ratios beyond these selected source configurations.

Build locally first. The command is resumable at completed configuration boundaries:

```bash
uv run python -m data.preprocessing.rebuild_3m \
  --input data/materialized/full-v0 \
  --output data/materialized/univi-3M-v0 \
  --text-samples 1000000 \
  --max-output-tokens 1024 \
  --num-proc 8
```

To replace only LibriSpeech in an existing four-configuration artifact while reusing the other configurations:

```bash
uv run python -m data.preprocessing.rebuild_3m \
  --output data/materialized/univi-3M-v0-split \
  --force-librispeech \
  --num-proc 8
```

After validating the local artifact, rerun with `--push-repo hungphongtrn/univi-3M-v0`. Existing local configurations are reused and pushed without rerendering.

### Render Policy

- Text render policy: follow DeepSeek-OCR's lesson that optical compression works best below roughly 10x text-token-to-vision-token compression; start with page-like 1024px canvases and avoid tiny/high-density modes for the first Gemma E2B run.
- Audio render policy: use mono 16 kHz audio, a 25 ms Hann window, a 10 ms hop, 80 mel bins, Whisper log-mel normalization, and one to four chronological 1000 × 160 pages at a fixed ten-second scale. Gemma's aspect-ratio-preserving processor maps this geometry to about 246 visual tokens per page.

## Open Decisions

- Exact `BAAI/DenseFusion-1M` subset.
- Exact subset sizes for the first training run.
- LoRA hyperparameters.
- Whether OmniInteract is the Phase 4 dataset after Valor32k results are available.
