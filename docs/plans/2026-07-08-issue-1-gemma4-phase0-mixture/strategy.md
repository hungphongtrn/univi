# Train Gemma 4 E2B on Phase 0 Visual-Unification Mixture — Strategy

## Goal
Fine-tune `unsloth/gemma-4-E2B-it` with LoRA on a five-source visual-input-only training mixture and evaluate whether a VLM can learn useful reasoning over text-as-image, audio-as-image, and natural image inputs when every answer-bearing input is a pixel.

## Architecture
Five independent preprocessing scripts convert each source dataset into the Gemma 4 multimodal `messages` format (images as HF Image objects, short task instruction, assistant target). The preprocessed splits are concatenated and shuffled into one materialized Hugging Face dataset. Training uses Unsloth `FastVisionModel`, `UnslothVisionDataCollator`, and TRL `SFTTrainer` with the Gemma 4 non-thinking template. Evaluation compares an image-only lane (withheld native text/audio) against a native upper-bound lane.

## Tech Stack
- **Model:** `unsloth/gemma-4-E2B-it` via Unsloth `FastVisionModel`
- **Training:** Unsloth + TRL `SFTTrainer`, QLoRA/LoRA, `UnslothVisionDataCollator`
- **Data:** Hugging Face `datasets` with materialized images
- **Audio:** `librosa` for log-mel spectrogram rendering
- **Text rendering:** Pillow / `matplotlib` for DeepSeek-OCR-style page packing
- **Compute:** RTX 3060 12 GB (smoke), A100 40 GB (full run)

## Constraints & Assumptions
- Gemma 4 E2B `audio_tower` is kept in the first run (no model surgery).
- All answer-bearing inputs are images; a short native text instruction is allowed but must not contain the answer.
- Images are stored as materialized HF `Image` objects, not local paths.
- The concatenate-and-shuffle mixture rule is used (no hand-tuned modality ratios in the first run).
- Evaluation uses strict first-letter A/B/C/D parsing; invalid outputs count as wrong.
- The same materialized dataset serves both smoke and full runs (differ only in schedule/rank).
- Text rendering follows DeepSeek-OCR-style page packing with conservative compression (≥14 pt, 1024 px wide).
- Audio rendering uses Whisper-style log-mel: 16 kHz mono, 25 ms window, 10 ms hop, 80 bins.

## ADR References
- [ADR-0001](../../adr/0001-use-fully-visual-transcripts.md) — Multi-turn context as pixels
- [ADR-0002](../../adr/0002-use-valor32k-for-first-trimodal-cut.md) — Valor32k as dataset of record
- [ADR-0003](../../adr/0003-train-first-phase-0-mixture.md) — Train-first Phase 0 mixture

## Phases (High-Level)

### Phase 1: Preprocessing Pipeline — Foundation
**Outcome:** All five source datasets are materialized as HF datasets in Gemma 4 multimodal `messages` format, concatenated and shuffled into one training dataset.
**Rough scope:** Write 5 preprocessing scripts + 1 merge script + validation that each source produces correct schema.

### Phase 2: Smoke Training — Core Feature
**Outcome:** RTX 3060 smoke run validates data loading, image collation, loss masking, loss trending down, and checkpoint save/load without crashes.
**Rough scope:** Unsloth training script, smoke config, short-run validation.
**Depends on:** Phase 1

### Phase 3: Full Training & Evaluation
**Outcome:** A100 full run completes with checkpoint save/reload. Image-only lane and native upper-bound metrics reported for each modality label.
**Rough scope:** Full training script, evaluation harness, metric collection.
**Depends on:** Phase 2

### Phase 4: Diagnostics & Publication
**Outcome:** Retention metrics, shuffled-option diagnostics, invalid output rate published. Render settings and source splits documented for reproduction.
**Rough scope:** Analysis notebook, report generation, dataset publication to HF Hub.
**Depends on:** Phase 3

## Open Questions
1. **DenseFusion subset selection** — Which subset of BAAI/DenseFusion-1M? Exact subset size for the first run?
2. **LibriSpeech split mapping** — Which LibriSpeech configs/subset sizes map to train/validation/test?
3. **FineWeb-Edu chunk strategy** — How many characters per rendered page? How to handle truncation for long documents?
4. **SmolTalk subset size** — How many examples from which SmolTalk configs?
5. **Valor32k media retrieval** — Are media files (video, audio) accessible for the train split via HF dataset, or do they require external download?
6. **LoRA hyperparameters** — Rank, alpha, target modules, learning rate, batch size for both smoke and full runs? Needs first-pass defaults; tune after smoke.
7. **Spectrogram colormap** — Which colormap and dynamic range for the log-mel image? Should match convention the model can interpret visually.
8. **RTX 3060 context length** — What max sequence length fits in 12 GB with QLoRA? May constrain text rendering density or number of Valor32k frames.
