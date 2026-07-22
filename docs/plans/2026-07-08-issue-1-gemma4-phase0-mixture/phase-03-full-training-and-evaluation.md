# Phase 3: Full Training & Evaluation

## Phase Goal

Run a larger A100-oriented training schedule for the four active Phase 0 sources, save and reload the checkpoint, and evaluate an **Image-Only Lane** against honest **Native Upper Bound** baselines where native inputs are available. Valor32k remains deferred behind issue #2 and must not block Phase 3.

Phase 3 is intentionally split into code-bearing tasks before GPU execution. The first implementation target is a reproducible local evaluation harness that can run on the existing smoke checkpoint and tiny datasets; the A100 full run is GPU-gated.

## Scope Constraints

- Active sources only: LibriSpeech ASR, DenseFusion, FineWeb-Edu, and SmolTalk.
- No Valor32k training or evaluation unless issue #2 resolves media retrieval first.
- Evaluation must not infer native inputs from rendered images. Eval materialization must preserve `native_user_content`, `target_text`, split/source metadata, and image messages.
- Retention must compare Image-Only Lane and Native Upper Bound for the same model/checkpoint when both lanes are available. Base-model image-only metrics are reported separately as an improvement baseline.
- DenseFusion's Native Upper Bound is identical to the Image-Only Lane because the answer-bearing input is already a natural image; report this explicitly.
- LibriSpeech native audio is optional and defaults to unsupported until a one-example native-audio forward pass succeeds.
- All Gemma 4 calls must use the **Original Gemma E2B Template** through the existing Unsloth/processor path; no hand-rolled prompt wrapper.

## Project Structure After Phase 3

```text
univi/
├── configs/
│   ├── smoke.yaml
│   ├── full.yaml
│   └── eval.yaml
├── train_smoke.py
├── train_full.py
├── eval_lane.py
├── univi/
│   ├── __init__.py
│   ├── trainer.py
│   └── metrics.py
├── data/
│   ├── preprocessing/
│   │   ├── prepare_eval.py
│   │   └── merge_mixture.py
│   ├── materialized/
│   │   ├── smoke-v0/
│   │   ├── full-v0/
│   │   └── eval-v0/{librispeech,densefusion,fineweb,smoltalk}/
│   ├── checkpoints/
│   │   ├── smoke-v0/
│   │   └── full-v0/
│   └── eval/full-v0/results.json
└── tests/
    ├── test_training.py
    ├── test_metrics.py
    └── test_eval.py
```

## Evaluation Matrix

For each source, report these lanes when available:

| Lane | Model | Input form | Purpose |
| --- | --- | --- | --- |
| `base_image` | base `unsloth/gemma-4-E2B-it` | rendered image messages | measures pre-finetune Image-Only Lane behavior |
| `trained_image` | Phase 3 checkpoint | rendered image messages | primary trained Image-Only Lane result |
| `base_native` | base model | native input if available | native capability reference |
| `trained_native` | Phase 3 checkpoint | native input if available | same-checkpoint Native Upper Bound for retention |

Retention is computed as `trained_image / trained_native * 100` for higher-is-better metrics and `trained_native / trained_image * 100` for lower-is-better metrics. If `trained_native` is unavailable, retention is `null` with a `reason` field.

## Per-Source Experiments

### LibriSpeech ASR

- **Dataset source:** `openslr/librispeech_asr`, train split for training, held-out `test.clean` or documented fallback split for evaluation.
- **Rendering method:** Whisper-style Log-Mel Spectrogram Image, mono 16 kHz, 25 ms window, 10 ms hop, 80 mel bins, central 10 s window.
- **Image-only instruction:** `Transcribe the speech represented by this spectrogram image.`
- **Native Upper Bound:** unsupported by default. Enable only after a documented one-example raw-audio forward pass through Gemma 4 succeeds.
- **Evaluation metric:** held-out loss and WER.
- **Failure criterion:** trained image-only WER is above 80%, loss is NaN, or trained image-only loss is worse than base image-only loss.

### DenseFusion Image-Text Description

- **Dataset source:** `BAAI/DenseFusion-1M`, deterministic train/eval slices from the selected subset.
- **Rendering method:** preserve source Natural Image.
- **Image-only instruction:** `Describe this image.`
- **Native Upper Bound:** identical to Image-Only Lane; report `native_equals_image_only: true` and retention `100.0` only for metrics where both values are identical.
- **Evaluation metric:** held-out loss and ROUGE-L.
- **Failure criterion:** trained image-only loss is worse than base image-only loss or ROUGE-L is below 0.2 on the small eval slice.

### FineWeb-Edu Text-Compressed Raw Text

- **Dataset source:** `HuggingFaceFW/fineweb-edu`, `sample-10BT`, deterministic non-overlapping train/eval slices.
- **Rendering method:** DeepSeek-OCR-Style Text Packing, 1024 px wide, at least 14 pt, bounded `max_chars` target.
- **Image-only instruction:** `Transcribe the text shown in the image.`
- **Native Upper Bound:** native text user content `Transcribe this text exactly:\n\n{native_user_content}` with the same bounded target.
- **Evaluation metric:** held-out loss and normalized character error rate; exact match is secondary for short examples only.
- **Failure criterion:** normalized character error rate is above 0.8, loss is NaN, or trained image-only loss is worse than base image-only loss.

### SmolTalk Text-Compressed Instruction Following

- **Dataset source:** `HuggingFaceTB/smoltalk`, deterministic non-overlapping train/eval slices.
- **Rendering method:** DeepSeek-OCR-Style Text Packing of the user instruction, 1024 px wide, at least 14 pt.
- **Image-only instruction:** `Follow the instruction shown in the image.`
- **Native Upper Bound:** original bounded instruction text as native user content, target is the assistant response.
- **Evaluation metric:** held-out loss and ROUGE-L.
- **Failure criterion:** trained image-only loss is worse than base image-only loss or ROUGE-L is below 0.2 on the small eval slice.

## Tasks

### Task 1: Shared Training Module

**Goal:** Extract reusable training helpers without breaking the existing smoke entry point.

**Files:**
- Create: `univi/__init__.py`
- Create: `univi/trainer.py`
- Modify: `train_smoke.py`
- Modify: `tests/test_training.py`

**Steps:**

1. Add a failing import test for `univi.trainer` exposing `load_config`, `build_model`, `apply_lora`, `load_dataset`, and `train`.
2. Move the existing functions from `train_smoke.py` into `univi/trainer.py` with minimal behavior changes.
3. Keep `train_smoke.py` as a thin wrapper that re-exports the moved functions so existing tests and operators still work.
4. Do not implement unverified adapter magic in this task. Checkpoint reload support belongs in Task 5 and must be tested with the smoke checkpoint.
5. Run `uv run pytest tests/test_training.py -v`.

**Completion criterion:** all existing training tests pass or GPU-only tests skip on CPU.

### Task 2: Deterministic Full And Eval Materialization

**Goal:** Build non-overlapping full-train and eval datasets, and preserve native eval fields that current materialized rows do not contain.

**Files:**
- Modify: `data/preprocessing/merge_mixture.py`
- Create: `data/preprocessing/prepare_eval.py`
- Modify/add tests under `tests/test_merge_mixture.py` or `tests/test_eval.py`

**Required behavior:**

- `merge_mixture.py` accepts per-source `split`, `offset`, and `max_samples` controls or an equivalent deterministic slicing config.
- The full training dataset writes `data/materialized/full-v0` with non-overlap metadata sufficient to prove it does not reuse eval rows.
- `prepare_eval.py` writes one dataset per source under `data/materialized/eval-v0/`.
- Each eval row contains:
  - `messages`: Image-Only Lane Gemma 4 multimodal messages.
  - `native_user_content`: text for native text baselines, or `null` when unavailable/identical.
  - `target_text`: assistant target.
  - `source_dataset_id`, `split`, `row_id`, `render_config`, `modality_label`, `preprocessing_version`.
  - `native_available`: boolean.
  - `native_equals_image_only`: boolean.

**Commands:**

```bash
uv run python -m data.preprocessing.merge_mixture \
    --librispeech-samples 200 \
    --densefusion-samples 200 \
    --fineweb-samples 200 \
    --smoltalk-samples 200 \
    --output ./data/materialized/full-v0 \
    --seed 42

uv run python -m data.preprocessing.prepare_eval \
    --output-dir ./data/materialized/eval-v0 \
    --max-samples 50 \
    --offset 1000
```

**Completion criterion:** tests prove eval rows include native fields and train/eval row identifiers do not overlap for the same source/split.

### Task 3: Metrics Module

**Goal:** Provide small dependency-light metrics used by `eval_lane.py`.

**Files:**
- Create: `univi/metrics.py`
- Create: `tests/test_metrics.py`

**Required functions:**

- `strict_first_letter_parse(text: str) -> str | None` for later Valor32k use.
- `compute_wer(reference: str, hypothesis: str) -> float`.
- `compute_normalized_cer(reference: str, hypothesis: str) -> float`.
- `compute_rougel(reference: str, hypothesis: str) -> float`.
- `compute_exact_match(reference: str, hypothesis: str) -> float`.
- `compute_retention(image_score, native_score, higher_is_better=True) -> float | None`.
- `write_eval_output(results, output_path)`.

**Commands:**

```bash
uv run pytest tests/test_metrics.py -v
```

**Completion criterion:** metric tests cover perfect, partial, empty, and zero-denominator cases.

### Task 4: Full Training Config And Entry Point

**Goal:** Add the A100 full-run entry point while preserving smoke behavior.

**Files:**
- Create: `configs/full.yaml`
- Create: `train_full.py`
- Modify: `tests/test_training.py`

**Config defaults:**

```yaml
model:
  name: "unsloth/gemma-4-E2B-it"
  load_in_4bit: true
  use_gradient_checkpointing: true

lora:
  r: 32
  alpha: 64
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  finetune_vision_layers: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 500
  max_seq_length: 2048
  learning_rate: 2.0e-4
  warmup_steps: 20
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  logging_steps: 10
  save_steps: 100
  output_dir: "data/checkpoints/full-v0"
  report_to: "none"
  remove_unused_columns: false
  dataloader_num_workers: 2

dataset:
  path: "data/materialized/full-v0"
  hf_hub_repo_id: "hungphongtrn/univi-phase0-dataset-full-v0"

hub:
  model_repo_id: "hungphongtrn/univi-phase0-full-model-v0"
```

**Decision note:** this intentionally introduces `full-v0` rather than reusing `smoke-v0`. Add a decision-log entry explaining that Phase 2 validated the pipeline on the same materialized dataset shape, and Phase 3 now scales sample count with deterministic non-overlap.

**Commands:**

```bash
uv run python -c "from train_full import main; print('OK')"
uv run pytest tests/test_training.py -v
```

**Completion criterion:** config loads and `train_full.py` imports without GPU work.

### Task 5: Evaluation Config And Harness

**Goal:** Evaluate base and trained checkpoints across image/native lanes and write structured JSON.

**Files:**
- Create: `configs/eval.yaml`
- Create: `eval_lane.py`
- Create: `tests/test_eval.py`

**Output schema:**

```json
{
  "checkpoint": "data/checkpoints/full-v0/final",
  "base_model": "unsloth/gemma-4-E2B-it",
  "per_source": {
    "fineweb": {
      "base_image": {},
      "trained_image": {},
      "base_native": {},
      "trained_native": {},
      "retention": {},
      "native_note": null
    }
  }
}
```

**Required behavior:**

- Load Image-Only Lane rows from `messages` without prompt rewriting.
- Build native text messages only from `native_user_content` and `target_text`.
- Treat DenseFusion native as identical with explicit metadata.
- Treat LibriSpeech native as `null` unless `native_available` is true and a tested audio path exists.
- Compute loss for every lane that can run.
- Generate at least one deterministic prediction per source for qualitative inspection.
- Reload the smoke checkpoint in a GPU-marked test when available before attempting full checkpoint reload.

**Commands:**

```bash
uv run pytest tests/test_eval.py -v
uv run python eval_lane.py --config configs/eval.yaml --max-examples 2
```

**Completion criterion:** `eval_lane.py` can run a two-example CPU-safe config as far as imports/config validation, and GPU evaluation either passes on CUDA or skips with a clear message.

### Task 6: Full Training Execution

**Goal:** Run the full A100 schedule and validate checkpoint reload.

**Prerequisites:** A100 40 GB or equivalent CUDA device, `data/materialized/full-v0`, passing tests.

**Commands:**

```bash
nvidia-smi
uv run python train_full.py --config configs/full.yaml
```

**Expected output:**

- 500 training steps complete without OOM.
- Loss trends downward over the final 100 steps.
- Checkpoints exist under `data/checkpoints/full-v0/` and final checkpoint under `data/checkpoints/full-v0/final/`.
- CUDA peak memory is printed.

**Reload check:**

```bash
uv run python -c "from unsloth import FastVisionModel; FastVisionModel.from_pretrained('data/checkpoints/full-v0/final', load_in_4bit=True); print('OK')"
```

**Failure fallback:** if A100 OOMs, reduce in this order: batch size 2 to 1, max sequence length 2048 to 1024, LoRA rank 32 to 16. Document the actual working values in this file and `decisions.md`.

### Task 7: Evaluation Execution And Report Artifact

**Goal:** Produce the structured Phase 3 metrics artifact for Phase 4 diagnostics.

**Commands:**

```bash
uv run python eval_lane.py --config configs/eval.yaml
uv run python -c "import json; r=json.load(open('data/eval/full-v0/results.json')); print(r.keys()); print(r['per_source'].keys())"
```

**Completion criterion:** `data/eval/full-v0/results.json` contains all four active sources, all available lanes, retention values or explicit null reasons, and no Valor32k dependency.

## Phase Completion Criteria

- [ ] `univi/trainer.py` exists and `train_smoke.py` remains compatible with existing tests.
- [ ] `data/materialized/full-v0` is materialized with deterministic source counts.
- [ ] `data/materialized/eval-v0/{librispeech,densefusion,fineweb,smoltalk}` exists with native eval fields.
- [ ] Train/eval non-overlap is asserted by tests for every active source.
- [ ] `configs/full.yaml` and `train_full.py` exist and import/config tests pass.
- [ ] `univi/metrics.py` and `tests/test_metrics.py` exist and pass.
- [ ] `configs/eval.yaml`, `eval_lane.py`, and `tests/test_eval.py` exist.
- [ ] Full A100 run completes or a documented smaller fallback run completes.
- [ ] Checkpoint save and reload succeeds.
- [ ] Evaluation writes `data/eval/full-v0/results.json` with base image, trained image, native lanes where available, retention/null reason, and per-source failure flags.

## Phase Failure Criteria

- Full training cannot complete on available GPU after fallback reductions.
- Checkpoint reload fails or produces NaN loss/logits.
- Evaluation depends on Valor32k media or any deferred issue #2 artifact.
- Native text evaluation uses rendered-only rows instead of preserved native fields.
- Any source lacks dataset source, rendering method, metric, or failure criterion in the report.
- Trained Image-Only Lane loss is worse than base Image-Only Lane loss for two or more active sources.

## Handoff To Phase 4

Phase 4 consumes `data/eval/full-v0/results.json` and adds diagnostics: invalid-rate reporting, shuffled-option or low-prior text controls, retention summary publication, and dataset/checkpoint publication metadata.
