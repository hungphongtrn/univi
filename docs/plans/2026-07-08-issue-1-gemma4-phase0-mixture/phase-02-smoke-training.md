# Phase 2: Smoke Training

## Phase Goal
Run Gemma 4 E2B on the RTX 3060 (12 GB) with QLoRA/LoRA, batch size 1, short max sequence length, and small max steps to validate dataset loading, image collation, loss masking, checkpoint save/load, and that loss trends down without divergence.

**Image-Only Lane only during training.** All native text/audio paths are withheld — every answer-bearing input is a rendered image. The instruction text is the **Generic A-D Instruction** (`Describe this image.`, `Transcribe the speech represented by this spectrogram image.`, `Transcribe the text shown in the image.`, `Follow the instruction shown in the image.`) placed via the **Original Gemma E2B Template**. No custom chat template.

> **Original Gemma E2B Template** (ADR-flagged decision, see `CONTEXT.md:183`):
> The official Gemma 4 E2B chat/input template used by the model and processor
> without custom prompt wrapping. In practice this means calling
> `tokenizer.get_chat_template()` from Unsloth to load the Gemma 4 non-thinking
> template, and formatting each row's `messages` list via
> `tokenizer.apply_chat_template(..., tokenize=False)` before collation. The
> template is responsible for role markers (user/assistant), image token
> insertion, and loss-masking boundaries.

**Dataset:** 4-source materialized mixture (LibriSpeech, DenseFusion, FineWeb-Edu, SmolTalk) produced by Phase 1. Valor32k remains deferred behind issue #2.

**Compute:** RTX 3060 12 GB. QLoRA with 4-bit NF4 quantization to fit the ~27B model. (E2B is the largest Gemma 4 variant; do not describe it as "small" — fitting it at all on 12 GB depends on aggressive quantization and LoRA.)

**Model ID:** `unsloth/gemma-4-E2B-it` is the pinned default. The alternative `google/gemma-4-E2B` does not include the `-it` chat-tuned suffix and may lack the Gemma 4 chat template hooks Unsloth expects. If `unsloth/gemma-4-E2B-it` becomes unavailable, the model id is configurable via `smoke.yaml` so a substitute (e.g. `google/gemma-4-E2B` with manual template injection) can be swapped in without code changes.

**Native paths:** Preprocessing still decodes raw audio/text at materialization time (allowed per `decisions.md`), but the training loop receives only images. This is documented in case of concerns about preprocessing leakage.

## Project Structure After Phase 2

```
univi/
├── configs/
│   └── smoke.yaml                # QLoRA/LoRA hyperparameters, dataset path, etc.
├── train_smoke.py                # Training script entry point
├── data/
│   ├── preprocessing/            # (from Phase 1 — unchanged)
│   └── materialized/
│       └── smoke-v0/             # Pre-existing materialized dataset (from Phase 1 or rebuilt)
├── tests/
│   ├── test_training.py          # Training component tests (dataset loading, collation, loss masking)
│   └── fixtures/                 # (from Phase 1 — unchanged)
```

## Tasks

### Task 0: Add training dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Unsloth and training framework dependencies**

Run:
```bash
uv add 'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130' \
       'unsloth>=2025.3' 'trl>=0.16.0' 'peft>=0.15.0' 'transformers>=4.48.0' \
       'accelerate>=1.5.0' 'bitsandbytes>=0.45.0' 'scipy>=1.15.0' 'sentencepiece>=0.2.0' \
       'huggingface-hub>=0.30.0'
```

Expected: dependencies recorded in `pyproject.toml` and locked in `uv.lock`. The Phase 2 GPU target is CUDA 13 with `torch==2.10.0+cu130` and `torchvision==0.25.0+cu130`; do not allow the resolver to upgrade Torch past 2.10 or fall back to PyPI's non-CUDA-13 wheels for this smoke run. Configure the PyTorch CUDA 13 wheel index in `pyproject.toml`:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

- [ ] **Step 2: Verify GPU-dependent import**

Run: `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
Expected: Torch `2.10.0+cu130`, CUDA runtime `13.x`, and `True` on the RTX 3060 machine. CUDA 13 + Torch 2.10 is required for smoke training.

- [ ] **Step 3: Verify Unsloth import**

Run: `uv run python -c "from unsloth import FastVisionModel; print('OK')"`
Expected: prints `OK`. May require a CUDA device visible even for import; if the import fails on non-GPU CI, the task is still valid — training tests run only on the GPU machine.

### Task 1: Training configuration

**Files:**
- Create: `configs/smoke.yaml`

- [ ] **Step 1: Write the smoke YAML config**

```yaml
# configs/smoke.yaml — RTX 3060 smoke run on CUDA 13 + Torch 2.10
model:
  name: "unsloth/gemma-4-E2B-it"          # pinned default; swap if unavailable
  load_in_4bit: true
  use_gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  finetune_vision_layers: true            # vision encoder via flag, not target_modules

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  max_steps: 50
  max_seq_length: 1024
  learning_rate: 2.0e-4
  warmup_steps: 5
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  logging_steps: 5
  save_steps: 25
  output_dir: "data/checkpoints/smoke-v0"
  report_to: "none"              # no wandb/tensorboard for smoke
  remove_unused_columns: false
  dataloader_num_workers: 0

dataset:
  path: "data/materialized/smoke-v0"       # produced by Phase 1 merge CLI
  hf_hub_repo_id: "hungphongtrn/univi-phase0-dataset-smoke-v0"  # dataset on HF Hub

hub:
  model_repo_id: "hungphongtrn/univi-phase0-smoke-model-v0"    # separate from dataset id
```

Rationale:
- LoRA rank 16: conservative for 12 GB memory with 4-bit base.
- `finetune_vision_layers: true` — the research question is whether the model can learn from rendered images; freezing the vision encoder would undermine this. Using the dedicated Unsloth flag rather than listing `"vision_encoder"` in `target_modules` avoids API drift (the vision encoder may not use standard `Linear` layers that LoRA can graft onto).
- `max_seq_length` 1024: informed guess for 12 GB. May need adjustment lower (512) if OOM. Document the actual working value after smoke.
- `gradient_accumulation_steps` 4: compensates for batch size 1 to stabilise gradients.
- `adamw_8bit`: lower memory than full-precision AdamW.
- Dataset and model checkpoint repos have distinct IDs (`univi-phase0-dataset-smoke-v0` vs `univi-phase0-smoke-model-v0`) to avoid confusion.

### Task 2: Smoke training script

**Files:**
- Create: `train_smoke.py`
- Test: `tests/test_training.py`

- [ ] **Step 1: Write the failing test**

```python
def test_training_dataset_loads():
    """Verify that the training script can load the materialized dataset."""
    from datasets import load_from_disk
    ds = load_from_disk("data/materialized/smoke-v0")
    assert len(ds) > 0
    assert "messages" in ds[0]
    assert len(ds[0]["messages"]) == 2  # user + assistant
    user_content = ds[0]["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[-1]["type"] == "text"

def test_training_schema_conforms_to_fastvisionmodel():
    """Verify messages schema matches FastVisionModel expectations."""
    from datasets import load_from_disk
    ds = load_from_disk("data/materialized/smoke-v0")
    for row in ds:
        msgs = row["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        # All image types must be PIL Image objects after loading
        for c in msgs[0]["content"]:
            if c["type"] == "image":
                from PIL import Image
                assert isinstance(c["image"], Image.Image)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_training.py::test_training_dataset_loads -v
```

Expected: FAIL (dataset may not exist at `data/materialized/smoke-v0` — either rebuild via Phase 1 merge CLI or the test path needs updating).

- [ ] **Step 3: Ensure materialized dataset is available**

If `data/materialized/smoke-v0` does not exist, rebuild it:

```bash
uv run python -m data.preprocessing.merge_mixture \
    --librispeech-samples 10 \
    --densefusion-samples 10 \
    --fineweb-samples 10 \
    --smoltalk-samples 10 \
    --output ./data/materialized/smoke-v0 \
    --seed 42
```

Expected: 40 examples written to `data/materialized/smoke-v0/`.

- [ ] **Step 4: Write the training script**

`train_smoke.py` — CLI entry point that loads config, model, dataset, and runs training.

```python
"""
Smoke training entry point for Gemma 4 E2B Phase 0 visual-unification mixture.

Usage:
    uv run python train_smoke.py --config configs/smoke.yaml

Loss masking:
    UnslothVisionDataCollator applies response-only loss masking automatically
    via the chat template processing — user-turn tokens receive -100 labels
    so they do not contribute to the loss. No explicit masking step is needed.
"""
import argparse
import os
from pathlib import Path

import yaml
from datasets import load_dataset as hf_load
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config["model"]["name"],
        load_in_4bit=config["model"].get("load_in_4bit", True),
        use_gradient_checkpointing=config["model"].get(
            "use_gradient_checkpointing", True
        ),
    )
    return model, tokenizer


def apply_lora(model, lora_cfg: dict):
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        finetune_vision_layers=lora_cfg.get("finetune_vision_layers", True),
    )
    return model


def load_dataset(config: dict):
    path = config["dataset"]["path"]
    if Path(path).exists():
        return load_from_disk(path)
    hub_repo = config["dataset"].get("hf_hub_repo_id")
    if hub_repo:
        return hf_load(hub_repo, split="train")
    raise FileNotFoundError(
        f"Dataset not found at {path} and no hf_hub_repo_id configured."
    )


def train(config: dict):
    model, tokenizer = build_model(config)
    model = apply_lora(model, config["lora"])

    # Call for_training to prepare model for SFTTrainer
    model = FastVisionModel.for_training(model)

    dataset = load_dataset(config)
    train_cfg = config["training"]

    # Gemma 4 non-thinking template via Unsloth
    tokenizer = tokenizer.get_chat_template()

    # Use SFTConfig from TRL with dataset_kwargs for SFT-specific setup
    training_args = SFTConfig(
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg.get(
            "gradient_accumulation_steps", 1
        ),
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        optim=train_cfg.get("optim", "adamw_8bit"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 25),
        output_dir=train_cfg["output_dir"],
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=train_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=training_args,
    )

    # Trainer loop. Loss masking (response-only) is handled automatically by
    # UnslothVisionDataCollator — no explicit mask_loss step needed.
    trainer.train()

    # Save final checkpoint
    trainer.save_model(train_cfg["output_dir"] + "/final")
    tokenizer.save_pretrained(train_cfg["output_dir"] + "/final")

    # Push to HF Hub if token available (separate model repo from dataset repo)
    hub_cfg = config.get("hub", {})
    model_repo_id = hub_cfg.get("model_repo_id", "")
    hf_token = os.environ.get("HF_TOKEN")
    if model_repo_id and hf_token:
        model.push_to_hub(model_repo_id, token=True)
        tokenizer.push_to_hub(model_repo_id, token=True)

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = train(config)
    log = trainer.state.log_history
    if log:
        print(f"Training complete. Log history: {log[-1]}")
    print(f"Checkpoint saved to {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add `pyyaml` dependency**

Run: `uv add pyyaml>=6.0`

- [ ] **Step 6: Run dataset-loading tests to verify they pass**

```bash
uv run pytest tests/test_training.py -v
```

Expected: PASS.

### Task 3: Training component tests

**Files:**
- Append to: `tests/test_training.py`

- [ ] **Step 1: Write collation test (GPU-required)**

This test validates that `UnslothVisionDataCollator` produces correctly shaped batches. Loss masking is handled automatically by UnslothVisionDataCollator's chat-template processing (user-turn tokens receive -100 labels); no explicit masking test is needed. The test requires a GPU because `FastVisionModel.from_pretrained` for E2B needs CUDA.

```python
import pytest
import torch


def has_gpu() -> bool:
    import torch
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="requires CUDA GPU")


@requires_gpu
def test_unsloth_vision_data_collator_outputs():
    """Verify collator produces expected batch structure from one example."""
    from datasets import load_from_disk
    from unsloth import FastVisionModel, UnslothVisionDataCollator

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/gemma-4-E2B-it",
        load_in_4bit=True,
    )
    tokenizer = tokenizer.get_chat_template()
    collator = UnslothVisionDataCollator(model, tokenizer)

    ds = load_from_disk("data/materialized/smoke-v0")
    batch = [ds[i] for i in range(min(2, len(ds)))]

    result = collator(batch)

    assert "input_ids" in result
    assert "labels" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].shape[0] == len(batch)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_training.py -v
```

Expected: SKIPPED (requires GPU) or PASS if run on GPU machine. Dataset-loading/schema tests (without `@requires_gpu`) pass on CPU.

- [ ] **Step 3: Add GPU-conditional marker**

```python
import pytest

def has_gpu() -> bool:
    import torch
    return torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(not has_gpu(), reason="requires CUDA GPU")
```

Apply `@requires_gpu` to all collation and model-load tests. These tests cannot run on CPU because `FastVisionModel.from_pretrained(device_map="cpu")` is not supported by Unsloth for E2B (model architecture requires CUDA). Dataset-loading and schema tests remain CPU-safe.

### Task 4: Smoke training execution

**Execution on RTX 3060 machine (manual / not auto-CI).**

- [x] **Step 1: Verify environment**

```bash
nvidia-smi | grep -E "RTX 3060|CUDA Version"
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected: RTX 3060 visible, CUDA Version 13.x from `nvidia-smi`, and Torch reports CUDA runtime 13.x with `torch==2.10.0+cu130`.

**CUDA-aware Unsloth install guidance:** If `import unsloth` fails after the initial dependency install, the Unsloth CUDA kernel build may need a manual step. Prefer a CUDA 13-compatible Unsloth/PyTorch install and keep `torch==2.10.0+cu130` pinned:
```bash
# Install Unsloth from source against the local CUDA 13 + Torch 2.10 stack if the wheel misses kernels.
uv pip install --no-deps "unsloth @ git+https://github.com/unslothai/unsloth.git"
```
If only the PyPI wheel is available, ensure `torch==2.10.0+cu130` was installed with CUDA 13 support (not the CPU-only or CUDA 12 variant). The `uv add 'unsloth>=2025.3'` step installs the PyPI release; CUDA dispatch should work on a CUDA-visible system but may need the source install if triton/FlashAttention kernels fail to load.

- [x] **Step 2: Rebuild materialized dataset with smoke-sized subset**

```bash
uv run python -m data.preprocessing.merge_mixture \
    --librispeech-samples 10 \
    --densefusion-samples 10 \
    --fineweb-samples 10 \
    --smoltalk-samples 10 \
    --output ./data/materialized/smoke-v0 \
    --seed 42
```

Expected: 40 examples written.

- [x] **Step 3: Run smoke training**

```bash
uv run python train_smoke.py --config configs/smoke.yaml
```

Expected:
- Model loads without OOM.
- Training loop runs 50 steps.
- Loss logs at every 5 steps.
- Loss trends downward (e.g. starts ~13-15 per Unsloth guidance, trends lower).
- Checkpoint saved at steps 25 and 50.
- Final model saved to `data/checkpoints/smoke-v0/final/`.

If OOM at `max_seq_length=1024`, reduce to 512 in `configs/smoke.yaml`:

```yaml
training:
  max_seq_length: 512
```

Document the working `max_seq_length` in the config and in this plan.

- [x] **Step 4: Verify checkpoint save and reload**

```bash
# Load the saved checkpoint and run one forward pass on a held-out batch
uv run python -c "
from unsloth import FastVisionModel
from datasets import load_from_disk
model, tokenizer = FastVisionModel.from_pretrained(
    'data/checkpoints/smoke-v0/final',
    load_in_4bit=True,
)
ds = load_from_disk('data/materialized/smoke-v0')
print('Checkpoint loaded successfully.')
print(f'Dataset has {len(ds)} examples.')
"
```

Expected: Checkpoint loads without error.

- [x] **Step 5: Push checkpoints to HF Hub**

If `HF_TOKEN` environment variable is set, the training script pushes automatically. Otherwise:

```bash
huggingface-cli login
uv run python -c "
from unsloth import FastVisionModel
model = FastVisionModel.from_pretrained('data/checkpoints/smoke-v0/final')
model.push_to_hub('hungphongtrn/univi-phase0-smoke-model-v0', token=True)
"
```

Expected: Model checkpoint available at `https://huggingface.co/hungphongtrn/univi-phase0-smoke-model-v0`.

### Task 5: Integration validation

- [x] **Step 1: All training tests pass**

```bash
uv run pytest tests/test_training.py -v
```

Expected: PASS.

- [x] **Step 2: Loss curve trends downward (soft check)**

Verify the loss trends downward over the training run — the last logged loss should be lower than the first logged loss, but a strict monotonic assertion is too brittle (short runs may show fluctuations). Use a trend check:

```python
# Inspect trainer.state.log_history
first_loss = log_history[0]["loss"]
last_loss = log_history[-1]["loss"]
# Soft assertion — allow small noise but flag if loss increased
assert last_loss < first_loss * 1.05, (
    f"Loss did not trend down: first={first_loss:.3f}, last={last_loss:.3f}. "
    "Check learning rate, LoRA rank, or dataset quality."
)
```

- [x] **Step 3: Checkpoint reload produces output on a held-out batch**

Run the checkpoint through a forward pass on one example from the materialized dataset. Verify no shape errors, no NaN logits.

- [x] **Step 4: No CUDA OOM during entire run**

Verify `nvidia-smi` peak memory stayed within 12 GB. Record peak for Phase 3 capacity planning.

- [x] **Step 5: Document actual max_seq_length**

Record the working `max_seq_length` value. If 1024 worked, note it. If reduced to 512, note it and flag that Phase 3 (A100) should restore the higher value.

## Actual Smoke Results

- Date: 2026-07-09.
- Hardware: NVIDIA GeForce RTX 3060, 11.631 GB reported max memory, CUDA Toolkit 13.0, Torch `2.10.0+cu130`, Unsloth `2026.7.2`.
- Dataset: `data/materialized/smoke-v0`, 40 examples, 10 each from LibriSpeech, DenseFusion, FineWeb-Edu, and SmolTalk.
- Config: `max_seq_length=1024`, batch size 1, gradient accumulation 4, LoRA rank 16, LoRA alpha 32, `finetune_vision_layers=true`, 50 steps.
- Training result: completed 50/50 steps without CUDA OOM or trainer crash; final checkpoint saved to `data/checkpoints/smoke-v0/final/`.
- Loss trend: first logged loss `0.5781` at step 5; last logged loss `0.1559` at step 50; train loss `0.2806`.
- Checkpoint reload validation: `FastVisionModel.from_pretrained("data/checkpoints/smoke-v0/final", load_in_4bit=True)` succeeded; one materialized example produced finite logits with shape `(1, 668, 262144)` and loss `0.4459`.
- Memory: exact training peak was not captured in the original run; reload-forward validation peak allocation was `8.854 GiB`. `train_smoke.py` now prints CUDA peak allocated/reserved memory for future reruns.
- Tests: `uv run pytest tests/test_training.py` passed (`4 passed`).
- Hub checkpoint: uploaded to `https://huggingface.co/hungphongtrn/univi-phase0-smoke-model-v0`, commit `e8cc7a25286bd467a5e96ba0e9b07cdae1226b8f`.
- Known warnings: Unsloth fell back to a pre-forward hook for `audio_tower`; Gemma4 default image size was absent so Unsloth used 512; tokenizer BOS aligned to tokenizer value. None blocked training, checkpoint save, reload, or forward validation.

## Phase Completion Criteria

- [x] Training dependencies added and importable (`FastVisionModel`, `SFTTrainer`, `UnslothVisionDataCollator`).
- [x] `configs/smoke.yaml` defines all hyperparameters with documented rationale; uses `finetune_vision_layers: true` (not `vision_encoder` in target_modules); omits `use_rslora`; separates dataset and model repo ids.
- [x] `train_smoke.py` loads the materialized dataset, builds model, applies `for_training`, applies LoRA with `finetune_vision_layers`, uses `SFTConfig` with `dataset_text_field=""` and `dataset_kwargs={"skip_prepare_dataset": True}`, runs training, saves checkpoint.
- [x] Smoke training completes on RTX 3060 without CUDA OOM or trainer crash.
- [x] Loss decreases over 50 steps (first loss > last loss with tolerance for noise).
- [x] Checkpoint saved, reloaded, and produces a valid forward pass.
- [x] Unit tests for dataset loading and schema pass on CPU; collation test uses `@requires_gpu` and passes on GPU (skipped on CPU).
- [x] Model checkpoint pushed to HF Hub (`hungphongtrn/univi-phase0-smoke-model-v0`); dataset remains at separate repo id (`hungphongtrn/univi-phase0-dataset-smoke-v0`).
- [x] Working `max_seq_length` and peak GPU memory documented for Phase 3 capacity planning.

## Handoff Notes for Phase 3

- The same materialized dataset (`data/materialized/smoke-v0` or HF Hub `hungphongtrn/univi-phase0-dataset-smoke-v0`) is used for Phase 3, scaled to larger sample counts. Dataset and model checkpoints live at separate repo ids.
- Use `max_seq_length=1024` as the confirmed smoke value. Treat `8.854 GiB` as the reload-forward peak allocation and rerun with the new `train_smoke.py` memory print if exact training peak is needed for A100 capacity planning.
- LoRA hyperparameters that worked for smoke (rank, alpha, target modules, finetune_vision_layers) are the starting point for Phase 3 tuning.
- Phase 3 runs the same `train_smoke.py` script (or a `train_full.py` variant) on the A100 with longer schedule, higher rank, and longer context.
- Phase 3 adds evaluation harnesses (`eval_image_only.py`, `eval_native_upper_bound.py`) — smoke does not evaluate held-out metrics.
- If Phase 1's materialized dataset has not been pushed to HF Hub yet, do it before Phase 2 starts so smoke can load from a stable remote URL.
- The `train_smoke.py` entry point design enables Phase 3 to reuse the same config format with a different YAML file (`configs/full.yaml`).
- Loss masking is handled automatically by `UnslothVisionDataCollator` — Phase 3 does not need explicit masking logic.

## Non-Goals (explicitly out of scope for Phase 2)

- No held-out evaluation metrics (accuracy, retention). Phase 3 handles this.
- No native upper-bound comparison. Phase 3 handles this.
- No shuffled-option or low-prior diagnostics. Phase 4 handles this.
- No Valor32k integration (deferred to issue #2).
- No model surgery (audio_tower stays). Phase 3+ may revisit.
- No hyperparameter search. Single fixed config.
- No Weights & Biases or TensorBoard reporting. Console logging only.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CUDA OOM at max_seq_length=1024 | High (12 GB + 27B param model) | Reduce to 512. Document working value. |
| Unsloth Gemma 4 notebook API drift | Medium | Pin `unsloth>=2025.3`. Re-check against notebook example. |
| `torchcodec` or `datasets` version conflict with training deps | Medium | Lock all deps with `uv lock`. Test import before training. |
| Materialized dataset not available locally | Medium | Rebuild with merge CLI, or push to HF Hub and configure fallback. |
| Training divergence in 50 steps | Low | Loss ~13-15 expected initially. Verify trend decreases (with noise tolerance). |
| `FastVisionModel` CPU load fails for unit tests | Medium | Mark GPU-only tests with `@requires_gpu`. Dataset tests remain CPU-safe. |

## Failure Criteria

- RTX 3060 cannot fit the model even at rank 16, max_seq_length=512, batch size 1, 4-bit QLoRA → Phase 2 is BLOCKED. Mitigation: switch to a smaller baseline model (Gemma 4 9B variant if available) or rent an A100 for smoke.
- Loss diverges (increases over 50 steps with no recovery) → Phase 2 is FAILED. Mitigation: check learning rate, LoRA rank, dataset quality, or gradient accumulation settings.
- Checkpoint save/reload produces different loss on the same batch → Phase 2 is FAILED. Mitigation: fix save/load path or PEFT adapter handling.
- Unit tests for dataset schema fail → Phase 2 is BLOCKED. Regenerate materialized dataset from Phase 1.
