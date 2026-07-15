"""
Training component tests for Phase 2 smoke training.

Test taxonomy:
- Dataset-loading and schema tests are CPU-safe (no GPU required).
- Collation tests require CUDA GPU (@requires_gpu) because
  FastVisionModel.from_pretrained for E2B needs CUDA.
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import json

import pytest
import torch
from datasets import Dataset
from PIL import Image


# --- GPU detection ---


def has_gpu() -> bool:
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="requires CUDA GPU")


# --- Helpers ---


def _materialized_path() -> Path:
    return Path("data/materialized/smoke-v0")


def _dataset_exists() -> bool:
    return _materialized_path().is_dir()


def _skip_if_no_dataset():
    if not _dataset_exists():
        pytest.skip("materialized dataset not found at data/materialized/smoke-v0")


def _assert_image_payload(value):
    if isinstance(value, Image.Image):
        return
    if isinstance(value, dict) and value.get("bytes"):
        with Image.open(BytesIO(value["bytes"])) as image:
            image.verify()
        return
    raise AssertionError(f"unexpected image payload type: {type(value)!r}")


# --- Dataset-loading tests (CPU-safe) ---


def test_training_dataset_loads():
    """Verify that the training script can load the materialized dataset."""
    _skip_if_no_dataset()
    from datasets import load_from_disk

    ds = load_from_disk(str(_materialized_path()))
    assert len(ds) > 0
    assert "messages" in ds[0]
    assert len(ds[0]["messages"]) == 2  # user + assistant
    user_content = ds[0]["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[-1]["type"] == "text"


def test_training_schema_conforms_to_fastvisionmodel():
    """Verify messages schema matches FastVisionModel expectations."""
    _skip_if_no_dataset()
    from datasets import load_from_disk

    ds = load_from_disk(str(_materialized_path()))
    for row in ds:
        msgs = row["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        for c in msgs[0]["content"]:
            if c["type"] == "image":
                _assert_image_payload(c["image"])


# --- Collation test (GPU required) ---


@requires_gpu
def test_unsloth_vision_data_collator_outputs():
    """Verify collator produces expected batch structure from one example."""
    _skip_if_no_dataset()
    from datasets import load_from_disk
    from unsloth import FastVisionModel, UnslothVisionDataCollator

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/gemma-4-E2B-it",
        load_in_4bit=True,
    )
    collator = UnslothVisionDataCollator(model, tokenizer)

    ds = load_from_disk(str(_materialized_path()))
    batch = [ds[i] for i in range(min(2, len(ds)))]

    result = collator(batch)

    assert "input_ids" in result
    assert "labels" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].shape[0] == len(batch)


# --- Import-level tests (CPU-safe) ---


def test_train_smoke_imports():
    """Verify that train_smoke.py's key imports resolve without CUDA."""
    import importlib

    mod = importlib.import_module("train_smoke")
    assert hasattr(mod, "load_config")
    assert hasattr(mod, "build_model")
    assert hasattr(mod, "load_dataset")
    assert hasattr(mod, "train")


def test_univi_trainer_imports():
    """Verify that univi.trainer exposes all shared training helpers."""
    import importlib

    mod = importlib.import_module("univi.trainer")
    assert hasattr(mod, "load_config")
    assert hasattr(mod, "build_model")
    assert hasattr(mod, "apply_lora")
    assert hasattr(mod, "load_dataset")
    assert hasattr(mod, "train")


def test_load_dataset_combines_local_source_configurations(tmp_path):
    from univi.trainer import load_dataset

    rows = []
    for name in ("fineweb-edu", "smoltalk"):
        subset_path = tmp_path / name / "train"
        subset_path.parent.mkdir()
        Dataset.from_list([{"source_dataset_id": name, "value": name}]).save_to_disk(
            subset_path
        )
        rows.append(
            {
                "config_name": name,
                "path": name,
                "split": "train",
                "path": f"{name}/train",
                "rows": 1,
                "source_dataset_id": name,
                "is_training_split": True,
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"subsets": rows}), encoding="utf-8"
    )

    dataset = load_dataset(
        {
            "dataset": {
                "path": str(tmp_path),
                "subsets": ["fineweb-edu", "smoltalk"],
                "shuffle_seed": 42,
            }
        }
    )

    assert len(dataset) == 2
    assert set(dataset["source_dataset_id"]) == {"fineweb-edu", "smoltalk"}


def test_load_dataset_ignores_local_held_out_splits(tmp_path):
    from univi.trainer import load_dataset

    entries = []
    for split, is_training in (("train", True), ("test", False)):
        path = tmp_path / "smoltalk" / split
        path.parent.mkdir(exist_ok=True)
        Dataset.from_list([{"split": split}]).save_to_disk(path)
        entries.append(
            {
                "config_name": "smoltalk",
                "split": split,
                "path": f"smoltalk/{split}",
                "is_training_split": is_training,
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"subsets": entries}), encoding="utf-8"
    )

    dataset = load_dataset(
        {"dataset": {"path": str(tmp_path), "subsets": ["smoltalk"]}}
    )

    assert dataset["split"] == ["train"]


def test_load_dataset_uses_configured_hub_training_splits(tmp_path, monkeypatch):
    from univi import trainer

    calls = []

    def fake_load(repo, name, split):
        calls.append((repo, name, split))
        return Dataset.from_list([{"split": split}])

    monkeypatch.setattr(trainer, "hf_load", fake_load)
    dataset = trainer.load_dataset(
        {
            "dataset": {
                "path": str(tmp_path / "missing"),
                "hf_hub_repo_id": "hungphongtrn/univi-3M-v0",
                "subsets": ["librispeech", "smoltalk"],
                "train_splits": {
                    "librispeech": ["train"],
                    "smoltalk": ["train"],
                },
            }
        }
    )

    assert len(dataset) == 2
    assert calls == [
        ("hungphongtrn/univi-3M-v0", "librispeech", "train"),
        ("hungphongtrn/univi-3M-v0", "smoltalk", "train"),
    ]


# --- Full-config tests ---


def test_full_config_loads():
    """Verify that configs/full.yaml loads and contains expected values."""
    from univi.trainer import load_config

    cfg = load_config("configs/full.yaml")

    assert cfg["model"]["name"] == "unsloth/gemma-4-E2B-it"
    assert cfg["model"]["load_in_4bit"] is True
    assert cfg["model"]["use_gradient_checkpointing"] is True

    assert cfg["lora"]["r"] == 32
    assert cfg["lora"]["alpha"] == 64
    assert "q_proj" in cfg["lora"]["target_modules"]
    assert "k_proj" in cfg["lora"]["target_modules"]
    assert "v_proj" in cfg["lora"]["target_modules"]
    assert "o_proj" in cfg["lora"]["target_modules"]
    assert "gate_proj" in cfg["lora"]["target_modules"]
    assert "up_proj" in cfg["lora"]["target_modules"]
    assert "down_proj" in cfg["lora"]["target_modules"]
    assert cfg["lora"]["finetune_vision_layers"] is True

    tr = cfg["training"]
    assert tr["per_device_train_batch_size"] == 2
    assert tr["gradient_accumulation_steps"] == 4
    assert tr["max_steps"] == 500
    assert tr["max_seq_length"] == 2048
    assert tr["learning_rate"] == 2.0e-4
    assert tr["warmup_steps"] == 20
    assert tr["lr_scheduler_type"] == "cosine"
    assert tr["optim"] == "adamw_8bit"
    assert tr["logging_steps"] == 10
    assert tr["save_steps"] == 100
    assert tr["output_dir"] == "data/checkpoints/full-v0"
    assert tr["report_to"] == "none"
    assert tr["remove_unused_columns"] is False
    assert tr["dataloader_num_workers"] == 2

    assert cfg["dataset"]["path"] == "data/materialized/univi-3M-v0"
    assert cfg["dataset"]["hf_hub_repo_id"] == "hungphongtrn/univi-3M-v0"
    assert cfg["dataset"]["subsets"] == [
        "fineweb-edu",
        "densefusion",
        "smoltalk",
        "librispeech",
    ]
    assert "librispeech" in cfg["dataset"]["train_splits"]
    assert cfg["dataset"]["train_splits"]["librispeech"] == ["train"]

    assert cfg["hub"]["model_repo_id"] == "hungphongtrn/univi-phase0-full-model-v0"


def test_train_full_imports():
    """Verify that train_full.py resolves all imports."""
    import importlib

    mod = importlib.import_module("train_full")
    assert hasattr(mod, "load_config")
    assert hasattr(mod, "train")
