"""
Training component tests for Phase 1 training foundation.

Test taxonomy:
- Import-level and config tests are CPU-safe.
- Collation tests require CUDA GPU (@requires_gpu).
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
    if isinstance(value, dict) and "bytes" in value:
        return
    raise AssertionError(f"unexpected image payload type: {type(value)!r}")


# --- Import-level tests (CPU-safe) ---


def test_train_smoke_imports():
    """Verify that train_smoke.py's key imports resolve without CUDA."""
    import importlib

    mod = importlib.import_module("train_smoke")
    assert hasattr(mod, "main")


def test_univi_trainer_imports():
    """Verify that univi.trainer exposes all shared training helpers."""
    import importlib

    mod = importlib.import_module("univi.trainer")
    assert hasattr(mod, "train")
    assert hasattr(mod, "build_model")
    assert hasattr(mod, "apply_lora")
    assert hasattr(mod, "load_dataset")


def test_train_full_imports():
    """Verify that train_full.py resolves all imports."""
    import importlib

    mod = importlib.import_module("train_full")
    assert hasattr(mod, "main")


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


# --- New API tests ---


def test_resolve_config_via_univi():
    """resolve_config is importable from univi top-level."""
    import univi

    cfg = univi.resolve_config("configs/3060_1epoch.yaml")
    assert cfg["training"]["max_length"] == 8192
    assert "config_hash" in cfg


def test_new_trainer_api():
    """New trainer API includes expected function signatures."""
    from univi.trainer import (
        make_training_args,
        resolve_model_config,
        verify_checkpointing_config,
        detect_markers,
        check_active_labels,
    )

    cfg = {
        "training": {"max_length": 8192, "output_dir": "/tmp/test"},
        "model": {
            "name": "unsloth/gemma-4-E2B-it",
            "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
            "max_lora_rank": 8,
            "use_gradient_checkpointing": "unsloth",
        },
    }
    args = make_training_args(cfg)
    assert args.max_length == 8192

    resolved = resolve_model_config(cfg)
    assert resolved["revision"] == "4abfca14e6c6bfb5888b80288185b1243fb8d539"

    mode = verify_checkpointing_config(cfg)
    assert mode == "unsloth"


# --- Full-config tests ---


def test_full_config_loads():
    """Verify that configs/full.yaml loads and contains expected values."""
    from univi.config import resolve_config

    cfg = resolve_config("configs/full.yaml")

    assert cfg["model"]["name"] == "unsloth/gemma-4-E2B-it"
    assert cfg["model"]["load_in_4bit"] is True
    assert cfg["model"]["use_gradient_checkpointing"] == "unsloth"

    assert cfg["lora"]["r"] == 8
    assert cfg["lora"]["alpha"] == 16
    assert "q_proj" in cfg["lora"]["target_modules"]
    assert cfg["lora"]["finetune_vision_layers"] is True

    tr = cfg["training"]
    assert tr["max_length"] == 8192
    assert "max_seq_length" not in tr
    assert tr["learning_rate"] == 2.0e-4
    assert tr["output_dir"] == "data/checkpoints/full-v0"
    assert tr["report_to"] == ["wandb"]

    assert cfg["dataset"]["path"] == "data/materialized/univi-3M-v0-split"
    assert cfg["dataset"]["hf_hub_repo_id"] == "hungphongtrn/univi-3M-v0"
    assert cfg["dataset"]["subsets"] == [
        "fineweb-edu",
        "densefusion",
        "smoltalk",
        "librispeech",
    ]

    assert cfg["hub"]["model_repo_id"] == "hungphongtrn/univi-phase0-full-model-v0"
