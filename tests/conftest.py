"""
Shared fixtures and helpers for training foundation tests.

Owned by Task 1. Other tasks consume these but MUST NOT modify this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a minimal 2-subset dataset at tmp_path with a manifest.

    Each subset has 4 rows with proper message structure and metadata.
    Returns the tmp_path.
    """
    subsets = {
        "fineweb-edu": [
            {"messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is AI?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Artificial intelligence."}]},
            ], "source_dataset_id": "fineweb-edu", "split": "train", "row_id": f"fe-{i}",
              "render_config": "{}", "modality_label": "text", "preprocessing_version": "1.0.0"}
            for i in range(4)
        ],
        "densefusion": [
            {"messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A scenic view."}]},
            ], "source_dataset_id": "densefusion", "split": "train", "row_id": f"df-{i}",
              "render_config": "{}", "modality_label": "image-text", "preprocessing_version": "1.0.0"}
            for i in range(4)
        ],
    }

    manifest_entries = []
    for name, rows in subsets.items():
        subset_path = tmp_path / name / "train"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).save_to_disk(str(subset_path))
        manifest_entries.append({
            "config_name": name,
            "path": f"{name}/train",
            "split": "train",
            "rows": len(rows),
            "source_dataset_id": name,
            "is_training_split": True,
        })

    (tmp_path / "manifest.json").write_text(
        json.dumps({"subsets": manifest_entries}), encoding="utf-8"
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Helper functions (not fixtures)
# ---------------------------------------------------------------------------


def minimal_config(dataset_path: str = "") -> dict:
    """Return a minimal resolved config dict for dataset/trainer tests.

    This is a regular Python function so tests can call it directly.
    Includes 'config_hash' for W&B and manifest tests.
    """
    return {
        "model": {
            "name": "unsloth/gemma-4-E2B-it",
            "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
        },
        "dataset": {
            "path": dataset_path or "",
            "subsets": ["fineweb-edu", "densefusion"],
            "train_splits": {
                "fineweb-edu": ["train"],
                "densefusion": ["train"],
            },
            "shuffle_seed": 42,
        },
        "training": {"max_length": 2048},
        "config_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    }
