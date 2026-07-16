"""
Dataset loader tests for univi.trainer — subset validation, local/Hub loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset


def test_load_dataset_local_detects_manifest(synthetic_dataset: Path):
    """Local loading uses manifest.json when present."""
    from univi.trainer import load_dataset

    cfg = {
        "dataset": {
            "path": str(synthetic_dataset),
            "subsets": ["fineweb-edu", "densefusion"],
            "train_splits": {"fineweb-edu": ["train"], "densefusion": ["train"]},
            "shuffle_seed": 42,
        }
    }
    ds = load_dataset(cfg, source="auto")
    assert len(ds) > 0
    assert "source_dataset_id" in ds[0]


def test_load_dataset_local_fallback(tmp_path: Path):
    """Local loading falls back to load_from_disk when no manifest."""
    from univi.trainer import load_dataset

    rows = [{"text": "hello world"}]
    Dataset.from_list(rows).save_to_disk(str(tmp_path))

    cfg = {"dataset": {"path": str(tmp_path)}}
    ds = load_dataset(cfg, source="local")
    assert len(ds) == 1


def test_load_dataset_rejects_invalid_subset():
    """Unknown subset names raise ValueError."""
    from univi.trainer import load_dataset

    cfg = {
        "dataset": {
            "path": "/tmp/nonexistent",
            "subsets": ["nonexistent-subset"],
        }
    }
    with pytest.raises(ValueError, match="Unknown subset"):
        load_dataset(cfg, source="local")


def test_load_dataset_uses_configured_training_splits(tmp_path: Path, monkeypatch):
    """Hub loading respects training split configuration."""
    from univi import trainer

    calls = []

    def fake_load(repo, name, split, **kwargs):
        calls.append((repo, name, split))
        return Dataset.from_list([{"split": split}])

    monkeypatch.setattr(trainer, "hf_load", fake_load)

    cfg = {
        "dataset": {
            "path": str(tmp_path / "missing"),
            "hf_hub_repo_id": "hungphongtrn/univi-3M-v0",
            "subsets": ["librispeech", "smoltalk"],
            "train_splits": {"librispeech": ["train"], "smoltalk": ["train"]},
        }
    }
    ds = trainer.load_dataset(cfg, source="hub")
    assert len(ds) == 2


def test_load_dataset_rejects_revision_mismatch_on_resume():
    """Resume with changed dataset revision is rejected."""
    from univi.trainer import load_dataset, RevisionMismatchError

    with pytest.raises(RevisionMismatchError, match="revision"):
        load_dataset(
            {
                "dataset": {
                    "path": "/tmp/test",
                    "revision": "abc123",
                    "_expected_revision": "def456",
                }
            },
            source="local",
        )


def test_load_dataset_ignores_non_training_splits(tmp_path: Path):
    """Local loading ignores held-out test splits (uses valid subset name)."""
    from univi.trainer import load_dataset

    entries = []
    for split, is_training in (("train", True), ("test", False)):
        path = tmp_path / "fineweb-edu" / split
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list([{"split": split}]).save_to_disk(str(path))
        entries.append(
            {
                "config_name": "fineweb-edu",
                "split": split,
                "path": f"fineweb-edu/{split}",
                "is_training_split": is_training,
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"subsets": entries}), encoding="utf-8"
    )

    cfg = {
        "dataset": {
            "path": str(tmp_path),
            "subsets": ["fineweb-edu"],
            "shuffle_seed": 42,
        }
    }
    ds = load_dataset(cfg, source="auto")
    assert all(s == "train" for s in ds["split"])


def test_load_dataset_filters_training_rows_with_four_or_more_images(
    tmp_path: Path,
):
    """The local training path retains only rows with at most three images."""
    from univi.trainer import load_dataset

    split_path = tmp_path / "fineweb-edu" / "train"
    split_path.parent.mkdir(parents=True)
    Dataset.from_list(
        [
            {"row_id": "one", "images": [b"1"]},
            {"row_id": "three", "images": [b"1", b"2", b"3"]},
            {"row_id": "four", "images": [b"1", b"2", b"3", b"4"]},
            {"row_id": "seven", "images": [b"1"] * 7},
        ]
    ).save_to_disk(str(split_path))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "subsets": [
                    {
                        "config_name": "fineweb-edu",
                        "split": "train",
                        "path": "fineweb-edu/train",
                        "is_training_split": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "dataset": {
            "path": str(tmp_path),
            "subsets": ["fineweb-edu"],
            "max_train_images": 3,
        }
    }

    loaded = load_dataset(cfg, source="local")

    assert sorted(loaded["row_id"]) == ["one", "three"]


def test_load_dataset_applies_image_filter_to_hub_training_rows(
    tmp_path: Path, monkeypatch,
):
    """Remote and local training sources enforce the same image ceiling."""
    from univi import trainer

    remote = Dataset.from_list(
        [
            {"row_id": "keep", "images": [b"1", b"2", b"3"]},
            {"row_id": "drop", "images": [b"1", b"2", b"3", b"4"]},
        ]
    )
    monkeypatch.setattr(trainer, "hf_load", lambda *args, **kwargs: remote)
    cfg = {
        "dataset": {
            "path": str(tmp_path / "missing"),
            "hf_hub_repo_id": "hungphongtrn/univi-3M-v0",
            "subsets": ["fineweb-edu"],
            "train_splits": {"fineweb-edu": ["train"]},
            "max_train_images": 3,
        }
    }

    loaded = trainer.load_dataset(cfg, source="hub")

    assert loaded["row_id"] == ["keep"]
