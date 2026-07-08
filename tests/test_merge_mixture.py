from __future__ import annotations

import json
import tempfile
from argparse import ArgumentError

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.merge_mixture import merge_and_shuffle, main as cli_main


def _make_row(source_id: str, idx: int) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (8, 8)), "text": None},
                    {"type": "text", "image": None, "text": f"instruction {idx}"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "image": None, "text": f"response {idx}"},
                ],
            },
        ],
        "source_dataset_id": source_id,
        "split": "train",
        "row_id": f"{source_id}-{idx}",
        "render_config": json.dumps({"key": "val"}, sort_keys=True),
        "modality_label": "text",
        "preprocessing_version": "0.1.0",
    }


def _make_dataset(source_id: str, n: int) -> Dataset:
    return Dataset.from_list([_make_row(source_id, i) for i in range(n)])


def _make_mock_preprocessor(source_id: str):
    def _mock_preprocessor(max_samples=0, **kwargs):
        return _make_dataset(source_id, max_samples)

    return _mock_preprocessor


# --- merge_and_shuffle ---


def test_merge_mixture_concatenates_and_shuffles():
    sources = {
        "librispeech": _make_dataset("libri", 5),
        "densefusion": _make_dataset("dense", 5),
        "fineweb": _make_dataset("fineweb", 5),
        "smoltalk": _make_dataset("smoltalk", 5),
    }
    merged = merge_and_shuffle(sources, seed=42)
    assert len(merged) == 20
    assert "messages" in merged[0]
    assert "source_dataset_id" in merged[0]
    assert "split" in merged[0]
    assert "row_id" in merged[0]
    assert "render_config" in merged[0]
    assert "modality_label" in merged[0]
    assert "preprocessing_version" in merged[0]


def test_merge_mixture_preserves_metadata():
    sources = {"librispeech": _make_dataset("libri", 2)}
    merged = merge_and_shuffle(sources, seed=42)
    row = merged[0]
    assert "source_dataset_id" in row
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert "modality_label" in row
    assert "preprocessing_version" in row


def test_merge_mixture_all_six_fields_round_trip():
    ds = _make_dataset("test", 3)
    merged = merge_and_shuffle({"test": ds}, seed=0)
    for row in merged:
        assert set(row.keys()) == {
            "messages",
            "source_dataset_id",
            "split",
            "row_id",
            "render_config",
            "modality_label",
            "preprocessing_version",
        }


def test_merge_mixture_raises_on_all_empty():
    sources = {
        "a": _make_dataset("a", 0),
        "b": _make_dataset("b", 0),
    }
    with pytest.raises(ValueError, match="No non-empty sources"):
        merge_and_shuffle(sources, seed=42)


def test_merge_mixture_raises_on_empty_dict():
    with pytest.raises(ValueError, match="No non-empty sources"):
        merge_and_shuffle({}, seed=42)


def test_merge_mixture_ignores_zero_length_source():
    sources = {
        "empty": _make_dataset("empty", 0),
        "nonempty": _make_dataset("nonempty", 3),
    }
    merged = merge_and_shuffle(sources, seed=42)
    assert len(merged) == 3
    for row in merged:
        assert row["source_dataset_id"] == "nonempty"


def test_merge_mixture_zero_length_sources_only_raises():
    sources = {
        "a": _make_dataset("a", 0),
    }
    with pytest.raises(ValueError, match="No non-empty sources"):
        merge_and_shuffle(sources, seed=42)


def test_merge_mixture_deterministic_shuffle():
    ds = _make_dataset("test", 10)
    m1 = merge_and_shuffle({"test": ds}, seed=42)
    m2 = merge_and_shuffle({"test": ds}, seed=42)
    assert m1[0]["row_id"] == m2[0]["row_id"]
    assert m1[5]["row_id"] == m2[5]["row_id"]
    assert [r["row_id"] for r in m1] == [r["row_id"] for r in m2]


def test_merge_mixture_different_seed_changes_order():
    ds = _make_dataset("test", 10)
    m1 = merge_and_shuffle({"test": ds}, seed=42)
    m2 = merge_and_shuffle({"test": ds}, seed=99)
    order1 = [r["row_id"] for r in m1]
    order2 = [r["row_id"] for r in m2]
    assert order1 != order2


def test_merge_mixture_does_not_mutate_input():
    ds = _make_dataset("test", 5)
    original_rows = [dict(r) for r in ds]
    merge_and_shuffle({"test": ds}, seed=42)
    assert [dict(r) for r in ds] == original_rows


def test_merge_mixture_save_load_round_trip():
    sources = {
        "a": _make_dataset("a", 3),
        "b": _make_dataset("b", 3),
    }
    merged = merge_and_shuffle(sources, seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        merged.save_to_disk(tmpdir)
        loaded = load_from_disk(tmpdir)
    assert len(loaded) == 6
    for row in loaded:
        assert "messages" in row
        assert "source_dataset_id" in row
        assert "split" in row
        assert "row_id" in row
        assert "render_config" in row
        assert "modality_label" in row
        assert "preprocessing_version" in row


# --- CLI ---


def test_cli_with_two_per_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.merge_mixture._load_source",
        lambda name, samples: _make_dataset(name, samples),
    )
    outdir = tmp_path / "merged"
    cli_main([
        "--librispeech-samples", "2",
        "--densefusion-samples", "2",
        "--fineweb-samples", "2",
        "--smoltalk-samples", "2",
        "--output", str(outdir),
        "--seed", "42",
    ])
    assert outdir.is_dir()
    loaded = load_from_disk(str(outdir))
    assert len(loaded) == 8
    for row in loaded:
        assert "messages" in row


def test_cli_zero_per_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.merge_mixture._load_source",
        lambda name, samples: _make_dataset(name, samples),
    )
    outdir = tmp_path / "merged"
    with pytest.raises(ValueError, match="No non-empty sources"):
        cli_main([
            "--librispeech-samples", "0",
            "--output", str(outdir),
        ])


def test_cli_rejects_valor32k():
    with pytest.raises((ArgumentError, SystemExit)):
        cli_main([
            "--valor32k-samples", "2",
            "--output", "/tmp/ignored",
        ])


def test_cli_missing_output_raises():
    with pytest.raises(SystemExit):
        cli_main(["--librispeech-samples", "1"])


def test_cli_single_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.merge_mixture._load_source",
        lambda name, samples: _make_dataset(name, samples),
    )
    outdir = tmp_path / "single"
    cli_main([
        "--librispeech-samples", "3",
        "--output", str(outdir),
        "--seed", "0",
    ])
    loaded = load_from_disk(str(outdir))
    assert len(loaded) == 3


def test_cli_output_message(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.merge_mixture._load_source",
        lambda name, samples: _make_dataset(name, samples),
    )
    outdir = tmp_path / "msg"
    cli_main([
        "--librispeech-samples", "1",
        "--densefusion-samples", "1",
        "--output", str(outdir),
        "--seed", "7",
    ])
    captured = capsys.readouterr()
    assert str(outdir) in captured.out
    assert "2" in captured.out
