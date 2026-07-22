from __future__ import annotations

import io
import json
import tempfile

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.densefusion import (
    _SOURCE_DATASET_ID,
    preprocess_densefusion,
)


def _make_image_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _MockFineVision:
    def __init__(self, num_rows=2, rows=None):
        if rows is not None:
            self._rows = rows
        else:
            self._rows = [
                {
                    "images": [{"bytes": _make_image_bytes(Image.new("RGB", (32, 32), color=(100 + i * 50, 50, 200))), "path": None}],
                    "texts": [{"user": "What do you observe in this photograph?", "assistant": f"a photo of sample number {i}"}],
                    "source": "densefusion_1m",
                    "relevance_ratings": [5],
                    "relevance_min": 5,
                    "image_correspondence_ratings": [2],
                    "image_correspondence_min": 2,
                    "formatting_ratings": [4],
                    "formatting_min": 4,
                    "visual_dependency_ratings": [5],
                    "visual_dependency_min": 5,
                }
                for i in range(num_rows)
            ]

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockFineVision.__new__(_MockFineVision)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, batched=False, remove_columns=None, batch_size=1000, **kwargs):
        remove_set = set(remove_columns or [])
        if batched:
            batch = {col: [row.get(col) for row in self._rows] for col in self.column_names}
            indices = list(range(len(self._rows)))
            new_columns = function(batch, indices)
            result_rows = []
            for i in range(len(self._rows)):
                merged = {k: v for k, v in self._rows[i].items() if k not in remove_set}
                for key in new_columns:
                    merged[key] = new_columns[key][i]
                result_rows.append(merged)
            return Dataset.from_list(result_rows)
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            merged = {k: v for k, v in row.items() if k not in remove_set}
            merged.update(new_row)
            result_rows.append(merged)
        return Dataset.from_list(result_rows)


# --- Functional tests ---


def test_densefusion_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert "photograph" in user_content[1]["text"].lower()


def test_densefusion_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(1),
    )

    dataset = preprocess_densefusion(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "HuggingFaceM4/FineVision"
    assert "split" in row
    assert isinstance(row["row_id"], str)
    assert isinstance(row["render_config"], str)
    assert row["modality_label"] == "image-text"
    assert "preprocessing_version" in row


def test_render_config_is_valid_json_densefusion(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(1),
    )

    dataset = preprocess_densefusion(max_samples=1)
    config = json.loads(dataset[0]["render_config"])
    assert config["render_method"] == "preserve_source_image"


def test_row_id_is_index_based(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(3),
    )

    dataset = preprocess_densefusion(max_samples=3)
    assert dataset[0]["row_id"] == "0"
    assert dataset[1]["row_id"] == "1"
    assert dataset[2]["row_id"] == "2"


def test_row_id_respects_offset(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(10),
    )

    dataset = preprocess_densefusion(max_samples=2, offset=5)
    assert dataset[0]["row_id"] == "5"
    assert dataset[1]["row_id"] == "6"


def test_user_instruction_does_not_contain_description(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    for row in dataset:
        user_text = row["messages"][0]["content"][1]["text"]
        description = row["messages"][1]["content"][0]["text"]
        assert description not in user_text


def test_original_token_length_densefusion(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(2),
    )

    class _CharTok:
        name_or_path = "char-tok"
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(chr(t) for t in tokens)

    dataset = preprocess_densefusion(max_samples=2, tokenizer=_CharTok())
    for row in dataset:
        assert "original_token_length" in row
        assert isinstance(row["original_token_length"], int)
        assert row["original_token_length"] > 0


def test_original_token_length_minus_one_without_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(1),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert dataset[0]["original_token_length"] == -1


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(2),
    )

    dataset = preprocess_densefusion(max_samples=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_to_disk(tmpdir)
        loaded = load_from_disk(tmpdir)

    assert len(loaded) == 2
    row = loaded[0]

    assert "messages" in row
    assert len(row["messages"]) == 2
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"

    assert row["source_dataset_id"] == "HuggingFaceM4/FineVision"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "image-text"
    assert "preprocessing_version" in row


def test_subset_empty_no_name_param(monkeypatch):
    recorded_kwargs = {}

    def recording_load(*args, **kwargs):
        recorded_kwargs["args"] = args
        recorded_kwargs["kwargs"] = kwargs
        return _MockFineVision(1)

    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        recording_load,
    )

    preprocess_densefusion(subset="", max_samples=1)
    assert "name" not in recorded_kwargs["kwargs"]


def test_subset_none_no_name_param(monkeypatch):
    recorded_kwargs = {}

    def recording_load(*args, **kwargs):
        recorded_kwargs["args"] = args
        recorded_kwargs["kwargs"] = kwargs
        return _MockFineVision(1)

    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        recording_load,
    )

    preprocess_densefusion(subset=None, max_samples=1)  # type: ignore[arg-type]
    assert "name" not in recorded_kwargs["kwargs"]


def test_map_used_not_direct_iteration(monkeypatch):
    map_kwargs: dict = {}

    def recording_map(self, function, *, with_indices=False, batched=False, remove_columns=None, batch_size=1000, **kwargs):
        map_kwargs["with_indices"] = with_indices
        map_kwargs["remove_columns"] = remove_columns
        remove_set = set(remove_columns or [])
        if batched:
            batch = {col: [row.get(col) for row in self._rows] for col in self.column_names}
            indices = list(range(len(self._rows)))
            new_columns = function(batch, indices)
            result_rows = []
            for i in range(len(self._rows)):
                merged = {k: v for k, v in self._rows[i].items() if k not in remove_set}
                for key in new_columns:
                    merged[key] = new_columns[key][i]
                result_rows.append(merged)
            return Dataset.from_list(result_rows)
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            merged = {k: v for k, v in row.items() if k not in remove_set}
            merged.update(new_row)
            result_rows.append(merged)
        return Dataset.from_list(result_rows)

    monkeypatch.setattr(_MockFineVision, "map", recording_map)

    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    assert len(dataset) == 2
    assert map_kwargs.get("with_indices") is True


def test_max_samples_none(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(5),
    )

    dataset = preprocess_densefusion(max_samples=None)
    assert len(dataset) == 5


def test_densefusion_accepts_decoded_pil_images(monkeypatch):
    rows = [
        {
            "images": [Image.new("RGB", (16, 16), color=(20, 40, 60))],
            "texts": [{"user": "describe", "assistant": "a small color patch"}],
            "source": "densefusion_1m",
            "relevance_ratings": [],
            "relevance_min": 0,
            "image_correspondence_ratings": [],
            "image_correspondence_min": 0,
            "formatting_ratings": [],
            "formatting_min": 0,
            "visual_dependency_ratings": [],
            "visual_dependency_min": 0,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)

    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"][0]["text"] == "a small color patch"


def test_raises_on_empty_texts(monkeypatch):
    img = Image.new("RGB", (16, 16))
    rows = [
        {
            "images": [{"bytes": _make_image_bytes(img), "path": None}],
            "texts": [],
            "source": "densefusion_1m",
            "relevance_ratings": [],
            "relevance_min": 0,
            "image_correspondence_ratings": [],
            "image_correspondence_min": 0,
            "formatting_ratings": [],
            "formatting_min": 0,
            "visual_dependency_ratings": [],
            "visual_dependency_min": 0,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockFineVision(rows=rows),
    )

    with pytest.raises(ValueError, match="texts"):
        preprocess_densefusion(max_samples=1)
