from __future__ import annotations

import json
import tempfile
import zipfile

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.densefusion import (
    _SOURCE_DATASET_ID,
    _find_key,
    _resolve_densefusion_image,
    _resolve_image,
    preprocess_densefusion,
)


class _MockDenseFusion:
    def __init__(self, num_rows=2, rows=None):
        if rows is not None:
            self._rows = rows
        else:
            self._rows = [
                {
                    "image": Image.new(
                        "RGB", (32, 32), color=(100 + i * 50, 50, 200)
                    ),
                    "description": f"a photo of sample number {i}",
                    "id": f"test-{i}",
                }
                for i in range(num_rows)
            ]

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockDenseFusion.__new__(_MockDenseFusion)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None):
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            result_rows.append(new_row)
        return Dataset.from_list(result_rows)


# --- _find_key unit tests ---


def test_find_key_finds_first_match():
    row = {"png": "x", "image": "y", "jpg": "z"}
    assert _find_key(row, ["image", "jpg", "png"], "image") == "image"


def test_find_key_falls_back():
    row = {"png": "x"}
    assert _find_key(row, ["image", "jpg", "png"], "image") == "png"


def test_find_key_raises_on_missing():
    row = {"other": "x"}
    with pytest.raises(KeyError, match="image"):
        _find_key(row, ["image", "jpg", "png"], "image")


# --- _resolve_image unit tests ---


def test_resolve_image_passthrough():
    img = Image.new("RGB", (16, 16))
    assert _resolve_image(img) is img


def test_resolve_image_from_path(tmp_path):
    path = str(tmp_path / "test.png")
    img = Image.new("RGB", (16, 16))
    img.save(path)
    result = _resolve_image(path)
    assert isinstance(result, Image.Image)
    assert result.size == (16, 16)


def test_resolve_image_raises_on_bad_path(monkeypatch):
    def mock_download(repo_id, repo_type, filename):
        raise ValueError("File not found in HF repo")

    monkeypatch.setattr(
        "data.preprocessing.densefusion.hf_hub_download",
        mock_download,
    )

    with pytest.raises(ValueError):
        _resolve_image("/nonexistent/image.png")


def test_resolve_image_raises_on_bad_type():
    with pytest.raises(TypeError, match="Expected PIL Image or file path"):
        _resolve_image(42)


def test_resolve_image_hf_repo_path(monkeypatch, tmp_path):
    img_path = str(tmp_path / "test.png")
    img = Image.new("RGB", (16, 16))
    img.save(img_path)

    def mock_download(repo_id, repo_type, filename):
        assert repo_id == _SOURCE_DATASET_ID
        assert repo_type == "dataset"
        assert filename == "images/DenseFusion-4V-100K/test.png"
        return img_path

    monkeypatch.setattr(
        "data.preprocessing.densefusion.hf_hub_download",
        mock_download,
    )

    result = _resolve_image("images/DenseFusion-4V-100K/test.png")
    assert isinstance(result, Image.Image)
    assert result.size == (16, 16)


def test_resolve_image_raises_on_url():
    with pytest.raises(ValueError, match="URL"):
        _resolve_image("https://example.com/image.jpg")
    with pytest.raises(ValueError, match="URL"):
        _resolve_image("hf://datasets/foo/bar")


def test_resolve_densefusion_image_from_zip_with_bare_member_name(
    monkeypatch, tmp_path
):
    img = Image.new("RGB", (16, 16), color=(10, 20, 30))
    image_path = "DenseFusion-4V-100K/000000/1000866019454.png"
    archive_path = tmp_path / "000000.zip"
    member_name = "1000866019454.png"

    with zipfile.ZipFile(archive_path, "w") as zf:
        image_file = tmp_path / member_name
        img.save(image_file)
        zf.write(image_file, member_name)

    def mock_download(repo_id, repo_type, filename):
        assert repo_id == _SOURCE_DATASET_ID
        assert repo_type == "dataset"
        assert filename == "images/DenseFusion-4V-100K/000000.zip"
        return archive_path

    monkeypatch.setattr(
        "data.preprocessing.densefusion.hf_hub_download",
        mock_download,
    )

    result = _resolve_densefusion_image(image_path)
    assert isinstance(result, Image.Image)
    assert result.size == (16, 16)


# --- Functional tests ---


def test_densefusion_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert "describe" in user_content[1]["text"].lower()


def test_densefusion_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(1),
    )

    dataset = preprocess_densefusion(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "BAAI/DenseFusion-1M"
    assert "split" in row
    assert isinstance(row["row_id"], str)
    assert isinstance(row["render_config"], str)
    assert row["modality_label"] == "image-text"
    assert "preprocessing_version" in row


def test_render_config_is_valid_json_densefusion(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(1),
    )

    dataset = preprocess_densefusion(max_samples=1)
    config = json.loads(dataset[0]["render_config"])
    assert config["render_method"] == "preserve_source_image"


def test_row_id_is_string_with_numeric_source_id_densefusion(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (32, 32), color=(100, 50, 200)),
            "description": "a description",
            "id": 999,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "999"


def test_row_id_falls_back_to_index_when_id_is_none(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (32, 32), color=(100, 50, 200)),
            "description": "a description with null id",
            "id": None,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_row_id_falls_back_to_index_when_id_is_empty(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (32, 32), color=(100, 50, 200)),
            "description": "a description with empty id",
            "id": "",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_user_instruction_does_not_contain_description(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    for row in dataset:
        user_text = row["messages"][0]["content"][1]["text"]
        description = row["messages"][1]["content"][0]["text"]
        assert description not in user_text


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(2),
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

    assert row["source_dataset_id"] == "BAAI/DenseFusion-1M"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "image-text"
    assert "preprocessing_version" in row


def test_fallback_to_caption_key(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (32, 32), color=(100, 50, 200)),
            "caption": "a cat sitting on a chair",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert "cat" in dataset[0]["messages"][1]["content"][0]["text"]


def test_fallback_to_jpg_key(monkeypatch):
    rows = [
        {
            "jpg": Image.new("RGB", (32, 32), color=(200, 100, 50)),
            "description": "a dog in the park",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    dataset = preprocess_densefusion(max_samples=1)
    assert dataset[0]["messages"][0]["content"][0]["type"] == "image"


def test_missing_text_key_raises_key_error(monkeypatch):
    rows = [
        {
            "image": Image.new("RGB", (32, 32), color=(100, 50, 200)),
            "irrelevant": "whatever",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    with pytest.raises(KeyError, match="description|caption|text|output"):
        preprocess_densefusion(max_samples=1)


def test_missing_image_key_raises_key_error(monkeypatch):
    rows = [
        {
            "description": "a description with no image",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(rows=rows),
    )

    with pytest.raises(KeyError, match="image|jpg|png"):
        preprocess_densefusion(max_samples=1)


def test_subset_empty_no_name_param(monkeypatch):
    recorded_kwargs = {}

    def recording_load(*args, **kwargs):
        recorded_kwargs["args"] = args
        recorded_kwargs["kwargs"] = kwargs
        return _MockDenseFusion(1)

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
        return _MockDenseFusion(1)

    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        recording_load,
    )

    preprocess_densefusion(subset=None, max_samples=1)  # type: ignore[arg-type]
    assert "name" not in recorded_kwargs["kwargs"]


def test_map_used_not_direct_iteration(monkeypatch):
    map_kwargs: dict = {}

    def recording_map(self, function, *, with_indices=False, remove_columns=None):
        map_kwargs["with_indices"] = with_indices
        map_kwargs["remove_columns"] = remove_columns
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            result_rows.append(new_row)
        return Dataset.from_list(result_rows)

    monkeypatch.setattr(_MockDenseFusion, "map", recording_map)

    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(2),
    )

    dataset = preprocess_densefusion(max_samples=2)
    assert len(dataset) == 2
    assert map_kwargs.get("with_indices") is True
    assert map_kwargs.get("remove_columns") == ["image", "description", "id"]


def test_max_samples_none(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _MockDenseFusion(5),
    )

    dataset = preprocess_densefusion(max_samples=None)
    assert len(dataset) == 5
