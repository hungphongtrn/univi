from __future__ import annotations

import json
import tempfile

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.fineweb_edu import preprocess_fineweb_edu


class _MockFineWebEdu:
    def __init__(self, num_rows=2, rows=None):
        if rows is not None:
            self._rows = rows
        else:
            self._rows = [
                {
                    "text": f"Sample educational text number {i}. " * 20,
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
        ds = _MockFineWebEdu.__new__(_MockFineWebEdu)
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


def _mock_renderer(*args, **kwargs):
    return Image.new("RGB", (64, 64))


def test_fineweb_edu_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=2, max_chars=2000)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    assert len(row["messages"]) == 2
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert "transcribe" in user_content[1]["text"].lower()
    assistant_text = row["messages"][1]["content"][0]["text"]
    assert len(assistant_text) > 0


def test_fineweb_edu_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    row = dataset[0]
    assert row["source_dataset_id"] == "HuggingFaceFW/fineweb-edu"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row


def test_user_instruction_does_not_contain_raw_text(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=2, max_chars=2000)
    for row in dataset:
        user_text = row["messages"][0]["content"][1]["text"]
        raw_text = row["messages"][1]["content"][0]["text"]
        assert raw_text not in user_text


def test_render_config_is_valid_json_and_contains_expected_keys(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(
        max_samples=1, max_chars=2000, canvas_width=1024, font_size=14
    )
    config = json.loads(dataset[0]["render_config"])
    assert isinstance(config, dict)
    assert config["render_method"] == "text_page"
    assert config["max_chars"] == 2000
    assert config["canvas_width"] == 1024
    assert config["font_size"] == 14


def test_render_config_matches_renderer_kwargs(monkeypatch):
    recorded_kwargs: dict = {}

    def recording_renderer(*args, **kwargs):
        recorded_kwargs.clear()
        recorded_kwargs.update(kwargs)
        return Image.new("RGB", (64, 64))

    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        recording_renderer,
    )

    dataset = preprocess_fineweb_edu(
        max_samples=1, max_chars=500, canvas_width=800, font_size=16
    )

    row = dataset[0]
    render_config = json.loads(row["render_config"])
    assert render_config["canvas_width"] == recorded_kwargs.get("canvas_width")
    assert render_config["font_size"] == recorded_kwargs.get("font_size")


def test_row_id_is_string_with_numeric_source_id(monkeypatch):
    rows = [
        {
            "text": "Some educational content.",
            "id": 999,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "999"


def test_row_id_falls_back_to_index_when_no_id_column(monkeypatch):
    rows = [
        {
            "text": "Some educational content without id field.",
        }
    ]

    class _NoIdMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _NoIdMock.__new__(_NoIdMock)
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

    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _NoIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    assert dataset[0]["row_id"] == "0"


def test_text_truncation(monkeypatch):
    long_text = "A" * 5000

    class _LongTextMock:
        def __init__(self):
            self._rows = [{"text": long_text, "id": "0"}]

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _LongTextMock.__new__(_LongTextMock)
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

    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _LongTextMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=100)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert len(assistant_text) == 100


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=2, max_chars=2000)

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

    assert row["source_dataset_id"] == "HuggingFaceFW/fineweb-edu"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row


def test_fallback_to_content_key(monkeypatch):
    rows = [
        {
            "content": "Document content with educational material.",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert "Document content" in assistant_text


def test_fallback_to_document_key(monkeypatch):
    rows = [
        {
            "document": "Long document text here.",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    assert "document" in dataset[0]["messages"][1]["content"][0]["text"].lower()


def test_missing_text_key_raises_key_error(monkeypatch):
    rows = [
        {
            "irrelevant": "whatever",
            "id": "test-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    with pytest.raises(KeyError, match="text|content|document"):
        preprocess_fineweb_edu(max_samples=1, max_chars=2000)


def test_max_samples_none_loads_all(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(5),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=None, max_chars=2000)
    assert len(dataset) == 5


def test_max_samples_zero_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(5),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=0, max_chars=2000)
    assert len(dataset) == 0


def test_map_used_not_direct_iteration(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _MockFineWebEdu(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_fineweb_edu(max_samples=2, max_chars=2000)
    assert len(dataset) == 2
