from __future__ import annotations

import json
import tempfile

import numpy as np
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.librispeech_asr import preprocess_librispeech_asr


class _MockLibriSpeech:
    def __init__(self, num_rows: int = 2):
        self._rows = [
            {
                "audio": {
                    "array": np.zeros(16000 * 5, dtype=np.float32),
                    "sampling_rate": 16000,
                },
                "text": f"transcript number {i}",
                "id": f"test-{i}",
            }
            for i in range(num_rows)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockLibriSpeech.__new__(_MockLibriSpeech)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None, fn_kwargs=None):
        result_rows = []
        kwargs = fn_kwargs or {}
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i, **kwargs)
            else:
                new_row = function(row, **kwargs)
            result_rows.append(new_row)
        return Dataset.from_list(result_rows)


def _mock_renderer(*args, **kwargs):
    return Image.new("RGB", (64, 64))


def test_librispeech_asr_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(
        subset="clean-100", max_samples=2, sample_rate=16000, n_mels=80
    )
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    assert len(row["messages"]) == 2
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert "transcribe" in user_content[1]["text"].lower()
    assert row["messages"][1]["role"] == "assistant"
    assert len(row["messages"][1]["content"][0]["text"]) > 0


def test_librispeech_asr_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "openslr/librispeech_asr"
    assert "split" in row
    assert isinstance(row["row_id"], str)
    assert isinstance(row["render_config"], str)
    assert row["modality_label"] == "audio"
    assert "preprocessing_version" in row


def test_row_id_is_string_with_numeric_source_id(monkeypatch):
    rows = [
        {
            "audio": {
                "array": np.zeros(16000 * 5, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "text": "transcript",
            "id": 999,
        }
    ]

    class _NumericIdMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _NumericIdMock.__new__(_NumericIdMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self, function, *, with_indices=False, remove_columns=None, fn_kwargs=None
        ):
            result_rows = []
            kwargs = fn_kwargs or {}
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i, **kwargs)
                else:
                    new_row = function(row, **kwargs)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _NumericIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "999"


def test_row_id_falls_back_to_index_when_id_is_none(monkeypatch):
    rows = [
        {
            "audio": {
                "array": np.zeros(16000 * 5, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "text": "transcript with null id",
            "id": None,
        }
    ]

    class _NullIdMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _NullIdMock.__new__(_NullIdMock)
            ds._rows = [self._rows[i] for i in indices]

            return ds

        def map(
            self, function, *, with_indices=False, remove_columns=None, fn_kwargs=None
        ):
            result_rows = []
            kwargs = fn_kwargs or {}
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i, **kwargs)
                else:
                    new_row = function(row, **kwargs)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _NullIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_row_id_falls_back_to_index_when_id_is_empty(monkeypatch):
    rows = [
        {
            "audio": {
                "array": np.zeros(16000 * 5, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "text": "transcript with empty id",
            "id": "",
        }
    ]

    class _EmptyIdMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _EmptyIdMock.__new__(_EmptyIdMock)
            ds._rows = [self._rows[i] for i in indices]

            return ds

        def map(
            self, function, *, with_indices=False, remove_columns=None, fn_kwargs=None
        ):
            result_rows = []
            kwargs = fn_kwargs or {}
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i, **kwargs)
                else:
                    new_row = function(row, **kwargs)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _EmptyIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_user_instruction_does_not_contain_transcript(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=2)
    for row in dataset:
        user_text = row["messages"][0]["content"][1]["text"]
        transcript = row["messages"][1]["content"][0]["text"]
        assert transcript not in user_text


def test_render_config_is_valid_json(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    row = dataset[0]
    config = json.loads(row["render_config"])
    assert isinstance(config, dict)
    assert len(config) > 0


def test_render_config_matches_renderer_kwargs(monkeypatch):
    recorded_kwargs: dict = {}

    def recording_renderer(*args, **kwargs):
        recorded_kwargs.clear()
        recorded_kwargs.update(kwargs)
        return Image.new("RGB", (64, 64))

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        recording_renderer,
    )

    dataset = preprocess_librispeech_asr(
        subset="clean-100", max_samples=1, sample_rate=16000, n_mels=80
    )

    row = dataset[0]
    render_config = json.loads(row["render_config"])
    for k, v in recorded_kwargs.items():
        assert k in render_config, f"render_config missing key {k!r}"
        assert render_config[k] == v, (
            f"render_config[{k!r}] = {render_config[k]!r} != recorded {v!r}"
        )


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=2)

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

    assert row["source_dataset_id"] == "openslr/librispeech_asr"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "audio"
    assert "preprocessing_version" in row
