from __future__ import annotations

import io
import json
import tempfile

import numpy as np
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.librispeech_asr import preprocess_librispeech_asr


class _MockLibriSpeech:
    def __init__(self, num_rows: int = 2, duration_sec: float = 5.0):
        self._rows = [
            {
                "audio": {
                    "array": np.zeros(int(16000 * duration_sec), dtype=np.float32),
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

    def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
        result_rows = []
        for i, row in enumerate(self._rows):
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)

    def filter(self, function, **kwargs):
        kept = [row for row in self._rows if function(row)]
        ds = _MockLibriSpeech.__new__(_MockLibriSpeech)
        ds._rows = kept
        return ds


def _mock_renderer(audio, **kwargs):
    duration_sec = np.asarray(audio).shape[-1] / kwargs["source_sample_rate"]
    n_pages = max(1, int(np.ceil(duration_sec / kwargs["page_duration_sec"])))
    return [
        Image.new("RGB", (kwargs["output_width"], kwargs["output_height"]))
        for _ in range(n_pages)
    ]


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
        max_samples=2, sample_rate=16000, n_mels=80
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

    dataset = preprocess_librispeech_asr(max_samples=1)
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

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i)
                else:
                    new_row = function(row)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

        def filter(self, function, **kwargs):
            kept = [row for row in self._rows if function(row)]
            ds = _NumericIdMock.__new__(_NumericIdMock)
            ds._rows = kept
            return ds

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _NumericIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=1)
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

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i)
                else:
                    new_row = function(row)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

        def filter(self, function, **kwargs):
            kept = [row for row in self._rows if function(row)]
            ds = _NullIdMock.__new__(_NullIdMock)
            ds._rows = kept
            return ds

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _NullIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=1)
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

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i)
                else:
                    new_row = function(row)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

        def filter(self, function, **kwargs):
            kept = [row for row in self._rows if function(row)]
            ds = _EmptyIdMock.__new__(_EmptyIdMock)
            ds._rows = kept
            return ds

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _EmptyIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=1)
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

    dataset = preprocess_librispeech_asr(max_samples=2)
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

    dataset = preprocess_librispeech_asr(max_samples=1)
    row = dataset[0]
    config = json.loads(row["render_config"])
    assert isinstance(config, dict)
    assert len(config) > 0


def test_render_config_matches_renderer_kwargs(monkeypatch):
    recorded_kwargs: dict = {}

    def recording_renderer(*args, **kwargs):
        recorded_kwargs.clear()
        recorded_kwargs.update(kwargs)
        return _mock_renderer(*args, **kwargs)

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        recording_renderer,
    )

    dataset = preprocess_librispeech_asr(
        max_samples=1, sample_rate=16000, n_mels=80
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

    dataset = preprocess_librispeech_asr(max_samples=2)

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


def test_preprocessing_version_updated(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=1)
    row = dataset[0]
    assert row["preprocessing_version"] == "0.5.0"


def test_original_token_length_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    class _CharTok:
        name_or_path = "char-tok"
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(chr(t) for t in tokens)

    dataset = preprocess_librispeech_asr(max_samples=2, tokenizer=_CharTok())
    for row in dataset:
        assert "original_token_length" in row
        assert isinstance(row["original_token_length"], int)
        assert row["original_token_length"] > 0


def test_original_token_length_minus_one_without_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=1)
    assert dataset[0]["original_token_length"] == -1


# --- Gemma 4 fixed-page LibriSpeech materialization ---


def test_short_audio_produces_one_image_and_placeholder(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(3),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=3)
    for row in dataset:
        assert len(row["images"]) == 1
        placeholders = sum(
            item["type"] == "image"
            for message in row["messages"]
            if message["role"] == "user"
            for item in message["content"]
        )
        assert placeholders == 1
        assert len(row["messages"][0]["content"]) == 2


def test_output_images_are_1000x160_rgb(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1, duration_sec=15.0),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    row = preprocess_librispeech_asr(max_samples=1)[0]
    assert len(row["images"]) == 2
    for image in row["images"]:
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        assert image.size == (1000, 160)
        assert image.mode == "RGB"


def test_short_audio_uses_fixed_ten_second_scale(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1, duration_sec=3.0),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    config = json.loads(preprocess_librispeech_asr(max_samples=1)[0]["render_config"])
    assert config["source_duration_ms"] == 3000.0
    assert config["page_duration_sec"] == 10.0
    assert config["ms_per_horizontal_pixel"] == 10.0
    assert config["right_padded_final_page"] is True


def test_renderer_receives_audio_array_without_temporary_wav(monkeypatch):
    recorded: dict = {}

    def recording_renderer(audio, **kwargs):
        recorded["audio"] = audio
        recorded["kwargs"] = kwargs
        return _mock_renderer(audio, **kwargs)

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1, duration_sec=7.0),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        recording_renderer,
    )

    row = preprocess_librispeech_asr(max_samples=1)[0]
    assert isinstance(recorded["audio"], np.ndarray)
    assert recorded["audio"].shape == (7 * 16000,)
    assert recorded["kwargs"]["source_sample_rate"] == 16000
    config = json.loads(row["render_config"])
    assert config["render_method"] == "fixed_10s_rectangular_pages"


def test_excludes_utterances_over_40_seconds(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(num_rows=2, duration_sec=40.01),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=2)
    assert len(dataset) == 0


def test_mixed_duration_filtering_retains_up_to_40_seconds(monkeypatch):
    class _MixedDurationMock:
        def __init__(self):
            self._rows = [
                {
                    "audio": {
                        "array": np.zeros(int(16000 * duration), dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "text": f"{row_id} transcript",
                    "id": row_id,
                }
                for row_id, duration in (
                    ("short-1", 5),
                    ("four-pages-1", 35),
                    ("boundary-1", 40),
                    ("toolong-1", 40.01),
                )
            ]

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            dataset = _MixedDurationMock.__new__(_MixedDurationMock)
            dataset._rows = [self._rows[i] for i in indices]
            return dataset

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            rows = [
                function(row, index) if with_indices else function(row)
                for index, row in enumerate(self._rows)
            ]
            return Dataset.from_list(rows)

        def filter(self, function, **kwargs):
            dataset = _MixedDurationMock.__new__(_MixedDurationMock)
            dataset._rows = [row for row in self._rows if function(row)]
            return dataset

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MixedDurationMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_librispeech_asr(max_samples=4)
    assert {row["row_id"] for row in dataset} == {
        "short-1",
        "four-pages-1",
        "boundary-1",
    }
    image_counts = {row["row_id"]: len(row["images"]) for row in dataset}
    assert image_counts == {
        "short-1": 1,
        "four-pages-1": 4,
        "boundary-1": 4,
    }


def test_render_config_records_fixed_page_policy(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockLibriSpeech(1, duration_sec=25.0),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        _mock_renderer,
    )

    config = json.loads(preprocess_librispeech_asr(max_samples=1)[0]["render_config"])
    assert config["image_width"] == 1000
    assert config["image_height"] == 160
    assert config["render_method"] == "fixed_10s_rectangular_pages"
    assert config["n_pages"] == 3
    assert config["page_duration_sec"] == 10.0
    assert config["max_pages"] == 4
    assert config["normalization"] == "whisper_log_mel"
