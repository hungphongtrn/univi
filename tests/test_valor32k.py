from __future__ import annotations

import json
import tempfile

import numpy as np
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.valor32k import preprocess_valor32k


class _MockValor32k:
    def __init__(self, rows: list[dict] | None = None):
        self._rows = rows or [
            {
                "id": 1,
                "question": "What color is the car?",
                "options": ["Red", "Blue", "Green", "Yellow"],
                "correct_answer_idx": 0,
                "modality": "visual",
                "audio": None,
                "frames": [Image.new("RGB", (32, 32)) for _ in range(4)],
            },
            {
                "id": 2,
                "question": "What sound is heard?",
                "options": ["Honking", "Music", "Speech", "Silence"],
                "correct_answer_idx": 2,
                "modality": "audio",
                "audio": {
                    "array": np.zeros(16000 * 5, dtype=np.float32),
                    "sampling_rate": 16000,
                },
                "frames": None,
            },
            {
                "id": 3,
                "question": "What is happening in the video?",
                "options": ["Running", "Jumping", "Sitting", "Walking"],
                "correct_answer_idx": 3,
                "modality": "audio-visual",
                "audio": {
                    "array": np.zeros(16000 * 3, dtype=np.float32),
                    "sampling_rate": 16000,
                },
                "frames": [Image.new("RGB", (32, 32)) for _ in range(4)],
            },
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockValor32k.__new__(_MockValor32k)
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


def _mock_renderer(*args, **kwargs):
    return Image.new("RGB", (64, 64))


def test_valor32k_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockValor32k(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=3)
    assert len(dataset) == 3
    for row in dataset:
        assert "messages" in row
        user_content = row["messages"][0]["content"]
        assert user_content[0]["type"] == "image"
        assert user_content[-1]["type"] == "text"
        assert "answer" in user_content[-1]["text"].lower()


def test_valor32k_image_count_by_modality(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockValor32k(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=10)
    for row in dataset:
        modality = row.get("modality_label")
        images = [
            c for c in row["messages"][0]["content"] if c["type"] == "image"
        ]
        if modality == "visual":
            assert len(images) == 5, f"visual: expected 5, got {len(images)}"
        elif modality == "audio":
            assert len(images) == 2, f"audio: expected 2, got {len(images)}"
        elif modality == "audio-visual":
            assert len(images) == 6, (
                f"audio-visual: expected 6, got {len(images)}"
            )


def test_valor32k_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockValor32k(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "inesriahi/valor32k-avqa-v2"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] in ("visual", "audio", "audio-visual")
    assert "preprocessing_version" in row


def test_generic_instruction_has_no_native_content(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "What color is the car?",
            "options": ["Red", "Blue", "Green", "Yellow"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(4)],
        },
    ]

    class _SingleMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _SingleMock.__new__(_SingleMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _SingleMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    user_text = dataset[0]["messages"][0]["content"][-1]["text"]
    assert "What color" not in user_text
    assert "Red" not in user_text
    assert "car" not in user_text
    assert "A)" not in user_text


def test_render_config_is_valid_json(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockValor32k(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    config = json.loads(dataset[0]["render_config"])
    assert isinstance(config, dict)
    assert config["render_method"] == "visual_bundle"
    assert "canvas_width" in config
    assert "font_size" in config
    assert "frame_count" in config
    assert "spectrogram" in config


def test_row_id_numeric_fallback(monkeypatch):
    rows_with_none_id = [
        {
            "id": None,
            "question": "Test?",
            "options": ["A1", "B1", "C1", "D1"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
        },
        {
            "id": "",
            "question": "Test2?",
            "options": ["A2", "B2", "C2", "D2"],
            "correct_answer_idx": 1,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
        },
    ]

    class _IdMock:
        def __init__(self):
            self._rows = rows_with_none_id

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _IdMock.__new__(_IdMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _IdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=2)
    assert dataset[0]["row_id"] == "0"
    assert dataset[1]["row_id"] == "1"


def test_row_id_numeric_source_id(monkeypatch):
    rows = [
        {
            "id": 999,
            "question": "Test?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
        },
    ]

    class _MockWithId:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MockWithId.__new__(_MockWithId)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockWithId(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    assert dataset[0]["row_id"] == "999"


def test_map_call_args(monkeypatch):
    captured_kwargs = {}

    class _CaptureMock:
        def __init__(self):
            self._rows = [
                {
                    "id": 1,
                    "question": "Test?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer_idx": 0,
                    "modality": "visual",
                    "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
                },
            ]

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _CaptureMock.__new__(_CaptureMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
        ):
            captured_kwargs.clear()
            captured_kwargs.update(
                with_indices=with_indices, remove_columns=remove_columns
            )
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _CaptureMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    preprocess_valor32k(max_samples=1)
    assert captured_kwargs["with_indices"] is True
    assert captured_kwargs["remove_columns"] is not None


def test_modality_normalization(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "V?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(4)],
        },
        {
            "id": 2,
            "question": "A?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 1,
            "modality": "audio",
            "audio": {
                "array": np.zeros(16000 * 3, dtype=np.float32),
                "sampling_rate": 16000,
            },
        },
        {
            "id": 3,
            "question": "AV?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 2,
            "modality": "audio-visual",
            "audio": {
                "array": np.zeros(16000 * 3, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "frames": [Image.new("RGB", (32, 32)) for _ in range(4)],
        },
    ]

    class _ModMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _ModMock.__new__(_ModMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _ModMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=3)
    labels = [r["modality_label"] for r in dataset]
    assert labels == ["visual", "audio", "audio-visual"]


def test_answer_normalization(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(1)],
        },
    ]

    class _AnsMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _AnsMock.__new__(_AnsMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _AnsMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    answer_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert answer_text == "A"


def test_frame_padding(monkeypatch):
    single_frame = Image.new("RGB", (32, 32))
    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": [single_frame],
        },
    ]

    class _PadMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _PadMock.__new__(_PadMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _PadMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1, frame_count=4)
    images = [
        c for c in dataset[0]["messages"][0]["content"] if c["type"] == "image"
    ]
    assert len(images) == 5


def test_frame_truncation(monkeypatch):
    frames = [Image.new("RGB", (32, 32), color=c) for c in ["red", "green", "blue"]]
    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "frames": frames,
        },
    ]

    class _TruncMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _TruncMock.__new__(_TruncMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _TruncMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1, frame_count=2)
    images = [
        c for c in dataset[0]["messages"][0]["content"] if c["type"] == "image"
    ]
    assert len(images) == 3


def test_temp_wav_for_audio_dict(monkeypatch):
    recorded_paths = []

    def _recording_spectrogram(audio_path, **kwargs):
        recorded_paths.append(audio_path)
        return Image.new("RGB", (64, 64))

    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "audio",
            "audio": {
                "array": np.zeros(16000 * 3, dtype=np.float32),
                "sampling_rate": 16000,
            },
        },
    ]

    class _AudioMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _AudioMock.__new__(_AudioMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _AudioMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _recording_spectrogram,
    )

    preprocess_valor32k(max_samples=1)
    assert len(recorded_paths) == 1
    assert recorded_paths[0].endswith(".wav")


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MockValor32k(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_to_disk(tmpdir)
        loaded = load_from_disk(tmpdir)

    assert len(loaded) == 3
    row = loaded[0]

    assert "messages" in row
    assert len(row["messages"]) == 2
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[-1]["type"] == "text"

    assert row["source_dataset_id"] == "inesriahi/valor32k-avqa-v2"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] in ("visual", "audio", "audio-visual")
    assert "preprocessing_version" in row


def test_question_query_prompt_keys(monkeypatch):
    rows = [
        {
            "id": 1,
            "query": "What is the capital of France?",
            "options": ["Paris", "London", "Berlin", "Madrid"],
            "correct_answer_idx": 0,
            "modality_label": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
        },
    ]

    class _QueryMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _QueryMock.__new__(_QueryMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _QueryMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    assert len(dataset) == 1
    assert dataset[0]["modality_label"] == "visual"


def test_modality_inference_from_audio_and_frames(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "audio": {
                "array": np.zeros(16000 * 3, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "frames": [Image.new("RGB", (32, 32)) for _ in range(2)],
        },
    ]

    class _InfMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _InfMock.__new__(_InfMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _InfMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    assert dataset[0]["modality_label"] == "audio-visual"


def test_option_a_b_c_d_keys(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Best color?",
            "option_a": "Red",
            "option_b": "Blue",
            "option_c": "Green",
            "option_d": "Yellow",
            "correct_answer_idx": 2,
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(1)],
        },
    ]

    class _OptMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _OptMock.__new__(_OptMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _OptMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    assert dataset[0]["messages"][1]["content"][0]["text"] == "C"


def test_answer_preserves_original_string_when_not_normalizable(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": -1,
            "answer": "the quick brown fox",
            "modality": "visual",
            "frames": [Image.new("RGB", (32, 32)) for _ in range(1)],
        },
    ]

    class _PreserveMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _PreserveMock.__new__(_PreserveMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _PreserveMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    answer_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert answer_text == "the quick brown fox"


def test_missing_audio_for_audio_row_raises(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Sound?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "audio",
            "video_id": "test-video-001",
        },
    ]

    class _MissingAudioMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MissingAudioMock.__new__(_MissingAudioMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MissingAudioMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    import pytest
    with pytest.raises(RuntimeError) as exc:
        preprocess_valor32k(max_samples=1)
    assert "video_id" in str(exc.value)
    assert "audio" in str(exc.value)


def test_missing_frames_for_visual_row_raises(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Visual?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "visual",
            "video_id": "test-video-002",
        },
    ]

    class _MissingFramesMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MissingFramesMock.__new__(_MissingFramesMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _MissingFramesMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )

    import pytest
    with pytest.raises(RuntimeError) as exc:
        preprocess_valor32k(max_samples=1)
    assert "video_id" in str(exc.value)
    assert "visual" in str(exc.value)


def test_no_frames_for_pure_audio(monkeypatch):
    rows = [
        {
            "id": 1,
            "question": "Sound?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_idx": 0,
            "modality": "audio",
            "audio": {
                "array": np.zeros(16000 * 3, dtype=np.float32),
                "sampling_rate": 16000,
            },
        },
    ]

    class _AudioOnlyMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _AudioOnlyMock.__new__(_AudioOnlyMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(
            self,
            function,
            *,
            with_indices=False,
            remove_columns=None,
            fn_kwargs=None,
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
        "data.preprocessing.valor32k.load_dataset",
        lambda *a, **kw: _AudioOnlyMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_text_page",
        _mock_renderer,
    )
    monkeypatch.setattr(
        "data.preprocessing.valor32k.render_log_mel_spectrogram",
        _mock_renderer,
    )

    dataset = preprocess_valor32k(max_samples=1)
    images = [
        c for c in dataset[0]["messages"][0]["content"] if c["type"] == "image"
    ]
    assert len(images) == 2
