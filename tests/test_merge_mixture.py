from __future__ import annotations

import json
import tempfile

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.densefusion import preprocess_densefusion
from data.preprocessing.fineweb_edu import preprocess_fineweb_edu
from data.preprocessing.librispeech_asr import preprocess_librispeech_asr
from data.preprocessing.merge_mixture import merge_and_shuffle, main as cli_main
from data.preprocessing.smoltalk import preprocess_smoltalk


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
        lambda name, samples, offset=0, **kwargs: _make_dataset(name, samples),
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
        lambda name, samples, offset=0, **kwargs: _make_dataset(name, samples),
    )
    outdir = tmp_path / "merged"
    with pytest.raises(ValueError, match="No non-empty sources"):
        cli_main([
            "--librispeech-samples", "0",
            "--output", str(outdir),
        ])


def test_cli_rejects_valor32k():
    with pytest.raises(SystemExit):
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
        lambda name, samples, offset=0, **kwargs: _make_dataset(name, samples),
    )
    outdir = tmp_path / "single"
    cli_main([
        "--librispeech-samples", "3",
        "--output", str(outdir),
        "--seed", "0",
    ])
    loaded = load_from_disk(str(outdir))
    assert len(loaded) == 3


def test_merge_mixture_real_preprocessors_with_mocks(tmp_path, monkeypatch):
    _mock_image = Image.new("RGB", (64, 64))

    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _LibriMock(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        lambda *a, **kw: _mock_image,
    )
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _DenseMock(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        lambda *a, **kw: _FineWebMock(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        lambda *a, **kw: _mock_image,
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _SmolTalkMock(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: _mock_image,
    )

    libri = preprocess_librispeech_asr(max_samples=1)
    dense = preprocess_densefusion(max_samples=1)
    fineweb = preprocess_fineweb_edu(max_samples=1)
    smoltalk = preprocess_smoltalk(max_samples=1)

    sources = {
        "librispeech": libri,
        "densefusion": dense,
        "fineweb": fineweb,
        "smoltalk": smoltalk,
    }
    merged = merge_and_shuffle(sources, seed=42)
    assert len(merged) == 4
    for row in merged:
        assert "messages" in row
        assert len(row["messages"]) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        merged.save_to_disk(tmpdir)
        loaded = load_from_disk(tmpdir)
    assert len(loaded) == 4
    for row in loaded:
        assert "messages" in row
        assert "source_dataset_id" in row
        assert "split" in row
        assert "row_id" in row
        assert "render_config" in row


class _LibriMock:
    def __init__(self, num_rows):
        self._rows = [
            {
                "audio": {"array": __import__("numpy").zeros(16000, dtype="float32"), "sampling_rate": 16000},
                "text": f"transcript {i}",
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
        ds = _LibriMock.__new__(_LibriMock)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
        result_rows = []
        for i, row in enumerate(self._rows):
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)

    def filter(self, function, **kwargs):
        dataset = _LibriMock.__new__(_LibriMock)
        dataset._rows = [row for row in self._rows if function(row)]
        return dataset


class _DenseMock:
    def __init__(self, num_rows):
        import io
        self._rows = []
        for i in range(num_rows):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color=(100, 50, 200)).save(buf, format="PNG")
            self._rows.append({
                "images": [{"bytes": buf.getvalue(), "path": None}],
                "texts": [{"user": "What do you observe?", "assistant": f"a photo of sample {i}"}],
                "source": "densefusion_1m",
                "relevance_ratings": [5],
                "relevance_min": 5,
                "image_correspondence_ratings": [2],
                "image_correspondence_min": 2,
                "formatting_ratings": [4],
                "formatting_min": 4,
                "visual_dependency_ratings": [5],
                "visual_dependency_min": 5,
            })

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _DenseMock.__new__(_DenseMock)
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
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)

class _FineWebMock:
    def __init__(self, num_rows):
        self._rows = [
            {"text": f"educational text {i} " * 30, "id": f"test-{i}"}
            for i in range(num_rows)
        ]

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _FineWebMock.__new__(_FineWebMock)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
        result_rows = []
        for i, row in enumerate(self._rows):
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)


class _SmolTalkMock:
    def __init__(self, num_rows):
        self._rows = [
            {
                "messages": [
                    {"role": "user", "content": f"Write about {i}."},
                    {"role": "assistant", "content": f"Here is text about {i}."},
                ],
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
        ds = _SmolTalkMock.__new__(_SmolTalkMock)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
        result_rows = []
        for i, row in enumerate(self._rows):
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)


def test_cli_output_message(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.merge_mixture._load_source",
        lambda name, samples, offset=0, **kwargs: _make_dataset(name, samples),
    )
    outdir = tmp_path / "msg"
    cli_main([
        "--librispeech-samples", "1",
        "--densefusion-samples", "1",
        "--output", str(outdir),
        "--seed", "7",
    ])
    captured = capsys.readouterr()
    assert f"saved to {outdir}" in captured.out
    assert "with 2 rows" in captured.out


# --- Offset selection ---


def test_preprocessor_offset_selects_different_rows(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    rows = [
        {"instruction": f"instr {i}", "response": f"resp {i}", "id": str(i)}
        for i in range(20)
    ]

    class _MockDS:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MockDS.__new__(_MockDS)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                result_rows.append(function(row, i) if with_indices else function(row))
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockDS(rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: mock_image,
    )
    from data.preprocessing.smoltalk import preprocess_smoltalk

    ds_a = preprocess_smoltalk(max_samples=5, offset=0)
    ds_b = preprocess_smoltalk(max_samples=5, offset=5)

    assert len(ds_a) == 5
    assert len(ds_b) == 5
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert ids_a.isdisjoint(ids_b), "offset groups must not overlap"


def test_offset_non_overlap_via_cli(tmp_path, monkeypatch):
    mock_image = Image.new("RGB", (8, 8))

    class _MockSmolTalk:
        def __init__(self, num_rows):
            self._rows = [
                {"instruction": f"instr {i}", "response": f"resp {i}", "id": f"st-{i}"}
                for i in range(num_rows)
            ]

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MockSmolTalk.__new__(_MockSmolTalk)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                result_rows.append(function(row, i) if with_indices else function(row))
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(10),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: mock_image,
    )
    outdir_a = tmp_path / "set_a"
    cli_main([
        "--smoltalk-samples", "3",
        "--smoltalk-offset", "0",
        "--output", str(outdir_a),
        "--seed", "0",
    ])
    outdir_b = tmp_path / "set_b"
    cli_main([
        "--smoltalk-samples", "3",
        "--smoltalk-offset", "3",
        "--output", str(outdir_b),
        "--seed", "0",
    ])
    from datasets import load_from_disk
    set_a = load_from_disk(str(outdir_a))
    set_b = load_from_disk(str(outdir_b))
    ids_a = {r["row_id"] for r in set_a}
    ids_b = {r["row_id"] for r in set_b}
    assert len(set_a) == 3
    assert len(set_b) == 3
    assert ids_a.isdisjoint(ids_b), "offset groups via CLI must not overlap"


# --- No-id offset fallback (row_id uniqueness) ---


def _make_no_id_mock(rows: list[dict]):
    """Return a mock dataset class whose rows lack an ``id`` column."""

    class _MockNoId:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MockNoId.__new__(_MockNoId)
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
                result_rows.append(function(row, i) if with_indices else function(row))
            return Dataset.from_list(result_rows)

        def filter(self, function, **kwargs):
            dataset = _MockNoId.__new__(_MockNoId)
            dataset._rows = [row for row in self._rows if function(row)]
            return dataset

    return _MockNoId(rows)


def _no_id_rows_libri(n: int):
    return [
        {
            "audio": {"array": __import__("numpy").zeros(16000, dtype="float32"), "sampling_rate": 16000},
            "text": f"transcript {i}",
        }
        for i in range(n)
    ]


def _no_id_rows_dense(n: int):
    import io
    rows = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        rows.append({
            "images": [{"bytes": buf.getvalue(), "path": None}],
            "texts": [{"user": "What do you see?", "assistant": f"desc {i}"}],
            "source": "densefusion_1m",
            "relevance_ratings": [5],
            "relevance_min": 5,
            "image_correspondence_ratings": [2],
            "image_correspondence_min": 2,
            "formatting_ratings": [4],
            "formatting_min": 4,
            "visual_dependency_ratings": [5],
            "visual_dependency_min": 5,
        })
    return rows


def _no_id_rows_fineweb(n: int):
    return [{"text": f"content {i} " * 20} for i in range(n)]


def _no_id_rows_smoltalk(n: int):
    return [
        {"instruction": f"instr {i}", "response": f"resp {i}"}
        for i in range(n)
    ]


def test_librispeech_offset_no_id_non_overlap(monkeypatch):
    n = 20
    mock_ds = _make_no_id_mock(_no_id_rows_libri(n))
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: mock_ds,
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        lambda *a, **kw: Image.new("RGB", (8, 8)),
    )
    ds_a = preprocess_librispeech_asr(max_samples=5, offset=0)
    ds_b = preprocess_librispeech_asr(max_samples=5, offset=5)
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert ids_a == {"0", "1", "2", "3", "4"}
    assert ids_b == {"5", "6", "7", "8", "9"}
    assert ids_a.isdisjoint(ids_b)


def test_densefusion_offset_no_id_non_overlap(monkeypatch):
    n = 20
    mock_ds = _make_no_id_mock(_no_id_rows_dense(n))
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: mock_ds,
    )
    ds_a = preprocess_densefusion(max_samples=4, offset=0)
    ds_b = preprocess_densefusion(max_samples=4, offset=6)
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert ids_a == {"0", "1", "2", "3"}
    assert ids_b == {"6", "7", "8", "9"}
    assert ids_a.isdisjoint(ids_b)


def test_fineweb_offset_no_id_non_overlap(monkeypatch):
    n = 20
    mock_cls = _make_no_id_mock(_no_id_rows_fineweb(n))

    class _FineWebSplitMock:
        """FineWeb uses split syntax for slicing, so we handle that."""

        def __init__(self, num_rows):
            self._num_rows = num_rows

        def __len__(self):
            return self._num_rows

        @property
        def column_names(self):
            return ["text"]

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i in range(self._num_rows):
                row = {"text": f"content {i} " * 20}
                result_rows.append(function(row, i) if with_indices else function(row))
            return Dataset.from_list(result_rows)

    def _load_side_effect(*args, **kw):
        split = kw.get("split", "train")
        if "[" in split:
            offset, rest = split.split("[")[1].split(":")
            # ignore rest for mock purposes
            start = int(offset)
            end = int(rest.rstrip("]"))
            return _FineWebSplitMock(end - start)
        return _FineWebSplitMock(100)

    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        _load_side_effect,
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        lambda *a, **kw: Image.new("RGB", (8, 8)),
    )
    ds_a = preprocess_fineweb_edu(max_samples=3, offset=0)
    ds_b = preprocess_fineweb_edu(max_samples=3, offset=4)
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert ids_a == {"0", "1", "2"}
    assert ids_b == {"4", "5", "6"}
    assert ids_a.isdisjoint(ids_b)


def test_smoltalk_offset_no_id_non_overlap(monkeypatch):
    n = 20
    mock_ds = _make_no_id_mock(_no_id_rows_smoltalk(n))
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: mock_ds,
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: Image.new("RGB", (8, 8)),
    )
    ds_a = preprocess_smoltalk(max_samples=5, offset=0)
    ds_b = preprocess_smoltalk(max_samples=5, offset=5)
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert ids_a == {"0", "1", "2", "3", "4"}
    assert ids_b == {"5", "6", "7", "8", "9"}
    assert ids_a.isdisjoint(ids_b)


def test_preprocessor_offset_preserves_existing_behavior(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    rows = [
        {"instruction": f"instr {i}", "response": f"resp {i}", "id": str(i)}
        for i in range(10)
    ]

    class _MockDS:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _MockDS.__new__(_MockDS)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
            result_rows = []
            for i, row in enumerate(self._rows):
                result_rows.append(function(row, i) if with_indices else function(row))
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockDS(rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: mock_image,
    )
    from data.preprocessing.smoltalk import preprocess_smoltalk

    ds_default = preprocess_smoltalk(max_samples=5)
    ds_offset0 = preprocess_smoltalk(max_samples=5, offset=0)

    assert len(ds_default) == len(ds_offset0)
    assert [r["row_id"] for r in ds_default] == [r["row_id"] for r in ds_offset0]

