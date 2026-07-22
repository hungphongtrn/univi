from __future__ import annotations

import json

import pytest
from datasets import Dataset
from PIL import Image

from data.preprocessing.prepare_eval import prepare_source_eval

_REQUIRED_EVAL_FIELDS = {
    "messages",
    "native_user_content",
    "target_text",
    "source_dataset_id",
    "split",
    "row_id",
    "render_config",
    "modality_label",
    "preprocessing_version",
    "native_available",
    "native_equals_image_only",
}


def _make_smoltalk_raw(num_rows: int):
    return [
        {
            "messages": [
                {"role": "user", "content": f"instruction {i}"},
                {"role": "assistant", "content": f"response {i}"},
            ],
            "id": f"test-{i}",
        }
        for i in range(num_rows)
    ]


def _make_fineweb_raw(num_rows: int):
    return [
        {"text": f"educational text {i} " * 30, "id": f"test-{i}"}
        for i in range(num_rows)
    ]


def _make_densefusion_raw(num_rows: int):
    import io

    rows = []
    for i in range(num_rows):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buf, format="PNG")
        rows.append({
            "images": [{"bytes": buf.getvalue(), "path": None}],
            "texts": [{"user": "What do you see?", "assistant": f"a photo of sample {i}"}],
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


def _make_librispeech_raw(num_rows: int):
    import numpy as np

    return [
        {
            "audio": {"array": np.zeros(16000, dtype="float32"), "sampling_rate": 16000},
            "text": f"transcript {i}",
            "id": f"test-{i}",
        }
        for i in range(num_rows)
    ]


class _MockIterable:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockIterable.__new__(_MockIterable)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None, **kwargs):
        result_rows = []
        for i, row in enumerate(self._rows):
            result_rows.append(function(row, i) if with_indices else function(row))
        return Dataset.from_list(result_rows)

    def filter(self, function, **kwargs):
        dataset = _MockIterable.__new__(_MockIterable)
        dataset._rows = [row for row in self._rows if function(row)]
        return dataset


def _mock_fineweb_load(rows):
    def _load(*args, **kwargs):
        split = kwargs.get("split", "train")
        import re
        m = re.match(r".*\[(\d+):(\d+)\]", split)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            return _MockIterable(rows[start:end])
        return _MockIterable(rows)
    return _load


# --- Eval schema tests ---


def test_eval_fineweb_schema(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        _mock_fineweb_load(_make_fineweb_raw(5)),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        lambda *a, **kw: mock_image,
    )
    ds = prepare_source_eval("fineweb", max_samples=3, offset=0)
    assert len(ds) == 3
    for row in ds:
        assert set(row.keys()) >= _REQUIRED_EVAL_FIELDS
        assert row["native_available"] is True
        assert row["native_equals_image_only"] is False
        assert isinstance(row["native_user_content"], str)
        assert isinstance(row["target_text"], str)
        assert row["native_user_content"] == row["target_text"]


def test_eval_smoltalk_schema(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockIterable(_make_smoltalk_raw(5)),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        lambda *a, **kw: mock_image,
    )
    ds = prepare_source_eval("smoltalk", max_samples=3, offset=0)
    assert len(ds) == 3
    for row in ds:
        assert set(row.keys()) >= _REQUIRED_EVAL_FIELDS
        assert row["native_available"] is True
        assert row["native_equals_image_only"] is False
        assert isinstance(row["native_user_content"], str)
        assert isinstance(row["target_text"], str)
        assert "instruction" in row["native_user_content"]
        assert "response" in row["target_text"]


def test_eval_densefusion_schema(monkeypatch):

    class _DenseMock:
        def __init__(self, rows):
            self._rows = rows

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

    raw = _make_densefusion_raw(5)
    monkeypatch.setattr(
        "data.preprocessing.densefusion.load_dataset",
        lambda *a, **kw: _DenseMock(raw),
    )
    ds = prepare_source_eval("densefusion", max_samples=3, offset=0)
    assert len(ds) == 3
    for row in ds:
        assert set(row.keys()) >= _REQUIRED_EVAL_FIELDS
        assert row["native_available"] is True
        assert row["native_equals_image_only"] is True
        assert row["native_user_content"] is None
        assert isinstance(row["target_text"], str)


def test_eval_librispeech_schema(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.load_dataset",
        lambda *a, **kw: _MockIterable(_make_librispeech_raw(5)),
    )
    monkeypatch.setattr(
        "data.preprocessing.librispeech_asr.render_log_mel_spectrogram",
        lambda *a, **kw: mock_image,
    )
    ds = prepare_source_eval("librispeech", max_samples=3, offset=0)
    assert len(ds) == 3
    for row in ds:
        assert set(row.keys()) >= _REQUIRED_EVAL_FIELDS
        assert row["native_available"] is False
        assert row["native_equals_image_only"] is False
        assert row["native_user_content"] is None
        assert isinstance(row["target_text"], str)
        assert "transcript" in row["target_text"]


def test_eval_offset_selection(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        _mock_fineweb_load(_make_fineweb_raw(20)),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        lambda *a, **kw: mock_image,
    )
    ds_a = prepare_source_eval("fineweb", max_samples=5, offset=0)
    ds_b = prepare_source_eval("fineweb", max_samples=5, offset=5)
    ids_a = {r["row_id"] for r in ds_a}
    ids_b = {r["row_id"] for r in ds_b}
    assert len(ds_a) == 5
    assert len(ds_b) == 5
    assert ids_a.isdisjoint(ids_b)


def test_eval_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        prepare_source_eval("unknown_source", max_samples=5)


def test_eval_field_types_are_consistent(monkeypatch):
    mock_image = Image.new("RGB", (8, 8))
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset",
        _mock_fineweb_load(_make_fineweb_raw(3)),
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_page",
        lambda *a, **kw: mock_image,
    )
    ds = prepare_source_eval("fineweb", max_samples=3, offset=0)
    for row in ds:
        assert isinstance(row["source_dataset_id"], str)
        assert isinstance(row["split"], str)
        assert isinstance(row["row_id"], str)
        assert isinstance(row["render_config"], str)
        json.loads(row["render_config"])
        assert isinstance(row["modality_label"], str)
        assert isinstance(row["preprocessing_version"], str)
        assert isinstance(row["messages"], list)
        assert len(row["messages"]) == 2


# --- Eval harness tests ---


def test_eval_config_loading(tmp_path):
    import yaml

    cfg = {
        "base_model": "unsloth/gemma-4-E2B-it",
        "checkpoint_path": "data/checkpoints/full-v0/final",
        "eval_datasets": {
            "librispeech": "data/materialized/eval-v0/librispeech",
            "densefusion": "data/materialized/eval-v0/densefusion",
            "fineweb": "data/materialized/eval-v0/fineweb",
            "smoltalk": "data/materialized/eval-v0/smoltalk",
        },
        "output": "data/eval/results.json",
        "generation": {"max_new_tokens": 256, "temperature": 0.0, "do_sample": False},
    }
    path = tmp_path / "eval.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    from eval_lane import load_eval_config

    loaded = load_eval_config(str(path))
    assert loaded["base_model"] == "unsloth/gemma-4-E2B-it"
    assert loaded["checkpoint_path"] == "data/checkpoints/full-v0/final"
    assert set(loaded["eval_datasets"]) == {
        "librispeech", "densefusion", "fineweb", "smoltalk"
    }
    assert loaded["output"] == "data/eval/results.json"
    assert loaded["generation"]["max_new_tokens"] == 256


def test_build_native_messages_returns_none_when_no_content():
    from eval_lane import build_native_messages

    row = {"native_user_content": None, "target_text": "hello"}
    assert build_native_messages(row) is None

    row = {"target_text": "hello"}
    assert build_native_messages(row) is None


def test_build_native_messages_constructs_correctly():
    from eval_lane import build_native_messages

    row = {
        "native_user_content": "What is 2+2?",
        "target_text": "4",
    }
    msgs = build_native_messages(row)
    assert msgs is not None
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["type"] == "text"
    assert msgs[0]["content"][0]["text"] == "What is 2+2?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["text"] == "4"


def test_should_skip_native():
    from eval_lane import should_skip_native

    skip, reason = should_skip_native({"native_available": False})
    assert skip is True
    assert "native_available" in reason

    skip, reason = should_skip_native(
        {"native_available": True, "native_user_content": None}
    )
    assert skip is True
    assert "native_user_content" in reason

    skip, reason = should_skip_native(
        {"native_available": True, "native_user_content": "hello"}
    )
    assert skip is False
    assert reason is None


def test_get_native_note():
    from eval_lane import get_native_note

    assert "Native audio not implemented" in get_native_note("librispeech")
    assert "Native identical" in get_native_note("densefusion")
    assert get_native_note("fineweb") is None
    assert get_native_note("smoltalk") is None
    assert get_native_note("unknown") is None


def test_retention_wiring(monkeypatch):
    from eval_lane import evaluate_source

    rows = [
        {
            "messages": [{"role": "user", "content": "img"}],
            "native_user_content": "hi",
            "target_text": "hello world",
            "native_available": True,
            "native_equals_image_only": False,
        },
    ]

    def _fake_loss(model, tokenizer, messages):
        return 0.5

    def _fake_generate(model, tokenizer, messages, **kw):
        return "hello world"

    monkeypatch.setattr("eval_lane.compute_loss", _fake_loss)
    monkeypatch.setattr("eval_lane.generate_text", _fake_generate)

    result = evaluate_source(
        "fineweb", rows, object(), object(), object(), object(), {},
    )
    assert result["base_image"] is not None
    assert result["trained_image"] is not None
    assert result["base_native"] is not None
    assert result["trained_native"] is not None
    assert result["retention"] == 100.0
    assert result["native_note"] is None


def test_output_schema(monkeypatch, tmp_path):
    from eval_lane import run_eval
    from datasets import Dataset

    fake_row = {
        "messages": [{"role": "user", "content": "img"}],
        "native_user_content": "hello",
        "target_text": "world",
        "native_available": True,
        "native_equals_image_only": False,
        "source_dataset_id": "test",
        "split": "eval",
        "row_id": "0",
        "render_config": "{}",
        "modality_label": "text",
        "preprocessing_version": "0.1.0",
    }
    ds = Dataset.from_list([fake_row])

    for source in ("fineweb", "smoltalk"):
        (tmp_path / source).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(tmp_path / source))

    (tmp_path / "librispeech").mkdir(parents=True, exist_ok=True)
    libri_row = dict(fake_row)
    libri_row["native_available"] = False
    libri_row["native_user_content"] = None
    ds_libri = Dataset.from_list([libri_row])
    ds_libri.save_to_disk(str(tmp_path / "librispeech"))

    (tmp_path / "densefusion").mkdir(parents=True, exist_ok=True)
    dense_row = dict(fake_row)
    dense_row["native_user_content"] = None
    dense_row["native_equals_image_only"] = True
    ds_dense = Dataset.from_list([dense_row])
    ds_dense.save_to_disk(str(tmp_path / "densefusion"))

    monkeypatch.setattr("eval_lane.compute_loss", lambda *a, **kw: 0.5)
    monkeypatch.setattr("eval_lane.generate_text", lambda *a, **kw: "world")
    monkeypatch.setattr(
        "eval_lane.load_model",
        lambda p: (object(), object()),
    )

    config = {
        "base_model": "unsloth/gemma-4-E2B-it",
        "checkpoint_path": str(tmp_path / "ckpt"),
        "eval_datasets": {
            "librispeech": str(tmp_path / "librispeech"),
            "densefusion": str(tmp_path / "densefusion"),
            "fineweb": str(tmp_path / "fineweb"),
            "smoltalk": str(tmp_path / "smoltalk"),
        },
        "output": str(tmp_path / "results.json"),
    }
    results = run_eval(config)

    assert "checkpoint" in results
    assert "base_model" in results
    assert "per_source" in results

    for src in ("librispeech", "densefusion", "fineweb", "smoltalk"):
        assert src in results["per_source"], f"Missing {src}"
        entry = results["per_source"][src]
        for key in (
            "base_image", "trained_image", "base_native",
            "trained_native", "retention", "native_note",
            "base_image_loss", "trained_image_loss",
            "base_native_loss", "trained_native_loss",
        ):
            assert key in entry, f"Missing {key} in {src}"

    libri = results["per_source"]["librispeech"]
    assert libri["base_native"] is None
    assert libri["trained_native"] is None
    assert libri["native_note"] == "Native audio not implemented"

    dense = results["per_source"]["densefusion"]
    assert dense["native_note"] == "Native identical to image lane; no separate native score"
    # DenseFusion native mirrors image lane when native_equals_image_only is True
    assert dense["base_native"] is not None
    assert dense["trained_native"] is not None
    assert dense["base_native"] == dense["base_image"]
    assert dense["trained_native"] == dense["trained_image"]

    # Image lanes should have scores since compute_loss/generate_text are monkeypatched
    for src in ("fineweb", "smoltalk"):
        entry = results["per_source"][src]
        assert entry["base_image"] is not None
        assert entry["trained_image"] is not None
        assert entry["base_native"] is not None
        assert entry["trained_native"] is not None
        assert entry["retention"] is not None


# ---------------------------------------------------------------------------
# Review-finding tests for Phase 3 Task 5
# ---------------------------------------------------------------------------


def test_densefusion_native_mirrors_image(monkeypatch):
    """When native_equals_image_only is True, native lanes mirror image lanes."""
    from eval_lane import evaluate_source

    rows = [
        {
            "messages": [{"role": "user", "content": "img"}],
            "native_user_content": None,
            "target_text": "hello world",
            "native_available": True,
            "native_equals_image_only": True,
        },
    ]

    def _fake_loss(model, tokenizer, messages):
        return 0.5

    def _fake_generate(model, tokenizer, messages, **kw):
        return "hello world"

    monkeypatch.setattr("eval_lane.compute_loss", _fake_loss)
    monkeypatch.setattr("eval_lane.generate_text", _fake_generate)

    result = evaluate_source(
        "densefusion", rows, object(), object(), object(), object(), {},
    )
    assert result["base_native"] == result["base_image"]
    assert result["trained_native"] == result["trained_image"]
    assert result["native_note"] == "Native identical to image lane; no separate native score"


def test_retention_uses_trained_lanes(monkeypatch):
    """Retention compares trained_image to trained_native, not base lanes."""
    from eval_lane import evaluate_source

    rows = [
        {
            "messages": [{"role": "user", "content": "img"}],
            "native_user_content": "hi",
            "target_text": "hello world",
            "native_available": True,
            "native_equals_image_only": False,
        },
    ]

    def _fake_loss(model, tokenizer, messages):
        return 0.5

    call_count = [0]

    def _fake_generate(model, tokenizer, messages, **kw):
        call_count[0] += 1
        c = call_count[0]
        if c in (2, 4):  # trained_image, trained_native
            return "hello world"
        return "wrong prediction"

    monkeypatch.setattr("eval_lane.compute_loss", _fake_loss)
    monkeypatch.setattr("eval_lane.generate_text", _fake_generate)

    result = evaluate_source(
        "fineweb", rows, object(), object(), object(), object(), {},
    )
    # Calls order: 1=base_image, 2=trained_image, 3=base_native, 4=trained_native
    # trained_image (2) = "hello world" → ROUGE-L 1.0
    # trained_native (4) = "hello world" → ROUGE-L 1.0
    # retention = trained_image / trained_native = 1.0 / 1.0 * 100 = 100.0
    assert result["trained_image"] == 1.0
    assert result["base_image"] == 0.0  # base lanes differ
    assert result["base_native"] == 0.0
    assert result["retention"] == 100.0


def test_lane_output_includes_loss(monkeypatch):
    """Each lane result includes loss in the output artifact."""
    from eval_lane import evaluate_source

    rows = [
        {
            "messages": [{"role": "user", "content": "img"}],
            "native_user_content": "hi",
            "target_text": "hello world",
            "native_available": True,
            "native_equals_image_only": False,
        },
    ]

    def _fake_loss(model, tokenizer, messages):
        return 2.718

    def _fake_generate(model, tokenizer, messages, **kw):
        return "hello world"

    monkeypatch.setattr("eval_lane.compute_loss", _fake_loss)
    monkeypatch.setattr("eval_lane.generate_text", _fake_generate)

    result = evaluate_source(
        "smoltalk", rows, object(), object(), object(), object(), {},
    )
    assert result["base_image_loss"] == 2.718
    assert result["trained_image_loss"] == 2.718
    assert result["base_native_loss"] == 2.718
    assert result["trained_native_loss"] == 2.718


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="No CUDA")
def test_eval_gpu_skip_if_no_cuda():
    """GPU smoke test — only runs when CUDA is available."""
    import torch
    assert torch.cuda.is_available()
