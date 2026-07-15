from __future__ import annotations

import json
import io

from datasets import Dataset
from PIL import Image

from data.preprocessing.rebuild_3m import (
    TRAINING_SPLITS,
    VALIDATION_SPLITS,
    _save_split,
    _deterministic_split,
    _select_source_from_fullv0,
    _check_row_images,
    _convert_old_librispeech_row_to_tiles,
    _add_token_length,
    _reject_legacy_output,
    _SOURCE_IDS,
)


class _CharTokenizer:
    name_or_path = "test-char-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)


def _row(source: str, image_count: int = 1, text: str = "target") -> dict:
    return {
        "images": [_make_img_bytes() for _ in range(image_count)],
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(image_count)],
                    {"type": "text", "text": "instruction"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ],
        "source_dataset_id": source,
        "split": "train",
        "row_id": "0",
        "render_config": json.dumps({"key": "val"}, sort_keys=True),
        "modality_label": "text",
        "preprocessing_version": "0.1.0",
    }


def _make_img_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=128).save(buf, format="PNG")
    return buf.getvalue()


def test_select_source_subset_retains_only_requested_source():
    dataset = Dataset.from_list([_row("keep"), _row("drop")])
    selected = _select_source_from_fullv0(dataset, "keep", num_proc=1)
    assert len(selected) == 1
    assert selected[0]["source_dataset_id"] == "keep"


def test_check_row_images_passes():
    row = _row("test", image_count=2)
    _check_row_images(row)


def test_check_row_images_fails_on_mismatch():
    row = _row("test", image_count=1)
    row["images"] = [_make_img_bytes(), _make_img_bytes()]
    try:
        _check_row_images(row)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_deterministic_split_90_10():
    ds = Dataset.from_list([_row("test", text=str(i)) for i in range(100)])
    train, val = _deterministic_split(ds, 0.9)
    assert len(train) == 90
    assert len(val) == 10
    train_ids = {row["messages"][1]["content"][0]["text"] for row in train}
    val_ids = {row["messages"][1]["content"][0]["text"] for row in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {str(i) for i in range(100)}


def test_deterministic_split_80_20():
    ds = Dataset.from_list([_row("test", text=str(i)) for i in range(50)])
    train, val = _deterministic_split(ds, 0.8)
    assert len(train) == 40
    assert len(val) == 10


def test_save_subset_uses_config_and_exact_split(tmp_path, monkeypatch):
    dataset = Dataset.from_list([_row("openslr/librispeech_asr", image_count=2)])
    pushed = []
    monkeypatch.setattr(
        Dataset,
        "push_to_hub",
        lambda self, repo, **kwargs: pushed.append((repo, kwargs)),
    )

    entry = _save_split(
        dataset,
        tmp_path,
        "librispeech",
        "train",
        "openslr/librispeech_asr",
        "hungphongtrn/univi-3M-v0",
    )

    assert (tmp_path / "librispeech" / "train").is_dir()
    assert entry["path"] == "librispeech/train"
    assert entry["split"] == "train"
    assert entry["is_training_split"] is True
    assert pushed == [
        (
            "hungphongtrn/univi-3M-v0",
            {"config_name": "librispeech", "split": "train"},
        )
    ]


def test_save_subset_densefusion_validation(tmp_path, monkeypatch):
    dataset = Dataset.from_list([_row("HuggingFaceM4/FineVision", image_count=1)])
    pushed = []
    monkeypatch.setattr(
        Dataset,
        "push_to_hub",
        lambda self, repo, **kwargs: pushed.append((repo, kwargs)),
    )

    entry = _save_split(
        dataset,
        tmp_path,
        "densefusion",
        "validation",
        "HuggingFaceM4/FineVision",
        "hungphongtrn/univi-3M-v0",
    )

    assert (tmp_path / "densefusion" / "validation").is_dir()
    assert entry["is_training_split"] is False


def test_training_splits_contract():
    assert TRAINING_SPLITS == {
        "librispeech": ("train",),
        "densefusion": ("train",),
        "fineweb-edu": ("train",),
        "smoltalk": ("train",),
    }


def test_validation_splits_contract():
    assert VALIDATION_SPLITS == {
        "librispeech": ("validation",),
        "densefusion": ("validation",),
        "fineweb-edu": ("validation",),
        "smoltalk": ("validation",),
    }


def test_source_ids_present():
    assert _SOURCE_IDS["librispeech"] == "openslr/librispeech_asr"
    assert _SOURCE_IDS["densefusion"] == "HuggingFaceM4/FineVision"
    assert _SOURCE_IDS["fineweb-edu"] == "HuggingFaceFW/fineweb-edu"
    assert _SOURCE_IDS["smoltalk"] == "HuggingFaceTB/smoltalk"


def test_add_token_length_adds_column():
    tokenizer = _CharTokenizer()
    row = {"target_text": "hello", "original_token_length": -1}
    result = _add_token_length(row, tokenizer, "target_text")
    assert result["original_token_length"] > 0


def test_add_token_length_skips_if_already_set():
    tokenizer = _CharTokenizer()
    row = {"target_text": "hello", "original_token_length": 42}
    result = _add_token_length(row, tokenizer, "target_text")
    assert result["original_token_length"] == 42


def test_convert_old_librispeech_row_to_tiles():
    old_img = _make_img_bytes((192, 64))
    row = {
        "images": [old_img],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe the speech."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "transcript"}],
            },
        ],
        "source_dataset_id": "openslr/librispeech_asr",
        "split": "train",
        "row_id": "test-0",
        "render_config": json.dumps({"sample_rate": 16000}),
        "modality_label": "audio",
        "preprocessing_version": "0.1.0",
    }
    result = _convert_old_librispeech_row_to_tiles(row)
    assert len(result["images"]) == 3
    assert len(result["messages"][0]["content"]) == 4
    assert sum(c["type"] == "image" for c in result["messages"][0]["content"]) == 3
    assert result["messages"][0]["content"][-1]["type"] == "text"
    assert all(c["type"] != "image" for c in result["messages"][1]["content"])
    _check_row_images(result)
    for img_bytes in result["images"]:
        img = Image.open(io.BytesIO(img_bytes))
        assert img.width == img.height
    rc = json.loads(result["render_config"])
    assert rc["render_method"] == "chronological_square_tiles"


def test_convert_old_librispeech_pads_final_tile():
    old_img = _make_img_bytes((100, 64))
    row = {
        "images": [old_img],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "transcript"}],
            },
        ],
        "source_dataset_id": "openslr/librispeech_asr",
        "split": "train",
        "row_id": "test-1",
        "render_config": json.dumps({"sample_rate": 16000}),
        "modality_label": "audio",
        "preprocessing_version": "0.1.0",
    }
    result = _convert_old_librispeech_row_to_tiles(row)
    assert len(result["images"]) == 2
    for img_bytes in result["images"]:
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (64, 64)


def test_convert_raises_on_no_images():
    row = _row("test", image_count=0)
    try:
        _convert_old_librispeech_row_to_tiles(row)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_reject_legacy_flat_output(tmp_path):
    legacy = tmp_path / "densefusion"
    legacy.mkdir()
    (legacy / "state.json").write_text("{}", encoding="utf-8")

    try:
        _reject_legacy_output(tmp_path)
        assert False, "Expected FileExistsError"
    except FileExistsError as error:
        assert "Legacy flat dataset" in str(error)


def test_accept_new_split_layout(tmp_path):
    (tmp_path / "densefusion" / "train").mkdir(parents=True)
    _reject_legacy_output(tmp_path)
