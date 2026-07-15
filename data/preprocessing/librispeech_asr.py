from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import (
    render_log_mel_spectrogram,
    tile_spectrogram_image,
)
from data.preprocessing.text_token_utils import compute_token_length

_PREPROCESSING_VERSION = "0.3.0"
_USER_INSTRUCTION = "Transcribe the speech represented by these spectrogram tiles."


def _render_defaults(*, sample_rate: int, n_mels: int) -> dict:
    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": 400,
        "hop_length": 160,
        "window": "hann",
        "central_duration": 10.0,
        "power": 2.0,
    }


def preprocess_librispeech_asr(
    subset: str = "clean",
    split: str = "train.100",
    max_samples: int | None = None,
    offset: int = 0,
    sample_rate: int = 16000,
    n_mels: int = 80,
    include_native: bool = False,
    tokenizer=None,
    tokenizer_name: str = "unsloth/gemma-4-E2B-it",
    num_proc: int = 4,
) -> Dataset:
    source = load_dataset("openslr/librispeech_asr", subset, split=split, num_proc=num_proc)
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    render_kwargs = _render_defaults(sample_rate=sample_rate, n_mels=n_mels)

    def _process_row(row, index: int):
        audio_data = row["audio"]
        audio_array = audio_data["array"]
        audio_sr = audio_data["sampling_rate"]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio_array, audio_sr)
            spectrogram = render_log_mel_spectrogram(tmp_path, **render_kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        tiles = tile_spectrogram_image(spectrogram)
        image_bytes_list = []
        for tile in tiles:
            buf = io.BytesIO()
            tile.save(buf, format="PNG")
            image_bytes_list.append(buf.getvalue())

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in tiles
                ] + [
                    {"type": "text", "text": _USER_INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": row["text"]},
                ],
            },
        ]

        source_id = row.get("id")
        row_id = str(source_id) if source_id not in (None, "") else str(offset + index)

        if tokenizer is not None:
            original_len = compute_token_length(row["text"], tokenizer)
        else:
            original_len = -1

        result = {
            "images": image_bytes_list,
            "messages": messages,
            "source_dataset_id": "openslr/librispeech_asr",
            "split": split,
            "row_id": row_id,
            "render_config": json.dumps(
                {
                    **render_kwargs,
                    "render_method": "chronological_square_tiles",
                    "tile_size": spectrogram.height,
                    "tile_pad_color": 255,
                },
                sort_keys=True,
            ),
            "modality_label": "audio",
            "preprocessing_version": _PREPROCESSING_VERSION,
            "original_token_length": original_len,
        }
        if include_native:
            result["native_user_content"] = None
            result["target_text"] = row["text"]
            result["native_available"] = False
            result["native_equals_image_only"] = False
        return result

    ds = source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
        num_proc=num_proc,
    )
    if len(ds) > 0:
        ds = ds.cast_column("images", List(HfImage()))
    return ds


def add_token_length_to_librispeech_row(
    row: dict,
    tokenizer,
    text_key: str = "text",
) -> dict:
    """Add original_token_length to a single row (used for reuse path).
    
    The *row* already has a ``target_text`` from a previous pass.
    This computes the untruncated length from the source transcript.
    """
    source_text = row.get(text_key, row.get("target_text", ""))
    if tokenizer is not None and source_text:
        row["original_token_length"] = compute_token_length(source_text, tokenizer)
    else:
        row["original_token_length"] = -1
    return row
