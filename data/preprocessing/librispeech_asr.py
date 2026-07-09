from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import render_log_mel_spectrogram

_PREPROCESSING_VERSION = "0.1.0"
_USER_INSTRUCTION = "Transcribe the speech represented by this spectrogram image."


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
) -> Dataset:
    source = load_dataset("openslr/librispeech_asr", subset, split=split)
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

        buf = io.BytesIO()
        spectrogram.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
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

        result = {
            "images": [image_bytes],
            "messages": messages,
            "source_dataset_id": "openslr/librispeech_asr",
            "split": split,
            "row_id": row_id,
            "render_config": json.dumps(render_kwargs, sort_keys=True),
            "modality_label": "audio",
            "preprocessing_version": _PREPROCESSING_VERSION,
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
        num_proc=4,
    )
    if len(ds) > 0:
        ds = ds.cast_column("images", List(HfImage()))
    return ds
