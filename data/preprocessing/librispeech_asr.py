from __future__ import annotations

import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Dataset, load_dataset

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
    subset: str = "clean-100",
    split: str = "train",
    max_samples: int | None = None,
    sample_rate: int = 16000,
    n_mels: int = 80,
) -> Dataset:
    source = load_dataset("openslr/librispeech_asr", subset, split=split)
    if max_samples is not None:
        source = source.select(range(min(max_samples, len(source))))

    render_kwargs = _render_defaults(sample_rate=sample_rate, n_mels=n_mels)

    rows = []
    for i, row in enumerate(source):
        audio = row["audio"]
        audio_array = audio["array"]
        audio_sr = audio["sampling_rate"]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio_array, audio_sr)
            spectrogram = render_log_mel_spectrogram(tmp_path, **render_kwargs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": spectrogram, "text": None},
                    {"type": "text", "image": None, "text": _USER_INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "image": None, "text": row["text"]},
                ],
            },
        ]

        rows.append({
            "messages": messages,
            "source_dataset_id": "openslr/librispeech_asr",
            "split": split,
            "row_id": row.get("id", str(i)),
            "render_config": dict(render_kwargs),
            "modality_label": "audio",
            "preprocessing_version": _PREPROCESSING_VERSION,
        })

    return Dataset.from_list(rows)
