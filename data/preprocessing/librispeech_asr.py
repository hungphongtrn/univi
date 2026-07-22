from __future__ import annotations

import io
import json
import soundfile as sf

from datasets import Audio, Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import render_log_mel_spectrogram
from data.preprocessing.text_token_utils import compute_token_length

_PREPROCESSING_VERSION = "0.5.0"
_USER_INSTRUCTION = "Transcribe the speech represented by this spectrogram image."


def _render_defaults(*, sample_rate: int, n_mels: int) -> dict:
    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": 400,
        "hop_length": 160,
        "window": "hann",
        "power": 2.0,
        "page_duration_sec": 10.0,
        "max_pages": 4,
        "output_width": 1000,
        "output_height": 160,
        "normalization": "whisper_log_mel",
    }


def _audio_duration_seconds(audio) -> float:
    if isinstance(audio, dict):
        if "array" in audio:
            return len(audio["array"]) / audio["sampling_rate"]
        encoded = audio.get("bytes")
        source = io.BytesIO(encoded) if encoded is not None else audio["path"]
        info = sf.info(source)
        return info.frames / info.samplerate
    metadata = getattr(audio, "metadata", None)
    if metadata is not None:
        return float(metadata.duration_seconds)
    samples = audio.get_all_samples()
    return float(samples.duration_seconds)


def _decode_audio(audio) -> tuple:
    if isinstance(audio, dict):
        return audio["array"], int(audio["sampling_rate"])
    samples = audio.get_all_samples()
    array = samples.data.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array, int(samples.sample_rate)


def preprocess_librispeech_asr(
    subset: str = "clean",
    split: str = "train.360",
    max_samples: int | None = None,
    offset: int = 0,
    sample_rate: int = 16000,
    n_mels: int = 80,
    include_native: bool = False,
    tokenizer=None,
    tokenizer_name: str = "unsloth/gemma-4-E2B-it",
    num_proc: int = 4,
) -> Dataset:
    source = load_dataset(
        "openslr/librispeech_asr",
        subset,
        split=split,
        num_proc=num_proc,
    )
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    render_kwargs = _render_defaults(sample_rate=sample_rate, n_mels=n_mels)
    max_duration_sec = (
        render_kwargs["page_duration_sec"] * render_kwargs["max_pages"]
    )
    map_num_proc = num_proc if num_proc > 1 else None
    audio_feature = getattr(source, "features", {}).get("audio")
    if isinstance(audio_feature, Audio):
        source = source.cast_column(
            "audio",
            Audio(sampling_rate=audio_feature.sampling_rate, decode=False),
        )
    source = source.filter(
        lambda row: _audio_duration_seconds(row["audio"]) <= max_duration_sec,
        num_proc=map_num_proc,
        desc=f"Filtering LibriSpeech audio over {max_duration_sec:g}s",
    )
    if isinstance(audio_feature, Audio):
        source = source.cast_column(
            "audio",
            Audio(sampling_rate=sample_rate, decode=True),
        )

    def _process_row(row, index: int):
        audio_array, audio_sr = _decode_audio(row["audio"])
        duration_sec = len(audio_array) / audio_sr
        row_render_kwargs = {
            **render_kwargs,
            "source_sample_rate": audio_sr,
        }
        rendered = render_log_mel_spectrogram(
            audio_array,
            **row_render_kwargs,
        )
        pages = rendered if isinstance(rendered, list) else [rendered]

        image_bytes_list: list[bytes] = []
        for page in pages:
            buffer = io.BytesIO()
            page.save(buffer, format="PNG", compress_level=1)
            image_bytes_list.append(buffer.getvalue())

        user_content: list[dict] = [
            {"type": "image"} for _ in image_bytes_list
        ]
        user_content.append({"type": "text", "text": _USER_INSTRUCTION})
        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["text"]}],
            },
        ]

        source_id = row.get("id")
        row_id = (
            str(source_id)
            if source_id not in (None, "")
            else str(offset + index)
        )
        original_len = (
            compute_token_length(row["text"], tokenizer)
            if tokenizer is not None
            else -1
        )
        page_duration_sec = render_kwargs["page_duration_sec"]
        output_width = render_kwargs["output_width"]

        result = {
            "images": image_bytes_list,
            "messages": messages,
            "source_dataset_id": "openslr/librispeech_asr",
            "split": split,
            "row_id": row_id,
            "render_config": json.dumps(
                {
                    **row_render_kwargs,
                    "source_duration_ms": round(duration_sec * 1000, 2),
                    "ms_per_horizontal_pixel": round(
                        page_duration_sec * 1000 / output_width,
                        4,
                    ),
                    "image_width": output_width,
                    "image_height": render_kwargs["output_height"],
                    "n_pages": len(image_bytes_list),
                    "right_padded_final_page": (
                        duration_sec < len(image_bytes_list) * page_duration_sec
                    ),
                    "render_method": "fixed_10s_rectangular_pages",
                    "render_version": _PREPROCESSING_VERSION,
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

    dataset = source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
        num_proc=map_num_proc,
        desc="Rendering fixed LibriSpeech spectrogram pages",
    )
    if len(dataset) > 0:
        dataset = dataset.cast_column("images", List(HfImage()))
    return dataset


def add_token_length_to_librispeech_row(
    row: dict,
    tokenizer,
    text_key: str = "target_text",
) -> dict:
    """Add original_token_length to a single row (used for reuse path)."""
    if "original_token_length" in row and row["original_token_length"] != -1:
        return row
    source = row.get(text_key, "")
    if not source:
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                source = "".join(
                    item.get("text", "")
                    for item in message.get("content", [])
                    if item.get("type") == "text"
                )
                break
    if tokenizer is not None and source:
        row["original_token_length"] = compute_token_length(source, tokenizer)
    else:
        row["original_token_length"] = -1
    return row
