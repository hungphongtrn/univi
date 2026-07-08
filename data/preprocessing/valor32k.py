from __future__ import annotations

import json
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import Dataset, load_dataset
from PIL import Image

from data.preprocessing.render_utils import render_log_mel_spectrogram, render_text_page

_PREPROCESSING_VERSION = "0.1.0"
_SOURCE_DATASET_ID = "inesriahi/valor32k-avqa-v2"

_GENERIC_A_D_INSTRUCTION = (
    "Answer the multiple-choice question shown in the images. "
    "Reply with only A, B, C, or D."
)

_QUESTION_CANDIDATES = ["question", "query", "prompt"]
_MODALITY_CANDIDATES = ["modality", "modality_label", "type"]
_FRAME_CANDIDATES = ["frames", "images", "video_frames", "image"]
_ANSWER_LETTERS = ["A", "B", "C", "D"]


def _find_key(row: dict, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in row:
            return key
    return None


def _get_options(row: dict) -> list[str] | None:
    if "options" in row:
        opts = row["options"]
        if isinstance(opts, list):
            return opts
        if isinstance(opts, dict):
            return [opts.get(k, "") for k in _ANSWER_LETTERS]
    found = []
    for letter in _ANSWER_LETTERS:
        key = _find_key(row, [f"option_{letter.lower()}", letter])
        if key is not None:
            found.append((letter, key))
    if found:
        result = []
        for letter in _ANSWER_LETTERS:
            key = _find_key(row, [f"option_{letter.lower()}", letter])
            if key is not None:
                result.append(row[key])
            else:
                result.append("")
        if any(r for r in result):
            return result
    return None


def _build_question_text(question: str, options: list[str] | None) -> str:
    lines = [f"Question: {question}", ""]
    if options:
        for letter, opt in zip(_ANSWER_LETTERS, options):
            lines.append(f"{letter}) {opt}")
    return "\n".join(lines)


def _normalize_answer(row: dict) -> str:
    if "correct_answer_idx" in row:
        idx = row["correct_answer_idx"]
        if isinstance(idx, int) and 0 <= idx <= 3:
            return _ANSWER_LETTERS[idx]
    answer_key = _find_key(row, ["answer", "label", "correct_answer"])
    if answer_key is not None:
        val = row[answer_key]
        if isinstance(val, str):
            val_upper = val.strip().upper()
            if val_upper in _ANSWER_LETTERS:
                return val_upper
            for letter in _ANSWER_LETTERS:
                if val_upper.startswith(letter):
                    return letter
        if isinstance(val, int) and 0 <= val <= 3:
            return _ANSWER_LETTERS[val]
    raise KeyError(
        f"Cannot normalize answer from row. "
        f"Available keys: {list(row.keys())}"
    )


def _normalize_modality(row: dict) -> str:
    key = _find_key(row, _MODALITY_CANDIDATES)
    if key is not None:
        val = str(row[key]).strip().lower()
        if "audio-visual" in val or "audiovisual" in val or val in ("av", "both"):
            return "audio-visual"
        if "audio" in val:
            return "audio"
        if "visual" in val or "video" in val or "vision" in val:
            return "visual"
    has_audio = isinstance(row.get("audio"), dict) and "array" in row["audio"]
    has_visual = any(k in row for k in _FRAME_CANDIDATES)
    if has_audio and has_visual:
        return "audio-visual"
    if has_audio:
        return "audio"
    if has_visual:
        return "visual"
    return "visual"


def _get_audio_dict(row: dict) -> dict | None:
    audio = row.get("audio")
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        return audio
    return None


def _get_frames(row: dict, frame_count: int) -> list[Image.Image]:
    frame_key = _find_key(row, _FRAME_CANDIDATES)
    if frame_key is None:
        return []
    frames = row[frame_key]
    if frames is None:
        return []
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        return []
    if len(frames) < frame_count:
        last = frames[-1]
        frames = frames + [last] * (frame_count - len(frames))
    elif len(frames) > frame_count:
        frames = frames[:frame_count]
    return frames


def preprocess_valor32k(
    split: str = "train",
    max_samples: int | None = None,
    canvas_width: int = 1024,
    font_size: int = 14,
    frame_count: int = 4,
    sample_rate: int = 16000,
    n_mels: int = 80,
) -> Dataset:
    source = load_dataset(_SOURCE_DATASET_ID, split=split)
    if max_samples is not None:
        source = source.select(range(min(max_samples, len(source))))

    spectrogram_kwargs = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": 400,
        "hop_length": 160,
        "window": "hann",
        "central_duration": 10.0,
        "power": 2.0,
    }

    def _process_row(row, index: int):
        question_key = _find_key(row, _QUESTION_CANDIDATES)
        if question_key is None:
            raise KeyError(
                f"No question key found. "
                f"Searched: {_QUESTION_CANDIDATES}. "
                f"Available: {list(row.keys())}"
            )
        question = row[question_key]
        options = _get_options(row)
        question_text = _build_question_text(question, options)

        question_img = render_text_page(
            question_text,
            canvas_width=canvas_width,
            font_size=font_size,
        )

        modality = _normalize_modality(row)
        answer = _normalize_answer(row)

        audio_dict = _get_audio_dict(row)
        frames = _get_frames(row, frame_count)

        spectrogram_img = None
        if modality in ("audio", "audio-visual") and audio_dict is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                sf.write(tmp_path, audio_dict["array"], audio_dict["sampling_rate"])
                spectrogram_img = render_log_mel_spectrogram(
                    tmp_path, **spectrogram_kwargs
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        user_content = []
        user_content.append({"type": "image", "image": question_img, "text": None})
        if spectrogram_img is not None:
            user_content.append(
                {"type": "image", "image": spectrogram_img, "text": None}
            )
        for frame in frames:
            user_content.append({"type": "image", "image": frame, "text": None})
        user_content.append(
            {"type": "text", "image": None, "text": _GENERIC_A_D_INSTRUCTION}
        )

        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "image": None, "text": answer}],
            },
        ]

        source_id = row.get("id")
        row_id = str(source_id) if source_id not in (None, "") else str(index)

        render_config = json.dumps(
            {
                "render_method": "visual_bundle",
                "canvas_width": canvas_width,
                "font_size": font_size,
                "frame_count": frame_count,
                "sample_rate": sample_rate,
                "n_mels": n_mels,
                "spectrogram": {
                    "sample_rate": sample_rate,
                    "n_mels": n_mels,
                    "n_fft": 400,
                    "hop_length": 160,
                    "window": "hann",
                    "central_duration": 10.0,
                    "power": 2.0,
                },
                "generic_a_d_instruction": _GENERIC_A_D_INSTRUCTION,
            },
            sort_keys=True,
        )

        return {
            "messages": messages,
            "source_dataset_id": _SOURCE_DATASET_ID,
            "split": split,
            "row_id": row_id,
            "render_config": render_config,
            "modality_label": modality,
            "preprocessing_version": _PREPROCESSING_VERSION,
        }

    return source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
    )
