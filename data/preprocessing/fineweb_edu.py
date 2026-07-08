from __future__ import annotations

import json

from datasets import Dataset, load_dataset

from data.preprocessing.render_utils import render_text_page

_PREPROCESSING_VERSION = "0.1.0"
_USER_INSTRUCTION = "Transcribe the text shown in the image."
_SOURCE_DATASET_ID = "HuggingFaceFW/fineweb-edu"
_TEXT_CANDIDATES = ["text", "content", "document"]


def _find_text_key(row: dict) -> str:
    for key in _TEXT_CANDIDATES:
        if key in row:
            return key
    raise KeyError(
        f"None of {_TEXT_CANDIDATES} found in row. "
        f"Available keys: {list(row.keys())}"
    )


def preprocess_fineweb_edu(
    split: str = "train",
    max_samples: int | None = None,
    max_chars: int = 2000,
    canvas_width: int = 1024,
    font_size: int = 14,
) -> Dataset:
    source = load_dataset(_SOURCE_DATASET_ID, split=split)
    if max_samples is not None:
        source = source.select(range(min(max_samples, len(source))))

    def _process_row(row, index: int):
        text_key = _find_text_key(row)
        raw_text = row[text_key][:max_chars]

        image = render_text_page(
            raw_text, canvas_width=canvas_width, font_size=font_size
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "text": None},
                    {"type": "text", "image": None, "text": _USER_INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "image": None, "text": raw_text},
                ],
            },
        ]

        source_id = row.get("id")
        row_id = str(source_id) if source_id not in (None, "") else str(index)

        return {
            "messages": messages,
            "source_dataset_id": _SOURCE_DATASET_ID,
            "split": split,
            "row_id": row_id,
            "render_config": json.dumps(
                {
                    "render_method": "text_page",
                    "max_chars": max_chars,
                    "canvas_width": canvas_width,
                    "font_size": font_size,
                },
                sort_keys=True,
            ),
            "modality_label": "text",
            "preprocessing_version": _PREPROCESSING_VERSION,
        }

    return source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
    )
