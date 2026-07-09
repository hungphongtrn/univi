from __future__ import annotations

import io
import json

from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import render_text_page

_PREPROCESSING_VERSION = "0.1.0"
_USER_INSTRUCTION = "Transcribe the text shown in the image."
_SOURCE_DATASET_ID = "HuggingFaceFW/fineweb-edu"
_SOURCE_CONFIG = "sample-10BT"
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
    config: str = _SOURCE_CONFIG,
    split: str = "train",
    max_samples: int | None = None,
    offset: int = 0,
    max_chars: int = 2000,
    canvas_width: int = 1024,
    font_size: int = 14,
    include_native: bool = False,
) -> Dataset:
    if max_samples is not None:
        source = load_dataset(_SOURCE_DATASET_ID, config, split=f"{split}[{offset}:{offset + max_samples}]")
    else:
        source = load_dataset(_SOURCE_DATASET_ID, config, split=split)

    def _process_row(row, index: int):
        text_key = _find_text_key(row)
        raw_text = row[text_key][:max_chars]

        image = render_text_page(
            raw_text, canvas_width=canvas_width, font_size=font_size
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
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
                    {"type": "text", "text": raw_text},
                ],
            },
        ]

        source_id = row.get("id")
        row_id = str(source_id) if source_id not in (None, "") else str(offset + index)

        result = {
            "images": [image_bytes],
            "messages": messages,
            "source_dataset_id": _SOURCE_DATASET_ID,
            "split": split,
            "row_id": row_id,
            "render_config": json.dumps(
                {
                    "render_method": "text_page",
                    "source_config": config,
                    "max_chars": max_chars,
                    "canvas_width": canvas_width,
                    "font_size": font_size,
                },
                sort_keys=True,
            ),
            "modality_label": "text",
            "preprocessing_version": _PREPROCESSING_VERSION,
        }
        if include_native:
            result["native_user_content"] = raw_text
            result["target_text"] = raw_text
            result["native_available"] = True
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
