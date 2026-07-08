from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_dataset
from PIL import Image

_PREPROCESSING_VERSION = "0.1.0"
_USER_INSTRUCTION = "Describe this image."
_SOURCE_DATASET_ID = "BAAI/DenseFusion-1M"


def _find_key(row: dict, candidates: list[str], purpose: str) -> str:
    for key in candidates:
        if key in row:
            return key
    raise KeyError(
        f"None of {candidates} found in row for {purpose}. "
        f"Available keys: {list(row.keys())}"
    )


def _resolve_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return Image.open(path)
        raise ValueError(
            f"Image value is a string but not a valid file path: {value!r}"
        )
    raise TypeError(
        f"Expected PIL Image or file path string, got {type(value).__name__}: {value!r}"
    )


def preprocess_densefusion(
    subset: str = "default",
    split: str = "train",
    max_samples: int | None = None,
) -> Dataset:
    load_kwargs = {"split": split}
    if subset:
        load_kwargs["name"] = subset
    source = load_dataset("BAAI/DenseFusion-1M", **load_kwargs)
    if max_samples is not None:
        source = source.select(range(min(max_samples, len(source))))

    def _process_row(row, index: int):
        image_key = _find_key(row, ["image", "jpg", "png"], "image")
        text_key = _find_key(
            row, ["description", "caption", "text", "output"], "description"
        )

        image = _resolve_image(row[image_key])

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
                    {"type": "text", "image": None, "text": row[text_key]},
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
                {"render_method": "preserve_source_image"}, sort_keys=True
            ),
            "modality_label": "image-text",
            "preprocessing_version": _PREPROCESSING_VERSION,
        }

    return source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
    )
