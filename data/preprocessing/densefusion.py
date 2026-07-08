from __future__ import annotations

from datasets import Dataset, load_dataset

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row[image_key], "text": None},
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

        return {
            "messages": messages,
            "source_dataset_id": _SOURCE_DATASET_ID,
            "split": split,
            "row_id": row.get("id", str(index)),
            "render_config": {"render_method": "preserve_source_image"},
            "modality_label": "image-text",
            "preprocessing_version": _PREPROCESSING_VERSION,
        }

    return source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
    )
