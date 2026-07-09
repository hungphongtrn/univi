from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from datasets import Dataset, Image as HfImage, List, load_dataset
from huggingface_hub import hf_hub_download
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


def _resolve_densefusion_image(image_path: str) -> Image.Image:
    parts = image_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            "DenseFusion image_path must have format "
            f"'<subset>/<batch_id>/<filename>', got {image_path!r}"
        )
    zip_path = f"images/{parts[0]}/{parts[1]}.zip"
    local_zip = hf_hub_download(
        repo_id=_SOURCE_DATASET_ID,
        repo_type="dataset",
        filename=zip_path,
    )
    with zipfile.ZipFile(local_zip) as zf:
        candidates = list(
            dict.fromkeys(
                [
                    image_path,
                    "/".join(parts[1:]),
                    parts[-1],
                ]
            )
        )
        names = set(zf.namelist())
        for candidate in candidates:
            if candidate in names:
                with zf.open(candidate) as img_file:
                    return Image.open(img_file).convert("RGB")
        raise ValueError(
            f"Could not find DenseFusion image {image_path!r} in {zip_path!r}; "
            f"tried archive members {candidates!r}"
        )


def _resolve_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return Image.open(path).convert("RGB")
        if value.startswith(("http://", "https://", "hf://")):
            raise ValueError(
                f"Image value is a URL which is not supported: {value!r}. "
                "Use a local file path or HF dataset repo-relative path instead."
            )
        local_path = hf_hub_download(
            repo_id=_SOURCE_DATASET_ID,
            repo_type="dataset",
            filename=value,
        )
        return Image.open(local_path).convert("RGB")
    raise TypeError(
        f"Expected PIL Image or file path string, got {type(value).__name__}: {value!r}"
    )


def preprocess_densefusion(
    subset: str = "DenseFusion-4V-100K",
    split: str = "train",
    max_samples: int | None = None,
    offset: int = 0,
    include_native: bool = False,
) -> Dataset:
    load_kwargs = {"split": split}
    if subset:
        load_kwargs["name"] = subset
    source = load_dataset("BAAI/DenseFusion-1M", **load_kwargs)
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    def _process_row(row, index: int):
        image_key = _find_key(row, ["image_path", "image", "jpg", "png"], "image")
        text_key = _find_key(
            row, ["description", "caption", "text", "output"], "description"
        )

        if image_key == "image_path":
            image = _resolve_densefusion_image(row[image_key])
        else:
            image = _resolve_image(row[image_key])

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
                    {"type": "text", "text": row[text_key]},
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
                {"render_method": "preserve_source_image"}, sort_keys=True
            ),
            "modality_label": "image-text",
            "preprocessing_version": _PREPROCESSING_VERSION,
        }
        if include_native:
            result["native_user_content"] = None
            result["target_text"] = row[text_key]
            result["native_available"] = True
            result["native_equals_image_only"] = True
        return result

    ds = source.map(
        _process_row,
        with_indices=True,
        remove_columns=source.column_names,
    )
    if len(ds) > 0:
        ds = ds.cast_column("images", List(HfImage()))
    return ds
