from __future__ import annotations

import io
import json
from pathlib import Path

from datasets import Dataset, Image as HfImage, List, load_dataset
from pyarrow.parquet import ParquetFile

from data.preprocessing.render_utils import render_text_page, render_text_pages
from data.preprocessing.text_token_utils import compute_token_length, truncate_to_token_limit

_PREPROCESSING_VERSION = "0.3.0"
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
    max_chars: int | None = 2000,
    canvas_width: int = 1024,
    canvas_height: int | None = None,
    font_size: int = 14,
    max_output_tokens: int | None = None,
    tokenizer=None,
    tokenizer_name: str = "unsloth/gemma-4-E2B-it",
    include_native: bool = False,
    num_proc: int = 4,
    source_path: str | None = None,
) -> Dataset:
    if max_output_tokens is not None and tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if source_path is not None:
        config_dir = Path(source_path) / "sample" / config.removeprefix("sample-")
        parquet_files = sorted(config_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No cached FineWeb-Edu parquet files in {config_dir}")

        selected_files = []
        required_rows = offset + max_samples if max_samples is not None else None
        available_rows = 0
        for parquet_file in parquet_files:
            selected_files.append(str(parquet_file))
            available_rows += ParquetFile(parquet_file).metadata.num_rows
            if required_rows is not None and available_rows >= required_rows:
                break

        source = load_dataset(
            "parquet", data_files=selected_files, split=split, num_proc=num_proc
        )
        if max_samples is not None:
            source = source.select(range(offset, min(required_rows, len(source))))
    elif max_samples is not None:
        source = load_dataset(_SOURCE_DATASET_ID, config, split=f"{split}[{offset}:{offset + max_samples}]", num_proc=num_proc)
    else:
        source = load_dataset(_SOURCE_DATASET_ID, config, split=split, num_proc=num_proc)

    def _process_row(row, index: int):
        text_key = _find_text_key(row)
        raw_text = row[text_key]
        if max_chars is not None:
            raw_text = raw_text[:max_chars]

        images = (
            render_text_pages(
                raw_text,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                font_size=font_size,
            )
            if canvas_height is not None
            else [
                render_text_page(
                    raw_text, canvas_width=canvas_width, font_size=font_size
                )
            ]
        )
        image_bytes_list = []
        for image in images:
            buf = io.BytesIO()
            image.convert("L").save(buf, format="PNG", compress_level=1)
            image_bytes_list.append(buf.getvalue())

        target_text = truncate_to_token_limit(
            raw_text, tokenizer, max_output_tokens
        )

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in image_bytes_list],
                    {"type": "text", "text": _USER_INSTRUCTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            },
        ]

        source_id = row.get("id")
        row_id = str(source_id) if source_id not in (None, "") else str(offset + index)

        if tokenizer is not None:
            original_len = compute_token_length(raw_text, tokenizer)
        else:
            original_len = -1

        result = {
            "images": image_bytes_list,
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
                    "canvas_height": canvas_height,
                    "font_size": font_size,
                    "image_mode": "L",
                    "png_compression_level": 1,
                    "max_output_tokens": max_output_tokens,
                    "tokenizer_name": tokenizer_name if max_output_tokens else None,
                },
                sort_keys=True,
            ),
            "modality_label": "text",
            "preprocessing_version": _PREPROCESSING_VERSION,
            "original_token_length": original_len,
        }
        if include_native:
            result["native_user_content"] = raw_text
            result["target_text"] = target_text
            result["native_available"] = True
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
