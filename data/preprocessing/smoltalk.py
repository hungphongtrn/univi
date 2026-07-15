from __future__ import annotations

import io
import json

from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import render_text_page, render_text_pages
from data.preprocessing.text_token_utils import compute_token_length, truncate_to_token_limit

_PREPROCESSING_VERSION = "0.3.0"
_USER_INSTRUCTION = "Follow the instruction shown in the image."
_SOURCE_DATASET_ID = "HuggingFaceTB/smoltalk"


def _extract_instruction_and_response(row: dict) -> tuple[str, str]:
    if "messages" in row:
        msgs = row["messages"]
        user_idx = None
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            raise KeyError(
                f"No user message found in 'messages'. "
                f"Available roles: {[m.get('role', '') for m in msgs]}"
            )
        instruction = msgs[user_idx].get("content", "")
        if not isinstance(instruction, str):
            instruction = ""
        for m in msgs[user_idx + 1 :]:
            if m.get("role") == "assistant":
                response = m.get("content", "")
                if not isinstance(response, str):
                    response = ""
                return instruction, response
        raise KeyError(
            "No assistant message found after user message in 'messages'."
        )

    if "conversations" in row:
        convs = row["conversations"]
        user_roles = {"human", "user"}
        assistant_roles = {"assistant", "gpt"}

        user_idx = None
        for i, c in enumerate(convs):
            role = c.get("role") or c.get("from") or ""
            if role.lower() in user_roles:
                user_idx = i
                break
        if user_idx is None:
            raise KeyError(
                f"No human/user turn found in 'conversations'. "
                f"Available roles: {[c.get('role', c.get('from', '')) for c in convs]}"
            )
        instruction = (
            convs[user_idx].get("value") or convs[user_idx].get("content") or ""
        )
        if not isinstance(instruction, str):
            instruction = ""
        for c in convs[user_idx + 1 :]:
            role = c.get("role") or c.get("from") or ""
            if role.lower() in assistant_roles:
                response = c.get("value") or c.get("content") or ""
                if not isinstance(response, str):
                    response = ""
                return instruction, response
        raise KeyError(
            "No assistant/gpt turn found after user turn in 'conversations'."
        )

    instruction_keys = ["instruction", "prompt", "question"]
    response_keys = ["response", "answer", "output"]

    instruction = None
    for key in instruction_keys:
        if key in row:
            instruction = row[key]
            break

    response = None
    for key in response_keys:
        if key in row:
            response = row[key]
            break

    if instruction is None and response is None:
        raise KeyError(
            f"Cannot find instruction or response. "
            f"Searched instruction keys: {instruction_keys}, "
            f"response keys: {response_keys}. "
            f"Available keys: {list(row.keys())}"
        )
    if instruction is None:
        raise KeyError(
            f"Cannot find instruction. "
            f"Searched keys: {instruction_keys}. "
            f"Available keys: {list(row.keys())}"
        )
    if response is None:
        raise KeyError(
            f"Cannot find response. "
            f"Searched keys: {response_keys}. "
            f"Available keys: {list(row.keys())}"
        )

    return instruction, response


def preprocess_smoltalk(
    config: str = "all",
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
    source = load_dataset(
        source_path or _SOURCE_DATASET_ID, config, split=split, num_proc=num_proc
    )
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    def _process_row(row, index: int):
        instruction, response = _extract_instruction_and_response(row)

        if max_chars is not None:
            instruction = instruction[:max_chars]

        images = (
            render_text_pages(
                instruction,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                font_size=font_size,
            )
            if canvas_height is not None
            else [
                render_text_page(
                    instruction, canvas_width=canvas_width, font_size=font_size
                )
            ]
        )
        image_bytes_list = []
        for image in images:
            buf = io.BytesIO()
            image.convert("L").save(buf, format="PNG", compress_level=1)
            image_bytes_list.append(buf.getvalue())

        target_text = truncate_to_token_limit(
            response, tokenizer, max_output_tokens
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
            original_len = compute_token_length(response, tokenizer)
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
                    "config": config,
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
            result["native_user_content"] = instruction
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
