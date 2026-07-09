from __future__ import annotations

import io
import json

from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.render_utils import render_text_page

_PREPROCESSING_VERSION = "0.1.0"
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
    max_chars: int = 2000,
    canvas_width: int = 1024,
    font_size: int = 14,
    include_native: bool = False,
) -> Dataset:
    source = load_dataset(_SOURCE_DATASET_ID, config, split=split)
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    def _process_row(row, index: int):
        instruction, response = _extract_instruction_and_response(row)

        if max_chars is not None:
            instruction = instruction[:max_chars]

        image = render_text_page(
            instruction, canvas_width=canvas_width, font_size=font_size
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
                    {"type": "text", "text": response},
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
                    "config": config,
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
            result["native_user_content"] = instruction
            result["target_text"] = response
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
