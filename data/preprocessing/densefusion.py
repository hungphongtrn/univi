from __future__ import annotations

import json

from datasets import Dataset, Image as HfImage, List, load_dataset

from data.preprocessing.text_token_utils import compute_token_length

_PREPROCESSING_VERSION = "0.2.0"
_SOURCE_DATASET_ID = "HuggingFaceM4/FineVision"


def preprocess_densefusion(
    subset: str = "densefusion_1m",
    split: str = "train",
    max_samples: int | None = None,
    offset: int = 0,
    include_native: bool = False,
    tokenizer=None,
    tokenizer_name: str = "unsloth/gemma-4-E2B-it",
    num_proc: int = 4,
) -> Dataset:
    load_kwargs = {"split": split, "num_proc": num_proc}
    if subset:
        load_kwargs["name"] = subset
    source = load_dataset(_SOURCE_DATASET_ID, **load_kwargs)
    if max_samples is not None:
        source = source.select(range(offset, min(offset + max_samples, len(source))))

    def _process_batch(batch, indices):
        num_rows = len(indices)
        messages_list = []
        source_dataset_ids = [_SOURCE_DATASET_ID] * num_rows
        splits = [split] * num_rows
        row_ids = [str(offset + idx) for idx in indices]
        render_configs = [
            json.dumps({"render_method": "preserve_source_image"}, sort_keys=True)
        ] * num_rows
        modality_labels = ["image-text"] * num_rows
        preprocessing_versions = [_PREPROCESSING_VERSION] * num_rows

        for i in range(num_rows):
            texts = batch["texts"][i]
            if not texts:
                raise ValueError("Row has no texts")
            user_text, assistant_text = texts[0]["user"], texts[0]["assistant"]

            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                },
            ])

        if tokenizer is not None:
            original_lengths = []
            for i in range(num_rows):
                raw = batch["texts"][i][0]["assistant"]
                original_lengths.append(compute_token_length(raw, tokenizer))
        else:
            original_lengths = [-1] * num_rows

        result = {
            "messages": messages_list,
            "source_dataset_id": source_dataset_ids,
            "split": splits,
            "row_id": row_ids,
            "render_config": render_configs,
            "modality_label": modality_labels,
            "preprocessing_version": preprocessing_versions,
            "original_token_length": original_lengths,
        }
        if include_native:
            result["native_user_content"] = [None] * num_rows
            result["target_text"] = []
            for i in range(num_rows):
                result["target_text"].append(batch["texts"][i][0]["assistant"])
            result["native_available"] = [True] * num_rows
            result["native_equals_image_only"] = [True] * num_rows
        return result

    columns_to_drop = [c for c in source.column_names if c != "images"]
    ds = source.map(
        _process_batch,
        batched=True,
        with_indices=True,
        remove_columns=columns_to_drop,
        num_proc=num_proc,
    )
    if len(ds) > 0:
        ds = ds.cast_column("images", List(HfImage()))
    return ds
