"""
Assembly helpers for the issue-#6 hybrid model.

Turns two *config* sources — a Gemma 4 12B checkpoint (for its encoder-free
vision-embedder architecture) and a Qwen 3 checkpoint (for the text backbone +
tokenizer) — into a **randomly initialized**
:class:`UniViHybridForConditionalGeneration`, plus the tokenizer, image
processor and image-placeholder token id they must agree on.

No pretrained model weights are downloaded or loaded here: only ``config.json``
and tokenizer files are read (all tiny). The image processor is instantiated
locally (:class:`univi.hybrid.vision.Gemma4UnifiedImageProcessor`) rather than
fetched, since the 12B checkpoint's ``model_type`` (``gemma4_unified``) isn't
registered in this repo's pinned ``transformers`` release yet — see
:mod:`univi.hybrid.vision` for why. This keeps the build fast and, crucially,
keeps the model free of any inherited language/vision prior — the whole point
of issue #6.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from transformers import AutoTokenizer, Qwen3Config

from univi.hybrid.model import (
    UniViHybridConfig,
    UniViHybridForConditionalGeneration,
)
from univi.hybrid.vision import Gemma4UnifiedImageProcessor, Gemma4UnifiedVisionConfig

# Placeholder token spliced into the Qwen sequence, one per image soft token.
IMAGE_PLACEHOLDER = "<|univi_image|>"


def load_vision_config(vision_source: str) -> Gemma4UnifiedVisionConfig:
    """Read just the ``vision_config`` sub-dict out of a Gemma-4-12B ``config.json``.

    ``AutoConfig.from_pretrained`` can't be used here: the checkpoint's
    ``model_type`` (``gemma4_unified``) isn't recognized by this repo's pinned
    ``transformers`` release, so we fetch the raw JSON and parse it ourselves.
    Only the tiny ``config.json`` is fetched — never weights.
    """
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=vision_source, filename="config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    vision_dict = raw_config.get("vision_config")
    if vision_dict is None:
        raise ValueError(f"{vision_source!r} has no vision_config to replicate.")
    return Gemma4UnifiedVisionConfig(**vision_dict)


def load_tokenizer_and_processor(
    text_source: str,
    image_placeholder: str = IMAGE_PLACEHOLDER,
) -> tuple[Any, Any, int]:
    """Load the Qwen tokenizer + the encoder-free image processor, add the image token.

    Returns ``(tokenizer, image_processor, image_token_id)``. The image token is
    registered as an additional special token if not already present. The image
    processor needs no remote source — it has no learned parameters (patchify +
    merge is pure arithmetic), so it's built locally with the Gemma-4-12B
    defaults (``patch_size=16``, ``pooling_kernel_size=3``, ``max_soft_tokens=280``).
    """
    tokenizer = AutoTokenizer.from_pretrained(text_source)
    if image_placeholder not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [image_placeholder]}
        )
    image_token_id = tokenizer.convert_tokens_to_ids(image_placeholder)
    image_processor = Gemma4UnifiedImageProcessor()
    return tokenizer, image_processor, image_token_id


def build_model(
    vision_source: str,
    text_source: str,
    image_token_id: int,
    vocab_size: int,
    dtype: torch.dtype | None = None,
) -> UniViHybridForConditionalGeneration:
    """Instantiate the random-init hybrid model from config sources.

    Args:
        vision_source: repo/path whose ``config.json`` carries a
            ``vision_config`` — the Gemma 4 12B encoder-free image-embedder
            architecture to replicate (e.g. ``google/gemma-4-12B-it``).
        text_source: a Qwen 3 repo/path (e.g. the 1.7B config).
        image_token_id: id of the image-placeholder token in the tokenizer.
        vocab_size: tokenizer length *after* adding the image token; the text
            config's vocab and the tied embeddings are sized to match.
        dtype: optional dtype for the instantiated weights.
    """
    vision_config = load_vision_config(vision_source)

    text_config = Qwen3Config.from_pretrained(text_source)
    text_config.vocab_size = vocab_size

    config = UniViHybridConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=image_token_id,
    )
    model = UniViHybridForConditionalGeneration(config)
    if dtype is not None:
        model = model.to(dtype)
    return model


def build_all(
    vision_source: str,
    text_source: str,
    dtype: torch.dtype | None = None,
) -> tuple[UniViHybridForConditionalGeneration, Any, Any, int]:
    """One-call assembly: ``(model, tokenizer, image_processor, image_token_id)``.

    The model's token embeddings are resized to the tokenizer length so the added
    image-placeholder token has a (random) embedding row.
    """
    tokenizer, image_processor, image_token_id = load_tokenizer_and_processor(
        text_source
    )
    model = build_model(
        vision_source=vision_source,
        text_source=text_source,
        image_token_id=image_token_id,
        vocab_size=len(tokenizer),
        dtype=dtype,
    )
    # Ensure embeddings cover the (possibly newly added) image token id.
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, image_processor, image_token_id
