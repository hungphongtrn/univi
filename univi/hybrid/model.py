"""
Issue #6 — a *randomly initialized* hybrid vision-language model that replicates
the **Gemma 4 12B image-embedder architecture** (``google/gemma-4-12B-it``, the
encoder-free ``gemma4_unified`` vision path) and attaches it to a small
**Qwen 3 1.7B** language backbone.

Rationale (research): every prior Univi finding was confounded by the pretrained
language prior of ``gemma-4-E2B-it`` (the "learnability-limited" verdict in
CLAUDE.md turned on the model falling into an ignore-the-image basin *because* it
already had strong text priors). Instantiating the *architecture* from config
with **random weights** — Gemma 4's vision embedder + multimodal projector, plus
a Qwen 3 1.7B text model — removes that confound: any pixel-reading the model
acquires is learned here, from these images, with no inherited prior.

What is replicated (architecture only, no pretrained weights):

- ``Gemma4UnifiedVisionEmbedder`` (:mod:`univi.hybrid.vision`) — the Gemma 4 12B
  image embedder. This is **not** a ViT: the 12B "unified" line's vision path is
  encoder-free — patchify, merge small patches into larger ones, and project raw
  merged pixels directly into LM space (Dense + LayerNorm + factorized position
  embeddings, no self-attention between patches). The 12B ``vision_config`` has
  no ``num_hidden_layers``/``num_attention_heads`` at all — it is architecturally
  distinct from E2B's 16-layer ViT ``gemma4_vision`` tower this repo used in the
  first cut of issue #6. "Only use the projector and process image directly."
  The projector (RMSNorm → Linear mapping the embedder's output into the
  language model's embedding space) is baked into
  ``Gemma4UnifiedVisionEmbedder`` as its last step, parameterized by
  ``(vision_config, text_config)`` so it targets Qwen's hidden size unchanged.
- ``Qwen3ForCausalLM`` — the smaller/faster LLM requested in issue #6, built from
  its config (``hidden_size=2048``, 28 layers) with random weights.

The soft-token merge mirrors Gemma 4 exactly: embed ``input_ids``, then
``masked_scatter`` the projected image soft tokens into the positions holding the
image placeholder token.

CPU-safe: heavy classes are imported at module load (they are pure-Python
transformers modules and import fine without CUDA); no weights are downloaded —
everything is instantiated from config.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from univi.hybrid.vision import Gemma4UnifiedVisionConfig, Gemma4UnifiedVisionEmbedder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class UniViHybridConfig(PretrainedConfig):
    """Config composing a Gemma 4 12B vision embedder with a Qwen 3 text backbone.

    Args:
        vision_config: a ``Gemma4UnifiedVisionConfig`` (or dict) — the
            encoder-free image embedder architecture to replicate.
        text_config: a ``Qwen3Config`` (or dict) — the language backbone.
        image_token_id: the placeholder token id in the *Qwen* vocabulary whose
            positions are overwritten by projected image soft tokens.
    """

    model_type = "univi_hybrid"
    sub_configs = {"vision_config": Gemma4UnifiedVisionConfig, "text_config": Qwen3Config}

    def __init__(
        self,
        vision_config: Any = None,
        text_config: Any = None,
        image_token_id: int = 151936,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = Gemma4UnifiedVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Gemma4UnifiedVisionConfig(**vision_config)

        if text_config is None:
            text_config = Qwen3Config()
        elif isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)

        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class UniViHybridForConditionalGeneration(PreTrainedModel):
    """Random-init Gemma-4-vision + Qwen-3 VLM with Gemma's soft-token merge."""

    config_class = UniViHybridConfig
    _supports_flash_attn = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate to the sub-towers, which each own their checkpointing."""
        for module in (self.language_model, self.vision_tower):
            if getattr(module, "supports_gradient_checkpointing", False):
                module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        for module in (self.language_model, self.vision_tower):
            if getattr(module, "supports_gradient_checkpointing", False):
                module.gradient_checkpointing_disable()

    def __init__(self, config: UniViHybridConfig) -> None:
        super().__init__(config)
        # Encoder-free: patch embedder + factorized position embeddings +
        # projector (RMSNorm -> Linear into text hidden size), no ViT layers.
        self.vision_tower = Gemma4UnifiedVisionEmbedder(
            config.vision_config, config.text_config
        )
        self.language_model = Qwen3ForCausalLM(config.text_config)
        self.image_token_id = config.image_token_id
        self.post_init()

    # -- embedding accessors (needed by generate / resize / Trainer) --
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    # -- vision path --
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode images and return their projected soft tokens, flattened.

        The encoder-free vision tower returns soft tokens for every merged
        patch *including padding* (padding rows carry ``image_position_ids ==
        -1``). We mask those out and flatten to
        ``[total_soft_tokens, text_hidden]``, in image-then-patch order, which
        maps one-to-one onto the image-placeholder positions the collator wrote
        into ``input_ids`` (it processes images in the same order).
        """
        hidden_states = self.vision_tower(
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
        valid = (image_position_ids != -1).any(dim=-1)
        return hidden_states[valid]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        num_soft_tokens_per_image: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, image_position_ids
            ).to(inputs_embeds.dtype)

            n_placeholders = int((input_ids == self.image_token_id).sum())
            if n_placeholders != image_features.shape[0]:
                raise ValueError(
                    f"Image placeholder count ({n_placeholders}) does not "
                    f"match projected soft tokens ({image_features.shape[0]}). "
                    "Collator/model soft-token accounting is out of sync."
                )
            special_image_mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def build_hybrid_config(
    vision_source: str,
    text_source: str,
    image_token_id: int,
) -> UniViHybridConfig:
    """Build a hybrid config from a Gemma-4-12B vision source and a Qwen-3 source.

    ``vision_source`` is a repo/path whose ``config.json`` carries a
    ``gemma4_unified`` ``vision_config`` (e.g. ``google/gemma-4-12B-it``).
    ``text_source`` is a Qwen 3 repo/path. Only configs are read; **no weights**
    are loaded.
    """
    from univi.hybrid.build import load_vision_config

    vision_config = load_vision_config(vision_source)
    text_config = Qwen3Config.from_pretrained(text_source)
    return UniViHybridConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=image_token_id,
    )
