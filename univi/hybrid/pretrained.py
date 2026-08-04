"""
Pretrained-vision hybrid: **pretrained** Gemma-4-12B unified vision embedder +
a from-scratch linear adapter + **pretrained** Qwen3-1.7B decoder.

This is the follow-up to the encoder-free thread (H1/H2/H3), which showed a
*from-scratch* tiny linear embedder + a strong pretrained decoder collapses to an
ignore-the-pixels basin every time (grounding → 0). Diagnosis: the vision path
needs to be *pretrained*, not the decoder's problem. So here we swap the
from-scratch embedder for the **pretrained** Gemma-4-12B "unified" vision embedder
(``model.vision_embedder.*`` + ``model.embed_vision.embedding_projection``, ~50M
params, range-downloaded to ``data/hybrid/gemma12b-vision/vision_embedder.pt``),
keep the reused Qwen3 decoder, and bridge the two with a single from-scratch
``Linear(3840 → qwen_hidden)`` adapter.

Architecture (the user's diagram):

    image → Gemma-12B unified vision embedder (PRETRAINED, → 3840 soft tokens)
          → Linear(3840 → 2048)         [from-scratch adapter — the only random part]
          → Qwen3-1.7B decoder (PRETRAINED)
          → Gemma-style masked_scatter merge into the placeholder positions

The Gemma embedder keeps its NATIVE 3840 output (its own pretrained
``embedding_projection`` [3840,3840]); the adapter — not Gemma's projector —
re-maps into Qwen's 2048 space. That preserves the most pretrained vision
computation and matches the checkpoint shapes exactly (see
``scratchpad/fetch_gemma12b_vision.py``).

Reuses the issue-#6 machinery: :class:`~univi.hybrid.vision.Gemma4UnifiedVisionEmbedder`,
:class:`~univi.hybrid.data.HybridCollator`, and the Gemma soft-token merge from
:class:`~univi.hybrid.model.UniViHybridForConditionalGeneration`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from univi.hybrid.vision import (
    DEFAULT_MAX_SOFT_TOKENS,
    Gemma4UnifiedVisionConfig,
    Gemma4UnifiedVisionEmbedder,
)

logger = logging.getLogger(__name__)

VISION_WEIGHTS = "data/hybrid/gemma12b-vision/vision_embedder.pt"


class UniViHybridPretrainedConfig(PretrainedConfig):
    """Config for the pretrained-vision hybrid.

    The vision embedder keeps its native output width (``output_proj_dims``, 3840);
    ``adapter_in`` records it so the from-scratch ``Linear(adapter_in → text hidden)``
    is sized correctly.

    ``max_soft_tokens`` is the per-image soft-token budget the *image processor*
    was run at during training (H17). It has no effect on module shapes — the
    vision embedder is budget-agnostic and its factorized position table already
    covers 1120 — but it MUST be recorded on the checkpoint, because a probe that
    re-processes images at a different budget feeds the model a geometry it was
    never trained on and silently produces garbage. Absent from a config (every
    checkpoint written before H17) ⇒ 280, the historical default.
    """

    model_type = "univi_hybrid_pretrained"
    sub_configs = {"vision_config": Gemma4UnifiedVisionConfig, "text_config": Qwen3Config}

    def __init__(
        self,
        vision_config: Any = None,
        text_config: Any = None,
        image_token_id: int = 151936,
        max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
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
        self.adapter_in = vision_config.output_proj_dims
        self.max_soft_tokens = int(max_soft_tokens)
        super().__init__(**kwargs)


class UniViHybridPretrained(PreTrainedModel):
    """Pretrained Gemma-12B vision embedder + from-scratch adapter + pretrained Qwen3."""

    config_class = UniViHybridPretrainedConfig
    _supports_flash_attn = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: UniViHybridPretrainedConfig) -> None:
        super().__init__(config)
        vc = config.vision_config
        # Build the Gemma embedder at its NATIVE output width so its pretrained
        # 3840→3840 projector loads verbatim. A shim stands in for "text_config"
        # (the embedder only reads ``.hidden_size`` for its projector's out dim).
        native = SimpleNamespace(hidden_size=vc.output_proj_dims)
        self.vision_tower = Gemma4UnifiedVisionEmbedder(vc, native)
        # From-scratch bridge into the decoder's space — the only random module.
        self.adapter = nn.Linear(vc.output_proj_dims, config.text_config.hidden_size)
        self.language_model = Qwen3ForCausalLM(config.text_config)
        self.image_token_id = config.image_token_id
        # Recorded, not enforced: the budget the images must be processed at.
        # ``getattr`` keeps pre-H17 configs (no such key) at the 280 default.
        self.max_soft_tokens = getattr(config, "max_soft_tokens", DEFAULT_MAX_SOFT_TOKENS)
        self.post_init()

    def freeze_vision(self) -> None:
        """Hard-freeze the pretrained vision embedder (LLaVA-style; the data shows
        it barely moves at 5e-5 anyway — rel-delta ~0 — so freezing is free and
        drops ~50M params of gradients/optimizer state)."""
        for p in self.vision_tower.parameters():
            p.requires_grad_(False)

    # -- param groups for the LR schedule (adapter > vision > decoder) --
    # Frozen (requires_grad=False) params are dropped, and empty groups omitted, so
    # this works for both the joint 3-group recipe and the frozen-vision 2-group one.
    def param_groups(self, lr_adapter: float, lr_vision: float, lr_decoder: float):
        candidates = [
            (self.adapter.parameters(), lr_adapter),
            (self.vision_tower.parameters(), lr_vision),
            (self.language_model.parameters(), lr_decoder),
        ]
        groups = []
        for params, lr in candidates:
            trainable = [p for p in params if p.requires_grad]
            if trainable:
                groups.append({"params": trainable, "lr": lr})
        return groups

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if getattr(self.language_model, "supports_gradient_checkpointing", False):
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if getattr(self.language_model, "supports_gradient_checkpointing", False):
            self.language_model.gradient_checkpointing_disable()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_position_ids: torch.LongTensor
    ) -> torch.Tensor:
        """Gemma soft tokens (native 3840) → adapter → decoder space, padding masked."""
        hidden = self.vision_tower(
            pixel_values=pixel_values, image_position_ids=image_position_ids
        )
        hidden = self.adapter(hidden.to(self.adapter.weight.dtype))
        valid = (image_position_ids != -1).any(dim=-1)
        return hidden[valid]

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
            n_ph = int((input_ids == self.image_token_id).sum())
            if n_ph != image_features.shape[0]:
                raise ValueError(
                    f"Image placeholder count ({n_ph}) != projected soft tokens "
                    f"({image_features.shape[0]}). Collator/model out of sync."
                )
            mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Build: assemble with PRETRAINED weights for both towers
# ---------------------------------------------------------------------------

IMAGE_PLACEHOLDER = "<|univi_image|>"


def resolve_max_soft_tokens(model_or_config: Any, override: int | None = None) -> int:
    """Soft-token budget a probe should process images at.

    ``override`` (an explicit ``--max-soft-tokens``) wins; otherwise the value the
    checkpoint recorded; otherwise :data:`DEFAULT_MAX_SOFT_TOKENS` (280), which is
    what every checkpoint written before H17 was trained at.
    """
    if override is not None:
        return int(override)
    config = getattr(model_or_config, "config", model_or_config)
    return int(getattr(config, "max_soft_tokens", DEFAULT_MAX_SOFT_TOKENS))


def build_image_processor(max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS):
    """The vendored Gemma-4 unified image processor at an explicit budget."""
    from univi.hybrid.vision import Gemma4UnifiedImageProcessor

    return Gemma4UnifiedImageProcessor(max_soft_tokens=int(max_soft_tokens))


def build_pretrained_hybrid(
    vision_source: str,
    text_source: str,
    vision_weights: str = VISION_WEIGHTS,
    dtype: torch.dtype = torch.bfloat16,
    max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
):
    """Assemble ``(model, tokenizer, image_processor, image_token_id)`` with both
    towers pretrained: Gemma-12B vision embedder (from ``vision_weights``) and
    Qwen3 decoder (``from_pretrained(text_source)``). Only the adapter is random.

    ``max_soft_tokens`` sets the image processor's per-image budget AND is stored
    on the model config, so a checkpoint trained at 1120 reloads at 1120.
    """
    from transformers import AutoTokenizer
    from univi.hybrid.build import load_vision_config

    tokenizer = AutoTokenizer.from_pretrained(text_source)
    if IMAGE_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    image_processor = build_image_processor(max_soft_tokens)

    vision_config = load_vision_config(vision_source)
    text_config = Qwen3Config.from_pretrained(text_source)
    text_config.vocab_size = len(tokenizer)

    config = UniViHybridPretrainedConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=image_token_id,
        max_soft_tokens=max_soft_tokens,
    )
    model = UniViHybridPretrained(config)

    # --- load PRETRAINED Qwen3 decoder weights over the random init ---
    pre_qwen = Qwen3ForCausalLM.from_pretrained(text_source, dtype=dtype)
    if pre_qwen.get_input_embeddings().weight.shape[0] != len(tokenizer):
        pre_qwen.resize_token_embeddings(len(tokenizer))
    missing, unexpected = model.language_model.load_state_dict(
        pre_qwen.state_dict(), strict=False
    )
    logger.info(
        "Qwen3 decoder: loaded pretrained (missing=%d unexpected=%d)",
        len(missing), len(unexpected),
    )
    del pre_qwen

    # --- load PRETRAINED Gemma-12B vision embedder weights ---
    if not Path(vision_weights).is_file():
        raise FileNotFoundError(
            f"{vision_weights} not found — run scratchpad/fetch_gemma12b_vision.py"
        )
    vsd = torch.load(vision_weights, map_location="cpu")
    vmiss, vunexp = model.vision_tower.load_state_dict(vsd, strict=False)
    if vmiss or vunexp:
        raise RuntimeError(
            f"vision embedder weight mismatch: missing={vmiss} unexpected={vunexp}"
        )
    logger.info("Gemma-12B vision embedder: loaded %d pretrained tensors", len(vsd))

    model = model.to(dtype)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, image_processor, image_token_id
