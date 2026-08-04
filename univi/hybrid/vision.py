"""
Encoder-free vision embedder for the issue-#6 hybrid model, replicating the
**Gemma 4 12B** ``gemma4_unified`` image path (``google/gemma-4-12B-it``) rather
than the ViT-based ``gemma4`` (E2B) vision tower used in the rest of this repo.

Correction vs. the first cut of issue #6: the 12B "unified" line does **not**
run a 16-layer ViT encoder over image patches at all. Its vision path is
encoder-free (see the HuggingFaceM4 "Encoder-free VLM" writeup this replicates
the spirit of): patchify the image, merge small patches into larger ones, and
project the raw merged pixels straight into the language-model embedding space
with a Dense + LayerNorm stack — no self-attention between patches. "Only use
the projector and process image directly" means exactly this: no vision
transformer, only a patch embedder + linear projector.

Provenance: ``google/gemma-4-12B-it``'s ``config.json`` carries a
``gemma4_unified_vision`` sub-config with no ``num_hidden_layers`` /
``num_attention_heads`` fields at all (confirmed by inspecting the checkpoint's
raw config) — architecturally distinct from the E2B ``gemma4_vision`` config
(16-layer ViT) this repo used previously. The classes below are adapted from
``transformers.models.gemma4_unified`` (``modeling_gemma4_unified.py`` /
``image_processing_gemma4_unified.py`` / ``configuration_gemma4_unified.py``,
Apache-2.0, Copyright 2026 The HuggingFace Team), which is not yet present in
this repo's pinned ``transformers==5.5.0`` — vendored here (config + the
vision-only embedder + image processor; no audio/text/generation code) rather
than bumping the shared ``transformers`` dependency, since the rest of the repo
(Unsloth ``FastVisionModel`` training) pins against the installed version.

No pretrained weights: everything here is instantiated from config with random
weights, same as the rest of ``univi/hybrid_*``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, PILImageResampling
from transformers.processing_utils import ImagesKwargs, Unpack


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@strict
class Gemma4UnifiedVisionConfig(PreTrainedConfig):
    """Encoder-free vision config (``google/gemma-4-12B-it``'s ``vision_config``).

    Args:
        patch_size: teacher-patch size in pixels, before merging.
        pooling_kernel_size: side of the square block of teacher patches merged
            into one model patch. Merged patch side = ``patch_size *
            pooling_kernel_size`` (default 16*3 = 48px).
        mm_embed_dim: hidden width of the patch-embedding Dense projection.
        mm_posemb_size: size of the factorized row/col positional-embedding
            table (shape ``(mm_posemb_size, 2, mm_embed_dim)``).
        output_proj_dims: input width of the final multimodal-embedder
            projection; must equal ``mm_embed_dim`` (the vision pipeline's
            actual output width) for the projector's Linear to line up — the
            projector's *output* width is the text model's ``hidden_size``,
            supplied separately at construction time.
    """

    model_type = "gemma4_unified_vision"

    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 3840
    initializer_range: float = 0.02

    @property
    def model_patch_size(self) -> int:
        return self.patch_size * self.pooling_kernel_size

    @model_patch_size.setter
    def model_patch_size(self, value: int) -> None:
        if value != self.patch_size * self.pooling_kernel_size:
            raise ValueError(
                f"`model_patch_size` needs to be equal to {self.patch_size = } "
                f"* {self.pooling_kernel_size = }"
            )


# ---------------------------------------------------------------------------
# Modeling: patch embedder + projector, no self-attention between patches
# ---------------------------------------------------------------------------


class Gemma4UnifiedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class Gemma4UnifiedMultimodalEmbedder(nn.Module):
    """RMSNorm -> Linear, projecting soft tokens into the language model space."""

    def __init__(self, multimodal_config: Any, text_config: Any) -> None:
        super().__init__()
        self.multimodal_hidden_size = multimodal_config.output_proj_dims
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = nn.Linear(
            self.multimodal_hidden_size, self.text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = Gemma4UnifiedRMSNorm(
            self.multimodal_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if (target_dtype := self.embedding_projection.weight.dtype).is_floating_point:
            inputs_embeds = inputs_embeds.to(target_dtype)
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision embedder: projects raw merged pixel patches into LM space.

    Replaces the entire ViT-style vision tower. Instead of attention layers, a
    Dense projection with LayerNorm and factorized positional embeddings turns
    raw merged patch pixels directly into soft tokens.

    Pipeline: raw_patches -> LN -> Dense -> LN -> +factorized_posemb -> LN -> RMSNorm -> Linear
    """

    def __init__(self, vision_config: Gemma4UnifiedVisionConfig, text_config: Any) -> None:
        super().__init__()
        patch_dim = vision_config.model_patch_size**2 * 3
        mm_embed_dim = vision_config.mm_embed_dim

        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, 2, mm_embed_dim)
        )
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

        self.multimodal_embedder = Gemma4UnifiedMultimodalEmbedder(vision_config, text_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: ``(num_images, num_patches, model_patch_size**2 * 3)``
                — raw merged pixel patches.
            image_position_ids: ``(num_images, num_patches, 2)`` — integer XY
                positions (``-1`` for padding).

        Returns:
            ``(num_images, num_patches, text_hidden_size)`` — soft tokens,
            including padding positions (the caller must mask them out).
        """
        if (target_dtype := self.patch_dense.weight.dtype).is_floating_point:
            pixel_values = pixel_values.to(target_dtype)
        hidden_states = self.patch_ln1(pixel_values)
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        hidden_states = hidden_states + pos_embs
        hidden_states = self.pos_norm(hidden_states)

        return self.multimodal_embedder(hidden_states)


# ---------------------------------------------------------------------------
# Image processing: patchify + merge into model patches, no resampling/conv
# ---------------------------------------------------------------------------


class Gemma4UnifiedImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
#: Budget every code path uses when nothing else is specified. Raising it (H17)
#: is a per-run *config* choice — this constant must stay 280 so that every
#: existing checkpoint/probe keeps its original geometry.
DEFAULT_MAX_SOFT_TOKENS = 280


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """Largest aspect-ratio-preserving size fitting the patch budget.

    Target dimensions are the largest that (1) produce at most ``max_patches``
    patches when patchified with ``patch_size`` and (2) have height and width
    divisible by ``pooling_kernel_size * patch_size``.
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be "
            f"divisible by `pooling_kernel_size * patch_size`={side_mult}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(channels, height, width)`` -> ``(num_patches, patch_size**2 * channels)``."""
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    return patched_image.reshape(num_patches_height * num_patches_width, -1)


def pad_along_first_dim(
    image: torch.Tensor, positions: torch.Tensor, target_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return image, positions


def patches_merge(
    patches: torch.Tensor,
    positions_xy: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge k x k groups of small patches into larger patches.

    Given ``L`` input patches of dimension ``D = patch_size**2 * 3``, merge
    groups of ``k x k`` spatially adjacent patches into ``length`` output
    patches of dimension ``(k * patch_size)**2 * 3``. Grouping is determined by
    integer-dividing the XY positions by ``k``.
    """
    patch_size = math.isqrt(patches.shape[-1] // 3)
    if patches.shape[-1] != patch_size * patch_size * 3:
        raise ValueError(f"Patch dimension {patches.shape[-1]} is not a valid `patch_size * patch_size * 3`")

    k = math.isqrt(patches.shape[-2] // length)
    if k * k * length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {length}")

    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]

    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = position_within_kernel[..., 0] + position_within_kernel[..., 1] * k
    target_ordering = num_patches_from_top_left_of_kernel + num_patches_from_top_left

    perm = target_ordering.long().argsort(dim=-1)
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)

    batch_shape = patches.shape[:-2]
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * k, patch_size, patch_size, 3)
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k, k, patch_size, patch_size, 3)
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )
    merged_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * patch_size * k * patch_size * 3)

    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())

    padding = (positions_xy == -1).all(dim=-1, keepdim=True)
    kernel_ordered_positions = kernel_ordered_positions * (~padding).float() + positions_xy.float() * padding.float()

    kernel_ordered_positions = kernel_ordered_positions.reshape(*batch_shape, length, k * k, 2)
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)

    return merged_patches, new_positions


class Gemma4UnifiedImageProcessor(TorchvisionBackend):
    """Patchify + merge, no resampling filters or conv — raw pixels straight through."""

    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    patch_size = 16
    max_soft_tokens = DEFAULT_MAX_SOFT_TOKENS
    pooling_kernel_size = 3
    valid_kwargs = Gemma4UnifiedImageProcessorKwargs
    model_input_names = ["pixel_values", "image_position_ids", "num_soft_tokens_per_image"]

    def __init__(self, **kwargs: Unpack[Gemma4UnifiedImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)
        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}."
            )

    def _validate_preprocess_kwargs(self, **kwargs: Any) -> None:
        # Aspect-ratio-preserving resize is driven by patch_size / max_soft_tokens
        # / pooling_kernel_size, not the standard `size` kwarg.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        image: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: Any,
    ) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )
        if target_height == height and target_width == width:
            return image
        return tvF.resize(
            image, size=[target_height, target_width], interpolation=resample, antialias=True
        )

    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[Gemma4UnifiedImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        resample: Any,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: Any,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")

        max_patches = max_soft_tokens * pooling_kernel_size**2

        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            if do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image=image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            patch_height = image.shape[-2] // patch_size
            patch_width = image.shape[-1] // patch_size
            teacher_patches = convert_image_to_patches(image, patch_size)

            device = image.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            teacher_positions = torch.stack(patch_grid, dim=-1).reshape(teacher_patches.shape[0], 2)

            num_model_patches = teacher_patches.shape[0] // (pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(
                teacher_patches.unsqueeze(0), teacher_positions.unsqueeze(0), num_model_patches
            )
            merged_patches = merged_patches.squeeze(0)
            merged_positions = merged_positions.squeeze(0)
            num_soft_tokens_per_image.append(merged_patches.shape[0])

            merged_patches, merged_positions = pad_along_first_dim(merged_patches, merged_positions, max_soft_tokens)
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        pixel_values = torch.stack(pixel_values, dim=0)
        position_ids = torch.stack(position_ids, dim=0)

        data = {
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = [
    "DEFAULT_MAX_SOFT_TOKENS",
    "Gemma4UnifiedVisionConfig",
    "Gemma4UnifiedRMSNorm",
    "Gemma4UnifiedMultimodalEmbedder",
    "Gemma4UnifiedVisionEmbedder",
    "Gemma4UnifiedImageProcessor",
    "Gemma4UnifiedImageProcessorKwargs",
]
