"""
Visual-grounding ablation for the issue-#6 hybrid model (random-init Gemma
vision embedder + Qwen 3 1.7B).

Reuses the model-*independent* machinery in ``eval_ablation.py`` (donor-image
permutation, blank-image construction, paired-loss deltas, grounding flag,
report printing) by monkeypatching its model-dependent seams --
``load_model`` / ``compute_loss`` -- for :class:`UniViHybridForConditionalGeneration`
+ :class:`~univi.hybrid.data.HybridCollator` instead of Unsloth's
``FastVisionModel``. Everything else (``run_ablation``, ``print_report``,
``build_permutation``, ``build_blanks``) is untouched.

There is no pretrained checkpoint for this architecture, so "base_model" in
the config means a *freshly random-initialized* model (same vision/text
config sources, different random weights) -- the meaningful baseline is
"does training move Δperm away from the random-init model's Δperm", not
"does this model differ from some pretrained donor".

Usage:
    uv run python -m univi.hybrid.ablation --config configs/hybrid_ablation.yaml \
        --checkpoint data/checkpoints/hybrid-bounded-v0/checkpoint-3000 --no-base
"""

from __future__ import annotations

import torch

import eval_ablation
from eval_lane import _merge_images, load_source_dataset

from univi.hybrid.build import build_all
from univi.hybrid.data import HybridCollator
from univi.hybrid.model import UniViHybridForConditionalGeneration

_RANDOM_PREFIX = "random:"


def _extract_images(messages: list[dict]) -> list:
    """Undo ``_merge_images``: pull spliced ``image`` values back into a flat
    list, in encounter order -- the shape :class:`HybridCollator` expects."""
    images = []
    for msg in messages:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "image":
                images.append(content["image"])
    return images


def load_model(model_spec: str):
    """Load a hybrid model + a ready-to-use :class:`HybridCollator`.

    ``model_spec`` is either a checkpoint directory (``from_pretrained``), or
    ``"random:<vision_source>,<text_source>"`` for a fresh random-init model
    built the same way training does (:func:`univi.hybrid.build.build_all`).
    """
    if model_spec.startswith(_RANDOM_PREFIX):
        vision_source, text_source = model_spec[len(_RANDOM_PREFIX):].split(",", 1)
        model, tokenizer, image_processor, image_token_id = build_all(
            vision_source=vision_source, text_source=text_source
        )
    else:
        from transformers import AutoTokenizer

        from univi.hybrid.vision import Gemma4UnifiedImageProcessor

        model = UniViHybridForConditionalGeneration.from_pretrained(model_spec)
        tokenizer = AutoTokenizer.from_pretrained(model_spec)
        image_processor = Gemma4UnifiedImageProcessor()
        image_token_id = model.config.image_token_id

    if torch.cuda.is_available():
        model = model.to(torch.bfloat16).cuda()
    model.eval()

    collator = HybridCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_id=image_token_id,
        response_only=True,
    )
    return model, collator


def compute_loss(model, collator, messages: list[dict]) -> float:
    images = _extract_images(messages)
    batch = collator([{"messages": messages, "images": images}])
    batch = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in batch.items()}
    with torch.no_grad():
        return model(**batch).loss.item()


# Wire the hybrid seams into eval_ablation's module namespace -- its
# `evaluate_model_source` / `run_ablation` look up `load_model` / `compute_loss`
# / `_merge_images` / `load_source_dataset` as globals in *this* module at call
# time, so reassigning them here redirects the whole model-independent pipeline
# without touching eval_ablation.py itself.
eval_ablation.load_model = load_model
eval_ablation.compute_loss = compute_loss
eval_ablation._merge_images = _merge_images
eval_ablation.load_source_dataset = load_source_dataset


def main(argv: list[str] | None = None) -> None:
    eval_ablation.main(argv)


if __name__ == "__main__":
    main()
