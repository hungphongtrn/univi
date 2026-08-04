"""
Training entry point for the issue-#6 random-init hybrid model.

Unlike :mod:`univi.train` (Unsloth ``FastVisionModel`` + TRL), this path drives a
plain ``transformers.Trainer`` because the model is a bespoke
:class:`UniViHybridForConditionalGeneration`, not an Unsloth-wrapped pretrained
VLM. It reuses the repo's dataset loader (``univi.trainer.load_dataset``) so the
same materialized ``messages``+``images`` rows feed both experiments.

Usage:
    uv run python -m univi.hybrid.train --config configs/hybrid_smoke.yaml

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import argparse
import logging

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_training_args(tc: dict):
    from transformers import TrainingArguments

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=tc["output_dir"],
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 8),
        learning_rate=tc.get("learning_rate", 3e-4),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        weight_decay=tc.get("weight_decay", 0.01),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        num_train_epochs=tc.get("num_train_epochs", 1),
        max_steps=tc.get("max_steps", -1),
        logging_steps=tc.get("logging_steps", 10),
        save_steps=tc.get("save_steps", 1000),
        save_total_limit=tc.get("save_total_limit", 2),
        optim=tc.get("optim", "adamw_torch"),
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        dataloader_num_workers=tc.get("dataloader_num_workers", 0),
        report_to=tc.get("report_to", ["wandb"]),
        remove_unused_columns=False,
        seed=tc.get("seed", 3407),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train the hybrid Univi model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    mc = config["model"]
    tc = config["training"]
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps

    from transformers import Trainer

    from univi.hybrid.build import build_all
    from univi.hybrid.data import HybridCollator
    from univi.trainer import load_dataset

    logger.info("Assembling random-init hybrid model")
    model, tokenizer, image_processor, image_token_id = build_all(
        vision_source=mc["vision_source"],
        text_source=mc["text_source"],
    )
    if tc.get("gradient_checkpointing", True):
        model.config.use_cache = False
    logger.info(
        "Model assembled: %.1fM params (image_token_id=%d, vocab=%d)",
        sum(p.numel() for p in model.parameters()) / 1e6,
        image_token_id,
        len(tokenizer),
    )

    dataset = load_dataset(config, source="auto")
    logger.info("Dataset loaded: %d rows", len(dataset))

    collator = HybridCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_id=image_token_id,
        max_length=tc.get("max_length", 4096),
        response_only=tc.get("loss_masking", "response_only") == "response_only",
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(tc),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    out = tc["output_dir"] + "/final"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    image_processor.save_pretrained(out)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
