"""
Smoke training entry point for Gemma 4 E2B Phase 0 visual-unification mixture.

Usage:
    uv run python train_smoke.py --config configs/smoke.yaml

Loss masking:
    UnslothVisionDataCollator applies response-only loss masking automatically
    via the chat template processing — user-turn tokens receive -100 labels
    so they do not contribute to the loss. No explicit masking step is needed.
"""

from __future__ import annotations

import argparse

from univi.trainer import (
    apply_lora,
    build_model,
    load_config,
    load_dataset,
    train,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = train(config)
    log = trainer.state.log_history
    if log:
        print(f"Training complete. Log history: {log[-1]}")
    print(f"Checkpoint saved to {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
