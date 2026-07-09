"""
Full training entry point for Gemma 4 E2B Phase 0 visual-unification mixture.

Usage:
    uv run python train_full.py --config configs/full.yaml
"""

from __future__ import annotations

import argparse

from univi.trainer import load_config, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = train(config)
    log = trainer.state.log_history
    if log:
        print(f"Training complete. Log history: {log[-1]}")
    print(f"Checkpoint saved to {config['training']['output_dir']}")


if __name__ == "__main__":
    main()
