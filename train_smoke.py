"""
Smoke training entry point for Gemma 4 E2B visual-unification mixture.
Delegates to univi.cli.main for all mode dispatch.

Usage:
    uv run python train_smoke.py --config configs/smoke.yaml
    uv run python train_smoke.py --config configs/smoke.yaml --dry-run
    uv run python train_smoke.py --help
"""

from __future__ import annotations

import sys


def main():
    from univi.cli import main as cli_main

    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
