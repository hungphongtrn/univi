#!/usr/bin/env python3
"""
CLI entry point for UniVi Gemma 4 E2B training.

Pure delegation to `univi.cli.main` — no argument parsing, no config loading,
no trainer imports.

Usage:
    uv run python -m univi.train --config configs/smoke.yaml
    uv run python -m univi.train --config configs/3060_1epoch.yaml --dry-run
    uv run python -m univi.train --help
"""

from __future__ import annotations

import sys


def main() -> None:
    from univi.cli import main as cli_main

    sys.exit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
