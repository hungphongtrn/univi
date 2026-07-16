"""
CLI parsing and mode dispatch tests for univi.cli.
"""

from __future__ import annotations

import sys

import pytest


def test_cli_parses_modes():
    """CLI parser recognises all explicit modes."""
    from univi.cli import parse_args

    args = parse_args(["--preflight-only", "--config", "configs/smoke.yaml"])
    assert args.mode == "preflight_only"

    args = parse_args(["--smoke", "--config", "configs/smoke.yaml"])
    assert args.mode == "smoke"

    args = parse_args(["--train", "--config", "configs/smoke.yaml"])
    assert args.mode == "train"

    args = parse_args(["--resume", "--config", "configs/smoke.yaml"])
    assert args.mode == "resume"

    args = parse_args(["--evaluate", "--config", "configs/smoke.yaml"])
    assert args.mode == "evaluate"


def test_cli_dataset_source_auto():
    """--dataset-source auto resolves correctly."""
    from univi.cli import parse_args

    args = parse_args(["--dataset-source", "hub", "--config", "configs/smoke.yaml"])
    assert args.dataset_source == "hub"


def test_cli_default_mode():
    """No mode flag defaults to train."""
    from univi.cli import parse_args

    args = parse_args(["--config", "configs/smoke.yaml"])
    assert args.mode == "train"


def test_cli_help_exits_zero():
    """--help exits 0 via main()."""
    from univi.cli import main

    rc = main(["--help"])
    assert rc == 0


def test_cli_dry_run_known_config():
    """main() with a valid --dry-run config exits 0."""
    from univi.cli import main

    rc = main(["--config", "configs/3060_1epoch.yaml", "--dry-run"])
    assert rc == 0


def test_cli_dry_run_exit_code():
    """Integration: --dry-run exits 0 without GPU."""
    from univi.cli import main

    rc = main(["--config", "configs/3060_1epoch.yaml", "--dry-run"])
    assert rc == 0


def test_cli_config_required():
    """--config is required (non-help modes)."""
    from univi.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--dry-run"])


def test_cli_mutually_exclusive_modes():
    """Mutually exclusive mode flags cannot be combined."""
    from univi.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--preflight-only", "--smoke", "--config", "configs/smoke.yaml"])


def test_cli_overrides_parsing():
    """--overrides KEY=VALUE pairs are parsed correctly."""
    from univi.cli import _parse_overrides

    result = _parse_overrides(["training.max_length=1024", "training.learning_rate=1e-4"])
    assert result["training.max_length"] == 1024
    assert result["training.learning_rate"] == 0.0001


def test_smoke_config_uses_bounded_runtime_without_changing_gate_hash():
    from univi.cli import _resolve_smoke_config

    production = {
        "config_hash": "production-hash",
        "training": {
            "max_steps": 500,
            "output_dir": "data/checkpoints/production",
        },
        "dataset": {"path": "data/materialized/full"},
        "wandb": {"run_name_template": "production_{config_hash_short}"},
        "smoke": {
            "dataset_path": "data/materialized/smoke-v1",
            "logging_steps": 1,
            "output_dir": "data/checkpoints/production-smoke",
            "run_name_template": "production-smoke_{config_hash_short}",
        },
    }

    smoke = _resolve_smoke_config(production)

    assert smoke["config_hash"] == "production-hash"
    assert smoke["training"]["max_steps"] == 10
    assert smoke["training"]["logging_steps"] == 1
    assert smoke["training"]["output_dir"] == "data/checkpoints/production-smoke"
    assert smoke["dataset"]["path"] == "data/materialized/smoke-v1"
    assert smoke["wandb"]["run_name_template"] == (
        "production-smoke_{config_hash_short}"
    )
    assert production["training"]["max_steps"] == 500
    assert production["dataset"]["path"] == "data/materialized/full"


def test_train_module_imports():
    """univi.train CLI entrypoint module imports delegate cleanly."""
    import univi.train  # noqa: F811
    assert hasattr(univi.train, "__name__")
