"""
CPU-safe integration tests for the Phase 1 training foundation.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_dry_run_exit_code():
    """--dry-run exits 0 without GPU."""
    from univi.cli import main

    rc = main(["--config", "configs/3060_1epoch.yaml", "--dry-run"])
    assert rc == 0


def test_config_hash_stability():
    """Same config produces same hash across runs."""
    from univi.config import resolve_config

    cfg1 = resolve_config("configs/3060_1epoch.yaml")
    cfg2 = resolve_config("configs/3060_1epoch.yaml")
    assert cfg1["config_hash"] == cfg2["config_hash"]


def test_config_hash_changes_with_override():
    """Config hash changes when CLI override changes a value."""
    from univi.config import resolve_config

    cfg1 = resolve_config("configs/3060_1epoch.yaml")
    cfg2 = resolve_config("configs/3060_1epoch.yaml", overrides={"training.max_length": 1024})
    assert cfg1["config_hash"] != cfg2["config_hash"]


def test_all_subset_names_known():
    """All subsets in configs match VALID_SUBSETS."""
    from univi.config import resolve_config
    from univi.trainer import VALID_SUBSETS

    for cfg_name in ["smoke", "3060_full", "3060_1epoch", "full"]:
        cfg = resolve_config(f"configs/{cfg_name}.yaml")
        for s in cfg["dataset"]["subsets"]:
            assert s in VALID_SUBSETS, f"{cfg_name}: unknown subset {s}"


def test_full_preflight_pipeline_dry_run(synthetic_dataset: Path):
    """Dry-run preflight produces status success without model loading."""
    from univi.trainer import run_preflight

    cfg = {
        "model": {
            "name": "unsloth/gemma-4-E2B-it",
            "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
        },
        "dataset": {
            "path": str(synthetic_dataset),
            "subsets": ["fineweb-edu", "densefusion"],
            "train_splits": {"fineweb-edu": ["train"], "densefusion": ["train"]},
            "shuffle_seed": 42,
        },
        "training": {"max_length": 2048},
        "config_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    result = run_preflight(cfg, dry_run=True)
    assert result["status"] == "success"
    assert result["validation"]["valid_rows"] >= 0
    assert result["validation"]["failed_rows"] == 0


def test_train_module_imports():
    """univi.train CLI entrypoint module imports delegate cleanly."""
    import univi.train  # noqa: F401

    assert True


def test_public_api_imports():
    """All public API names are importable from univi."""
    import univi

    assert hasattr(univi, "train")
    assert hasattr(univi, "load_dataset")
    assert hasattr(univi, "build_model")
    assert hasattr(univi, "apply_lora")
    assert hasattr(univi, "resolve_config")
    assert hasattr(univi, "config_hash")
    assert hasattr(univi, "parse_args")
    assert hasattr(univi, "MODES")
    assert hasattr(univi, "validate_row_schema")
    assert hasattr(univi, "compute_fingerprint")
    assert hasattr(univi, "WandbManager")
    assert hasattr(univi, "Manifest")
    assert hasattr(univi, "UniViSFTTrainer")
