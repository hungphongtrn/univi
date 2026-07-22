"""
Config schema tests for univi.config.
"""

from __future__ import annotations

import pytest


def test_config_defaults():
    """Default config loads and produces a resolved config hash."""
    from univi.config import resolve_config

    cfg = resolve_config("configs/3060_1epoch.yaml")
    assert cfg["training"]["max_length"] == 8192
    assert "config_hash" in cfg
    assert isinstance(cfg["config_hash"], str)
    assert len(cfg["config_hash"]) == 64  # sha256 hex


def test_config_max_length_not_max_seq_length():
    """Resolved config must use max_length, not max_seq_length."""
    from univi.config import resolve_config

    cfg = resolve_config("configs/3060_1epoch.yaml")
    assert "max_length" in cfg["training"]
    assert "max_seq_length" not in cfg["training"]


def test_config_overrides_merge():
    """CLI overrides merge into resolved config and change hash."""
    from univi.config import resolve_config

    cfg = resolve_config("configs/3060_1epoch.yaml")
    cfg2 = resolve_config(
        "configs/3060_1epoch.yaml",
        overrides={"training.max_length": 1024},
    )
    assert cfg2["training"]["max_length"] == 1024
    assert cfg["config_hash"] != cfg2["config_hash"]


def test_config_hub_only_omits_path():
    """Hub-source validation passes when dataset.path is absent and hf_hub_repo_id is set."""
    from univi.config import resolve_config

    cfg = resolve_config(
        "configs/smoke.yaml",
        overrides={
            "dataset.path": None,
            "dataset.hf_hub_repo_id": "hungphongtrn/univi-3M-v0",
        },
    )
    assert "hf_hub_repo_id" in cfg["dataset"]


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
    cfg2 = resolve_config(
        "configs/3060_1epoch.yaml",
        overrides={"training.max_length": 1024},
    )
    assert cfg1["config_hash"] != cfg2["config_hash"]


def test_config_all_training_configs_load():
    """All training configs load and normalise correctly through resolve_config()."""
    from univi.config import resolve_config

    for name in ["smoke", "3060_full", "3060_1epoch", "full"]:
        cfg = resolve_config(f"configs/{name}.yaml")
        assert "config_hash" in cfg
        assert cfg["training"]["max_length"] == 8192
        assert cfg["dataset"]["max_train_images"] == 4
        assert "max_seq_length" not in cfg["training"]


def test_config_rejects_missing_required():
    """resolve_config raises ValueError when required keys are missing."""
    from univi.config import resolve_config

    with pytest.raises(ValueError, match="Missing required config keys"):
        resolve_config("configs/eval.yaml")


def test_validate_loss_masking():
    """Invalid loss_masking value raises ValueError."""
    from univi.config import resolve_config

    with pytest.raises(ValueError, match="Invalid loss_masking"):
        resolve_config("configs/3060_1epoch.yaml", overrides={"training.loss_masking": "invalid"})
