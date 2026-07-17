"""
Config schema, defaults, CLI override merging, and resolved-config hash.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "model.name",
    "model.revision",
    "dataset.subsets",
    "training.max_length",
}

VALID_DATASET_SOURCES = {"auto", "local", "hub"}
VALID_LOSS_MASKING = {"response_only", "full_sequence"}
VALID_MODES = ["train", "preflight_only", "smoke", "resume", "evaluate"]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _deep_set(d: dict, dotted: str, value: Any) -> None:
    """Set *dotted* key in a nested dict, creating intermediate dicts as needed."""
    parts = dotted.split(".")
    target = d
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _deep_get(d: dict, dotted: str) -> Any:
    """Get *dotted* key from a nested dict, returning None if missing."""
    parts = dotted.split(".")
    target = d
    for part in parts:
        if not isinstance(target, dict):
            return None
        target = target.get(part)
        if target is None:
            return None
    return target


def resolve_config(yaml_path: str, overrides: dict | None = None) -> dict:
    """Load and normalise a training YAML config, applying CLI overrides.

    Returns a resolved config dict with ``config_hash`` added.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Deep-merge overrides
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                _deep_set(cfg, k, v)

    # ---- Provide defaults for optional training keys ----
    training = cfg.setdefault("training", {})
    training.setdefault("loss_masking", "response_only")
    training.setdefault("packing", False)
    training.setdefault("eval_packing", False)
    training.setdefault("report_to", ["wandb"])
    training.setdefault("max_length", 8192)

    # ---- Validate required keys ----
    missing = [k for k in REQUIRED_KEYS if _deep_get(cfg, k) is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # ---- Validate loss_masking ----
    lm = training.get("loss_masking", "response_only")
    if lm not in VALID_LOSS_MASKING:
        raise ValueError(
            f"Invalid loss_masking={lm!r}. Valid: {VALID_LOSS_MASKING}"
        )

    # ---- Conditional key validation ----
    dcfg = cfg.get("dataset", {})
    source = dcfg.get("source", "auto")
    if source == "local" and not dcfg.get("path"):
        raise ValueError(
            "Dataset source is 'local' but dataset.path is not set."
        )
    if source == "hub" and not dcfg.get("hf_hub_repo_id"):
        raise ValueError(
            "Dataset source is 'hub' but dataset.hf_hub_repo_id is not set."
        )

    # ---- Config hash ----
    cfg["config_hash"] = config_hash(cfg)

    return cfg


def config_hash(cfg: dict) -> str:
    """Deterministic SHA-256 hex digest of a resolved config."""
    normalized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()
