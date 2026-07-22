"""
W&B lifecycle: init, config logging, artifact upload, safe teardown.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)


import os
from dataclasses import dataclass, field
from typing import Any

import wandb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WandbConfig:
    """W&B run configuration extracted from resolved training config."""

    project: str = "univi-gemma4"
    entity: str | None = None
    run_name_template: str = "run_{config_hash_short}"
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def validate_wandb_available() -> bool:
    """Check W&B credentials/network without starting a run."""
    try:
        # Check if API key is available
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get(
            "WANDB_API_KEY_FILE"
        )
        if not api_key:
            return False
        # Try a lightweight API check
        api = wandb.Api(api_key=api_key) if api_key else None
        if api is None:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class WandbManager:
    """Lifecycle manager for a single W&B run.

    Usage::

        mgr = WandbManager.init(cfg)
        mgr.log_validation_artifacts(...)
        mgr.log_metrics({"loss": 0.5}, step=10)
        mgr.finish()
    """

    def __init__(
        self,
        run_id: str,
        config_hash: str,
        run: Any = None,
    ) -> None:
        self.id = run_id
        self.config_hash = config_hash
        self._run = run
        self._active = run is not None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        cfg: dict,
        resume: bool = False,
        run_id: str | None = None,
    ) -> WandbManager:
        """Create and initialize (or resume) a W&B run from resolved config.

        Tries online ``wandb.init`` first — picks up credentials from
        ``wandb login``, ``WANDB_API_KEY``, or ``WANDB_API_KEY_FILE``.
        Falls back to ``mode="disabled"`` (no-op) when no credentials are available,
        ensuring tests and CI pass without network access.
        """
        wc = cfg.get("wandb", {})
        wcfg = WandbConfig(
            project=wc.get("project", "univi-gemma4"),
            entity=wc.get("entity"),
            run_name_template=wc.get("run_name_template", "run_{config_hash_short}"),
        )

        config_hash = cfg.get("config_hash", "unknown")
        config_hash_short = config_hash[:8]

        # Generate run name from template
        run_name = wcfg.run_name_template.format(
            config_hash_short=config_hash_short
        )

        # Try online mode first — wandb picks up credentials from login, env, or config.
        # Falls back to disabled mode if no credentials are available.

        try:
            run = wandb.init(
                project=wcfg.project,
                entity=wcfg.entity,
                name=run_name,
                id=run_id,
                resume=resume if run_id else None,
                config=_flatten_config(cfg),
                tags=wcfg.tags,
            )
        except Exception:
            logger.warning(
                "W&B credentials not found — running in disabled mode. "
                "Use `wandb login` or set WANDB_API_KEY to enable logging."
            )
            run = wandb.init(mode="disabled")
            run_id_fallback = getattr(run, "id", None) or "disabled"
            return cls(run_id=str(run_id_fallback), config_hash=config_hash, run=run)

        return cls(
            run_id=str(getattr(run, "id", "unknown")),
            config_hash=config_hash,
            run=run,
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def log_config(self, resolved_cfg: dict) -> None:
        """Log flat resolved config to W&B run config."""
        if self._run is not None:
            self._run.config.update(_flatten_config(resolved_cfg))

    def log_validation_artifacts(
        self,
        summary_path: str,
        failures_path: str,
        review_path: str,
        manifest_path: str,
    ) -> dict[str, bool]:
        """Upload each artifact (summary, failures, review, manifest)."""
        result: dict[str, bool] = {}
        for name, path in [
            ("summary", summary_path),
            ("failures", failures_path),
            ("review", review_path),
            ("manifest", manifest_path),
        ]:
            try:
                if os.path.exists(path):
                    artifact = wandb.Artifact(
                        name=f"validation-{name}",
                        type="validation",
                    )
                    artifact.add_file(path)
                    if self._run is not None:
                        self._run.log_artifact(artifact)
                    result[f"{name}_path"] = True
                else:
                    result[f"{name}_path"] = False
            except Exception:
                result[f"{name}_path"] = False
        return result

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log training/validation scalars.

        Transforms ``eval_<subset>_loss`` → ``eval/<subset>_loss`` and
        ``eval_mean_total_loss`` → ``eval/mean_total_loss`` immediately
        before ``wandb.log``. All other keys pass through unchanged.
        """
        transformed = {}
        for k, v in metrics.items():
            if k.startswith("eval_"):
                new_key = "eval/" + k[len("eval_"):]
                transformed[new_key] = v
            else:
                transformed[k] = v

        wandb.log(transformed, step=step)

    def mark_failed(self, reason: str) -> None:
        """Mark the W&B run as failed with a reason string."""
        if self._run is not None:
            try:
                self._run.failed()
                self._run.summary["failure_reason"] = reason
            except Exception:
                pass

    def finish(self) -> None:
        """Close the W&B run cleanly."""
        if self._run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
        self._active = False

    def teardown_on_error(self) -> None:
        """Safe cleanup on exception: marks run failed if active, then finishes."""
        if self._active and self._run is not None:
            try:
                self.mark_failed("unexpected error")
            except Exception:
                pass
            try:
                self.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict into dot-separated keys for W&B."""
    flat: dict = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_config(v, prefix=key))
        else:
            flat[key] = v
    return flat
