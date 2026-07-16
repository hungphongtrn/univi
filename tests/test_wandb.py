"""
W&B lifecycle tests for univi.wandb_utils.
"""

from __future__ import annotations

from unittest.mock import patch, call

import pytest
import wandb


def _minimal_config(config_hash: str = "test") -> dict:
    """Helper to create a minimal config for W&B tests."""
    return {
        "model": {"name": "test", "revision": "abc"},
        "dataset": {"path": "", "subsets": ["fineweb-edu"]},
        "training": {"max_length": 2048},
        "config_hash": config_hash,
    }


def test_wandb_init_with_config():
    """WandbManager.init returns manager with config_hash matching resolved config."""
    from univi.wandb_utils import WandbManager

    cfg = _minimal_config("abc123def456")
    mgr = WandbManager.init(cfg, resume=False)
    assert mgr is not None
    assert mgr.config_hash == cfg["config_hash"]
    assert mgr.id is not None


def test_wandb_mark_failed():
    """mark_failed sets run to failed status without raising."""
    from univi.wandb_utils import WandbManager

    mgr = WandbManager.init(_minimal_config("test-fail"))
    mgr.mark_failed("test failure")
    mgr.finish()


def test_wandb_teardown_on_error():
    """teardown_on_error handles active-run cleanup without raising."""
    from univi.wandb_utils import WandbManager

    mgr = WandbManager.init(_minimal_config("test-cleanup"))
    mgr.teardown_on_error()  # should not raise


def test_wandb_log_metrics_transforms_keys():
    """WandbManager.log_metrics transforms eval_ prefix to eval/ for hierarchical W&B display."""
    from univi.wandb_utils import WandbManager

    mgr = WandbManager.init(_minimal_config("keytest"))
    metrics = {
        "eval_fineweb-edu_loss": 1.5,
        "eval_densefusion_loss": 2.5,
        "eval_mean_total_loss": 2.0,
        "loss": 0.5,
        "grad_norm": 1.2,
    }
    with patch.object(wandb, "log") as mock_log:
        mgr.log_metrics(metrics, step=100)
    expected_calls = [
        call(
            {
                "eval/fineweb-edu_loss": 1.5,
                "eval/densefusion_loss": 2.5,
                "eval/mean_total_loss": 2.0,
                "loss": 0.5,
                "grad_norm": 1.2,
            },
            step=100,
        ),
    ]
    mock_log.assert_has_calls(expected_calls)


def test_wandb_log_metrics_no_transform():
    """WandbManager.log_metrics passes through non-eval keys unchanged."""
    from univi.wandb_utils import WandbManager

    mgr = WandbManager.init(_minimal_config("notransform"))
    metrics = {"loss": 0.5, "grad_norm": 1.2, "learning_rate": 2e-4}
    with patch.object(wandb, "log") as mock_log:
        mgr.log_metrics(metrics, step=1)
    call_args = mock_log.call_args
    assert call_args is not None
    logged = call_args[0][0]
    assert set(logged.keys()) == {"loss", "grad_norm", "learning_rate"}
    assert logged["loss"] == 0.5


def test_validate_wandb_available():
    """validate_wandb_available returns bool without starting a run."""
    from univi.wandb_utils import validate_wandb_available

    result = validate_wandb_available()
    assert isinstance(result, bool)
