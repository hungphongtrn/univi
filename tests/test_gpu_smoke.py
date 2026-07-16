"""
GPU-gated smoke gate integration tests.

These tests require ``pytest -m requires_gpu`` and a CUDA-capable device
with at least 12 GiB VRAM.  CPU-safe smoke helper tests live in
``test_trainer.py``.

Usage::

    uv run pytest tests/test_gpu_smoke.py -m requires_gpu -v

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_gpu

# ============================================================================
# GPU gate
# ============================================================================


def test_check_gpu_gate_passes():
    """check_gpu_gate returns device metadata on a CUDA-capable system."""
    from univi.smoke import check_gpu_gate

    meta = check_gpu_gate()
    assert meta["cuda_available"] is True
    assert meta["total_vram_gib"] >= 12.0
    assert "device_name" in meta
    assert len(meta["compute_capability"]) == 2
    assert "torch_version" in meta


def test_check_gpu_gate_vram_capacity():
    """check_gpu_gate reports VRAM >= 12 GiB for smoke gate."""
    from univi.smoke import check_gpu_gate

    meta = check_gpu_gate()
    assert meta["total_vram_gib"] >= 12.0, (
        f"Smoke gate requires 12 GiB, got {meta['total_vram_gib']:.2f}"
    )


# ============================================================================
# Config validation
# ============================================================================


def test_validate_smoke_config_passes():
    """validate_smoke_config passes on a valid smoke config."""
    from univi.config import resolve_config
    from univi.smoke import validate_smoke_config

    cfg = resolve_config("configs/smoke.yaml")
    # Should not raise
    validate_smoke_config(cfg)


def test_validate_smoke_config_rejects_wrong_steps():
    """validate_smoke_config rejects max_steps != 10."""
    from univi.smoke import validate_smoke_config, SmokeGateError

    cfg = {
        "training": {
            "max_steps": 5,
            "max_length": 2048,
            "packing": False,
            "loss_masking": "response_only",
            "output_dir": "data/checkpoints/smoke-v0",
            "validation_subsets": ["fineweb-edu", "densefusion", "smoltalk", "librispeech"],
        },
        "dataset": {
            "subsets": ["fineweb-edu", "densefusion", "smoltalk", "librispeech"],
            "revision": "5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375",
        },
        "model": {
            "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
        },
        "wandb": {
            "run_name_template": "smoke-v0_{config_hash_short}",
        },
    }
    with pytest.raises(SmokeGateError, match="max_steps=10"):
        validate_smoke_config(cfg)


def test_validate_smoke_config_rejects_packing():
    """validate_smoke_config rejects packing=True."""
    from univi.smoke import validate_smoke_config, SmokeGateError

    cfg = {
        "training": {
            "max_steps": 10,
            "max_length": 2048,
            "packing": True,
            "loss_masking": "response_only",
            "output_dir": "data/checkpoints/smoke-v0",
            "validation_subsets": ["fineweb-edu", "densefusion", "smoltalk", "librispeech"],
        },
        "dataset": {
            "subsets": ["fineweb-edu", "densefusion", "smoltalk", "librispeech"],
            "revision": "5e05ffab5742acce5cb9a1c0e9ba2fc2cb35c375",
        },
        "model": {
            "revision": "4abfca14e6c6bfb5888b80288185b1243fb8d539",
        },
        "wandb": {
            "run_name_template": "smoke-v0_{config_hash_short}",
        },
    }
    with pytest.raises(SmokeGateError, match="packing=False"):
        validate_smoke_config(cfg)


# ============================================================================
# Collation validation (requires GPU for model loading)
# ============================================================================


def test_smoke_metrics_callback():
    """SmokeMetricsCallback collects and verifies per-step metrics."""
    from univi.smoke import SmokeMetricsCallback

    callback = SmokeMetricsCallback()

    from unittest.mock import MagicMock

    state = MagicMock()
    control = MagicMock()

    state.global_step = 2
    callback.on_log(None, state, control, {
        "loss": 2.5, "grad_norm": 0.8, "num_input_tokens_seen": 1024
    })

    state.global_step = 4
    callback.on_log(None, state, control, {
        "loss": 2.1, "grad_norm": 0.6, "num_input_tokens_seen": 2048
    })

    assert len(callback.steps) == 2
    callback.verify()


def test_smoke_metrics_callback_ignores_evaluation_logs():
    from unittest.mock import MagicMock

    from univi.smoke import SmokeMetricsCallback

    callback = SmokeMetricsCallback()
    state = MagicMock(global_step=10)
    callback.on_log(
        None,
        state,
        MagicMock(),
        {"loss": 0.4, "grad_norm": 0.5, "num_input_tokens_seen": 22036},
    )
    callback.on_log(
        None,
        state,
        MagicMock(),
        {"eval_librispeech_loss": 4.2, "num_input_tokens_seen": 22036},
    )

    assert len(callback.steps) == 1
    callback.verify()


def test_collect_endpoint_eval_metrics_merges_logged_subset_results():
    from univi.smoke import collect_endpoint_eval_metrics

    history = [
        {"eval_fineweb-edu_loss": 0.5, "eval_fineweb-edu_runtime": 1.0},
        {"eval_librispeech_loss": 4.3, "eval_librispeech_runtime": 2.0},
    ]

    assert collect_endpoint_eval_metrics(history) == {
        "eval_fineweb-edu_loss": 0.5,
        "eval_fineweb-edu_runtime": 1.0,
        "eval_librispeech_loss": 4.3,
        "eval_librispeech_runtime": 2.0,
    }


def test_smoke_metrics_callback_non_finite_loss():
    """SmokeMetricsCallback.verify() raises on NaN loss."""
    from univi.smoke import SmokeMetricsCallback, SmokeGateError

    callback = SmokeMetricsCallback()
    from unittest.mock import MagicMock

    state = MagicMock()
    state.global_step = 2
    callback.on_log(None, state, MagicMock(), {"loss": float("nan"), "grad_norm": 0.5})

    with pytest.raises(SmokeGateError, match="Non-finite loss"):
        callback.verify()


def test_smoke_metrics_callback_stagnant_tokens():
    """SmokeMetricsCallback.verify() raises on non-increasing tokens."""
    from univi.smoke import SmokeMetricsCallback, SmokeGateError

    callback = SmokeMetricsCallback()
    from unittest.mock import MagicMock

    state = MagicMock()
    state.global_step = 2
    callback.on_log(None, state, MagicMock(), {
        "loss": 2.5, "grad_norm": 0.5, "num_input_tokens_seen": 1024
    })
    callback.on_log(None, state, MagicMock(), {
        "loss": 2.5, "grad_norm": 0.5, "num_input_tokens_seen": 512
    })

    with pytest.raises(SmokeGateError, match="num_input_tokens_seen did not increase"):
        callback.verify()


# ============================================================================
# Checkpoint reload verification
# ============================================================================


def test_verify_checkpoint_reload(tmp_path):
    """verify_checkpoint_reload validates checkpoint structure."""
    from univi.smoke import verify_checkpoint_reload

    ckpt_dir = tmp_path / "final"
    ckpt_dir.mkdir(parents=True)

    (ckpt_dir / "adapter_model.safetensors").write_text("dummy")
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (ckpt_dir / "tokenizer_config.json").write_text(json.dumps({"model_id": "test"}))

    result = verify_checkpoint_reload(str(ckpt_dir), "abc123", {})
    assert result["model_files_present"] is True
    assert result["adapter_config_present"] is True
    assert result["fingerprint_match"] is None


def test_verify_checkpoint_reload_missing(tmp_path):
    """verify_checkpoint_reload raises on missing checkpoint."""
    from univi.smoke import verify_checkpoint_reload, SmokeGateError

    with pytest.raises(SmokeGateError, match="not found"):
        verify_checkpoint_reload(str(tmp_path / "nonexistent"), "abc", {})


def test_verify_checkpoint_reload_incomplete(tmp_path):
    """verify_checkpoint_reload raises on incomplete checkpoint."""
    from univi.smoke import verify_checkpoint_reload, SmokeGateError

    ckpt_dir = tmp_path / "final"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "adapter_config.json").write_text("{}")

    with pytest.raises(SmokeGateError, match="missing model"):
        verify_checkpoint_reload(str(ckpt_dir), "abc", {})


# ============================================================================
# Report writing
# ============================================================================


def test_write_smoke_report(tmp_path):
    """write_smoke_report creates JSON and summary files."""
    from univi.smoke import write_smoke_report

    report = {
        "status": "success",
        "config_hash": "abcdef",
        "device": {"device_name": "Test GPU", "total_vram_gib": 40.0},
        "memory": {"peak_allocated_gib": 3.5, "peak_reserved_gib": 4.0},
        "training_steps": [{"step": 10, "loss": 2.0, "grad_norm": 0.5}],
        "evaluation": {"eval_loss": 2.1},
        "wandb_run_id": "test-run-123",
        "wandb_url": "https://wandb.ai/test/test/runs/abc123",
    }

    json_path = write_smoke_report(report, "test-run-123")
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["status"] == "success"

    summary_path = json_path.parent / "summary.txt"
    assert summary_path.exists()
    content = summary_path.read_text()
    assert "Smoke Gate Report" in content
    assert "Peak allocated VRAM" in content
