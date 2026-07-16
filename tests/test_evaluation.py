"""
Evaluation aggregation unit tests for univi.evaluation.
"""

from __future__ import annotations

import pytest


def test_compute_macro_avg():
    """compute_macro_avg computes unweighted mean of per-subset NLL losses."""
    from univi.evaluation import compute_macro_avg

    losses = {"fineweb-edu": 1.0, "densefusion": 2.0, "smoltalk": 3.0, "librispeech": 4.0}
    avg = compute_macro_avg(losses)
    assert avg == 2.5


def test_compute_macro_avg_empty():
    """compute_macro_avg returns 0.0 for empty dict."""
    from univi.evaluation import compute_macro_avg

    assert compute_macro_avg({}) == 0.0


def test_compute_macro_avg_single():
    """compute_macro_avg returns the single value for single-entry dict."""
    from univi.evaluation import compute_macro_avg

    assert compute_macro_avg({"fineweb-edu": 3.5}) == 3.5


def test_uni_vi_sft_trainer_evaluate_adds_mean_total_loss():
    """UniViSFTTrainer.evaluate() adds eval_mean_total_loss to merged metrics."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch

    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {
        "eval_fineweb-edu_loss": 1.5,
        "eval_densefusion_loss": 2.5,
        "eval_smoltalk_loss": 3.5,
        "eval_librispeech_loss": 4.5,
    }
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    assert "eval_mean_total_loss" in result
    assert result["eval_mean_total_loss"] == 3.0


def test_uni_vi_sft_trainer_evaluate_no_subsets():
    """UniViSFTTrainer.evaluate() skips mean when no per-subset keys exist."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch

    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {"eval_loss": 2.0}
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    assert "eval_mean_total_loss" not in result


def test_uni_vi_sft_trainer_best_model_resolution():
    """UniViSFTTrainer.evaluate() emits eval_mean_total_loss so Trainer's
    metric_for_best_model='mean_total_loss' resolves against the merged dict."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch

    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {
        "eval_fineweb-edu_loss": 2.0,
        "eval_densefusion_loss": 4.0,
        "eval_smoltalk_loss": 6.0,
        "eval_librispeech_loss": 8.0,
    }
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    metric_to_check = "mean_total_loss"
    if not metric_to_check.startswith("eval_"):
        metric_to_check = f"eval_{metric_to_check}"
    assert metric_to_check in result
    assert result[metric_to_check] == 5.0


def test_uni_vi_sft_trainer_wandb_key_derived():
    """W&B key derivation works on UniViSFTTrainer's output metrics."""
    from univi.evaluation import UniViSFTTrainer
    from trl import SFTTrainer
    from unittest.mock import patch

    trainer = UniViSFTTrainer.__new__(UniViSFTTrainer)
    merged = {
        "eval_fineweb-edu_loss": 1.5,
        "eval_densefusion_loss": 2.5,
        "eval_smoltalk_loss": 3.5,
        "eval_librispeech_loss": 4.5,
    }
    with patch.object(SFTTrainer, "evaluate", return_value=merged) as mock_parent:
        result = trainer.evaluate()
    wandb_metrics = {
        k.replace("eval_", "eval/", 1): v
        for k, v in result.items()
        if k.startswith("eval_")
    }
    assert "eval/mean_total_loss" in wandb_metrics
    assert wandb_metrics["eval/mean_total_loss"] == 3.0


def test_prediction_loss_only_avoids_unsloth_logits_path():
    """Loss-only evaluation computes only loss instead of full-vocabulary logits."""
    from contextlib import nullcontext

    import torch

    from univi.evaluation import UniViSFTTrainer

    trainer = object.__new__(UniViSFTTrainer)
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.compute_loss_context_manager = nullcontext
    calls = []

    def compute_loss(model, inputs, return_outputs=False):
        calls.append(return_outputs)
        return torch.tensor(2.5)

    trainer.compute_loss = compute_loss
    loss, logits, labels = trainer.prediction_step(
        object(), {"labels": torch.tensor([[1]])}, True,
    )
    assert loss.item() == 2.5
    assert logits is None
    assert labels is None
    assert calls == [False]


def test_build_eval_mapping():
    """build_eval_mapping creates dict of named eval datasets."""
    from univi.evaluation import build_eval_mapping
    from datasets import Dataset

    ds1 = Dataset.from_list([{"x": 1}])
    ds2 = Dataset.from_list([{"x": 2}])
    eval_datasets = {"fineweb-edu": ds1, "densefusion": ds2}
    mapping = build_eval_mapping(eval_datasets)
    assert set(mapping.keys()) == {"fineweb-edu", "densefusion"}
