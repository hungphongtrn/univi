"""
Evaluation helpers: macro-average, eval mapping, and UniViSFTTrainer subclass.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------


def compute_macro_avg(per_subset_losses: dict[str, float]) -> float:
    """Unweighted mean of per-subset loss values."""
    if not per_subset_losses:
        return 0.0
    return sum(per_subset_losses.values()) / len(per_subset_losses)


def build_eval_mapping(
    eval_datasets: dict[str, Dataset],
) -> dict[str, Dataset]:
    """Return the eval dataset mapping as-is.

    Each key produces ``eval_<key>_loss`` in Trainer metrics.
    """
    return eval_datasets


# ---------------------------------------------------------------------------
# UniViSFTTrainer
# ---------------------------------------------------------------------------


class UniViSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that adds ``eval_mean_total_loss`` to evaluation metrics.

    When ``eval_dataset`` is supplied as a dict of named subsets, ``Trainer``
    produces per-subset loss keys (``eval_<subset>_loss``).  This subclass
    overrides ``evaluate()`` to additionally compute their unweighted mean
    and store it as ``eval_mean_total_loss`` — the key resolved by
    ``metric_for_best_model="mean_total_loss"``.
    """

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Evaluate labeled batches without materializing vocabulary logits."""
        if not prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss.mean().detach(), None, None


    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, Any]:
        """Evaluate and add ``eval_mean_total_loss`` to merged metrics."""
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Collect per-subset loss keys
        subset_losses = {}
        for key, value in metrics.items():
            if key.startswith(f"{metric_key_prefix}_") and key.endswith("_loss"):
                # Exclude the aggregate key itself if somehow present
                if key == f"{metric_key_prefix}_mean_total_loss":
                    continue
                subset_name = key[len(metric_key_prefix) + 1 : -len("_loss")]
                if subset_name:
                    subset_losses[subset_name] = float(value)

        # Add macro average if we have at least two subsets
        if len(subset_losses) >= 2:
            metrics[f"{metric_key_prefix}_mean_total_loss"] = compute_macro_avg(
                subset_losses
            )

        return metrics
