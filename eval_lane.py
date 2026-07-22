"""
Evaluation harness for visual modality unification.

Runs image-only and native lanes for every configured source
against both the base model and the trained checkpoint.

Usage:
    uv run python eval_lane.py --config configs/eval.yaml
    uv run python eval_lane.py --config configs/eval.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk

from univi.metrics import compute_retention, write_eval_output

_SOURCES = ["librispeech", "densefusion", "fineweb", "smoltalk"]


# ---------------------------------------------------------------------------
# Config / data loading
# ---------------------------------------------------------------------------

def load_eval_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_source_dataset(source_path: str):
    return load_from_disk(source_path)


# ---------------------------------------------------------------------------
# Native message construction
# ---------------------------------------------------------------------------

def _merge_images(messages: list[dict], images: list) -> list[dict]:
    """Inject images from a separate column into ``{"type": "image"}`` placeholders."""
    import copy
    messages = copy.deepcopy(messages)
    placeholders = [
        (msg_i, c_i)
        for msg_i, msg in enumerate(messages)
        for c_i, content in enumerate(msg.get("content", []))
        if isinstance(content, dict) and content.get("type") == "image"
    ]
    if len(placeholders) != len(images):
        raise ValueError(
            f"Image placeholder count ({len(placeholders)}) "
            f"does not match images column count ({len(images)})."
        )
    for (msg_i, c_i), img in zip(placeholders, images):
        messages[msg_i]["content"][c_i]["image"] = img
    return messages


def build_native_messages(row: dict) -> list[dict] | None:
    """Build native-text messages from *native_user_content* and *target_text*.

    Returns ``None`` when *native_user_content* is missing (e.g. LibriSpeech
    where native audio is not implemented, or DenseFusion where the image lane
    is the only lane).
    """
    native_user_content = row.get("native_user_content")
    target_text = row.get("target_text")
    if native_user_content is None:
        return None
    return [
        {"role": "user", "content": [{"type": "text", "text": native_user_content}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        },
    ]


def should_skip_native(row: dict) -> tuple[bool, str | None]:
    """Return ``(True, reason)`` when the native lane must be skipped."""
    if not row.get("native_available", False):
        return True, "native_available is False"
    if row.get("native_user_content") is None:
        return True, "native_user_content is None"
    return False, None


def get_native_note(source_name: str) -> str | None:
    notes = {
        "librispeech": "Native audio not implemented",
        "densefusion": "Native identical to image lane; no separate native score",
    }
    return notes.get(source_name)


# ---------------------------------------------------------------------------
# Model-dependent seams (designed for monkeypatching in tests)
# ---------------------------------------------------------------------------

def load_model(model_path_or_name: str) -> tuple[object, object]:
    """Load model and tokenizer via Unsloth FastVisionModel.

    Handles both HF model names and local checkpoint directories
    saved by ``trainer.save_model``.

    Unsloth import is deferred so CPU-only environments that never
    call this function (e.g. tests that monkeypatch) do not fail.
    """
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path_or_name,
        load_in_4bit=True,
    )
    return model, tokenizer


def _collate_one(model, tokenizer, messages: list[dict]) -> dict:
    """Collate a single messages example into model inputs.

    Uses the same ``UnslothVisionDataCollator`` path as training so
    response-only loss masking and image preprocessing are consistent.
    """
    from unsloth import UnslothVisionDataCollator

    import torch

    collator = UnslothVisionDataCollator(model, tokenizer)
    batch = collator([{"messages": messages}])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    return batch


def compute_loss(model, tokenizer, messages: list[dict]) -> float:
    """Compute per-example negative log-likelihood loss.

    Override in unit tests via monkeypatch.
    """
    batch = _collate_one(model, tokenizer, messages)
    outputs = model(**batch)
    return outputs.loss.item()


def generate_text(
    model, tokenizer, messages: list[dict], **gen_kwargs
) -> str:
    """Generate a deterministic text prediction.

    Override in unit tests via monkeypatch.
    """
    batch = _collate_one(model, tokenizer, messages)
    batch.pop("labels", None)

    output_ids = model.generate(**batch, **gen_kwargs)
    input_len = batch["input_ids"].shape[1]
    generated = output_ids[0][input_len:]

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metric(source_name: str, reference: str, hypothesis: str) -> float:
    if source_name == "librispeech":
        from univi.metrics import compute_wer
        return compute_wer(reference, hypothesis)
    from univi.metrics import compute_rougel
    return compute_rougel(reference, hypothesis)


def _aggregate_scores(scores: list[float]) -> float | None:
    return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# Evaluation per source
# ---------------------------------------------------------------------------

def evaluate_lane(
    source_name: str,
    rows: list[dict],
    model,
    tokenizer,
    lane: str,
    gen_kwargs: dict,
) -> dict:
    """Evaluate one lane across a list of rows.

    Returns a dict with keys ``score``, ``loss``, ``samples``.
    """
    scores: list[float] = []
    losses: list[float] = []

    for row in rows:
        if lane == "image":
            messages = _merge_images(row["messages"], row.get("images", []))
        else:
            messages = build_native_messages(row)
            if messages is None:
                continue

        try:
            loss_val = compute_loss(model, tokenizer, messages)
            losses.append(loss_val)
        except NotImplementedError:
            pass

        try:
            pred = generate_text(model, tokenizer, messages, **gen_kwargs)
            metric = _compute_metric(source_name, row["target_text"], pred)
            scores.append(metric)
        except NotImplementedError:
            pass

    return {
        "score": _aggregate_scores(scores),
        "loss": _aggregate_scores(losses),
        "samples": len(scores) or len(losses) or len(rows),
    }


def evaluate_source(
    source_name: str,
    rows: list[dict],
    base_model,
    base_tokenizer,
    trained_model,
    trained_tokenizer,
    gen_kwargs: dict,
) -> dict:
    """Evaluate one source across image and native lanes."""
    # Image lane
    base_image = evaluate_lane(
        source_name, rows, base_model, base_tokenizer, "image", gen_kwargs
    )
    trained_image = evaluate_lane(
        source_name, rows, trained_model, trained_tokenizer, "image", gen_kwargs
    )

    # Native lane – skip when dataset does not support it
    native_note = get_native_note(source_name)

    # Check if native is logically identical to image-only (DenseFusion)
    native_equals_image_only = any(
        row.get("native_equals_image_only", False) for row in rows
    )
    base_native = None
    trained_native = None

    if native_equals_image_only:
        base_native = dict(base_image)
        trained_native = dict(trained_image)
        if native_note is None:
            native_note = "Native identical to image lane; no separate native score"
    elif rows:
        skip, reason = should_skip_native(rows[0])
        if skip:
            if native_note is None:
                native_note = reason
        else:
            base_native = evaluate_lane(
                source_name, rows, base_model, base_tokenizer, "native", gen_kwargs
            )
            trained_native = evaluate_lane(
                source_name, rows,
                trained_model, trained_tokenizer, "native", gen_kwargs,
            )

    # Retention: compare trained-image score to trained-native score
    retention = None
    ti = trained_image["score"]
    tn = trained_native["score"] if trained_native else None
    if ti is not None and tn is not None:
        retention = compute_retention(ti, tn)

    def _score_if(lane_dict):
        return lane_dict["score"] if lane_dict["samples"] > 0 else None

    def _loss_if(lane_dict):
        return lane_dict["loss"] if lane_dict["samples"] > 0 else None

    return {
        "base_image": _score_if(base_image),
        "trained_image": _score_if(trained_image),
        "base_native": _score_if(base_native) if base_native else None,
        "trained_native": _score_if(trained_native) if trained_native else None,
        "base_image_loss": _loss_if(base_image),
        "trained_image_loss": _loss_if(trained_image),
        "base_native_loss": _loss_if(base_native) if base_native else None,
        "trained_native_loss": _loss_if(trained_native) if trained_native else None,
        "retention": retention,
        "native_note": native_note,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _empty_source_result(note: str) -> dict:
    return {
        "base_image": None,
        "trained_image": None,
        "base_native": None,
        "trained_native": None,
        "base_image_loss": None,
        "trained_image_loss": None,
        "base_native_loss": None,
        "trained_native_loss": None,
        "retention": None,
        "native_note": note,
    }

def run_eval(config: dict, dry_run: bool = False) -> dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    config :
        Loaded YAML configuration dict.
    dry_run :
        When True, skip model loading (for testing the pipeline wiring).
    """
    gen_kwargs = config.get("generation", {})
    eval_datasets = config.get("eval_datasets", {})

    if dry_run:
        base_model = base_tokenizer = object()
        trained_model = trained_tokenizer = object()
    else:
        base_model, base_tokenizer = load_model(config["base_model"])
        trained_model, trained_tokenizer = load_model(config["checkpoint_path"])

    per_source = {}
    for source_name in _SOURCES:
        source_path = eval_datasets.get(source_name)
        if not source_path:
            per_source[source_name] = _empty_source_result("No dataset configured")
            continue

        if dry_run:
            per_source[source_name] = _empty_source_result("Dataset not loaded (dry-run)")
            continue

        dataset = load_source_dataset(source_path)
        rows = list(dataset)

        per_source[source_name] = evaluate_source(
            source_name,
            rows,
            base_model,
            base_tokenizer,
            trained_model,
            trained_tokenizer,
            gen_kwargs,
        )

    return {
        "checkpoint": config.get("checkpoint_path"),
        "base_model": config.get("base_model"),
        "per_source": per_source,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate base and trained checkpoints across image/native lanes."
    )
    parser.add_argument(
        "--config", default="configs/eval.yaml",
        help="Path to eval config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading (for testing pipeline wiring).",
    )
    args = parser.parse_args(argv)

    config = load_eval_config(args.config)
    results = run_eval(config, dry_run=args.dry_run)

    output_path = config.get("output", "data/eval/results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_eval_output(results, output_path)
    print(f"Results written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
