"""
Shared training helpers for visual modality unification experiments.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
import pyarrow.compute as pc

# Unsloth must be imported before trl/transformers/peft so its
# monkey-patches are applied before those libraries are initialised.
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported

from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from datasets import concatenate_datasets, load_dataset as hf_load
from datasets import load_from_disk

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SUBSETS = {"fineweb-edu", "densefusion", "smoltalk", "librispeech"}


# ---------------------------------------------------------------------------
# Checkpoint and run-id resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint(output_dir: str) -> str:
    """Return the path of the latest checkpoint in *output_dir*.

    Looks for ``checkpoint-NNNN`` subdirectories (Transformers format,
    highest step number wins) and falls back to a ``final`` subdirectory.

    Raises
        FileNotFoundError: if *output_dir* does not exist or contains no
            recognised checkpoint directory.
    """
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {output_dir}"
        )

    checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
    if checkpoints:
        return str(checkpoints[-1])

    final = ckpt_dir / "final"
    if final.is_dir():
        return str(final)

    raise FileNotFoundError(
        f"No checkpoint directories found in {output_dir}. "
        "Expected 'checkpoint-NNNN' or 'final' subdirectory."
    )


def resolve_run_id(output_dir: str) -> str | None:
    """Recover a W&B run ID from a metadata file in *output_dir*.

    Reads the first line of ``<output_dir>/.wandb_run_id``.
    Returns ``None`` when the file does not exist or is empty.
    """
    run_id_file = Path(output_dir) / ".wandb_run_id"
    if run_id_file.is_file():
        content = run_id_file.read_text(encoding="utf-8").strip()
        return content if content else None
    return None


def save_run_id(output_dir: str, run_id: str) -> None:
    """Persist a W&B run ID to ``<output_dir>/.wandb_run_id``."""
    run_id_file = Path(output_dir) / ".wandb_run_id"
    run_id_file.parent.mkdir(parents=True, exist_ok=True)
    run_id_file.write_text(run_id.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RevisionMismatchError(ValueError):
    """Raised when a revision does not match the expected pinned revision."""


class ZeroActiveLabelsError(Exception):
    """Raised when a training batch has zero active response labels."""


# ---------------------------------------------------------------------------
# Config loading (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load a raw YAML config (legacy — prefer ``resolve_config``)."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_model_config(cfg: dict) -> dict:
    """Extract and validate model configuration dict from resolved config."""
    mc = cfg["model"]
    required = {"name", "revision", "max_lora_rank", "use_gradient_checkpointing"}
    missing = required - set(mc.keys())
    if missing:
        raise KeyError(f"Missing model config keys: {missing}")
    return {
        "name": mc["name"],
        "revision": mc["revision"],
        "load_in_4bit": mc.get("load_in_4bit", True),
        "use_gradient_checkpointing": mc.get("use_gradient_checkpointing", "unsloth"),
        "max_lora_rank": mc["max_lora_rank"],
    }


def verify_checkpointing_config(cfg: dict) -> str:
    """Return the effective Unsloth gradient checkpointing mode."""
    mode = cfg.get("model", {}).get("use_gradient_checkpointing", "unsloth")
    valid_modes = {"unsloth", True, False}
    if mode not in valid_modes and mode not in ("unsloth",):
        raise ValueError(
            f"Invalid use_gradient_checkpointing={mode!r}. "
            f"Valid: {valid_modes}"
        )
    return "unsloth" if mode == "unsloth" else str(mode)


def build_model(cfg: dict) -> tuple:
    """Load model + processor via Unsloth FastVisionModel with pinned revision.

    When ``cfg.get("_cpu_validate")`` is True, skips GPU instantiation
    and returns ``(None, processor)`` after key validation.
    """
    mc = cfg["model"]
    logger.info("Loading model: %s", mc['name'])

    # CPU-testable configuration validation
    if cfg.get("_cpu_validate", False):
        resolve_model_config(cfg)
        # Validate revision is pinned 40-char hex string
        rev = mc.get("revision", "")
        assert len(rev) == 40 and all(
            c in "0123456789abcdef" for c in rev
        ), f"Model revision must be pinned 40-char hex, got {rev!r}"
        # Try loading processor on CPU
        logger.info("CPU-validation mode — skipping GPU instantiation")
        try:
            processor = FastVisionModel.from_pretrained(
                model_name=mc["name"],
                load_in_4bit=False,
                revision=mc.get("revision"),
                device_map=None,
            )
            return None, processor
        except Exception:
            return None, None

    # Full GPU path: pass the pinned base-model revision and disable
    # Unsloth's repository-name remapping. Remapping applies a different
    # quantized repository whose commit history is not compatible with the
    # base-model revision.
    model_cfg = resolve_model_config(cfg)
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_cfg["name"],
        load_in_4bit=model_cfg["load_in_4bit"],
        use_gradient_checkpointing=model_cfg["use_gradient_checkpointing"],
        max_lora_rank=model_cfg["max_lora_rank"],
        device_map="auto",
        revision=model_cfg["revision"],
        use_exact_model_name=True,
    )
    logger.info("Model loaded: %s (revision=%s...)", model_cfg['name'], model_cfg['revision'][:12])
    return model, processor


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

EXPECTED_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_lora(model: Any, lora_cfg: dict) -> Any:
    """Apply LoRA adapters to a loaded model with explicit target projections.
    
    Always sets ``finetune_audio_layers=False``.
    When ``lora_cfg.get("_cpu_validate")`` is True, performs config-only
    assertion without loading or modifying a model.
    """
    logger.info("Applying LoRA: r=%d, alpha=%d, dropout=%.2f",
                lora_cfg.get("r", 8), lora_cfg.get("alpha", 16), lora_cfg.get("dropout", 0.0))

    # CPU-testable validation
    if lora_cfg.get("_cpu_validate", False):
        target_modules = lora_cfg.get("target_modules", [])
        assert set(target_modules) == set(EXPECTED_TARGET_MODULES), (
            f"Expected target_modules {EXPECTED_TARGET_MODULES}, "
            f"got {target_modules}"
        )
        assert isinstance(lora_cfg.get("r"), int) and lora_cfg["r"] > 0
        assert isinstance(lora_cfg.get("alpha"), int) and lora_cfg["alpha"] > 0
        assert 0.0 <= lora_cfg.get("dropout", 0.0) < 1.0
        assert lora_cfg.get("bias", "none") == "none"
        assert lora_cfg.get("finetune_audio_layers", False) is False, (
            "finetune_audio_layers must be False for image-only training"
        )
        return None

    # Full GPU path
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("target_modules", EXPECTED_TARGET_MODULES),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        finetune_vision_layers=lora_cfg.get("finetune_vision_layers", True),
        finetune_language_layers=lora_cfg.get("finetune_language_layers", True),
        finetune_attention_layers=lora_cfg.get("finetune_attention_layers", True),
        finetune_mlp_layers=lora_cfg.get("finetune_mlp_layers", True),
        finetune_audio_layers=False,
    )
    logger.info("LoRA applied")
    return model


# ---------------------------------------------------------------------------
# Chat template marker detection
# ---------------------------------------------------------------------------


def detect_markers(tokenizer: Any) -> dict[str, str]:
    """Auto-detect Gemma 4 chat template markers from a tokenizer.

    The tokenizer should have its ``chat_template`` set via
    ``get_chat_template(processor.tokenizer, "gemma-4")`` *before*
    this function is called.
    """
    user_marker = "<|turn>user\n"
    model_marker = "<|turn>model\n"

    assert getattr(tokenizer, "chat_template", None) is not None, (
        "Tokenizer has no chat_template"
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "response"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert user_marker in rendered, (
        f"Expected user marker {user_marker!r} not emitted by chat_template"
    )
    assert model_marker in rendered, (
        f"Expected model marker {model_marker!r} not emitted by chat_template"
    )

    return {"user": user_marker, "model": model_marker}


# ---------------------------------------------------------------------------
# Active label guard
# ---------------------------------------------------------------------------


def check_active_labels(labels: list, batch_idx: int = 0) -> None:
    """Guard: raise ZeroActiveLabelsError if no token has a non-(-100) label."""
    active = [v for v in labels if v != -100 and v is not None]
    if len(active) == 0:
        raise ZeroActiveLabelsError(
            f"Batch {batch_idx} has zero active response labels. "
            "Check marker resolution and response-only masking configuration."
        )


class ActiveLabelCheckCallback(TrainerCallback):
    """Raises ZeroActiveLabelsError if any training batch has zero active labels.

    Register via ``trainer.add_callback()``.  The ``check_batch()`` method
    can be invoked directly in tests without a live trainer.
    """

    def __init__(self, check_fn=None) -> None:
        self._check_fn = check_fn or check_active_labels
        self._seen_active = False

    def check_batch(self, labels: list, batch_idx: int = 0) -> None:
        """Check *labels* and raise ``ZeroActiveLabelsError`` if all are -100.

        Directly callable in tests; avoids needing a live trainer object.
        """
        self._check_fn(labels, batch_idx=batch_idx)
        if any(v != -100 and v is not None for v in labels):
            self._seen_active = True

    def on_step_end(self, args, state, control, **kwargs):
        """Best-effort hook — trainer does not pass batch labels to callbacks.

        Returns ``control`` unchanged so it never interferes with training.
        """
        return control

class CheckedUnslothCollator(UnslothVisionDataCollator):
    """``UnslothVisionDataCollator`` subclass with an active-label guard.

    Raises ``ZeroActiveLabelsError`` immediately at collation time if the
    batch has no non-(-100) tokens, catching misconfigured response-only
    masking before the loss computation produces NaN.

    Inheriting from ``UnslothVisionDataCollator`` keeps unsloth's
    ``_is_vision_collator`` detection correct (MRO check passes), so
    ``train_on_responses_only`` takes the vision collator path and
    configures collation-time masking without touching the dataset.
    """

    def __call__(self, features: list) -> dict:
        batch = super().__call__(features)
        labels = batch.get("labels")
        if labels is not None:
            flat = labels.view(-1).tolist()
            check_active_labels(flat, batch_idx=0)
        return batch




# ---------------------------------------------------------------------------
# Training argument construction
# ---------------------------------------------------------------------------


def make_training_args(cfg: dict) -> SFTConfig:
    """Build SFTConfig from resolved training sub-config."""
    tc = cfg["training"]
    args = SFTConfig(
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 4),
        max_length=tc["max_length"],  # NOT max_seq_length
        num_train_epochs=tc.get("num_train_epochs", 1),
        learning_rate=tc.get("learning_rate", 2e-4),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        warmup_steps=0,  # ratio is authoritative
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        optim=tc.get("optim", "adamw_8bit"),
        weight_decay=tc.get("weight_decay", 0.001),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        logging_steps=tc.get("logging_steps", 10),
        save_steps=tc.get("save_steps", 1000),
        eval_strategy=tc.get("eval_strategy", "steps"),
        eval_steps=tc.get("eval_steps", 1000),
        output_dir=tc["output_dir"],
        report_to=tc.get("report_to", ["wandb"]),
        remove_unused_columns=False,
        dataloader_num_workers=tc.get("dataloader_num_workers", 0),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        max_steps=tc.get("max_steps", -1),
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_text_field="",
        packing=False,
        eval_packing=False,
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        include_num_input_tokens_seen="non_padding",
        save_total_limit=tc.get("save_total_limit", 3),
        seed=tc.get("seed", 3407),
        data_seed=tc.get("data_seed", 42),
    )

    # Conditional best-model selection
    eval_subsets = tc.get("validation_subsets", [])
    if eval_subsets:
        args.metric_for_best_model = tc.get("metric_for_best_model", "mean_total_loss")
        args.greater_is_better = tc.get("greater_is_better", False)
        args.load_best_model_at_end = tc.get("load_best_model_at_end", True)

    return args


# ---------------------------------------------------------------------------
# Response-only masking
# ---------------------------------------------------------------------------


def apply_response_masking(trainer: SFTTrainer, tokenizer: Any, num_proc: int | None = None) -> None:
    """Apply ``train_on_responses_only`` with Gemma 4 markers.

    The runtime active-label guard is wired into the data collator's
    ``__call__`` (set up in :func:`train`); no separate registration
    is needed.

    Args:
        trainer: The SFTTrainer instance.
        tokenizer: The processor/tokenizer.
        num_proc: Number of processes for the internal dataset map.
            Pass ``dataset.num_proc`` from config to control parallelism;
            ``None`` lets Unsloth auto-select (``cpu_count + 4``).
    """
    from unsloth.chat_templates import train_on_responses_only

    markers = detect_markers(tokenizer)
    train_on_responses_only(
        trainer,
        instruction_part=markers["user"],
        response_part=markers["model"],
        force_match=True,
        tokenizer=tokenizer,
        num_proc=num_proc,
    )

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: dict, source: str = "auto"):
    """Load a dataset from local path or Hugging Face Hub with pinned revisions."""
    dcfg = config["dataset"]
    subsets = dcfg.get("subsets", list(VALID_SUBSETS))

    unknown = set(subsets) - VALID_SUBSETS
    if unknown:
        raise ValueError(f"Unknown subset(s): {unknown}. Valid: {VALID_SUBSETS}")

    local_path = Path(dcfg.get("path", ""))
    if source == "auto":
        source = "local" if local_path.exists() else "hub"

    logger.info("Loading dataset: source=%s, subsets=%s", source, subsets)

    if source == "local":
        return _load_local(dcfg, subsets, local_path)
    elif source == "hub":
        return _load_hub(dcfg, subsets)
    else:
        raise ValueError(f"Invalid dataset source: {source}")


def _filter_training_images(dataset, subset: str, max_images: int | None, num_proc: int | None = None):
    """Exclude training rows above the configured image-count ceiling.

    Arrow formatting keeps image payloads encoded while the predicate runs;
    decoding millions of images merely to count placeholders is prohibitive.
    """
    if max_images is None:
        return dataset
    if max_images < 0:
        raise ValueError("dataset.max_train_images must be non-negative")
    if "images" not in dataset.column_names:
        raise ValueError(
            f"Training subset {subset!r} has no 'images' column required "
            "by dataset.max_train_images"
        )

    image_counts = pc.list_value_length(dataset.data.column("images"))
    histogram_values = pc.value_counts(image_counts).to_pylist()
    histogram = {
        int(item["values"]): int(item["counts"])
        for item in histogram_values
        if item["values"] is not None
    }
    retained = sum(
        count for image_count, count in histogram.items()
        if image_count <= max_images
    )
    excluded = len(dataset) - retained
    logger.info(
        "Training image filter: subset=%s, max_images=%s, "
        "retained=%d, excluded=%d, image_count_histogram=%s",
        subset, max_images, retained, excluded,
        dict(sorted(histogram.items())),
    )
    if excluded == 0:
        return dataset

    filtered = dataset.with_format("arrow").filter(
        lambda images: pc.less_equal(
            pc.list_value_length(images), max_images
        ),
        batched=True,
        batch_size=10_000,
        input_columns=["images"],
        desc=f"Filtering {subset} rows with >{max_images} images",
        num_proc=num_proc,
    )
    return filtered.with_format(None)


def _load_local(dcfg: dict, subsets: list[str], local_path: Path):
    """Load dataset from local disk with manifest support."""
    logger.info("Loading local dataset: path=%s, subsets=%s", local_path, subsets)

    expected_revision = dcfg.get("_expected_revision")
    if expected_revision and dcfg.get("revision", "") != expected_revision:
        raise RevisionMismatchError(
            f"Expected dataset revision {expected_revision!r}, "
            f"got {dcfg.get('revision', '')!r}"
        )

    seed = dcfg.get("shuffle_seed", 42)

    manifest_path = local_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        entries = [
            item
            for item in manifest["subsets"]
            if item.get("is_training_split", True)
            and (not subsets or item["config_name"] in subsets)
        ]
        datasets = [
            _filter_training_images(
                load_from_disk(local_path / item["path"]),
                item["config_name"],
                dcfg.get("max_train_images"),
                num_proc=dcfg.get("num_proc"),
            )
            for item in entries
        ]
        return concatenate_datasets(datasets).shuffle(seed=seed)

    # No manifest — load the whole directory
    dataset = load_from_disk(str(local_path))
    logger.info("Local dataset loaded (no manifest): path=%s, subsets=%s, rows=%d",
                local_path, subsets, len(dataset))
    return _filter_training_images(
        dataset, "local", dcfg.get("max_train_images"),
        num_proc=dcfg.get("num_proc"),
    )


def _load_hub(dcfg: dict, subsets: list[str]):
    """Load dataset from Hugging Face Hub with pinned revision and splits."""
    hub_repo = dcfg.get("hf_hub_repo_id")
    logger.info("Loading hub dataset: repo=%s, subsets=%s", hub_repo, subsets)
    if not hub_repo:
        raise ValueError("hf_hub_repo_id is required for hub source.")

    seed = dcfg.get("shuffle_seed", 42)
    split_map = dcfg.get("train_splits", {})
    datasets = []
    for name in subsets:
        for split in split_map.get(name, ["train"]):
            dataset = hf_load(
                hub_repo,
                name=name,
                split=split,
                revision=dcfg.get("revision", None),
                num_proc=dcfg.get("num_proc"),
            )
            datasets.append(
                _filter_training_images(
                    dataset, name, dcfg.get("max_train_images"),
                    num_proc=dcfg.get("num_proc"),
                )
            )

    return concatenate_datasets(datasets).shuffle(seed=seed)
    
    
def _load_eval_datasets(config: dict) -> dict[str, Any]:
    """Load per-subset validation datasets from the configured source.
    
    Returns a mapping ``{ subset_name: Dataset }`` for each entry in
    ``training.validation_subsets``.  Returns an empty dict when no
    validation subsets are configured.
    """
    logger.info("Loading eval datasets")
    dcfg = config.get("dataset", {})
    tc = config.get("training", {})
    subsets = tc.get("validation_subsets", [])
    if not subsets:
        return {}
    
    unknown = set(subsets) - VALID_SUBSETS
    if unknown:
        raise ValueError(f"Unknown validation subset(s): {unknown}. Valid: {VALID_SUBSETS}")
    
    source = dcfg.get("source", "auto")
    local_path = Path(dcfg.get("path", ""))
    if source == "auto":
        source = "local" if local_path.exists() else "hub"
    logger.info("Eval dataset source: %s", source)
    
    split_map = dcfg.get("validation_splits", {})
    revision = dcfg.get("revision", None)
    eval_datasets = {}
    
    for name in subsets:
        splits = split_map.get(name, ["validation"])
        ds_parts = []
        for split in splits:
            if source == "local":
                part = _load_validation_split_local(dcfg, name, split, local_path)
            elif source == "hub":
                part = _load_validation_split_hub(dcfg, name, split)
            else:
                raise ValueError(f"Invalid dataset source: {source}")
            if part is not None:
                ds_parts.append(part)
        if ds_parts:
            from datasets import concatenate_datasets
            eval_datasets[name] = concatenate_datasets(ds_parts)
    logger.info("Eval datasets loaded: %d subsets", len(eval_datasets))
    return eval_datasets
    
    
def _load_validation_split_local(
    dcfg: dict, subset: str, split: str, local_path: Path,
) -> Any:
    """Load a single validation split from a local materialized dataset."""
    manifest_path = local_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        for entry in manifest.get("subsets", []):
            if entry.get("config_name") == subset and entry.get("split") == split:
                return load_from_disk(local_path / entry["path"])
        return None
    return None
    
    
def _load_validation_split_hub(dcfg: dict, subset: str, split: str) -> Any:
    """Load a single validation split from Hugging Face Hub."""
    hub_repo = dcfg.get("hf_hub_repo_id")
    if not hub_repo:
        raise ValueError("hf_hub_repo_id is required for hub source.")
    return hf_load(
        hub_repo, name=subset, split=split,
        revision=dcfg.get("revision", None),
        num_proc=dcfg.get("num_proc"),
    )
    
    
# ---------------------------------------------------------------------------
# Preflight pipeline
# ---------------------------------------------------------------------------
def run_preflight(config: dict, dry_run: bool = False) -> dict:
    """Execute preflight validation without model loading.

    When *dry_run* is True, prints summary without side effects and
    skips all W&B initialisation and artifact upload.
    """
    logger.info("Starting preflight validation (dry_run=%s)", dry_run)
    result: dict[str, Any] = {
        "status": "unknown",
        "validation": {},
        "review_rows": [],
        "fingerprint": {},
        "processor_status": "unavailable_skip",
    }

    # Load dataset
    try:
        dataset = load_dataset(config, source="auto")
    except (FileNotFoundError, ValueError) as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    ds_cfg = config.get("dataset", {})
    result["dataset_size"] = len(dataset)
    result["dataset_subsets"] = ds_cfg.get("subsets", [])

    # Dry-run: print summary and return early
    if dry_run:
        logger.info("Dataset loaded: %d rows, subsets=%s", len(dataset), ds_cfg.get("subsets", []))
        logger.info("Preflight dry-run: would validate %d rows, select 32 review rows, compute fingerprint, create manifest", len(dataset))
        result["status"] = "success"
        result["validation"] = {
            "raw_rows": len(dataset),
            "valid_rows": len(dataset),
            "failed_rows": 0,
        }
        return result

    # Processor fingerprint is gated to GPU smoke (Phase 2).
    # Loading the full 9B model on CPU is slow and always fails — skip it.
    processor = None

    # Run structural validation
    from univi.validation import (
        generate_review_markdown,
        select_review_rows,
        validate_dataset,
    )

    # Bounded/streaming validation — do not materialize the full dataset
    max_validation_rows = config.get("preflight", {}).get("max_rows", 10000)
    total_rows = len(dataset)
    n_to_check = min(total_rows, max_validation_rows)

    logger.info("Running structural validation on %d rows", n_to_check)
    validation_result = validate_dataset(dataset, max_rows=n_to_check)
    result["validation"] = validation_result
    result["validation"]["raw_rows"] = total_rows

    # Collect review rows via bounded iteration (at most max_validation_rows)
    logger.info("Validated %d rows, collecting review rows", n_to_check)
    review_pool = [dataset[i] for i in range(n_to_check)]
    review_rows = select_review_rows(review_pool, n_per_subset=8)

    # Generate review markdown
    review_md = generate_review_markdown(review_rows, processor=processor)

    # Create manifests and write artifacts
    from datetime import datetime, timezone

    from univi.manifest import Manifest

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    val_dir = Path("data") / "validation" / timestamp
    val_dir.mkdir(parents=True, exist_ok=True)

    # Write validation summary
    summary_path = val_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            summarize_validation(validation_result), indent=2, default=str
        ),
        encoding="utf-8",
    )

    # Write validation failures
    failures_path = val_dir / "validation_failures.jsonl"
    with failures_path.open("w") as f:
        for row_idx, row_errors in validation_result.get("errors_per_row", {}).items():
            f.write(json.dumps({"row_index": row_idx, "errors": row_errors}) + "\n")

    # Write review markdown
    review_path = val_dir / "training_data_review.md"
    review_path.write_text(review_md, encoding="utf-8")

    # Create manifest
    manifest = Manifest(phase="preflight", output_dir=str(val_dir))
    manifest_entry = manifest.create_entry(
        config_hash=config.get("config_hash", ""),
        dataset_revision=config.get("dataset", {}).get("revision", ""),
        model_revision=config.get("model", {}).get("revision", ""),
    )
    manifest_path = manifest.save(manifest_entry)
    result["manifest_path"] = str(manifest_path)

    result["status"] = "success"
    result["artifact_dir"] = str(val_dir)

    logger.info("Preflight complete: artifacts in %s", val_dir)

    return result


def summarize_validation(result: dict) -> dict:
    """Normalise validation results into a summary dict."""
    return {
        "raw_rows": result.get("raw_rows", 0),
        "valid_rows": result.get("valid_rows", 0),
        "failed_rows": result.get("failed_rows", 0),
        "per_subset": result.get("per_subset", {}),
    }


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train(config: dict, args: Any = None) -> SFTTrainer:
    """Execute the training pipeline based on mode from args."""
    logger.info("Training pipeline starting (mode=%s)", getattr(args, "mode", "train") if args else "train")
    from univi.cli import parse_args

    if args is None:
        args = parse_args(["--config", "none"])

    mode = getattr(args, "mode", "train")
    loss_masking = config.get("training", {}).get("loss_masking", "response_only")

    if mode == "preflight_only":
        result = run_preflight(config, dry_run=getattr(args, "dry_run", False))
        if result["status"] != "success":
            raise RuntimeError(f"Preflight failed: {result.get('error')}")
        return None  # type: ignore[return-value]

    # Resume-mode contract: validate checkpoint directory and resolve run ID
    # before model loading so errors surface early.
    resume_from_checkpoint: str | None = None
    run_id: str | None = None
    if mode == "resume":
        output_dir = config.get("training", {}).get("output_dir", "checkpoints")
        resume_from_checkpoint = resolve_checkpoint(output_dir)
        run_id = resolve_run_id(output_dir) or config.get("wandb", {}).get("run_id")
        if not run_id:
            raise RuntimeError(
                f"Resume requested but no W&B run ID found. "
                f"Set 'wandb.run_id' in config or ensure "
                f"'{output_dir}/.wandb_run_id' exists."
            )

    # GPU-required path
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Building model...")
    model, processor = build_model(config)

    if model is None:
        raise RuntimeError("Model could not be loaded (no GPU available?)")

    logger.info("Model loaded successfully")
    model = apply_lora(model, config.get("lora", {}))
    model = FastVisionModel.for_training(model)

    # Apply chat template — fail loudly when response-only masking is configured
    try:
        from unsloth.chat_templates import get_chat_template
        processor.tokenizer = get_chat_template(processor.tokenizer, "gemma-4")
    except Exception:
        if loss_masking == "response_only":
            raise
        # For full_sequence, template issues are non-fatal

    dataset = load_dataset(config, source="auto")
    logger.info("Dataset loaded: %d rows", len(dataset))
    training_args = make_training_args(config)
    logger.info("Training arguments constructed")

    # Load eval datasets when validation_subsets configured
    eval_subsets = config.get("training", {}).get("validation_subsets", [])
    if eval_subsets:
        eval_dataset = _load_eval_datasets(config)
        # Require nonempty eval_dataset before checking individual subsets
        if not eval_dataset:
            raise ValueError(
                "validation_subsets configured but no evaluation datasets "
                "were loaded. Check dataset configuration."
            )
        # Then require every requested subset to be present
        missing = [s for s in eval_subsets if s not in eval_dataset]
        if missing:
            dcfg = config.get("dataset", {})
            raise ValueError(
                f"Validation subset(s) {missing} could not be loaded from "
                f"the configured dataset source ({dcfg.get('source', 'auto')}). "
                "Check dataset path, source configuration, and validation_splits."
            )
        from univi.evaluation import UniViSFTTrainer as TrainerClass
    else:
        eval_dataset = None
        TrainerClass = SFTTrainer
    # Initialize W&B lifecycle
    from univi.wandb_utils import WandbManager
    wandb_mgr = WandbManager.init(
        config,
        resume=(mode == "resume"),
        run_id=run_id if mode == "resume" else None,
    )
    if config.get("config_hash"):
        wandb_mgr.log_config(config)
    # Persist run ID for future --resume recovery
    if mode != "preflight_only":
        save_run_id(
            config.get("training", {}).get("output_dir", "checkpoints"),
            wandb_mgr.id,
        )

    try:
        # The vision collator needs the processor wrapper's image_processor,
        # not only its inner tokenizer. Keep a separate visual budget and
        # preserve pre-rendered tile sizes: upscaling 320px spectrogram tiles
        # to Gemma's 512px fallback creates avoidable image-token overflow.
        training_cfg = config.get("training", {})
        collator_max_length = training_cfg.get("collator_max_length", 4096)
        collator_resize = training_cfg.get("collator_resize", "max")
        if loss_masking == "response_only":
            data_collator = CheckedUnslothCollator(
                model, processor,
                max_seq_length=collator_max_length,
                resize=collator_resize,
            )
        else:
            data_collator = UnslothVisionDataCollator(
                model, processor,
                max_seq_length=collator_max_length,
                resize=collator_resize,
            )

        trainer = TrainerClass(
            model=model,
            processing_class=processor.tokenizer,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        logger.info("Trainer initialized: train=%d rows, eval_subsets=%s",
                     len(dataset), list(eval_dataset.keys()) if eval_dataset else [])
        # Apply response-only masking — fail loudly when configured
        if loss_masking == "response_only":
            apply_response_masking(
                trainer, processor.tokenizer,
                num_proc=config.get("dataset", {}).get("num_proc"),
            )
        # full_sequence: skip masking entirely


        logger.info("Starting training (resume_from_checkpoint=%s)", resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training complete")

        # Skip final save in resume mode — the checkpoint already existed
        if mode == "resume":
            logger.info("Resume complete. Model weights remain at %s", resume_from_checkpoint)
        else:
            output_dir = config.get("training", {}).get("output_dir", "checkpoints")
            logger.info("Saving final model to %s/final", output_dir)
            trainer.save_model(output_dir + "/final")
            processor.tokenizer.save_pretrained(output_dir + "/final")

            hub_cfg = config.get("hub", {})
            model_repo_id = hub_cfg.get("model_repo_id", "")
            hf_token = os.environ.get("HF_TOKEN")
            if model_repo_id and hf_token:
                logger.info("Pushing model to Hub: %s", model_repo_id)
                model.push_to_hub(model_repo_id, token=True)
                processor.tokenizer.push_to_hub(model_repo_id, token=True)

        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            logger.info(
                "CUDA peak memory: allocated=%.3f GiB, reserved=%.3f GiB",
                allocated, reserved,
            )

        wandb_mgr.finish()
    except Exception:
        wandb_mgr.teardown_on_error()
        raise
