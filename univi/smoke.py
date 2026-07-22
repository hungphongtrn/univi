"""
GPU smoke gate harness.

Exposes ``run_smoke()`` — a callable that wraps production training
with smoke-specific GPU dependency checks, retained-row validation,
collator assertions, per-step metric verification, checkpoint
reload, and structured reporting.

Usage::

    python -m univi.train --config configs/smoke.yaml --smoke

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback
from unsloth import FastVisionModel

from univi.config import config_hash as compute_config_hash
from univi.fingerprint import compute_fingerprint, fingerprint_hash
from univi.manifest import Manifest
from univi.trainer import (
    VALID_SUBSETS,
    UnslothVisionDataCollator,
    _load_eval_datasets,
    apply_lora,
    apply_response_masking,
    build_model,
    check_active_labels,
    load_dataset,
    make_training_args,
    save_run_id,
)
from univi.wandb_utils import WandbManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _smoke_data_collator(features: list) -> dict:
    """A lightweight data collator that drops raw images for the smoke harness.

    The smoke harness validates token-level contracts (input_ids, labels,
    attention_mask) and does not test vision encoding.  Raw PIL images are
    dropped from the batch to avoid ``default_data_collator`` failures.
    """
    from transformers import default_data_collator

    # Remove raw images that default_data_collator cannot handle
    cleaned = []
    for f in features:
        row = {k: v for k, v in f.items() if k != "images"}
        cleaned.append(row)

    return default_data_collator(cleaned)

EXPECTED_SMOKE_STEPS = 10
EXPECTED_MIN_VRAM_GIB = 12.0
ACTIVE_SUBSETS = sorted(VALID_SUBSETS)
REQUIRED_REVISION_LEN = 40
SMOKE_DATA_ROOT = Path("data/smoke")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SmokeGateError(RuntimeError):
    """Raised when a smoke gate check fails."""

    def __init__(self, message: str, check: str = "") -> None:
        self.check = check
        super().__init__(message)


# ---------------------------------------------------------------------------
# GPU / device gate
# ---------------------------------------------------------------------------


def check_gpu_gate() -> dict:
    """Verify CUDA availability, device specs, and minimum VRAM.

    Returns a dict of device metadata.
    Raises SmokeGateError on failure.
    """
    if not torch.cuda.is_available():
        raise SmokeGateError(
            "CUDA is not available. GPU smoke gate requires a CUDA-capable device.",
            check="cuda_available",
        )

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise SmokeGateError(
            "No CUDA devices found despite torch.cuda.is_available()=True.",
            check="cuda_device_count",
        )

    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / 1024**3

    if vram_gib < EXPECTED_MIN_VRAM_GIB:
        raise SmokeGateError(
            f"Insufficient VRAM: {vram_gib:.2f} GiB < {EXPECTED_MIN_VRAM_GIB} GiB required.",
            check="vram_capacity",
        )

    device_meta = {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_gib": round(vram_gib, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "cuda_available": True,
    }
    return device_meta


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_smoke_config(config: dict) -> None:
    """Validate that the config matches the smoke contract."""
    tc = config.get("training", {})

    if tc.get("max_steps") != EXPECTED_SMOKE_STEPS:
        raise SmokeGateError(
            f"Smoke requires max_steps={EXPECTED_SMOKE_STEPS}, got {tc.get('max_steps')}",
            check="config_max_steps",
        )

    max_len = tc.get("max_length", 0)
    if max_len < 2048:
        raise SmokeGateError(
            f"Smoke requires max_length >= 2048, got {max_len}",
            check="config_max_length",
        )

    if tc.get("packing", False):
        raise SmokeGateError(
            "Smoke requires packing=False to avoid cross-example concatenation.",
            check="config_packing",
        )

    loss_masking = tc.get("loss_masking", "response_only")
    if loss_masking != "response_only":
        raise SmokeGateError(
            f"Smoke requires response-only masking, got {loss_masking}",
            check="config_loss_masking",
        )

    subsets = config.get("dataset", {}).get("subsets", [])
    for s in ACTIVE_SUBSETS:
        if s not in subsets:
            raise SmokeGateError(
                f"Smoke dataset missing required subset: {s}",
                check="config_missing_subset",
            )

    model_revision = config.get("model", {}).get("revision", "")
    if len(model_revision) != REQUIRED_REVISION_LEN or not all(
        c in "0123456789abcdef" for c in model_revision
    ):
        raise SmokeGateError(
            f"Model revision must be a {REQUIRED_REVISION_LEN}-char hex string: {model_revision}",
            check="config_model_revision",
        )

    dataset_revision = config.get("dataset", {}).get("revision", "")
    if len(dataset_revision) != REQUIRED_REVISION_LEN or not all(
        c in "0123456789abcdef" for c in dataset_revision
    ):
        raise SmokeGateError(
            f"Dataset revision must be a {REQUIRED_REVISION_LEN}-char hex string: {dataset_revision}",
            check="config_dataset_revision",
        )

    output_dir = tc.get("output_dir", "")
    if "smoke" not in output_dir:
        raise SmokeGateError(
            f"Smoke output directory must contain 'smoke', got: {output_dir}",
            check="config_output_dir",
        )

    run_name = config.get("wandb", {}).get("run_name_template", "")
    if "smoke" not in run_name:
        raise SmokeGateError(
            f"W&B run name must contain 'smoke', got: {run_name}",
            check="config_wandb_run_name",
        )


# ---------------------------------------------------------------------------
# Per-step metric collection callback
# ---------------------------------------------------------------------------


class SmokeMetricsCallback(TrainerCallback):
    """Collects per-step loss, grad_norm, and num_input_tokens_seen."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:  # noqa: ARG002
        if logs is None:
            return
        if "loss" not in logs and "grad_norm" not in logs:
            return
        entry: dict[str, Any] = {"step": state.global_step}
        if "loss" in logs:
            entry["loss"] = float(logs["loss"])
        if "grad_norm" in logs:
            entry["grad_norm"] = float(logs["grad_norm"])
        if "num_input_tokens_seen" in logs:
            entry["num_input_tokens_seen"] = int(logs["num_input_tokens_seen"])
        self.steps.append(entry)

    def verify(self) -> None:
        """Assert finite loss/grad_norm and increasing token counter."""
        if not self.steps:
            raise SmokeGateError(
                "No training log entries captured.", check="no_log_entries"
            )

        prev_tokens = -1
        for entry in self.steps:
            step = entry.get("step", 0)
            loss = entry.get("loss")
            if loss is not None and not (loss > 0.0 and loss < float("inf")):
                raise SmokeGateError(
                    f"Non-finite loss at step {step}: {loss}",
                    check="finite_loss",
                )
            gn = entry.get("grad_norm")
            if gn is not None and not (gn >= 0.0 and gn < float("inf")):
                raise SmokeGateError(
                    f"Non-finite grad_norm at step {step}: {gn}",
                    check="finite_grad_norm",
                )
            tokens = entry.get("num_input_tokens_seen")
            if tokens is not None:
                if tokens <= prev_tokens:
                    raise SmokeGateError(
                        f"num_input_tokens_seen did not increase: "
                        f"{prev_tokens} -> {tokens} at step {step}",
                        check="token_counter_increasing",
                    )
                prev_tokens = tokens


def collect_endpoint_eval_metrics(
    log_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect the endpoint evaluation emitted by Trainer without rerunning it."""
    metrics: dict[str, Any] = {}
    for entry in log_history:
        for key, value in entry.items():
            if not key.startswith("eval_"):
                continue
            if isinstance(value, (int, float)):
                if value != value:
                    raise SmokeGateError(
                        f"Evaluation metric {key} is NaN",
                        check="eval_metric_nan",
                    )
                if value == float("inf"):
                    raise SmokeGateError(
                        f"Evaluation metric {key} is Inf",
                        check="eval_metric_inf",
                    )
                metrics[key] = float(value)
            else:
                metrics[key] = str(value)
    if not any(key.endswith("_loss") for key in metrics):
        raise SmokeGateError(
            "No endpoint evaluation losses were logged.",
            check="missing_eval_metrics",
        )
    return metrics


# ---------------------------------------------------------------------------
# Collation validation
# ---------------------------------------------------------------------------


def validate_collation(
    model: Any,
    processor: Any,
    dataset: Any,
    subsets: list[str],
    *,
    max_seq_length: int | None = None,
    resize: str | tuple[int, int] = "max",
) -> dict:
    """Collate one batch per subset and assert image/label/mask contracts.

    ``UnslothVisionDataCollator`` requires the processor wrapper because it
    owns the image processor; the inner tokenizer alone is insufficient.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    SUBSET_SOURCE_ALIASES = {
        "densefusion": ("densefusion", "finevision", "huggingfacem4/finevision"),
        "fineweb-edu": ("fineweb-edu", "fineweb"),
        "smoltalk": ("smoltalk",),
        "librispeech": ("librispeech", "librispeech_asr"),
    }

    # Match config names to materialized source IDs by leaf name or the
    # canonical source alias used by the materialization manifest.
    def _matches_subset(row: dict, subset: str) -> bool:
        source_id = str(row.get("source_dataset_id", "")).strip().lower()
        requested = subset.strip().lower()
        aliases = SUBSET_SOURCE_ALIASES.get(requested, (requested,))
        return any(alias == source_id.rsplit("/", 1)[-1] or alias in source_id for alias in aliases)
    base_collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=max_seq_length,
        resize=resize,
    )
    results: dict[str, Any] = {}

    for subset in subsets:
        subset_rows = [row for row in dataset if _matches_subset(row, subset)]
        if not subset_rows:
            raise SmokeGateError(
                f"No rows for subset '{subset}' in loaded dataset",
                check="missing_subset_rows",
            )

        row = subset_rows[0]
        actual_images = len(row.get("images") or [])
        message_placeholders = sum(
            part.get("type") == "image"
            for message in row.get("messages", [])
            for part in message.get("content", [])
        )
        if actual_images != message_placeholders:
            raise SmokeGateError(
                f"Image/placeholder mismatch for {subset}: "
                f"{actual_images} images vs {message_placeholders} placeholders",
                check="image_placeholder_mismatch",
            )

        try:
            batch = base_collator([row])
        except Exception as exc:
            raise SmokeGateError(
                f"Collation failed for subset '{subset}': {exc}",
                check="collation_failed",
            ) from exc
        if "input_ids" not in batch or "labels" not in batch:
            raise SmokeGateError(
                f"Collator omitted input_ids/labels for subset '{subset}'",
                check="collation_schema",
            )

        labels = batch["labels"]
        flat_labels = labels.view(-1).tolist()
        check_active_labels(flat_labels, batch_idx=0)
        img_token_id = getattr(tokenizer, "image_token_id", None)
        token_placeholders = (
            int((batch["input_ids"] == img_token_id).sum().item())
            if img_token_id is not None
            else None
        )
        if token_placeholders is not None:
            if actual_images == 0 and token_placeholders != 0:
                raise SmokeGateError(
                    f"Token/image mismatch for {subset}: no images but "
                    f"{token_placeholders} image tokens",
                    check="token_image_mismatch",
                )
            if actual_images > 0 and (
                token_placeholders == 0
                or token_placeholders % actual_images != 0
            ):
                raise SmokeGateError(
                    f"Token/image mismatch for {subset}: "
                    f"{actual_images} images vs {token_placeholders} image tokens "
                    "(token expansion is not divisible by image count)",
                    check="token_image_mismatch",
                )
        token_expansion = (
            token_placeholders // actual_images
            if token_placeholders is not None and actual_images
            else None
        )
        results[subset] = {
            "subset": subset,
            "rows_available": len(subset_rows),
            "input_ids_shape": list(batch["input_ids"].shape),
            "labels_shape": list(batch["labels"].shape),
            "num_active_labels": sum(label != -100 for label in flat_labels),
            "num_masked_labels": sum(label == -100 for label in flat_labels),
            "actual_images": actual_images,
            "message_image_placeholders": message_placeholders,
            "token_image_placeholders": token_placeholders,
            "image_tokens_per_image": token_expansion,
        }
    return results



# ---------------------------------------------------------------------------
# Checkpoint reload verification
# ---------------------------------------------------------------------------


def verify_checkpoint_reload(
    checkpoint_path: str,
    pre_fingerprint_hash: str,
    config: dict,
    *,
    strict: bool = False,
) -> dict:
    """Verify checkpoint files exist, tokenizer is saved, fingerprint is consistent.

    Returns a dict of verification results.
    """
    result: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
    }

    ckpt_dir = Path(checkpoint_path)

    # Check directory exists
    if not ckpt_dir.exists():
        raise SmokeGateError(
            f"Checkpoint directory not found: {checkpoint_path}",
            check="checkpoint_not_found",
        )

    # Verify model weight files
    has_model = bool(
        (ckpt_dir / "adapter_model.safetensors").exists()
        or (ckpt_dir / "adapter_model.bin").exists()
        or any(ckpt_dir.glob("model*.safetensors"))
        or any(ckpt_dir.glob("*.safetensors"))
    )
    has_config = (ckpt_dir / "adapter_config.json").exists()

    if not has_model or not has_config:
        raise SmokeGateError(
            f"Checkpoint missing model files at {checkpoint_path}",
            check="checkpoint_incomplete",
        )

    result["model_files_present"] = has_model
    result["adapter_config_present"] = has_config

    tokenizer_files = [p.name for p in ckpt_dir.glob("*tokenizer*")]
    tokenizer_files += [p.name for p in ckpt_dir.glob("*token*")]
    result["tokenizer_files"] = sorted(set(tokenizer_files))
    result["tokenizer_saved"] = bool(tokenizer_files)

    metadata_path = ckpt_dir / "smoke_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SmokeGateError(
                f"Invalid smoke checkpoint metadata: {exc}",
                check="checkpoint_metadata_invalid",
            ) from exc
    result["metadata_present"] = bool(metadata)
    result["pre_fingerprint_hash"] = pre_fingerprint_hash
    stored_fingerprint = metadata.get("processor_fingerprint_hash")
    result["stored_fingerprint_hash"] = stored_fingerprint
    result["fingerprint_match"] = (
        stored_fingerprint == pre_fingerprint_hash
        if stored_fingerprint is not None
        else None
    )
    if strict and result["fingerprint_match"] is not True:
        raise SmokeGateError(
            "Checkpoint processor fingerprint does not match pre-training fingerprint",
            check="processor_fingerprint_mismatch",
        )
    if strict:
        expected_step = int(config.get("training", {}).get("max_steps", 10))
        trainer_state = json.loads((ckpt_dir / "trainer_state.json").read_text())
        result["global_step"] = trainer_state.get("global_step")
        if result["global_step"] != expected_step:
            raise SmokeGateError(
                f"Checkpoint global_step {result['global_step']} != {expected_step}",
                check="checkpoint_global_step",
            )
    return result



# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_smoke_report(report: dict, run_id: str) -> Path:
    """Write structured smoke report and human-readable summary.

    Returns the path to the JSON report.
    """
    report_dir = SMOKE_DATA_ROOT / (run_id or "unknown")
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = report_dir / "results.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Human-readable summary
    summary_path = report_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Smoke Gate Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Status: {report.get('status', 'unknown')}\n")
        f.write(f"Config hash: {str(report.get('config_hash', 'N/A'))[:16]}...\n")

        if "device" in report:
            d = report["device"]
            f.write(f"Device: {d.get('device_name', 'N/A')}\n")
            f.write(f"VRAM: {d.get('total_vram_gib', '?')} GiB\n")

        if "memory" in report:
            m = report["memory"]
            f.write(f"Peak allocated VRAM: {m.get('peak_allocated_gib', '?')} GiB\n")
            f.write(f"Peak reserved VRAM: {m.get('peak_reserved_gib', '?')} GiB\n")

        if "training_steps" in report:
            steps = report["training_steps"]
            f.write(f"Training steps logged: {len(steps)}\n")
            if steps:
                f.write(f"  Final loss: {steps[-1].get('loss', 'N/A')}\n")
                f.write(f"  Final grad_norm: {steps[-1].get('grad_norm', 'N/A')}\n")

        if "evaluation" in report:
            f.write("Evaluation metrics:\n")
            for k, v in report["evaluation"].items():
                f.write(f"  {k}: {v}\n")

        if "collation" in report:
            f.write("Collation results:\n")
            for subset, cr in report["collation"].items():
                f.write(f"  {subset}: {cr.get('num_active_labels', '?')} active labels, "
                       f"{cr.get('num_masked_labels', '?')} masked\n")

        if "failure" in report:
            f.write(f"\nFAILURE: {report['failure'].get('check', 'unknown')}\n")
            f.write(f"  {report['failure'].get('message', '')}\n")

        f.write(f"\nReport: {json_path}\n")
        f.write(f"W&B run ID: {report.get('wandb_run_id', 'N/A')}\n")
        if report.get("wandb_url"):
            f.write(f"W&B URL: {report['wandb_url']}\n")

    return json_path


# ---------------------------------------------------------------------------
# Main smoke entry point
# ---------------------------------------------------------------------------


def run_smoke(config: dict) -> dict:
    """Execute the full smoke gate and return a structured report.

    Steps:
        1. GPU gate
        2. Config validation
        3. Model + processor loading
        4. Dataset loading
        5. Collation validation
        6. W&B initialization
        7. 10-step training with per-step metric collection
        8. Endpoint evaluation
        9. Checkpoint save and reload verification
        10. Report writing
    """
    report: dict[str, Any] = {
        "status": "running",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "smoke",
        "config_hash": config.get("config_hash", ""),
    }
    wandb_mgr: WandbManager | None = None
    # output_dir captured early for error-reporting fallback
    output_dir: str = config.get("training", {}).get("output_dir", "data/checkpoints/smoke-v0")
    try:
        # Step 1: GPU gate
        device_meta = check_gpu_gate()
        report["device"] = device_meta

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Step 2: Validate config
        validate_smoke_config(config)
        tc = config.get("training", {})

        # Step 3: Load model and processor
        model, processor = build_model(config)
        model = apply_lora(model, config.get("lora", {}))
        model = FastVisionModel.for_training(model)


        # Step 4: Apply chat template
        from unsloth.chat_templates import get_chat_template

        processor.tokenizer = get_chat_template(processor.tokenizer, "gemma-4")
        # Fingerprint the exact processor state that will be saved.
        pre_fingerprint = compute_fingerprint(processor, processor.tokenizer)
        pre_fingerprint_hash = fingerprint_hash(pre_fingerprint)
        report["preprocessor_fingerprint_hash"] = pre_fingerprint_hash


        # Step 5: Load dataset
        dataset = load_dataset(config, source="auto")
        subsets = tc.get("validation_subsets", ACTIVE_SUBSETS)
        collation_results = validate_collation(
            model,
            processor,
            dataset,
            subsets,
            max_seq_length=tc.get("max_length", 8192),
            resize=tc.get("collator_resize", "max"),
        )
        report["collation"] = collation_results

        # Step 7: Load eval datasets
        eval_subsets = tc.get("validation_subsets", [])
        eval_dataset = None
        if eval_subsets:
            eval_dataset = _load_eval_datasets(config)

        # Step 8: Initialize W&B
        from univi.evaluation import UniViSFTTrainer

        wandb_mgr = WandbManager.init(config, resume=False)
        if config.get("config_hash"):
            wandb_mgr.log_config(config)
        output_dir = tc.get("output_dir", "data/checkpoints/smoke-v0")
        save_run_id(output_dir, wandb_mgr.id)

        report["wandb_run_id"] = wandb_mgr.id

        # Try to construct W&B URL
        try:
            import wandb

            run_obj = wandb.run
            if run_obj is not None:
                report["wandb_url"] = run_obj.get_url()
        except Exception:
            report["wandb_url"] = ""
        # Use the same processor-aware vision collator as production training.
        data_collator = UnslothVisionDataCollator(model, processor)
        from trl import SFTTrainer

        TrainerClass = UniViSFTTrainer if eval_dataset else SFTTrainer
        training_args = make_training_args(config)

        smoke_callback = SmokeMetricsCallback()

        trainer = TrainerClass(
            model=model,
            processing_class=processor.tokenizer,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # Add smoke metrics callback
        trainer.add_callback(smoke_callback)

        # Apply response-only masking
        apply_response_masking(trainer, processor.tokenizer)

        # Step 10: Run training (10 steps)
        trainer.train()

        # Verify per-step metrics
        smoke_callback.verify()
        report["training_steps"] = smoke_callback.steps

        # Step 11: Validate the endpoint evaluation already run at step 10.
        # Avoid a redundant second pass; endpoint metrics were already emitted.
        eval_metrics: dict[str, Any] = {}
        if eval_dataset:
            eval_metrics = collect_endpoint_eval_metrics(trainer.state.log_history)
        report["evaluation"] = eval_metrics

        peak_before_save = 0.0
        if torch.cuda.is_available():
            peak_before_save = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

        # Step 12: Save the adapter export and tokenizer for inference.
        final_dir = Path(output_dir) / "final"
        trainer.save_model(str(final_dir))
        processor.tokenizer.save_pretrained(str(final_dir))
        checkpoint_metadata = {
            "config_hash": config.get("config_hash", ""),
            "dataset_revision": config.get("dataset", {}).get("revision", ""),
            "model_revision": config.get("model", {}).get("revision", ""),
            "processor_fingerprint_hash": pre_fingerprint_hash,
            "wandb_run_id": wandb_mgr.id,
        }
        (final_dir / "smoke_metadata.json").write_text(
            json.dumps(checkpoint_metadata, indent=2),
            encoding="utf-8",
        )
        report["checkpoint_path"] = str(final_dir)

        # Trainer checkpoints contain optimizer/scheduler/global-step state.
        ckpt_trainer_state = Path(output_dir) / f"checkpoint-{tc.get('max_steps', 10)}"
        if not ckpt_trainer_state.exists():
            raise SmokeGateError(
                f"Trainer checkpoint not found at {ckpt_trainer_state}",
                check="checkpoint_state_missing",
            )
        (ckpt_trainer_state / "smoke_metadata.json").write_text(
            json.dumps(checkpoint_metadata, indent=2),
            encoding="utf-8",
        )
        report["trainer_checkpoint_path"] = str(ckpt_trainer_state)

        # Step 13: Verify checkpoint state and processor fingerprint.
        reload_result = verify_checkpoint_reload(
            str(ckpt_trainer_state), pre_fingerprint_hash, config, strict=True
        )
        report["checkpoint_reload"] = reload_result

        # Step 14: Collect memory stats
        memory_stats: dict[str, Any] = {}
        if torch.cuda.is_available():
            peak_after_save = torch.cuda.max_memory_allocated() / 1024**3
            peak_alloc = max(peak_before_save, peak_after_save)
            peak_resv = torch.cuda.max_memory_reserved() / 1024**3
            memory_stats["peak_before_save_gib"] = round(peak_before_save, 3)
            memory_stats["peak_after_save_gib"] = round(peak_after_save, 3)
            memory_stats["peak_allocated_gib"] = round(peak_alloc, 3)
            memory_stats["peak_reserved_gib"] = round(peak_resv, 3)
            report["memory"] = memory_stats
            if peak_alloc >= 14.0:
                raise SmokeGateError(
                    f"Peak allocated VRAM {peak_alloc:.3f} GiB >= 14.0 GiB limit",
                    check="peak_vram_exceeded",
                )
        else:
            report["memory"] = memory_stats

        # Step 15: Write smoke manifest
        manifest = Manifest(phase="smoke", output_dir=output_dir)
        manifest_entry = manifest.create_entry(
            config_hash=config.get("config_hash", ""),
            dataset_revision=config.get("dataset", {}).get("revision", ""),
            model_revision=config.get("model", {}).get("revision", ""),
        )
        manifest.save(manifest_entry)

        # Step 16: Write report
        report["status"] = "success"
        report["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        report_path = write_smoke_report(report, wandb_mgr.id)
        report["report_path"] = str(report_path)

        wandb_mgr.finish()
        return report

    except (SmokeGateError, Exception):
        report["status"] = "failed"
        report["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Capture the failure details
        import traceback

        if "failure" not in report:
            report["failure"] = {
                "check": "unexpected",
                "message": traceback.format_exc(),
            }

        try:
            report_path = write_smoke_report(report, report.get("wandb_run_id", "unknown"))
            report["report_path"] = str(report_path)
        except Exception:
            pass

        if wandb_mgr is not None:
            wandb_mgr.finish()
        raise
