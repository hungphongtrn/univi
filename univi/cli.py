"""
CLI argument parser, mode dispatch, and top-level orchestrator for training.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
import json
import sys
from typing import Sequence

from univi.config import resolve_config

# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODES = ["train", "preflight_only", "smoke", "resume", "evaluate"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build and return parsed CLI arguments."""
    return _build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Override parsing
# ---------------------------------------------------------------------------


def _parse_overrides(raw: list[str] | None) -> dict:
    """Convert ``KEY=VALUE`` list to a flat dict for merge."""
    if not raw:
        return {}
    overrides = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid override format (expected KEY=VALUE): {item!r}")
        key, value = item.split("=", 1)
        # Try numeric / boolean conversion
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        overrides[key] = value
    return overrides


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _print_help() -> None:
    """Print the CLI help text without triggering argparse sys.exit."""
    parser = _build_parser()
    parser.print_help()


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separated so _print_help can use it)."""
    parser = argparse.ArgumentParser(
        description="UniVi Gemma 4 E2B training entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We handle help ourselves
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config file.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preflight-only",
        dest="mode",
        action="store_const",
        const="preflight_only",
        help="Run preflight validation without model loading or training.",
    )
    mode_group.add_argument(
        "--smoke",
        dest="mode",
        action="store_const",
        const="smoke",
        help="Run a short smoke training run.",
    )
    mode_group.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help="Resume training from the latest checkpoint.",
    )
    mode_group.add_argument(
        "--evaluate",
        dest="mode",
        action="store_const",
        const="evaluate",
        help="Run evaluation on a trained checkpoint.",
    )
    mode_group.add_argument(
        "--train",
        dest="mode",
        action="store_const",
        const="train",
        help="Run full training (default).",
    )

    parser.add_argument(
        "--dataset-source",
        choices=["auto", "local", "hub"],
        default="auto",
        help="Dataset source: auto (default), local, or hub.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="Cache directory for dataset downloads.",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Override dataset revision.",
    )
    parser.add_argument(
        "--preflight-workers",
        type=int,
        default=1,
        help="Number of workers for preflight validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and print summary without executing training.",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=None,
        help="YAML key overrides as KEY=VALUE pairs.",
    )

    parser.set_defaults(mode="train")
    return parser
def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate mode.

    Returns an exit code (0 = success, 1 = failure, 2 = error).
    """
    try:
        # Resolve the argument list
        args_list = list(sys.argv[1:] if argv is None else argv)

        # Handle --help before argparse to avoid SystemExit propagation
        if "-h" in args_list or "--help" in args_list:
            _print_help()
            return 0

        args = parse_args(argv)
        # Resolve config
        overrides = _parse_overrides(args.overrides)
        cfg = resolve_config(args.config, overrides=overrides)

        # Apply CLI overrides to config for downstream use
        if args.dataset_source != "auto":
            cfg.setdefault("dataset", {})["source"] = args.dataset_source
        if args.dataset_cache_dir:
            cfg.setdefault("dataset", {})["cache_dir"] = args.dataset_cache_dir
        if args.dataset_revision:
            cfg.setdefault("dataset", {})["revision"] = args.dataset_revision

        # ---- Mode dispatch ----

        if args.dry_run:
            print("=== Resolved Config (JSON) ===")
            print(json.dumps(cfg, indent=2, default=str))

            ds_cfg = cfg.get("dataset", {})
            subsets = ds_cfg.get("subsets", [])
            print(f"\n=== Dataset Summary ===")
            print(f"  Subsets: {subsets}")
            print(f"  Source: {ds_cfg.get('source', 'auto')}")
            print(f"  Path: {ds_cfg.get('path', 'N/A')}")
            print(f"  Hub repo: {ds_cfg.get('hf_hub_repo_id', 'N/A')}")
            print(f"\n  Config hash: {cfg['config_hash']}")
            print(f"  Dry-run: no side effects, no model loaded.\n")
            return 0

        if args.mode == "preflight_only":
            return _run_preflight_only(cfg, args)
        elif args.mode == "smoke":
            return _run_smoke(cfg, args)
        elif args.mode == "resume":
            return _run_resume(cfg, args)
        elif args.mode == "evaluate":
            return _run_evaluate(cfg, args)
        else:
            return _run_train(cfg, args)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def _run_preflight_only(cfg: dict, args: argparse.Namespace) -> int:
    """Run preflight validation without GPU model loading."""
    from univi.trainer import run_preflight

    result = run_preflight(cfg, dry_run=False)
    status = result.get("status", "unknown")

    if status == "success":
        summ = result.get("validation", {})
        print(
            f"Preflight complete: {summ.get('valid_rows', 0)} valid, "
            f"{summ.get('failed_rows', 0)} failed "
            f"out of {summ.get('raw_rows', 0)} raw rows."
        )
        if summ.get("failed_rows", 0) > 0:
            print("  WARNING: some rows failed validation.", file=sys.stderr)
        print(f"  Processor status: {result.get('processor_status', 'N/A')}")
        return 0
    else:
        print(f"Preflight failed: {result.get('error', 'unknown error')}", file=sys.stderr)
        return 1


def _resolve_smoke_config(config: dict) -> dict:
    """Build bounded smoke runtime settings while retaining the production gate hash."""
    smoke = deepcopy(config)
    runtime = config.get("smoke", {})

    smoke.setdefault("training", {})["max_steps"] = runtime.get("max_steps", 10)
    smoke["training"]["logging_steps"] = runtime.get("logging_steps", 1)
    if runtime.get("output_dir"):
        smoke["training"]["output_dir"] = runtime["output_dir"]
    if runtime.get("dataset_path"):
        smoke.setdefault("dataset", {})["path"] = runtime["dataset_path"]
    if runtime.get("run_name_template"):
        smoke.setdefault("wandb", {})["run_name_template"] = runtime[
            "run_name_template"
        ]
    return smoke


def _run_smoke(cfg: dict, args: argparse.Namespace) -> int:
    """Run a short smoke training with GPU gate checks and structured report."""
    from univi.smoke import run_smoke, SmokeGateError

    try:
        report = run_smoke(_resolve_smoke_config(cfg))
        status = report.get("status", "unknown")
        print(f"Smoke gate completed: {status}")

        if "report_path" in report:
            print(f"Report: {report['report_path']}")

        if "wandb_run_id" in report:
            print(f"W&B run ID: {report['wandb_run_id']}")

        if "wandb_url" in report and report["wandb_url"]:
            print(f"W&B URL: {report['wandb_url']}")

        if "memory" in report:
            m = report["memory"]
            print(f"Peak VRAM: allocated={m.get('peak_allocated_gib', '?')} GiB, "
                  f"reserved={m.get('peak_reserved_gib', '?')} GiB")

        if status == "success":
            return 0
        else:
            failure = report.get("failure", {})
            print(f"Smoke failed: {failure.get('check', 'unknown')} — {failure.get('message', '')}",
                  file=sys.stderr)
            return 1

    except SmokeGateError as exc:
        print(f"Smoke gate error: {exc.check}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Smoke training failed: {exc}", file=sys.stderr)
        return 1


def _run_resume(cfg: dict, args: argparse.Namespace) -> int:
    """Resume training from checkpoint."""
    from univi.trainer import train

    try:
        trainer = train(cfg, args)
        print("Resume completed successfully.")
        return 0
    except Exception as exc:
        print(f"Resume failed: {exc}", file=sys.stderr)
        return 1


def _run_evaluate(cfg: dict, args: argparse.Namespace) -> int:
    """Evaluate a trained checkpoint."""
    print("Evaluation mode is a Phase 3 feature. Use eval_lane.py for now.", file=sys.stderr)
    return 0


def _run_train(cfg: dict, args: argparse.Namespace) -> int:
    """Run full training (requires preflight artifact with matching config_hash)."""
    from univi.trainer import train
    from univi.manifest import Manifest, ManifestMismatchError
    from pathlib import Path

    config_hash = cfg.get("config_hash", "")

    # Scan for most recent preflight manifest
    val_root = Path("data") / "validation"
    preflight_manifest = None
    if val_root.is_dir():
        entries = sorted(val_root.iterdir(), key=lambda p: p.name, reverse=True)
        for entry in entries:
            mp = entry / "manifest.json"
            if mp.exists():
                try:
                    manifest = Manifest(phase="preflight", output_dir=str(entry))
                    entry_data = manifest.load()
                    manifest.verify_freshness(entry_data, config_hash)
                    preflight_manifest = entry_data
                    break
                except (ManifestMismatchError, FileNotFoundError, ValueError):
                    continue

    if preflight_manifest is None:
        print("Error: Valid preflight manifest not found. Run `univi-train --preflight-only` first.", file=sys.stderr)
        return 1

    try:
        trainer = train(cfg, args)
        print("Training completed successfully.")
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
