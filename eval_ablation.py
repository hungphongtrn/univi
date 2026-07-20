"""
Visual-grounding ablation — does the model actually READ the image?

For every configured source this measures the target negative-log-likelihood
under three conditions that share the SAME messages / SAME target and differ
only in the pixels fed to the vision encoder:

    aligned   : the real image                       (baseline loss)
    permuted  : another example's image, same image  (Modality-Permutation
                count, target unchanged                Control — CONTEXT.md:179)
    blank     : a flat grey image, same size          (no visual information)

The signal is the *delta*:

    reliance = loss(permuted) - loss(aligned)

If corrupting the image barely moves the loss, the model is producing the
target from language priors rather than reading the pixels — exactly the
"Linguistic Prior Dependence" failure the project is designed to detect
(CONTEXT.md:67). Run against BOTH the base model and the trained checkpoint
to see whether training *increased* or *eroded* visual grounding.

Loss only (single forward pass per lane) — cheap enough for a 3060.

Usage:
    uv run python eval_ablation.py --config configs/3060_eval.yaml
    uv run python eval_ablation.py --config configs/eval.yaml --checkpoint data/checkpoints/full-v0/checkpoint-3000
    uv run python eval_ablation.py --config configs/eval.yaml --max-samples 150 --sources fineweb,librispeech
    uv run python eval_ablation.py --config configs/eval.yaml --no-base   # trained checkpoint only (halves runtime)
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse the exact training-consistent seams from the lane harness so image
# preprocessing, response-only masking and loss computation are identical.
from eval_lane import (
    _SOURCES,
    _merge_images,
    compute_loss,
    load_eval_config,
    load_model,
    load_source_dataset,
)
from univi.metrics import write_eval_output


# ---------------------------------------------------------------------------
# Image variants (model-independent, computed once)
# ---------------------------------------------------------------------------

def _blank_like(img):
    """A flat grey image with the same size/mode — occupies the same number of
    vision tokens as the original but carries no information."""
    from PIL import Image

    mode = img.mode if img.mode in ("RGB", "L") else "RGB"
    fill = 127 if mode == "L" else (127, 127, 127)
    return Image.new(mode, img.size, fill)


def build_permutation(rows: list[dict]) -> list[list | None]:
    """Deterministic modality-permutation donor images for each row.

    Rows are grouped by image count and the images are rolled by one position
    within each group. This guarantees ``_merge_images`` still sees a matching
    placeholder/image count while the visual content belongs to a *different*
    example. Rows whose image count is unique (group size 1) have no valid
    same-shape donor and get ``None`` (their permuted lane is skipped).
    """
    groups: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        n = len(row.get("images", []))
        groups.setdefault(n, []).append(i)

    donor: list[list | None] = [None] * len(rows)
    for _n, idxs in groups.items():
        if len(idxs) < 2:
            continue  # no other row with the same image count → skip
        for k, src_i in enumerate(idxs):
            donor_i = idxs[(k + 1) % len(idxs)]
            donor[src_i] = rows[donor_i]["images"]
    return donor


def build_blanks(rows: list[dict]) -> list[list]:
    return [[_blank_like(im) for im in row.get("images", [])] for row in rows]


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def evaluate_model_source(
    model,
    tokenizer,
    rows: list[dict],
    permuted_images: list[list | None],
    blank_images: list[list],
    lanes: list[str],
) -> dict:
    """Compute per-lane mean NLL for one model on one source.

    Losses are paired per row (identical target across lanes) so the deltas are
    honest paired differences, not differences of independently sampled means.
    """
    import torch

    per_row: list[dict] = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            base_messages = row["messages"]
            entry: dict[str, float] = {}

            # aligned is always computed — it is the baseline for every delta
            entry["aligned"] = compute_loss(
                model, tokenizer, _merge_images(base_messages, row.get("images", []))
            )

            if "permuted" in lanes and permuted_images[i] is not None:
                entry["permuted"] = compute_loss(
                    model, tokenizer, _merge_images(base_messages, permuted_images[i])
                )

            if "blank" in lanes:
                entry["blank"] = compute_loss(
                    model, tokenizer, _merge_images(base_messages, blank_images[i])
                )

            per_row.append(entry)

    aligned = [r["aligned"] for r in per_row]
    result = {
        "n": len(per_row),
        "aligned": _mean(aligned),
    }

    for lane in ("permuted", "blank"):
        pairs = [(r["aligned"], r[lane]) for r in per_row if lane in r]
        if not pairs:
            continue
        lane_loss = _mean([p[1] for p in pairs])
        deltas = [p[1] - p[0] for p in pairs]
        aligned_on_pairs = _mean([p[0] for p in pairs])
        result[lane] = lane_loss
        result[f"delta_{lane}"] = _mean(deltas)
        result[f"rel_{lane}"] = (
            _mean(deltas) / aligned_on_pairs
            if aligned_on_pairs not in (None, 0)
            else None
        )
        result[f"n_{lane}"] = len(pairs)

    return result


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def grounding_flag(rel_permuted: float | None) -> str:
    """Heuristic label from the relative permutation delta. Thresholds are a
    guide, not a verdict — read the raw deltas."""
    if rel_permuted is None:
        return "n/a"
    if rel_permuted < 0.05:
        return "IGNORES IMAGE"
    if rel_permuted < 0.20:
        return "WEAK grounding"
    return "uses image"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _source_paths(config: dict, sources: list[str]) -> dict[str, str]:
    eval_datasets = config.get("eval_datasets", {})
    return {s: eval_datasets[s] for s in sources if eval_datasets.get(s)}


def run_ablation(
    config: dict,
    max_samples: int,
    sources: list[str],
    lanes: list[str],
    include_base: bool,
) -> dict:
    paths = _source_paths(config, sources)

    # Load + cap each source once, and precompute the image variants (these are
    # model-independent, so we do it before touching any GPU weights).
    prepared: dict[str, dict] = {}
    for name, path in paths.items():
        try:
            rows = list(load_source_dataset(path))[:max_samples]
        except Exception as exc:  # missing/unreadable dataset → skip, don't abort
            print(f"[skip] {name}: could not load {path} ({exc})", flush=True)
            continue
        if not rows:
            print(f"[skip] {name}: empty dataset", flush=True)
            continue
        prepared[name] = {
            "rows": rows,
            "permuted": build_permutation(rows) if "permuted" in lanes else [None] * len(rows),
            "blank": build_blanks(rows) if "blank" in lanes else [[]] * len(rows),
        }
        print(f"[load] {name}: {len(rows)} rows", flush=True)

    models_to_run = (["base"] if include_base else []) + ["trained"]
    model_key_to_path = {
        "base": config["base_model"],
        "trained": config["checkpoint_path"],
    }

    per_source: dict[str, dict] = {name: {} for name in prepared}

    # One model in memory at a time (3060-friendly): load, sweep every source,
    # release, then the next model.
    for model_key in models_to_run:
        model_path = model_key_to_path[model_key]
        print(f"\n=== loading {model_key} model: {model_path} ===", flush=True)
        model, tokenizer = load_model(model_path)
        try:
            for name, data in prepared.items():
                print(f"  [{model_key}] {name} ...", flush=True)
                per_source[name][model_key] = evaluate_model_source(
                    model, tokenizer, data["rows"], data["permuted"], data["blank"], lanes
                )
        finally:
            del model, tokenizer
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    # Training effect: how much the permutation delta changed base -> trained.
    for name, models in per_source.items():
        if "base" in models and "trained" in models:
            bd = models["base"].get("delta_permuted")
            td = models["trained"].get("delta_permuted")
            if bd is not None and td is not None:
                models["training_effect_permuted"] = td - bd
        tr = models.get("trained", {})
        models["flag"] = grounding_flag(tr.get("rel_permuted"))

    return {
        "checkpoint": config.get("checkpoint_path"),
        "base_model": config.get("base_model"),
        "max_samples": max_samples,
        "lanes": lanes,
        "per_source": per_source,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  -   "


def print_report(results: dict) -> None:
    print("\n" + "=" * 78)
    print("VISUAL-GROUNDING ABLATION  (loss = mean NLL/token; higher delta = more image use)")
    print("=" * 78)
    for name, models in results["per_source"].items():
        print(f"\n■ {name}   [{models.get('flag', 'n/a')}]")
        header = f"    {'model':8} {'aligned':>9} {'permuted':>9} {'Δperm':>8} {'relΔ':>7} {'blank':>9} {'Δblank':>8}"
        print(header)
        for mk in ("base", "trained"):
            m = models.get(mk)
            if not m:
                continue
            print(
                f"    {mk:8} {_fmt(m.get('aligned')):>9} {_fmt(m.get('permuted')):>9} "
                f"{_fmt(m.get('delta_permuted')):>8} {_fmt(m.get('rel_permuted')):>7} "
                f"{_fmt(m.get('blank')):>9} {_fmt(m.get('delta_blank')):>8}"
            )
        te = models.get("training_effect_permuted")
        if te is not None:
            arrow = "improved" if te > 0 else "ERODED"
            print(f"    training effect on grounding (Δperm trained-base): {te:+.4f}  [{arrow}]")
    print("\nRead: Δperm near 0  -> model ignores the image (prior-driven).")
    print("      Δperm large    -> model relies on the pixels.")
    print("      trained < base -> training eroded grounding (stop-and-rethink signal).")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml", help="Eval config YAML (reuses eval_lane schema).")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint_path from the config.")
    parser.add_argument("--max-samples", type=int, default=200, help="Rows per source (first-N of the pre-shuffled split).")
    parser.add_argument("--sources", default=None, help="Comma list subset of: " + ",".join(_SOURCES))
    parser.add_argument("--lanes", default="permuted,blank", help="Corruption lanes to run (aligned is always the baseline).")
    parser.add_argument("--no-base", action="store_true", help="Skip the base model (trained checkpoint only).")
    parser.add_argument("--output", default=None, help="Override output JSON path.")
    args = parser.parse_args(argv)

    config = load_eval_config(args.config)
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint

    sources = args.sources.split(",") if args.sources else list(_SOURCES)
    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]

    results = run_ablation(
        config,
        max_samples=args.max_samples,
        sources=sources,
        lanes=lanes,
        include_base=not args.no_base,
    )

    print_report(results)

    default_out = Path(config.get("output", "data/eval/results.json")).parent / "ablation.json"
    output_path = args.output or str(default_out)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_eval_output(results, output_path)
    print(f"Ablation results written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
