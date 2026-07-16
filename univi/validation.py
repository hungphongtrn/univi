"""
Structural schema validation, 32-row stratified processor review, and sentinel counting.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Required metadata fields
# ---------------------------------------------------------------------------

REQUIRED_METADATA = {
    "source_dataset_id",
    "split",
    "row_id",
    "render_config",
    "modality_label",
    "preprocessing_version",
}

VALID_ROLES = {"user", "assistant"}


# ---------------------------------------------------------------------------
# Row-level validation
# ---------------------------------------------------------------------------


def validate_row_schema(row: dict) -> list[str]:
    """Validate a single dataset row's structure.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # Check messages presence
    if "messages" not in row:
        return ["Missing 'messages' column."]

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("Messages must be a list with at least 2 entries.")
        return errors

    # Check role alternation starts with user
    if messages[0].get("role") != "user":
        errors.append(
            f"First message role must be 'user', got {messages[0].get('role')!r}."
        )
    if messages[-1].get("role") != "assistant":
        errors.append(
            f"Last message role must be 'assistant', got {messages[-1].get('role')!r}."
        )

    # Check for unexpected roles
    for msg in messages:
        role = msg.get("role", "")
        if role not in VALID_ROLES:
            errors.append(f"Unexpected role {role!r}. Valid roles: {VALID_ROLES}.")

    # Count image placeholders vs images column
    placeholder_count = 0
    for msg in messages:
        for content in msg.get("content", []):
            if not isinstance(content, dict):
                continue
            ctype = content.get("type", "")
            if ctype not in ("image", "text"):
                errors.append(f"Invalid content type {ctype!r}.")
            elif ctype == "image":
                placeholder_count += 1

    images = row.get("images", [])
    if not isinstance(images, list):
        images = []

    if placeholder_count != len(images):
        errors.append(
            f"Image placeholder count ({placeholder_count}) does not match "
            f"images column length ({len(images)})."
        )

    # Check required metadata
    for field in REQUIRED_METADATA:
        if field not in row:
            errors.append(f"Missing required metadata field: {field!r}.")

    return errors


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def validate_dataset(
    rows: list[dict],
    max_rows: int = 0,
) -> dict:
    """Validate all rows in a dataset.

    Args:
        rows: List of dataset row dicts.
        max_rows: If > 0, stop after this many rows.

    Returns:
        Validation result dict with raw/valid/failed counts and per-subset breakdown.
    """
    valid_rows = 0
    failed_rows = 0
    errors_per_row: dict[str, list[str]] = {}
    per_subset: dict[str, dict[str, int]] = {}

    for i, row in enumerate(rows):
        if max_rows > 0 and i >= max_rows:
            break

        subset = row.get("source_dataset_id", "unknown")
        if subset not in per_subset:
            per_subset[subset] = {"raw": 0, "valid": 0, "failed": 0}
        per_subset[subset]["raw"] += 1

        row_errors = validate_row_schema(row)
        if row_errors:
            failed_rows += 1
            per_subset[subset]["failed"] += 1
            errors_per_row[str(i)] = row_errors
        else:
            valid_rows += 1
            per_subset[subset]["valid"] += 1

    # i is the last loop index; defined only if loop body executed
    if max_rows > 0:
        if 'i' in dir() or 'i' in locals():
            checked = i + 1
        else:
            checked = 0
    else:
        checked = len(rows)
    return {
        "raw_rows": checked,
        "valid_rows": valid_rows,
        "failed_rows": failed_rows,
        "per_subset": per_subset,
        "errors_per_row": errors_per_row,
    }


# ---------------------------------------------------------------------------
# Stratified review row selection
# ---------------------------------------------------------------------------


def select_review_rows(
    rows: list[dict],
    n_per_subset: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Select stratified review rows across subsets.

    Groups rows by ``source_dataset_id`` (subset) and selects
    up to *n_per_subset* rows per subset, balancing across
    image-count buckets.

    Returns a list of row dicts with an added ``subset`` key.
    """
    import random

    rng = random.Random(seed)

    # Group by subset
    by_subset: dict[str, list[dict]] = {}
    for row in rows:
        subset = row.get("source_dataset_id", "unknown")
        by_subset.setdefault(subset, []).append(row)

    selected: list[dict] = []
    for subset, subset_rows in by_subset.items():
        # Further bucket by image count
        buckets: dict[int, list[dict]] = {}
        for row in subset_rows:
            n_images = len(row.get("images", []))
            buckets.setdefault(n_images, []).append(row)

        # Select proportionally from each bucket
        remaining = min(n_per_subset, len(subset_rows))
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)

        # Round-robin through buckets
        bucket_order = sorted(buckets.keys(), key=lambda k: len(buckets[k]))
        while remaining > 0:
            for bkey in bucket_order:
                if remaining <= 0:
                    break
                blist = buckets[bkey]
                if blist:
                    row = blist.pop(0)
                    row["subset"] = subset
                    selected.append(row)
                    remaining -= 1
                if remaining <= 0:
                    break

    return selected


# ---------------------------------------------------------------------------
# Review markdown generation
# ---------------------------------------------------------------------------


def generate_review_markdown(
    rows: list[dict],
    processor: Any = None,
    tokenizer: Any = None,
) -> str:
    """Generate a markdown review of selected rows for human inspection.

    When *processor* is available, renders
    ``processor.apply_chat_template(...)`` for each row.
    """
    lines = [
        "# Training Data Review",
        "",
        f"**Rows reviewed:** {len(rows)}",
        "",
        "| # | Subset | Modality | Placeholders | Images | Target Tokens | Row ID |",
        "|---|--------|----------|-------------|--------|--------------|--------|",
    ]

    for i, row in enumerate(rows):
        subset = row.get("subset", row.get("source_dataset_id", "?"))
        modality = row.get("modality_label", "?")
        placeholder_count = sum(
            1
            for msg in row.get("messages", [])
            for c in msg.get("content", [])
            if isinstance(c, dict) and c.get("type") == "image"
        )
        n_images = len(row.get("images", []))
        target_tokens = "?"
        row_id = row.get("row_id", "?")
        lines.append(
            f"| {i} | {subset} | {modality} | {placeholder_count} | "
            f"{n_images} | {target_tokens} | {row_id} |"
        )

    lines.extend(["", "---", ""])

    if processor is not None and hasattr(processor, "apply_chat_template"):
        for i, row in enumerate(rows):
            try:
                rendered = processor.apply_chat_template(
                    row.get("messages", []), tokenize=False
                )
            except Exception:
                rendered = "<render error>"
            lines.append(f"### Row {i}")
            lines.append("")
            lines.append(f"```")
            lines.append(rendered)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_validation(results: dict) -> dict:
    """Normalise validation results into a summary dict."""
    return {
        "raw_rows": results.get("raw_rows", 0),
        "valid_rows": results.get("valid_rows", 0),
        "failed_rows": results.get("failed_rows", 0),
        "per_subset": results.get("per_subset", {}),
    }
