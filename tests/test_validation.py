"""
Structural validation and 32-row processor review tests.
"""

from __future__ import annotations

from PIL import Image

import pytest


def _make_valid_row():
    """Create a well-formed row for validation tests."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "A cat."}],
            },
        ],
        "images": [Image.new("RGB", (224, 224))],
        "source_dataset_id": "BAAI/DenseFusion-1M",
        "split": "train",
        "row_id": "0",
        "render_config": "{}",
        "modality_label": "image-text",
        "preprocessing_version": "1.0.0",
    }


def test_validator_accepts_valid_row():
    """A well-formed row passes structural validation."""
    from univi.validation import validate_row_schema

    errors = validate_row_schema(_make_valid_row())
    assert len(errors) == 0


def test_validator_rejects_missing_images():
    """Row missing images column fails validation."""
    from univi.validation import validate_row_schema

    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        ],
    }
    errors = validate_row_schema(row)
    assert any("images" in e for e in errors)


def test_validator_rejects_placeholder_count_mismatch():
    """Image placeholder count must match images column length."""
    from univi.validation import validate_row_schema

    row = _make_valid_row()
    row["images"] = [
        Image.new("RGB", (224, 224)),
        Image.new("RGB", (224, 224)),
    ]  # 2 images but only 1 placeholder
    errors = validate_row_schema(row)
    assert any("placeholder" in e.lower() for e in errors)


def test_validator_rejects_role_order():
    """Message roles must start with user, then assistant."""
    from univi.validation import validate_row_schema

    row = _make_valid_row()
    row["messages"] = [
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
        {"role": "user", "content": [{"type": "text", "text": "Describe."}]},
    ]
    errors = validate_row_schema(row)
    assert any("role" in e.lower() for e in errors)


def test_validator_rejects_missing_messages():
    """Row without messages fails."""
    from univi.validation import validate_row_schema

    errors = validate_row_schema({})
    assert any("messages" in e for e in errors)


def test_validator_rejects_unexpected_role():
    """Row with system role fails."""
    from univi.validation import validate_row_schema

    row = _make_valid_row()
    row["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": "Be helpful."}]},
        {"role": "user", "content": [{"type": "text", "text": "Hi."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
    ]
    errors = validate_row_schema(row)
    assert any("role" in e.lower() for e in errors)


def test_validation_summary_contains_expected_counts():
    """Validation summary has raw, valid, failed, and per-subset counts."""
    from univi.validation import summarize_validation

    results = {
        "raw_rows": 10,
        "valid_rows": 8,
        "failed_rows": 2,
        "per_subset": {
            "fineweb-edu": {"raw": 5, "valid": 4, "failed": 1},
            "densefusion": {"raw": 5, "valid": 4, "failed": 1},
        },
    }
    summary = summarize_validation(results)
    assert summary["raw_rows"] == 10
    assert summary["valid_rows"] == 8
    assert summary["failed_rows"] == 2
    assert "per_subset" in summary
    assert "fineweb-edu" in summary["per_subset"]


def test_dataset_validation():
    """validate_dataset validates a list of rows and returns counts."""
    from univi.validation import validate_dataset

    rows = [_make_valid_row() for _ in range(5)]
    # Add one invalid row
    rows.append({"messages": []})
    result = validate_dataset(rows, max_rows=0)
    assert result["raw_rows"] == 6
    assert result["valid_rows"] >= 5
    assert result["failed_rows"] >= 1
