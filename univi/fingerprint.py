"""
Processor/tokenizer fingerprint computation, pre/post model-load matching.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FingerprintMismatchError(ValueError):
    """Raised when pre- and post-model-load fingerprints do not match."""

    def __init__(self, message: str, diffs: list[str] | None = None) -> None:
        self.diffs = diffs or []
        super().__init__(message)


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def compute_fingerprint(processor: Any, tokenizer: Any) -> dict:
    """Compute a canonical processor/tokenizer fingerprint.

    The *processor* is the Gemma4Processor wrapper returned by
    ``FastVisionModel.from_pretrained``.  The *tokenizer* **must** be the
    inner tokenizer (``processor.tokenizer``) — the one that has had
    ``get_chat_template(processor.tokenizer, "gemma-4")`` applied.

    Returns a JSON-serializable dict with fields:
    - tokenizer_vocab_size
    - tokenizer_class
    - special_tokens_map
    - chat_template
    - chat_template_hash
    - image_sentinel_ids
    - processor_class
    - image_processor_config
    """
    fp: dict[str, Any] = {}

    # Tokenizer fields
    try:
        fp["tokenizer_vocab_size"] = len(tokenizer)
    except Exception:
        fp["tokenizer_vocab_size"] = None

    fp["tokenizer_class"] = type(tokenizer).__name__

    try:
        fp["special_tokens_map"] = _serialize(tokenizer.special_tokens_map)
    except Exception:
        fp["special_tokens_map"] = {}

    chat_template = getattr(tokenizer, "chat_template", None)
    fp["chat_template"] = chat_template
    fp["chat_template_hash"] = (
        hashlib.sha256((chat_template or "").encode()).hexdigest()
    )

    # Image sentinel IDs
    try:
        im_end = getattr(tokenizer, "image_token_id", None)
        fp["image_sentinel_ids"] = [im_end] if im_end is not None else []
    except Exception:
        fp["image_sentinel_ids"] = []

    # Processor fields
    fp["processor_class"] = type(processor).__name__

    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None and hasattr(ip, "to_dict"):
            fp["image_processor_config"] = _serialize(ip.to_dict())
        else:
            fp["image_processor_config"] = {}
    except Exception:
        fp["image_processor_config"] = {}

    return fp


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------


def fingerprint_hash(fp: dict) -> str:
    """Deterministic SHA-256 hex digest of a fingerprint dict."""
    normalized = json.dumps(fp, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def assert_fingerprints_match(fp1: dict, fp2: dict) -> None:
    """Raise ``FingerprintMismatchError`` if the two fingerprints differ."""
    diffs = fingerprint_diff(fp1, fp2)
    if diffs:
        raise FingerprintMismatchError(
            "Fingerprint mismatch: " + "; ".join(diffs), diffs=diffs
        )


def fingerprint_diff(fp1: dict, fp2: dict) -> list[str]:
    """Return a human-readable list of differences between two fingerprints."""
    diffs: list[str] = []
    all_keys = set(fp1) | set(fp2)

    for key in sorted(all_keys):
        v1 = fp1.get(key)
        v2 = fp2.get(key)

        if key not in fp1:
            diffs.append(f"'{key}' only in second fingerprint")
        elif key not in fp2:
            diffs.append(f"'{key}' only in first fingerprint")
        elif v1 != v2:
            diffs.append(f"'{key}': {v1!r} != {v2!r}")

    return diffs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Attempt to convert a value to a JSON-safe type."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
