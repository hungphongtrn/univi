"""
Phase artifact manifest creation, freshness hash computation, and verification.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASES = ["preflight", "smoke", "train", "resume", "evaluate"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ManifestMismatchError(Exception):
    """Raised when a manifest's freshness or phase does not match expectations."""

    def __init__(self, message: str, expected_phase: str = "", actual_phase: str = "") -> None:
        self.expected_phase = expected_phase
        self.actual_phase = actual_phase
        super().__init__(message)


# ---------------------------------------------------------------------------
# Manifest class
# ---------------------------------------------------------------------------


class Manifest:
    """Create, save, load, and verify phase artifact manifests."""

    def __init__(
        self,
        phase: str,
        output_dir: str | Path | None = None,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"Invalid phase {phase!r}. Valid phases: {PHASES}")
        self.phase = phase
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()

    @property
    def manifest_path(self) -> Path:
        return self._output_dir / "manifest.json"

    def create_entry(
        self,
        config_hash: str,
        dataset_revision: str,
        model_revision: str,
        **extra: Any,
    ) -> dict:
        """Create a manifest entry dict for the current phase."""
        entry = {
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "dataset_revision": dataset_revision,
            "model_revision": model_revision,
        }
        entry.update(extra)
        return entry

    def save(self, entry: dict) -> Path:
        """Write a manifest entry to ``manifest.json`` in the output directory."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_path
        path.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")
        return path

    def load(self) -> dict:
        """Load a manifest from disk."""
        path = self.manifest_path
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        entry = json.loads(path.read_text(encoding="utf-8"))

        loaded_phase = entry.get("phase", "")
        if loaded_phase != self.phase:
            raise ManifestMismatchError(
                f"Phase mismatch: expected {self.phase!r}, loaded {loaded_phase!r}.",
                expected_phase=self.phase,
                actual_phase=loaded_phase,
            )
        return entry

    def fingerprint(self, entry: dict) -> str:
        """Compute a SHA-256 hash of the phase-aware subset of a manifest entry."""
        subset = {k: v for k, v in entry.items() if k in self._fingerprint_keys()}
        normalized = json.dumps(subset, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def verify_freshness(self, entry: dict, current_hash: str) -> None:
        """Verify that a loaded manifest's config_hash matches *current_hash*.

        Raises ManifestMismatchError if the hashes differ.
        """
        stored_config_hash = entry.get("config_hash", "")
        if stored_config_hash != current_hash:
            raise ManifestMismatchError(
                f"Config hash mismatch: stored={stored_config_hash[:12]!r}, "
                f"current={current_hash[:12]!r}. "
                "Resolved config has changed since the manifest was created."
            )

    def _fingerprint_keys(self) -> set[str]:
        """Return the set of keys included in the fingerprint."""
        return {"phase", "config_hash", "dataset_revision", "model_revision"}
