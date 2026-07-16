"""
Manifest lifecycle tests for univi.manifest.
"""

from __future__ import annotations

import json
import tempfile

import pytest


def test_manifest_creates_entry():
    """Creating a manifest entry produces a valid manifest dict."""
    from univi.manifest import Manifest

    m = Manifest(phase="preflight", output_dir="/tmp/test-manifest")
    entry = m.create_entry(
        config_hash="abc123",
        dataset_revision="def456",
        model_revision="4abfca14e6c6bfb5888b80288185b1243fb8d539",
    )
    assert entry["phase"] == "preflight"
    assert "timestamp" in entry
    assert "config_hash" in entry
    assert "model_revision" in entry
    assert entry["model_revision"] == "4abfca14e6c6bfb5888b80288185b1243fb8d539"


def test_manifest_freshness_hash():
    """Manifest hash changes when config hash changes."""
    from univi.manifest import Manifest

    m1 = Manifest(phase="preflight", output_dir="/tmp/test-m1")
    m2 = Manifest(phase="preflight", output_dir="/tmp/test-m2")
    e1 = m1.create_entry(config_hash="aaa", dataset_revision="r1", model_revision="m1")
    e2 = m2.create_entry(config_hash="bbb", dataset_revision="r1", model_revision="m1")
    assert m1.fingerprint(e1) != m2.fingerprint(e2)


def test_manifest_rejects_stale_phase():
    """Loading a preflight manifest when smoke phase expected fails."""
    from univi.manifest import Manifest, ManifestMismatchError

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Manifest(phase="preflight", output_dir=tmpdir)
        entry = m.create_entry(config_hash="abc", dataset_revision="r1", model_revision="m1")
        m.save(entry)
        # Now try to load same path expecting a different phase
        m2 = Manifest(phase="smoke", output_dir=tmpdir)
        with pytest.raises(ManifestMismatchError, match="Phase mismatch"):
            m2.load()


def test_manifest_save_and_load():
    """Save and load a manifest entry round-trips."""
    from univi.manifest import Manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        m = Manifest(phase="preflight", output_dir=tmpdir)
        entry = m.create_entry(config_hash="abc", dataset_revision="r1", model_revision="m1")
        path = m.save(entry)
        assert path.exists()

        loaded = Manifest(phase="preflight", output_dir=tmpdir).load()
        assert loaded["phase"] == "preflight"
        assert loaded["config_hash"] == "abc"


def test_manifest_verify_freshness():
    """verify_freshness passes with matching hash, raises on mismatch."""
    from univi.manifest import Manifest, ManifestMismatchError

    m = Manifest(phase="preflight", output_dir="/tmp/t")
    entry = m.create_entry(config_hash="abc", dataset_revision="r1", model_revision="m1")
    m.verify_freshness(entry, current_hash="abc")  # no error

    with pytest.raises(ManifestMismatchError, match="Config hash mismatch"):
        m.verify_freshness(entry, current_hash="xyz")
