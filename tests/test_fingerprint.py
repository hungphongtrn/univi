"""
Processor/template fingerprint tests for univi.fingerprint.
"""

from __future__ import annotations

import pytest


def test_fingerprint_hash_is_stable():
    """Same fingerprint dict produces identical hash."""
    from univi.fingerprint import fingerprint_hash

    fp = {"a": 1, "b": "hello"}
    h1 = fingerprint_hash(fp)
    h2 = fingerprint_hash(fp)
    assert h1 == h2


def test_fingerprint_hash_is_deterministic():
    """fingerprint_hash produces the same result regardless of key order."""
    from univi.fingerprint import fingerprint_hash

    fp1 = {"a": 1, "b": 2}
    fp2 = {"b": 2, "a": 1}
    assert fingerprint_hash(fp1) == fingerprint_hash(fp2)


def test_fingerprint_diff_identical():
    """fingerprint_diff returns empty list for identical fingerprints."""
    from univi.fingerprint import fingerprint_diff

    fp = {"tokenizer_class": "GemmaTokenizerFast", "tokenizer_vocab_size": 256000}
    assert fingerprint_diff(fp, fp) == []


def test_fingerprint_diff_different():
    """fingerprint_diff returns differences for changed values."""
    from univi.fingerprint import fingerprint_diff

    fp1 = {"tokenizer_class": "GemmaTokenizerFast", "tokenizer_vocab_size": 256000}
    fp2 = {"tokenizer_class": "LlamaTokenizerFast", "tokenizer_vocab_size": 128000}
    diffs = fingerprint_diff(fp1, fp2)
    assert len(diffs) >= 2
    assert any("tokenizer_class" in d for d in diffs)
    assert any("tokenizer_vocab_size" in d for d in diffs)


def test_fingerprint_diff_missing_key():
    """fingerprint_diff reports missing keys."""
    from univi.fingerprint import fingerprint_diff

    diffs = fingerprint_diff({"a": 1}, {"b": 2})
    assert any("only" in d and "a" in d for d in diffs)
    assert any("only" in d and "b" in d for d in diffs)


def test_pre_post_fingerprint_match():
    """Pre- and post-model-load fingerprints must match."""
    from univi.fingerprint import assert_fingerprints_match

    fp1 = {"tokenizer_class": "GemmaTokenizerFast"}
    fp2 = {"tokenizer_class": "GemmaTokenizerFast"}
    assert_fingerprints_match(fp1, fp2)  # no error


def test_pre_post_fingerprint_mismatch_raises():
    """Mismatched fingerprints raise FingerprintMismatchError."""
    from univi.fingerprint import (
        assert_fingerprints_match,
        FingerprintMismatchError,
    )

    fp1 = {"tokenizer_class": "GemmaTokenizerFast"}
    fp2 = {"tokenizer_class": "LlamaTokenizerFast"}
    with pytest.raises(FingerprintMismatchError):
        assert_fingerprints_match(fp1, fp2)


def test_compute_fingerprint_mock():
    """compute_fingerprint works with a mock processor/tokenizer-like dict."""
    from univi.fingerprint import compute_fingerprint

    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 256000
            self.chat_template = "<|turn|>"

        def __len__(self):
            return self.vocab_size

        @property
        def special_tokens_map(self):
            return {"pad_token": "<pad>"}

    class MockProcessor:
        def __init__(self):
            self.tokenizer = MockTokenizer()

        @property
        def image_processor(self):
            return type("obj", (object,), {"to_dict": lambda self: {"size": 224}})()

    fp = compute_fingerprint(MockProcessor(), MockProcessor().tokenizer)
    assert "tokenizer_vocab_size" in fp
    assert fp["tokenizer_vocab_size"] == 256000
    assert "tokenizer_class" in fp
    assert "chat_template_hash" in fp
    assert "processor_class" in fp
