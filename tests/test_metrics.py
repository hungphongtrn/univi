from __future__ import annotations

import json
import math
import tempfile

import pytest

from univi.metrics import (
    compute_exact_match,
    compute_normalized_cer,
    compute_retention,
    compute_rougel,
    compute_wer,
    strict_first_letter_parse,
    write_eval_output,
)


class TestStrictFirstLetterParse:
    def test_uppercase_abcd(self):
        for letter in ("A", "B", "C", "D"):
            assert strict_first_letter_parse(letter) == letter

    def test_lowercase_not_accepted(self):
        for letter in ("a", "b", "c", "d"):
            assert strict_first_letter_parse(letter) is None

    def test_leading_whitespace(self):
        assert strict_first_letter_parse("  B") == "B"

    def test_empty_and_whitespace(self):
        assert strict_first_letter_parse("") is None
        assert strict_first_letter_parse("   ") is None
        assert strict_first_letter_parse("\t\n") is None

    def test_extra_chars_after_letter(self):
        assert strict_first_letter_parse("C something else") == "C"

    def test_non_abcd_first_char(self):
        for s in ("E", "X", "1", "?", "."):
            assert strict_first_letter_parse(s) is None

    def test_trailing_newline(self):
        assert strict_first_letter_parse("A\n") == "A"


class TestComputeWer:
    def test_perfect_match(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_empty_both(self):
        assert compute_wer("", "") == 0.0

    def test_empty_reference(self):
        assert compute_wer("", "hello") == 0.0

    def test_empty_hypothesis(self):
        assert compute_wer("hello world", "") == 1.0

    def test_substitution(self):
        assert compute_wer("hello world", "hello there") == 0.5

    def test_insertion(self):
        assert compute_wer("hello world", "hello beautiful world") == 0.5

    def test_deletion(self):
        assert compute_wer("hello world", "hello") == 0.5

    def test_complete_mismatch(self):
        assert compute_wer("hello world", "foo bar") == 1.0


class TestComputeNormalizedCer:
    def test_perfect_match(self):
        assert compute_normalized_cer("hello", "hello") == 0.0

    def test_empty_both(self):
        assert compute_normalized_cer("", "") == 0.0

    def test_empty_reference(self):
        assert compute_normalized_cer("", "abc") == 0.0

    def test_empty_hypothesis(self):
        assert compute_normalized_cer("abc", "") == 1.0

    def test_single_substitution(self):
        result = compute_normalized_cer("hello", "hollo")
        assert math.isclose(result, 0.2, rel_tol=1e-9)

    def test_insertion(self):
        result = compute_normalized_cer("hi", "hi!")
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    def test_deletion(self):
        result = compute_normalized_cer("hey!", "hey")
        assert math.isclose(result, 0.25, rel_tol=1e-9)

    def test_case_sensitive(self):
        assert compute_normalized_cer("Hello", "hello") > 0.0


class TestComputeRougel:
    def test_perfect_match(self):
        assert math.isclose(compute_rougel("hello world", "hello world"), 1.0)

    def test_empty_both(self):
        assert math.isclose(compute_rougel("", ""), 1.0)

    def test_empty_reference(self):
        assert math.isclose(compute_rougel("", "hello"), 0.0)

    def test_empty_hypothesis(self):
        assert math.isclose(compute_rougel("hello", ""), 0.0)

    def test_partial_overlap(self):
        result = compute_rougel("the cat sat on the mat", "the cat sat")
        assert 0.0 < result < 1.0

    def test_no_overlap(self):
        assert math.isclose(compute_rougel("hello world", "foo bar"), 0.0)

    def test_symmetric_single_word(self):
        r = compute_rougel("hello", "hello")
        assert math.isclose(r, 1.0)

    def test_out_of_order_penalized(self):
        ref = "the cat sat"
        hyp = "cat the sat"
        result = compute_rougel(ref, hyp)
        assert math.isclose(result, 2.0 / 3.0)


class TestComputeExactMatch:
    def test_exact_match(self):
        assert compute_exact_match("hello", "hello") == 1.0

    def test_case_differs(self):
        assert compute_exact_match("Hello", "hello") == 0.0

    def test_whitespace_differs(self):
        assert compute_exact_match("hello world", "hello  world") == 0.0

    def test_empty_both(self):
        assert compute_exact_match("", "") == 1.0

    def test_one_empty(self):
        assert compute_exact_match("", "hello") == 0.0
        assert compute_exact_match("hello", "") == 0.0

    def test_different_strings(self):
        assert compute_exact_match("abc", "xyz") == 0.0


class TestComputeRetention:
    def test_higher_is_better_equal(self):
        assert math.isclose(compute_retention(0.8, 0.8), 100.0)

    def test_higher_is_better_degradation(self):
        assert math.isclose(compute_retention(0.6, 0.8), 75.0)

    def test_higher_is_better_improvement(self):
        assert math.isclose(compute_retention(0.9, 0.8), 112.5)

    def test_lower_is_better(self):
        assert math.isclose(compute_retention(0.3, 0.1, higher_is_better=False), 100.0 / 3.0)

    def test_lower_is_better_zero_image_score(self):
        assert compute_retention(0.0, 0.1, higher_is_better=False) is None

    def test_none_image_score(self):
        assert compute_retention(None, 0.8) is None

    def test_none_native_score(self):
        assert compute_retention(0.8, None) is None

    def test_zero_native_score_returns_none(self):
        assert compute_retention(0.8, 0.0) is None

    def test_both_none(self):
        assert compute_retention(None, None) is None


class TestWriteEvalOutput:
    def test_writes_json(self):
        results = [{"id": 1, "score": 0.95}, {"id": 2, "score": 0.87}]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as f:
            write_eval_output(results, f.name)
            data = json.load(f)
        assert data == results

    def test_empty_list(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as f:
            write_eval_output([], f.name)
            data = json.load(f)
        assert data == []

    def test_round_trip_types(self):
        results = [
            {
                "wer": 0.3,
                "em": 0.0,
                "text": "hello",
                "count": 42,
                "flag": True,
            }
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as f:
            write_eval_output(results, f.name)
            loaded = json.load(f)
        assert loaded == results
