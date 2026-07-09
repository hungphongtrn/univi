from __future__ import annotations

import json


def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[m]


def _lcs_length(a: list, b: list) -> int:
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[m]


def strict_first_letter_parse(text: str) -> str | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    first = stripped[0]
    if first in ("A", "B", "C", "D"):
        return first
    return None


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def compute_normalized_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    return _levenshtein(ref_chars, hyp_chars) / len(reference)


def compute_rougel(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words and not hyp_words:
        return 1.0
    if not ref_words or not hyp_words:
        return 0.0
    lcs_len = _lcs_length(ref_words, hyp_words)
    precision = lcs_len / len(hyp_words) if hyp_words else 0.0
    recall = lcs_len / len(ref_words) if ref_words else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(reference: str, hypothesis: str) -> float:
    return 1.0 if reference == hypothesis else 0.0


def compute_retention(image_score, native_score, higher_is_better=True) -> float | None:
    if image_score is None or native_score is None:
        return None
    if higher_is_better:
        if native_score == 0:
            return None
        return image_score / native_score * 100
    else:
        if image_score == 0:
            return None
        return native_score / image_score * 100


def write_eval_output(results, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
