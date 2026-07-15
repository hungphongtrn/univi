from __future__ import annotations


def truncate_to_token_limit(text: str, tokenizer, max_tokens: int | None) -> str:
    if max_tokens is None:
        return text
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def compute_token_length(text: str, tokenizer) -> int:
    """Return the number of tokens in *text* without any truncation."""
    return len(tokenizer.encode(text, add_special_tokens=False))
