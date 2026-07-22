"""
Test tokenizer/processor can handle vision input formats correctly.

Tests both the placeholder format (images in separate ``images`` column)
and the inline format (images embedded in message content as
``{"type": "image", "image": <PIL.Image>}``), as well as multi-image
scenarios.
"""

import pytest
from PIL import Image

_IMG1 = Image.new("RGB", (64, 64), color=(255, 0, 0))
_IMG2 = Image.new("RGB", (64, 64), color=(0, 255, 0))
_IMG3 = Image.new("RGB", (64, 64), color=(0, 0, 255))

_MESSAGES_PLACEHOLDER_SINGLE = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A red square."}],
    },
]

_MESSAGES_INLINE_SINGLE = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": _IMG1},
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A red square."}],
    },
]

_MESSAGES_PLACEHOLDER_MULTI = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "Answer the question shown."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A"}],
    },
]

_MESSAGES_INLINE_MULTI = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": _IMG1},
            {"type": "image", "image": _IMG2},
            {"type": "image", "image": _IMG3},
            {"type": "text", "text": "Answer the question shown."},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "A"}],
    },
]

_IMAGE_TOKEN = "<|image|>"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-4-E2B-it", trust_remote_code=True
        )
    return _tokenizer


_processor = None


def _get_processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor
        _processor = AutoProcessor.from_pretrained(
            "google/gemma-4-E2B-it", trust_remote_code=True
        )
    return _processor


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return _get_tokenizer()
    except Exception:
        pytest.skip("cannot load tokenizer")


@pytest.fixture(scope="module")
def processor():
    try:
        return _get_processor()
    except Exception:
        pytest.skip("cannot load processor")


# ---------------------------------------------------------------------------
# Chat template tests (tokenizer-only, CPU-safe)
# ---------------------------------------------------------------------------


def _apply_template(messages):
    tok = _get_tokenizer()
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def test_chat_template_placeholder_single():
    prompt = _apply_template(_MESSAGES_PLACEHOLDER_SINGLE)
    assert _IMAGE_TOKEN in prompt
    assert prompt.count(_IMAGE_TOKEN) == 1
    assert "Describe this image." in prompt
    assert "A red square." in prompt


def test_chat_template_inline_single():
    prompt = _apply_template(_MESSAGES_INLINE_SINGLE)
    assert _IMAGE_TOKEN in prompt
    assert prompt.count(_IMAGE_TOKEN) == 1
    assert "Describe this image." in prompt
    assert "A red square." in prompt


def test_chat_template_placeholder_multi():
    prompt = _apply_template(_MESSAGES_PLACEHOLDER_MULTI)
    assert prompt.count(_IMAGE_TOKEN) == 3


def test_chat_template_inline_multi():
    prompt = _apply_template(_MESSAGES_INLINE_MULTI)
    assert prompt.count(_IMAGE_TOKEN) == 3


def test_chat_template_placeholder_and_inline_produce_identical_prompt():
    p1 = _apply_template(_MESSAGES_PLACEHOLDER_SINGLE)
    p2 = _apply_template(_MESSAGES_INLINE_SINGLE)
    assert p1 == p2


def test_chat_template_roles_present():
    prompt = _apply_template(_MESSAGES_PLACEHOLDER_SINGLE)
    assert "user" in prompt
    assert "model" in prompt


def test_chat_template_has_correct_image_token_count_per_placeholder():
    prompt = _apply_template(_MESSAGES_PLACEHOLDER_MULTI)
    assert prompt.count(_IMAGE_TOKEN) == 3


# ---------------------------------------------------------------------------
# Full tokenization tests (processor required)
# ---------------------------------------------------------------------------


def _tokenize(messages, images, proc):
    prompt = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return proc(images=images, text=prompt, return_tensors="pt", padding=True)


def test_tokenize_placeholder_single_image(processor):
    inputs = _tokenize(_MESSAGES_PLACEHOLDER_SINGLE, [_IMG1], processor)
    assert inputs["input_ids"].ndim == 2
    assert inputs["input_ids"].shape[0] == 1
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].shape[0] == 1


def test_tokenize_inline_single_image(processor):
    inputs = _tokenize(_MESSAGES_INLINE_SINGLE, [_IMG1], processor)
    assert inputs["input_ids"].shape[0] == 1
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].shape[0] == 1


def test_tokenize_multi_image(processor):
    inputs = _tokenize(
        _MESSAGES_PLACEHOLDER_MULTI, [_IMG1, _IMG2, _IMG3], processor
    )
    assert inputs["input_ids"].shape[0] == 1
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].shape[0] == 3


def test_tokenize_single_vs_multi_sequence_length(processor):
    single = _tokenize(_MESSAGES_PLACEHOLDER_SINGLE, [_IMG1], processor)
    multi = _tokenize(
        _MESSAGES_PLACEHOLDER_MULTI, [_IMG1, _IMG2, _IMG3], processor
    )
    assert multi["input_ids"].shape[1] > single["input_ids"].shape[1]
