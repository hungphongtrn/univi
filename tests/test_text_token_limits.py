from __future__ import annotations

from datasets import Dataset
from PIL import Image

from data.preprocessing.fineweb_edu import preprocess_fineweb_edu
from data.preprocessing.smoltalk import preprocess_smoltalk


class _CharacterTokenizer:
    name_or_path = "test-character-tokenizer"

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


def _two_page_renderer(text, **kwargs):
    assert len(text) == 200
    return [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]


def test_fineweb_keeps_full_visual_input_and_caps_target_tokens(monkeypatch):
    source = Dataset.from_list([{"text": "A" * 200, "id": "doc-1"}])
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.load_dataset", lambda *a, **kw: source
    )
    monkeypatch.setattr(
        "data.preprocessing.fineweb_edu.render_text_pages", _two_page_renderer
    )

    result = preprocess_fineweb_edu(
        max_samples=1,
        max_chars=None,
        max_output_tokens=100,
        tokenizer=_CharacterTokenizer(),
        canvas_height=1024,
        num_proc=1,
    )

    row = result[0]
    assert len(row["images"]) == 2
    assert [item["type"] for item in row["messages"][0]["content"]] == [
        "image",
        "image",
        "text",
    ]
    assert row["messages"][1]["content"][0]["text"] == "A" * 100


def test_smoltalk_keeps_full_instruction_and_caps_response_tokens(monkeypatch):
    source = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "I" * 200},
                    {"role": "assistant", "content": "R" * 200},
                ],
                "id": "talk-1",
            }
        ]
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset", lambda *a, **kw: source
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_pages", _two_page_renderer
    )

    result = preprocess_smoltalk(
        max_samples=1,
        max_chars=None,
        max_output_tokens=100,
        tokenizer=_CharacterTokenizer(),
        canvas_height=1024,
        num_proc=1,
    )

    row = result[0]
    assert len(row["images"]) == 2
    assert [item["type"] for item in row["messages"][0]["content"]] == [
        "image",
        "image",
        "text",
    ]
    assert row["messages"][1]["content"][0]["text"] == "R" * 100
