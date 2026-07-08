from __future__ import annotations

import json
import tempfile

import pytest
from datasets import Dataset, load_from_disk
from PIL import Image

from data.preprocessing.smoltalk import (
    _extract_instruction_and_response,
    preprocess_smoltalk,
)


class _MockSmolTalk:
    def __init__(self, num_rows=2, rows=None):
        if rows is not None:
            self._rows = rows
        else:
            self._rows = [
                {
                    "messages": [
                        {"role": "user", "content": f"Write a poem about {i}."},
                        {
                            "role": "assistant",
                            "content": f"Here is a poem about {i}.",
                        },
                    ],
                    "id": f"test-{i}",
                }
                for i in range(num_rows)
            ]

    def __len__(self):
        return len(self._rows)

    @property
    def column_names(self):
        return list(self._rows[0]) if self._rows else []

    def select(self, indices):
        ds = _MockSmolTalk.__new__(_MockSmolTalk)
        ds._rows = [self._rows[i] for i in indices]
        return ds

    def map(self, function, *, with_indices=False, remove_columns=None):
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            result_rows.append(new_row)
        return Dataset.from_list(result_rows)


def _mock_renderer(*args, **kwargs):
    return Image.new("RGB", (64, 64))


# --- _extract_instruction_and_response unit tests ---


def test_extract_messages_schema():
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "Hello"
    assert response == "Hi there"


def test_extract_messages_skips_system_and_finds_first_user():
    row = {
        "messages": [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Actual question"},
            {"role": "assistant", "content": "Actual answer"},
        ]
    }
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "Actual question"
    assert response == "Actual answer"


def test_extract_messages_multi_turn_uses_first_pair():
    row = {
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
    }
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "First question"
    assert response == "First answer"


def test_extract_messages_no_user_raises():
    row = {"messages": [{"role": "assistant", "content": "Hi"}]}
    with pytest.raises(KeyError, match="No user message"):
        _extract_instruction_and_response(row)


def test_extract_messages_no_assistant_raises():
    row = {"messages": [{"role": "user", "content": "Hi"}]}
    with pytest.raises(KeyError, match="No assistant message"):
        _extract_instruction_and_response(row)


def test_extract_conversations_role_value():
    row = {
        "conversations": [
            {"role": "user", "value": "What is AI?"},
            {"role": "assistant", "value": "Artificial intelligence is..."},
        ]
    }
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "What is AI?"
    assert response == "Artificial intelligence is..."


def test_extract_conversations_from_value():
    row = {
        "conversations": [
            {"from": "human", "value": "Tell me a joke"},
            {"from": "gpt", "value": "Why did the chicken..."},
        ]
    }
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "Tell me a joke"
    assert "chicken" in response


def test_extract_conversations_no_user_raises():
    row = {"conversations": [{"role": "assistant", "value": "Hi"}]}
    with pytest.raises(KeyError, match="No human/user turn"):
        _extract_instruction_and_response(row)


def test_extract_conversations_no_assistant_raises():
    row = {"conversations": [{"from": "human", "value": "Hi"}]}
    with pytest.raises(KeyError, match="No assistant/gpt turn"):
        _extract_instruction_and_response(row)


def test_extract_direct_columns_instruction_response():
    row = {"instruction": "Do this", "response": "Done"}
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "Do this"
    assert response == "Done"


def test_extract_direct_columns_prompt_answer():
    row = {"prompt": "Translate", "answer": "Translation"}
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "Translate"
    assert response == "Translation"


def test_extract_direct_columns_question_output():
    row = {"question": "What?", "output": "That"}
    instruction, response = _extract_instruction_and_response(row)
    assert instruction == "What?"
    assert response == "That"


def test_extract_direct_columns_missing_both_raises():
    row = {"irrelevant": "data"}
    with pytest.raises(KeyError, match="Cannot find instruction or response"):
        _extract_instruction_and_response(row)


def test_extract_direct_columns_missing_instruction_raises():
    row = {"response": "answer", "irrelevant": "data"}
    with pytest.raises(KeyError, match="Cannot find instruction"):
        _extract_instruction_and_response(row)


def test_extract_direct_columns_missing_response_raises():
    row = {"instruction": "do it", "irrelevant": "data"}
    with pytest.raises(KeyError, match="Cannot find response"):
        _extract_instruction_and_response(row)


# --- Functional tests ---


def test_smoltalk_preprocess_generates_messages(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=2)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert "follow" in user_content[1]["text"].lower()
    assert row["messages"][1]["role"] == "assistant"


def test_smoltalk_metadata_present(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "HuggingFaceTB/smoltalk"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row


def test_no_native_instruction_or_response_leakage(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=2)
    for row in dataset:
        user_text = row["messages"][0]["content"][1]["text"]
        assistant_text = row["messages"][1]["content"][0]["text"]
        assert user_text == "Follow the instruction shown in the image."
        assert assistant_text not in user_text


def test_render_config_is_valid_json_and_contains_expected_keys(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(
        max_samples=1, canvas_width=1024, font_size=14
    )
    config = json.loads(dataset[0]["render_config"])
    assert isinstance(config, dict)
    assert config["render_method"] == "text_page"
    assert config["config"] == "all"
    assert config["max_chars"] == 2000
    assert config["canvas_width"] == 1024
    assert config["font_size"] == 14


def test_render_config_matches_renderer_kwargs(monkeypatch):
    recorded_kwargs: dict = {}

    def recording_renderer(*args, **kwargs):
        recorded_kwargs.clear()
        recorded_kwargs.update(kwargs)
        return Image.new("RGB", (64, 64))

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(1),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        recording_renderer,
    )

    dataset = preprocess_smoltalk(
        max_samples=1, canvas_width=800, font_size=16
    )

    row = dataset[0]
    render_config = json.loads(row["render_config"])
    assert render_config["canvas_width"] == recorded_kwargs.get("canvas_width")
    assert render_config["font_size"] == recorded_kwargs.get("font_size")
    assert render_config["config"] == "all"
    assert render_config["max_chars"] == 2000


def test_row_id_is_string_with_numeric_source_id(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Done"},
            ],
            "id": 999,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "999"


def test_row_id_falls_back_to_index_when_id_is_none(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Done"},
            ],
            "id": None,
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_row_id_falls_back_to_index_when_id_is_empty(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Done"},
            ],
            "id": "",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assert isinstance(dataset[0]["row_id"], str)
    assert dataset[0]["row_id"] == "0"


def test_row_id_falls_back_to_index_when_no_id_column(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Done"},
            ],
        }
    ]

    class _NoIdMock:
        def __init__(self):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0]) if self._rows else []

        def select(self, indices):
            ds = _NoIdMock.__new__(_NoIdMock)
            ds._rows = [self._rows[i] for i in indices]
            return ds

        def map(self, function, *, with_indices=False, remove_columns=None):
            result_rows = []
            for i, row in enumerate(self._rows):
                if with_indices:
                    new_row = function(row, i)
                else:
                    new_row = function(row)
                result_rows.append(new_row)
            return Dataset.from_list(result_rows)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _NoIdMock(),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assert dataset[0]["row_id"] == "0"


def test_save_load_round_trip(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset.save_to_disk(tmpdir)
        loaded = load_from_disk(tmpdir)

    assert len(loaded) == 2
    row = loaded[0]

    assert "messages" in row
    assert len(row["messages"]) == 2
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"

    assert row["source_dataset_id"] == "HuggingFaceTB/smoltalk"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row


def test_map_used_not_direct_iteration(monkeypatch):
    map_kwargs: dict = {}

    def recording_map(self, function, *, with_indices=False, remove_columns=None):
        map_kwargs["with_indices"] = with_indices
        map_kwargs["remove_columns"] = remove_columns
        result_rows = []
        for i, row in enumerate(self._rows):
            if with_indices:
                new_row = function(row, i)
            else:
                new_row = function(row)
            result_rows.append(new_row)
        return Dataset.from_list(result_rows)

    monkeypatch.setattr(_MockSmolTalk, "map", recording_map)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(2),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=2)
    assert len(dataset) == 2
    assert map_kwargs.get("with_indices") is True
    assert map_kwargs.get("remove_columns") == ["messages", "id"]


def test_load_dataset_called_with_config_and_split(monkeypatch):
    captured: list = [(), {}]

    def recording_load(*args, **kwargs):
        captured[0] = args
        captured[1] = kwargs
        return _MockSmolTalk(1)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        recording_load,
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    preprocess_smoltalk(config="all", split="train", max_samples=1)
    assert len(captured[0]) >= 2
    assert captured[0][0] == "HuggingFaceTB/smoltalk"
    assert captured[0][1] == "all"
    assert captured[1].get("split") == "train"


def test_default_config_is_all(monkeypatch):
    captured: list = [()]

    def recording_load(*args, **kwargs):
        captured[0] = args
        return _MockSmolTalk(1)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        recording_load,
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    preprocess_smoltalk(max_samples=1)
    assert captured[0][1] == "all"


def test_custom_config_propagates(monkeypatch):
    captured: list = [()]

    def recording_load(*args, **kwargs):
        captured[0] = args
        return _MockSmolTalk(1)

    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        recording_load,
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    preprocess_smoltalk(config="everyday-conversations", max_samples=1)
    assert captured[0][1] == "everyday-conversations"


def test_instruction_truncated_before_rendering(monkeypatch):
    captured: list = [""]

    def recording_renderer(text, **kwargs):
        captured[0] = text
        return Image.new("RGB", (64, 64))

    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "A" * 5000,
                },
                {
                    "role": "assistant",
                    "content": "short response",
                },
            ],
            "id": "0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        recording_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1, max_chars=100)
    assert len(captured[0]) == 100
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert assistant_text == "short response"


def test_messages_schema_variant(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "User instruction here"},
                {"role": "assistant", "content": "Assistant response here"},
            ],
            "id": "msg-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert "Assistant response" in assistant_text


def test_conversations_schema_with_role_value(monkeypatch):
    rows = [
        {
            "conversations": [
                {"role": "user", "value": "Conversation question"},
                {"role": "assistant", "value": "Conversation answer"},
            ],
            "id": "conv-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert assistant_text == "Conversation answer"


def test_conversations_schema_with_from_value(monkeypatch):
    rows = [
        {
            "conversations": [
                {"from": "human", "value": "Human question"},
                {"from": "gpt", "value": "GPT answer"},
            ],
            "id": "conv-1",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert "GPT answer" in assistant_text


def test_direct_instruction_response_columns(monkeypatch):
    rows = [
        {
            "instruction": "Direct instruction",
            "response": "Direct response",
            "id": "dir-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert assistant_text == "Direct response"


def test_direct_prompt_answer_columns(monkeypatch):
    rows = [
        {
            "prompt": "Prompt text",
            "answer": "Answer text",
            "id": "dir-1",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert assistant_text == "Answer text"


def test_direct_question_output_columns(monkeypatch):
    rows = [
        {
            "question": "What is the capital?",
            "output": "Paris",
            "id": "dir-2",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    dataset = preprocess_smoltalk(max_samples=1)
    assistant_text = dataset[0]["messages"][1]["content"][0]["text"]
    assert assistant_text == "Paris"


def test_no_recognized_schema_raises_key_error(monkeypatch):
    rows = [
        {
            "unknown": "data",
            "id": "err-0",
        }
    ]
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.load_dataset",
        lambda *a, **kw: _MockSmolTalk(rows=rows),
    )
    monkeypatch.setattr(
        "data.preprocessing.smoltalk.render_text_page",
        _mock_renderer,
    )

    with pytest.raises(KeyError, match="Cannot find instruction or response"):
        preprocess_smoltalk(max_samples=1)
