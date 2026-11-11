import json
from pathlib import Path

import pytest
from conftest import make_sample

from emperor.export.gguf import find_converter
from emperor.train.cli import main as train_main
from emperor.train.config import ConfigError, load_config, parse_config
from emperor.train.data import (
    IGNORE,
    assistant_spans,
    chat_messages,
    labels_from_offsets,
    pad_batch,
    split_records,
)

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_loads_and_is_bf16_lora():
    config = load_config(ROOT / "configs" / "sft.toml")
    assert config.dtype == "bfloat16"
    assert config.data.max_length == 6144
    assert config.model.base.endswith("Qwen3-4B-Instruct-2507")
    assert config.merge.output_dir == "weights/emperor-4b-bf16"
    assert config.train.save_steps > 0 and config.train.gradient_checkpointing


def test_defaults_parse_from_an_empty_file():
    assert parse_config({}).lora.r == 32


@pytest.mark.parametrize("key", ["load_in_4bit", "load_in_8bit", "qlora", "bnb_4bit_quant_type"])
def test_quantised_training_is_refused(key):
    with pytest.raises(ConfigError, match="not supported"):
        parse_config({"model": {key: True}})


def test_unknown_keys_and_bad_values_are_refused():
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"train": {"epochz": 1}})
    with pytest.raises(ConfigError, match="unknown section"):
        parse_config({"trainer": {}})
    with pytest.raises(ConfigError):
        parse_config({"data": {"max_length": 10}})
    with pytest.raises(ConfigError):
        parse_config({"data": {"val_fraction": 0.9}})


def test_config_file_errors_are_reported(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[model\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    assert train_main(["show", "--config", str(bad)]) == 2
    assert train_main(["show", "--config", str(ROOT / "configs" / "sft.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["lora"]["r"] == 32


def test_split_is_deterministic_and_stable_under_additions():
    records = [{"id": f"p-{i}"} for i in range(400)]
    train, val = split_records(records, 0.1, seed=1)
    assert len(train) + len(val) == 400 and 15 <= len(val) <= 70
    again = split_records(records, 0.1, seed=1)
    assert (train, val) == again
    more = split_records(records + [{"id": "p-new"}], 0.1, seed=1)
    assert {r["id"] for r in val} <= {r["id"] for r in more[1]}
    assert split_records(records, 0.0, seed=1)[1] == []


def test_tool_call_arguments_become_objects_for_the_chat_template():
    messages = chat_messages(make_sample())
    arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, dict) and arguments["tier"] == "standard"


def fake_render(messages, add_generation_prompt):
    text = ""
    for message in messages:
        text += f"<|{message['role']}|>{message.get('content') or ''}<|end|>\n"
    if add_generation_prompt:
        text += "<|assistant|>"
    return text


def test_assistant_spans_cover_only_assistant_turns():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "bye"},
    ]
    # Template whose generation prompt is "<|assistant|>" and whose turns open the same way.
    full, spans = assistant_spans(messages, fake_render)
    assert [full[a:b] for a, b in spans] == ["hello<|end|>", "bye<|end|>"]


def test_labels_mask_everything_outside_assistant_spans():
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    ids = [10, 11, 12, 13]
    assert labels_from_offsets(offsets, ids, [(6, 12)]) == [IGNORE, IGNORE, 12, 13]
    # A token that straddles the boundary is not trained on.
    assert labels_from_offsets(offsets, ids, [(7, 12)]) == [IGNORE, IGNORE, IGNORE, 13]


def test_padding_masks_labels_and_attention():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4], "labels": [4], "attention_mask": [1]},
    ]
    out = pad_batch(batch, pad_token_id=0)
    assert out["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert out["labels"] == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


def test_gguf_export_is_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("EMPEROR_LLAMA_CPP", raising=False)
    assert find_converter() is None
    (tmp_path / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    assert find_converter(tmp_path) == tmp_path / "convert_hf_to_gguf.py"
