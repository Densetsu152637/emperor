"""Turning chat samples into token ids with assistant-only labels (no torch needed here).

The chat template renders each conversation once. Assistant turns are found by rendering the
prefix up to each assistant message with and without the generation prompt, which needs no
`{% generation %}` markers in the template; token labels are then taken from character offsets.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

IGNORE = -100


def load_records(path: Path | str) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_combined(paths: Sequence[Path | str]) -> list[dict]:
    """Samples from several files (one per data round), keeping the first of each sample id."""
    seen: set[str] = set()
    combined = []
    for path in paths:
        for record in load_records(path):
            if record["id"] not in seen:
                seen.add(record["id"])
                combined.append(record)
    return combined


def split_records(
    records: Sequence[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/validation split by sample id (stable when records are added)."""
    train, val = [], []
    for record in records:
        digest = hashlib.sha256(f"{seed}:{record['id']}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (val if bucket < val_fraction else train).append(record)
    return train, val


def chat_messages(sample: dict) -> list[dict]:
    """The sample's messages with tool-call arguments as objects, as chat templates expect."""
    messages = []
    for message in sample["messages"]:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"] or "{}")
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        messages.append(message)
    return messages


class TemplateError(ValueError):
    """The chat template does not render a conversation prefix-consistently."""


_PROBE = {"role": "user", "content": "<<end of conversation>>"}


def assistant_spans(
    messages: list[dict], render: Callable[..., str]
) -> tuple[str, list[tuple[int, int]]]:
    """Render the whole conversation and return it with the character span of each assistant turn.

    `render(messages, add_generation_prompt=bool)` is the tokenizer's chat template (tools bound).
    A span covers the assistant text through its end-of-turn token, not the newline after it.

    Qwen's templates add an empty `<think>` block to the assistant turn that ends the rendered
    conversation, which the served model never sees in history. A probe user message is therefore
    appended while rendering and cut off again, so every assistant turn is rendered as it will be
    at inference.
    """
    head = messages[:2]
    plain = render(head, add_generation_prompt=False)
    probed = render([*head, _PROBE], add_generation_prompt=False)
    if not probed.startswith(plain):
        raise TemplateError("cannot isolate the probe message from the template")
    suffix = probed[len(plain) :]

    def body(msgs: list[dict]) -> str:
        text = render([*msgs, _PROBE], add_generation_prompt=False)
        if not text.endswith(suffix):
            raise TemplateError("template changes how the probe message renders")
        return text[: -len(suffix)]

    full = body(messages)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = render(messages[:i], add_generation_prompt=True)
        through = body(messages[: i + 1]).rstrip("\n")
        if not through.startswith(before) or not full.startswith(through):
            raise TemplateError(f"template is not prefix-consistent at message {i}")
        spans.append((len(before), len(through)))
    return full, spans


def labels_from_offsets(
    offsets: Sequence[tuple[int, int]], input_ids: Sequence[int], spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Keep a token's id as its label when it lies inside an assistant span; mask the rest."""
    labels = []
    for (start, end), token in zip(offsets, input_ids, strict=True):
        inside = any(start >= a and end <= b and end > start for a, b in spans)
        labels.append(token if inside else IGNORE)
    return labels


def encode_sample(sample: dict, tokenizer, max_length: int) -> dict | None:
    """Tokenise one sample; None when it is longer than `max_length` (never truncated)."""
    tools = _tool_definitions(sample)
    messages = chat_messages(sample)

    def render(msgs, add_generation_prompt):
        return tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    text, spans = assistant_spans(messages, render)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    if len(ids) > max_length:
        return None
    labels = labels_from_offsets(encoded["offset_mapping"], ids, spans)
    if all(label == IGNORE for label in labels):
        return None
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def _tool_definitions(sample: dict) -> list[dict] | None:
    """The tool definitions the sample was written against; None for chat-only samples."""
    if not sample.get("tools"):
        return None
    from emperor.eval.contract import openai_tools

    return openai_tools(sample["tools"])


def pad_batch(batch: Sequence[dict], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch of encoded samples (lists, so the logic is testable without torch)."""
    longest = max(len(item["input_ids"]) for item in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        pad = longest - len(item["input_ids"])
        out["input_ids"].append(list(item["input_ids"]) + [pad_token_id] * pad)
        out["labels"].append(list(item["labels"]) + [IGNORE] * pad)
        out["attention_mask"].append(list(item["attention_mask"]) + [0] * pad)
    if all(all(row) for row in out["attention_mask"]):
        # Nothing is masked: positions instead of a mask let SDPA use its memory-efficient causal
        # kernel instead of the math path, whose L x L matrix overflows 16 GB on long samples.
        del out["attention_mask"]
        out["position_ids"] = [list(range(longest)) for _ in batch]
    return out
