"""Turning chat samples into token ids with assistant-only labels (no torch needed here).

The chat template renders each conversation once. Assistant turns are found by rendering the
prefix up to each assistant message with and without the generation prompt, which needs no
`{% generation %}` markers in the template; token labels are then taken from character offsets.

A thinking base (Qwen3-*-Thinking) opens `<think>` in its generation prompt and renders an
assistant turn's `reasoning_content` as `<think>...</think>` before the answer, but only for the
turns after the last real user message (the current tool-call loop); earlier turns lose their
reasoning. With such a base only those turns are trained, reasoning included, and a sample is
skipped when any of them has no reasoning: it would teach the model to answer without thinking.
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


class MissingReasoning(ValueError):
    """A turn that a thinking base would render with reasoning has none."""


THINK_OPEN = "<think>"
TOO_LONG = "too_long"
NO_REASONING = "no_reasoning"
NO_ASSISTANT_TOKENS = "no_assistant_tokens"


def renders_thinking(render: Callable[..., str]) -> bool:
    """True when the template's generation prompt opens a reasoning block (a thinking base)."""
    prompt = render([{"role": "user", "content": "x"}], add_generation_prompt=True)
    return prompt.rstrip().endswith(THINK_OPEN)


def _is_query(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, str)
        and not (content.startswith("<tool_response>") and content.endswith("</tool_response>"))
    )


def last_query_index(messages: list[dict]) -> int:
    """Index of the last real user message, as Qwen3's template finds it (tool responses
    wrapped as user text do not count); the last index when there is none."""
    for index in range(len(messages) - 1, -1, -1):
        if _is_query(messages[index]):
            return index
    return len(messages) - 1


def has_reasoning(message: dict) -> bool:
    reasoning = message.get("reasoning_content")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def reasoning_spans(
    messages: list[dict], render: Callable[..., str]
) -> tuple[str, list[tuple[int, int]]]:
    """Like `assistant_spans`, for a thinking template: the rendered conversation and the span
    of every assistant turn after the last user query, from the first reasoning character
    (the generation prompt already holds `<think>`) through the end-of-turn token.

    Raises MissingReasoning when the first of those turns has no `reasoning_content`. Later
    turns may have none: the teacher often acts on a tool result without thinking again, and
    Qwen3.5's template renders such a turn with an empty think block, which is trained as is.
    """
    query = last_query_index(messages)
    trained = [
        i for i, message in enumerate(messages) if message["role"] == "assistant" and i > query
    ]
    if trained and not has_reasoning(messages[trained[0]]):
        raise MissingReasoning("the first assistant turn after the user message has no reasoning")
    full = render(messages, add_generation_prompt=False)
    spans = []
    for i in trained:
        before = render(messages[:i], add_generation_prompt=True)
        through = render(messages[: i + 1], add_generation_prompt=False).rstrip("\n")
        if not through.startswith(before) or not full.startswith(through):
            raise TemplateError(f"template is not prefix-consistent at message {i}")
        spans.append((len(before), len(through)))
    return full, spans


def encode_sample(
    sample: dict, tokenizer, max_length: int, *, thinking: bool | None = None
) -> tuple[dict | None, str | None]:
    """Tokenise one sample: (row, None), or (None, why it is skipped). Never truncated.

    `thinking` says whether the base is a thinking model; None asks the chat template.
    """
    tools = _tool_definitions(sample)
    messages = chat_messages(sample)

    def render(msgs, add_generation_prompt):
        return tokenizer.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    if thinking is None:
        thinking = renders_thinking(render)
    if thinking:
        try:
            text, spans = reasoning_spans(messages, render)
        except MissingReasoning:
            return None, NO_REASONING
    else:
        text, spans = assistant_spans(messages, render)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    if len(ids) > max_length:
        return None, TOO_LONG
    labels = labels_from_offsets(encoded["offset_mapping"], ids, spans)
    if all(label == IGNORE for label in labels):
        return None, NO_ASSISTANT_TOKENS
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}, None


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
