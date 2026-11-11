"""Models under test: a scripted fake for dry runs, and any OpenAI-compatible chat endpoint."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from emperor.eval.contract import estimate_tokens


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, exactly as the model produced it


@dataclass
class Reply:
    content: str | None  # the answer, with any thinking block removed
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str | None = None  # the model's thinking, kept apart from the answer


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_thinking(content: str | None) -> tuple[str | None, str | None]:
    """(reasoning, answer) from a reply that may still hold its thinking block.

    A thinking model's generation prompt already opens `<think>`, so the text may hold only the
    closing tag. Text that opens a block and never closes it (the model ran out of tokens) is
    all reasoning and no answer.
    """
    if content is None:
        return None, None
    if THINK_CLOSE in content:
        reasoning, answer = content.split(THINK_CLOSE, 1)
        reasoning = reasoning.split(THINK_OPEN, 1)[-1]
        return reasoning.strip() or None, answer.lstrip("\n")
    stripped = content.lstrip()
    if stripped.startswith(THINK_OPEN):
        return stripped[len(THINK_OPEN) :].strip() or None, ""
    return None, content


class ModelError(RuntimeError):
    """The model could not be reached or answered with something unusable."""


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> Reply: ...


def _estimate_messages(messages: list[dict]) -> int:
    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def _raw_arguments(arguments: object) -> str:
    if arguments is None:
        return ""
    return arguments if isinstance(arguments, str) else json.dumps(arguments)


class ScriptedModel:
    """Plays back fixed assistant turns, ignoring what the tools return.

    Each turn is `{"content": str}` and/or
    `{"tool_calls": [{"name": str, "arguments": dict | str}]}`; string arguments are passed
    through verbatim so tests can script malformed calls. A `<think>` block in the content is
    split off as the reply's reasoning, as for a served model. Once the script runs out the model
    answers with an empty message, which ends the episode.
    """

    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        turn = self.turns[self.calls] if self.calls < len(self.turns) else {"content": ""}
        self.calls += 1
        calls = []
        for i, call in enumerate(turn.get("tool_calls", [])):
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                ToolCall(id=f"call_{self.calls}_{i}", name=call["name"], arguments=arguments)
            )
        reasoning, content = split_thinking(turn.get("content"))
        return Reply(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            prompt_tokens=_estimate_messages(messages),
            completion_tokens=estimate_tokens(content or "")
            + sum(estimate_tokens(c.arguments) for c in calls),
        )


# A thinking model writes its reasoning before the answer or tool call; 4096 tokens cut some
# replies off mid-thought.
DEFAULT_MAX_TOKENS = 16384


class OpenAICompatModel:
    """A chat-completions endpoint such as a llama.cpp or vLLM server.

    The server normally returns a thinking model's reasoning in `reasoning_content`; when it
    leaves the thinking block in `content` instead, it is split off here, so tool calls, routing
    and the final answer are read from the answer alone.

    The API key, if any, comes from `EMPEROR_EVAL_API_KEY` and is only ever sent as a header.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 600.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.api_key = api_key if api_key is not None else os.environ.get("EMPEROR_EVAL_API_KEY")

    def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise ModelError(f"{self.url} answered HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ModelError(f"cannot reach {self.url}: {error}") from error
        except json.JSONDecodeError as error:
            raise ModelError(f"{self.url} returned non-JSON") from error
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelError(f"{self.url} returned no choices") from error
        calls = [
            ToolCall(
                id=call.get("id") or f"call_{i}",
                name=call.get("function", {}).get("name", ""),
                arguments=_raw_arguments(call.get("function", {}).get("arguments")),
            )
            for i, call in enumerate(message.get("tool_calls") or [])
        ]
        usage = payload.get("usage") or {}
        reasoning, content = split_thinking(message.get("content"))
        served = message.get("reasoning_content")
        if isinstance(served, str) and served.strip():
            reasoning = served.strip() if reasoning is None else f"{served.strip()}\n{reasoning}"
        return Reply(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", _estimate_messages(messages)),
            completion_tokens=usage.get("completion_tokens", 0),
        )
