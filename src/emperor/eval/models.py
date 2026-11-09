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
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
    through verbatim so tests can script malformed calls. Once the script runs out the model
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
        content = turn.get("content")
        return Reply(
            content=content,
            tool_calls=calls,
            prompt_tokens=_estimate_messages(messages),
            completion_tokens=estimate_tokens(content or "")
            + sum(estimate_tokens(c.arguments) for c in calls),
        )


class OpenAICompatModel:
    """A chat-completions endpoint such as a llama.cpp or vLLM server.

    The API key, if any, comes from `EMPEROR_EVAL_API_KEY` and is only ever sent as a header.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 600.0,
        max_tokens: int = 4096,
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
        return Reply(
            content=message.get("content"),
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", _estimate_messages(messages)),
            completion_tokens=usage.get("completion_tokens", 0),
        )
