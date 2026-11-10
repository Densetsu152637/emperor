"""Training-sample shape and validation (roles, tool calls, briefs, secrets)."""

import re
from dataclasses import dataclass, field

from emperor.eval.contract import ALL_TOOLS, check_call, estimate_tokens
from emperor.eval.metrics import LEAK_PATTERNS

LICENCE = "Apache-2.0"
BRIEF_SECTIONS = ("## Goal", "## Acceptance")
MAX_BRIEF_TOKENS = 2000
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff\U0001f900-\U0001f9ff]"
)


def count_emojis(text: str) -> int:
    return len(_EMOJI.findall(text))


@dataclass
class Check:
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


def secret_hits(text: str) -> list[str]:
    return [kind for kind, pattern in LEAK_PATTERNS.items() if pattern.search(text)]


def validate_sample(sample: dict) -> Check:
    """Check one sample; every distinct problem is listed in `reasons` (empty means valid)."""
    check = Check()
    messages = sample.get("messages")
    tools = sample.get("tools")
    if not isinstance(messages, list) or len(messages) < 3:
        check.add("too_few_messages")
        return check
    if not isinstance(tools, list) or not tools or any(t not in ALL_TOOLS for t in tools):
        check.add("bad_tool_list")
        return check
    if not all(isinstance(m, dict) for m in messages):
        check.add("message_not_an_object")
        return check

    if messages[0].get("role") != "system":
        check.add("no_system_first")
    if messages[1].get("role") != "user":
        check.add("no_user_second")

    seen_ids: set[str] = set()
    i = 2
    while i < len(messages):
        message = messages[i]
        if message.get("role") != "assistant":
            check.add("roles_do_not_alternate")
            break
        calls = message.get("tool_calls") or []
        content = message.get("content") or ""
        if not calls and not content.strip():
            check.add("empty_assistant_turn")
        if secret_hits(content):
            check.add("secret_in_assistant_text")
        for call in calls:
            _check_call(call, tools, seen_ids, check)
        j = i + 1
        for call in calls:
            if (
                j >= len(messages)
                or messages[j].get("role") != "tool"
                or messages[j].get("tool_call_id") != call.get("id")
            ):
                check.add("tool_results_do_not_match_calls")
                return check
            if not isinstance(messages[j].get("content"), str):
                check.add("tool_result_not_text")
            j += 1
        if j < len(messages) and messages[j].get("role") == "tool":
            check.add("tool_result_without_call")
            return check
        i = j

    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls"):
        check.add("must_end_with_final_answer")
    return check


def _leaf_strings(value):
    """Every string inside parsed arguments (so JSON escapes cannot hide a secret)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaf_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_strings(item)


def _check_call(call: dict, offered: list[str], seen_ids: set[str], check: Check) -> None:
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
        check.add("bad_or_duplicate_call_id")
    else:
        seen_ids.add(call_id)
    function = call.get("function") or {}
    name = function.get("name", "")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        check.add("arguments_not_a_json_string")
        return
    arguments, errors = check_call(name, raw, offered)
    if errors:
        check.add("tool_call_invalid")
        return
    if any(secret_hits(text) for text in _leaf_strings(arguments)):
        check.add("secret_in_tool_arguments")
    if name in ("spawn_agent", "delegate_chunks") and arguments:
        brief = arguments.get("brief", "")
        if estimate_tokens(brief) > MAX_BRIEF_TOKENS:
            check.add("brief_over_limit")
        if name == "spawn_agent" and any(s not in brief for s in BRIEF_SECTIONS):
            check.add("brief_missing_sections")
