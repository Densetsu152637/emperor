import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

BRIEF = (
    "## Goal\nAdd pagination to the API.\n\n## Context\nRepo at ~/code/x.\n\n"
    "## Constraints\nKeep the public API.\n\n## Acceptance\n- tests pass\n\n"
    "## Output format\nChanged files and test output."
)


def call(call_id: str, name: str, arguments: dict | str) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


def make_sample(sample_id: str = "p-1") -> dict:
    return {
        "id": sample_id,
        "task_class": "coding",
        "specificity": "vague",
        "context_budget_k": 64,
        "tools": ["spawn_agent", "await_tasks", "update_plan", "web_search"],
        "messages": [
            {"role": "system", "content": "You are Emperor."},
            {"role": "user", "content": "Add pagination to my API."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("call_1", "spawn_agent", {"brief": BRIEF, "tier": "standard"})],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"task_id": "t1"}'},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("call_2", "await_tasks", {})],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": '{"reports": []}'},
            {"role": "assistant", "content": "Pagination is added; tests pass."},
        ],
        "meta": {"teacher": "gpt-oss-20b", "licence": "Apache-2.0", "seed": 7},
    }


@pytest.fixture
def sample():
    return make_sample()
