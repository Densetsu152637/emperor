import json

import pytest
from conftest import BRIEF

from emperor.data.generate import Environment, GenerationConfig, generate_trace, run_generation
from emperor.data.prompts import PromptRecord
from emperor.data.shards import ShardWriter, done_ids, iter_records
from emperor.data.teacher import TeacherError, require_local


class ScriptedTeacher:
    """Plays fixed assistant messages and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.kwargs = []

    def chat(self, messages, *, tools=None, **kwargs):
        self.seen.append((list(messages), tools))
        self.kwargs.append(kwargs)
        if not self.turns:
            raise TeacherError("out of script")
        return self.turns.pop(0)


class FakeEnv:
    def __init__(self):
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append(messages[-1]["content"])
        return {"content": "Report: done. Evidence: all checks pass."}


def record(**over):
    base = {
        "id": "p-abc",
        "seed": 11,
        "task_class": "coding",
        "specificity": "vague",
        "text": "Add pagination to my API.",
        "tools": ["spawn_agent", "await_tasks", "update_plan", "web_search"],
        "context_budget_k": 64,
    }
    return PromptRecord(**{**base, **over})


def tool_turn(name, arguments):
    return {
        "content": "",
        "tool_calls": [{"id": "x", "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }


def good_script():
    return [
        tool_turn("spawn_agent", {"brief": BRIEF, "tier": "standard"}),
        tool_turn("await_tasks", {}),
        {"content": "Pagination is added; the subagent's tests pass."},
    ]


def test_trace_is_multi_turn_valid_and_records_provenance():
    teacher, env = ScriptedTeacher(good_script()), FakeEnv()
    sample, reason = generate_trace(record(), teacher, env, GenerationConfig())
    assert reason is None
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert sample["meta"]["teacher"] == "gpt-oss-20b"
    assert sample["meta"]["licence"] == "Apache-2.0"
    assert sample["meta"]["seed"] == 11
    # The environment role answered the await; the spawn result is deterministic.
    assert json.loads(sample["messages"][3]["content"])["task_id"] == "t1"
    assert "Report: done" in sample["messages"][5]["content"]
    assert len(env.requests) == 1


def test_teacher_sees_the_style_but_the_sample_does_not():
    teacher = ScriptedTeacher(good_script())
    sample, _ = generate_trace(record(), teacher, FakeEnv())
    seen_system = teacher.seen[0][0][0]["content"]
    assert "Style requirements" in seen_system and "You are Emperor" in seen_system
    assert "Style requirements" not in sample["messages"][0]["content"]
    assert sample["messages"][0]["content"].rstrip().endswith("context budget: 64K tokens")
    assert [t["function"]["name"] for t in teacher.seen[0][1]] == record().tools


def test_invalid_tool_call_rejects_the_trace():
    teacher = ScriptedTeacher([tool_turn("spawn_agent", {"tier": "standard"})])
    sample, reason = generate_trace(record(), teacher, FakeEnv())
    assert sample is None and reason == "tool_call_invalid"


def test_turn_limit_and_empty_reply_are_rejected():
    loop = [tool_turn("update_plan", {"steps": []}) for _ in range(3)]
    sample, reason = generate_trace(
        record(), ScriptedTeacher(loop), FakeEnv(), GenerationConfig(max_turns=3)
    )
    assert (sample, reason) == (None, "turn_limit")
    _, reason = generate_trace(record(), ScriptedTeacher([{"content": ""}]), FakeEnv())
    assert reason == "empty_reply"


def test_brief_without_sections_fails_validation():
    teacher = ScriptedTeacher(
        [tool_turn("spawn_agent", {"brief": "just do it"}), {"content": "Done."}]
    )
    sample, reason = generate_trace(record(), teacher, FakeEnv())
    assert sample is None and reason == "brief_missing_sections"


def test_environment_tracks_tasks_deterministically():
    env = Environment(FakeEnv(), "task", seed=1)
    assert json.loads(env.run("spawn_agent", {"brief": BRIEF}))["task_id"] == "t1"
    assert json.loads(env.run("spawn_agent", {"brief": BRIEF}))["task_id"] == "t2"
    reports = json.loads(env.run("await_tasks", {}))["reports"]
    assert [r["task_id"] for r in reports] == ["t1", "t2"]
    assert "no running tasks" in env.run("await_tasks", {})
    chunks = json.loads(
        env.run(
            "delegate_chunks",
            {"artefact_id": "a", "chunk_tokens": 100000, "brief": BRIEF, "reduce": "merge"},
        )
    )
    assert len(chunks["task_ids"]) == 4


def test_generation_is_resumable_and_keeps_rejections(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    writer = ShardWriter(tmp_path, "unused")
    del writer
    with prompts.open("w", encoding="utf-8") as handle:
        for i in range(3):
            handle.write(json.dumps(record(id=f"p-{i}").to_dict()) + "\n")
    teacher = ScriptedTeacher(good_script() + [{"content": ""}])
    counts = run_generation(
        prompts, tmp_path, teacher, FakeEnv(), config=GenerationConfig(retries=0), limit=2
    )
    assert counts["accepted"] == 1 and counts["rejected"] == 1
    assert done_ids(tmp_path / "traces") == {"p-0"}
    assert done_ids(tmp_path / "rejects") == {"p-1"}
    teacher2 = ScriptedTeacher(good_script())
    counts = run_generation(prompts, tmp_path, teacher2, FakeEnv())
    assert counts["todo"] == 1 and counts["accepted"] == 1
    assert len(list(iter_records(tmp_path / "traces"))) == 2


def test_stop_and_server_loss_leave_ids_undone(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(record().to_dict()) + "\n", encoding="utf-8")
    counts = run_generation(
        prompts, tmp_path, ScriptedTeacher([]), FakeEnv(), should_stop=lambda: True
    )
    assert counts["stopped"] and not done_ids(tmp_path / "rejects")
    counts = run_generation(prompts, tmp_path, ScriptedTeacher([]), FakeEnv())
    assert counts["stopped"] and "error" in counts
    assert not done_ids(tmp_path / "rejects") and not done_ids(tmp_path / "traces")


def test_teacher_must_be_local():
    assert require_local("http://127.0.0.1:8081/v1") == "http://127.0.0.1:8081/v1"
    assert require_local("http://localhost:8081/v1/") == "http://localhost:8081/v1"
    for url in ("https://api.openai.com/v1", "https://api.anthropic.com", "http://10.0.0.5:8080"):
        with pytest.raises(TeacherError):
            require_local(url)
