import json

import pytest
from conftest import BRIEF

from emperor.data.generate import (
    _PREFLIGHT_USER,
    TEACHER_REASONING_FIELD,
    Environment,
    GenerationConfig,
    generate_trace,
    run_generation,
)
from emperor.data.prompts import PromptRecord
from emperor.data.shards import ShardWriter, done_ids, iter_records
from emperor.data.teacher import TeacherError, require_local

REASONING = "Think it through."


class ScriptedTeacher:
    """Plays fixed assistant messages and records what it was sent. A turn without a
    `reasoning_content` key gets one, as the real teacher's do; set it to None for none.
    The reasoning preflight is answered apart from the script (with reasoning unless
    `preflight_reasoning` is false) and recorded in `preflight`."""

    def __init__(self, turns, preflight_reasoning=True):
        self.turns = list(turns)
        self.seen = []
        self.kwargs = []
        self.preflight = []
        self.preflight_reasoning = preflight_reasoning

    def chat(self, messages, *, tools=None, **kwargs):
        if messages[-1].get("content") == _PREFLIGHT_USER:
            self.preflight.append(kwargs)
            reply = {"content": "", "tool_calls": []}
            if self.preflight_reasoning:
                reply["reasoning_content"] = REASONING
            return reply
        self.seen.append((list(messages), tools))
        self.kwargs.append(kwargs)
        if not self.turns:
            raise TeacherError("out of script")
        turn = self.turns.pop(0)
        return turn if "reasoning_content" in turn else {**turn, "reasoning_content": REASONING}


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
    assert sample["meta"]["teacher"] == "gpt-oss-120b"
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


def test_teacher_reasoning_is_stored_per_turn_and_kept_in_the_teachers_history():
    script = good_script()
    script[0] = {**script[0], "reasoning_content": "  Delegate the work.  "}
    teacher = ScriptedTeacher(script)
    sample, reason = generate_trace(record(), teacher, FakeEnv())
    assert reason is None
    turns = [m for m in sample["messages"] if m["role"] == "assistant"]
    assert [t.get("reasoning_content") for t in turns] == [
        "Delegate the work.",
        REASONING,
        REASONING,
    ]
    history = teacher.seen[-1][0]
    assert [m.get(TEACHER_REASONING_FIELD) for m in history if m["role"] == "assistant"] == [
        "Delegate the work.",
        REASONING,
    ]


def test_a_first_turn_without_reasoning_rejects_the_trace():
    script = good_script()
    script[0] = {**script[0], "reasoning_content": None}
    sample, reason = generate_trace(record(), ScriptedTeacher(script), FakeEnv())
    assert (sample, reason) == (None, "no_reasoning")
    config = GenerationConfig(require_reasoning=False)
    sample, reason = generate_trace(record(), ScriptedTeacher(script), FakeEnv(), config)
    assert reason is None and "reasoning_content" not in sample["messages"][2]


def test_a_later_turn_without_reasoning_is_kept_as_is():
    script = good_script()
    script[1] = {**script[1], "reasoning_content": None}
    sample, reason = generate_trace(record(), ScriptedTeacher(script), FakeEnv())
    assert reason is None
    assert "reasoning_content" not in sample["messages"][4]
    assert sample["messages"][2]["reasoning_content"] == REASONING


class ForcedToolHidesReasoning:
    """A teacher whose reasoning disappears when a tool call is forced."""

    def __init__(self, forced_reasoning=""):
        self.kwargs = []
        self.forced_reasoning = forced_reasoning

    def chat(self, messages, *, tools=None, **kwargs):
        self.kwargs.append(kwargs)
        forced = kwargs.get("tool_choice") == "required"
        return {"content": "ok", "reasoning_content": self.forced_reasoning if forced else "r"}


def test_generation_stops_at_once_when_the_teacher_returns_no_reasoning(tmp_path):
    from emperor.data.generate import check_teacher_reasoning
    from emperor.data.teacher import TeacherSetupError

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(record().to_dict()) + "\n", encoding="utf-8")
    silent = ScriptedTeacher(good_script(), preflight_reasoning=False)
    with pytest.raises(TeacherSetupError, match="tool_choice=required"):
        run_generation(prompts, tmp_path, silent, FakeEnv())
    assert silent.preflight[0]["tool_choice"] == "required" and silent.seen == []

    with pytest.raises(TeacherSetupError, match="tool_choice=required"):
        check_teacher_reasoning(ForcedToolHidesReasoning(), GenerationConfig())
    fine = ForcedToolHidesReasoning(forced_reasoning="r")
    check_teacher_reasoning(fine, GenerationConfig())
    assert [k.get("tool_choice") for k in fine.kwargs] == ["required", None]
    # Without forced tool calls in generation, only the unforced mode is checked.
    check_teacher_reasoning(ForcedToolHidesReasoning(), GenerationConfig(force_first_tool=False))


def test_generation_runs_after_a_passing_preflight(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(record().to_dict()) + "\n", encoding="utf-8")
    teacher = ScriptedTeacher(good_script())
    counts = run_generation(prompts, tmp_path, teacher, FakeEnv())
    assert counts["accepted"] == 1 and len(teacher.preflight) == 2


def test_generation_defaults_to_low_reasoning_effort():
    from emperor.data.cli import build_parser
    from emperor.data.teacher import ChatClient

    args = build_parser().parse_args(["retry"])
    assert (args.reasoning_effort, args.judge_reasoning_effort) == ("medium", "medium")
    assert args.environment_reasoning_effort == "medium"
    assert build_parser().parse_args(["judge"]).judge_reasoning_effort == "medium"
    assert ChatClient.__init__.__kwdefaults__["reasoning_effort"] == "medium"


def test_the_teacher_prompt_differs_between_tasks_only_at_its_end():
    from emperor.data.generate import teacher_system

    small, large = teacher_system(32 * 1024), teacher_system(128 * 1024)
    assert small.rstrip().endswith("context budget: 32K tokens")
    assert large.rstrip().endswith("context budget: 128K tokens")
    shared = len(small.rstrip().rsplit("\n", 1)[0])
    assert small[:shared] == large[:shared] and shared > 0.95 * len(small)


def test_a_trace_longer_than_the_training_length_is_rejected_early():
    teacher = ScriptedTeacher(good_script())
    sample, reason = generate_trace(
        record(), teacher, FakeEnv(), GenerationConfig(max_sample_tokens=1500)
    )
    assert (sample, reason) == (None, "too_long")
    assert len(teacher.seen) == 1  # stopped after the first turn's tool result
    sample, reason = generate_trace(record(), ScriptedTeacher(good_script()), FakeEnv())
    assert reason is None


class BatchEnv:
    """Answers a batched await with one section per subagent."""

    def __init__(self):
        self.requests = []

    def chat(self, messages, **kwargs):
        self.requests.append(messages[-1]["content"])
        return {"content": "### t1\nReport one.\n\n### `t2`\nReport two."}


def test_awaited_reports_come_from_one_simulator_call():
    client = BatchEnv()
    env = Environment(client, "task", seed=1)
    env.run("spawn_agent", {"brief": BRIEF})
    env.run("spawn_agent", {"brief": BRIEF})
    reports = json.loads(env.run("await_tasks", {}))["reports"]
    assert [(r["task_id"], r["report"]) for r in reports] == [
        ("t1", "Report one."),
        ("t2", "Report two."),
    ]
    assert len(client.requests) == 1 and "2 subagents" in client.requests[0]
