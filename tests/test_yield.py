"""Teacher yield: forced tool use, rejection sampling, judge robustness, extra data rounds."""

import json
from pathlib import Path

from conftest import make_sample
from test_data_generate import FakeEnv, ScriptedTeacher, good_script, record, tool_turn

from emperor.data.build import build_report, filter_kept
from emperor.data.generate import GenerationConfig, generate_trace, run_generation
from emperor.data.judge import JudgeResult, decide, judge_sample, run_judging
from emperor.data.prompts import normalise
from emperor.data.retry import prepare_round, retry_dirs, run_retry_rounds
from emperor.data.schema import table_rows
from emperor.data.shards import ShardWriter, done_ids, iter_records
from emperor.data.style import STYLE, judge_system
from emperor.data.teacher import ChatClient
from emperor.train.config import load_config, parse_config
from emperor.train.data import load_combined

CRITERIA = "concision delegation tool_use research safety style planning honesty".split()
GOOD = {"scores": dict.fromkeys(CRITERIA, 5), "pass": True, "reason": "ok"}
USER = [{"role": "user", "content": "x"}]
TOOLS = [{"type": "function"}]


# -- forced tool use and the style prompt ------------------------------------------------------


def test_first_turn_requires_a_tool_call_except_for_quick_answers():
    teacher = ScriptedTeacher(good_script())
    generate_trace(record(), teacher, FakeEnv(), GenerationConfig())
    assert teacher.kwargs[0].get("tool_choice") == "required"
    assert "tool_choice" not in teacher.kwargs[1]

    quick = ScriptedTeacher([{"content": "It is 4."}])
    sample, reason = generate_trace(record(task_class="quick_answer"), quick, FakeEnv())
    assert reason is None and "tool_choice" not in quick.kwargs[0]
    assert sample["messages"][-1]["content"] == "It is 4."

    off = ScriptedTeacher(good_script())
    generate_trace(record(), off, FakeEnv(), GenerationConfig(force_first_tool=False))
    assert "tool_choice" not in off.kwargs[0]


def test_work_without_any_tool_call_is_rejected_unless_quick_answer():
    teacher = ScriptedTeacher([{"content": "Here you go."}])
    sample, reason = generate_trace(record(), teacher, FakeEnv())
    assert sample is None and reason == "no_tool_use"


def test_markdown_tables_are_rejected_at_generation():
    table = "\n".join(["| a | b |"] * 6)
    script = good_script()
    script[-1] = {"content": table}
    sample, reason = generate_trace(record(), ScriptedTeacher(script), FakeEnv())
    assert sample is None and reason == "markdown_table"
    assert table_rows(table) == 6


def test_style_forbids_tables_and_demands_tools_first():
    assert "Markdown tables" in STYLE
    assert "first turn is a tool call" in STYLE
    assert "update_plan" in STYLE and "await_tasks" in STYLE


class _Response:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError("bad")
            error.response = self
            raise error

    def json(self):
        return {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}


class _Http:
    def __init__(self, reject_tool_choice):
        self.bodies = []
        self.reject = reject_tool_choice

    def post(self, url, json):
        self.bodies.append(dict(json))
        return _Response(400 if self.reject and "tool_choice" in json else 200)


def test_client_sends_tool_choice_and_falls_back_when_the_server_rejects_it():
    client = ChatClient.__new__(ChatClient)  # skips the httpx import (a separate dependency group)
    client.base_url, client.model, client.retries = "http://127.0.0.1:1/v1", "m", 3
    client.reasoning_effort, client.tool_choice_supported = None, True
    client._http = _Http(reject_tool_choice=False)
    client.chat(USER, tools=TOOLS, tool_choice="required")
    assert client._http.bodies[0]["tool_choice"] == "required"

    client._http = _Http(reject_tool_choice=True)
    assert client.chat(USER, tools=TOOLS, tool_choice="required")["content"] == "hi"
    assert "tool_choice" in client._http.bodies[0] and "tool_choice" not in client._http.bodies[1]
    assert not client.tool_choice_supported
    client.chat(USER, tools=TOOLS, tool_choice="required")
    assert "tool_choice" not in client._http.bodies[-1]


# -- rejection sampling ------------------------------------------------------------------------


def test_failed_schema_check_is_retried_with_another_seed(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(record().to_dict()) + "\n", encoding="utf-8")
    bad = [tool_turn("spawn_agent", {"tier": "standard"})]  # no brief: invalid call
    teacher = ScriptedTeacher(bad + good_script())
    counts = run_generation(prompts, tmp_path, teacher, FakeEnv())
    assert counts["accepted"] == 1 and counts["rejected"] == 0
    (sample,) = iter_records(tmp_path / "traces")
    assert sample["meta"]["attempts"] == 2
    assert sample["meta"]["attempt_failures"] == ["tool_call_invalid"]
    assert sample["meta"]["seed"] != 11  # the retry took a different seed
    assert teacher.kwargs[0]["seed"] != teacher.kwargs[1]["seed"]


def test_exhausted_retries_are_recorded_in_the_reject(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(record().to_dict()) + "\n", encoding="utf-8")
    teacher = ScriptedTeacher([{"content": ""}] * 5)
    config = GenerationConfig(retries=2)
    counts = run_generation(prompts, tmp_path, teacher, FakeEnv(), config=config)
    assert counts["rejected"] == 1
    (reject,) = iter_records(tmp_path / "rejects")
    assert reject["attempts"] == 3 and reject["attempt_failures"] == ["empty_reply"] * 3


# -- judge robustness --------------------------------------------------------------------------


class RecordingJudge:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"content": self.answers.pop(0)}


def test_unparseable_judgement_is_retried_colder_with_a_stricter_instruction():
    client = RecordingJudge(["I think it is fine.", json.dumps(GOOD)])
    assert judge_sample(make_sample(), client).passed
    (first, first_kw), (second, second_kw) = client.calls
    assert second_kw["temperature"] < first_kw["temperature"]
    assert "not valid JSON" in second[1]["content"] and "not valid JSON" not in first[1]["content"]
    assert second_kw["max_tokens"] >= first_kw["max_tokens"]


def test_quick_answer_and_clarify_are_not_penalised_for_not_delegating():
    scores = {
        **dict.fromkeys(CRITERIA, 5),
        "delegation": 1,
        "tool_use": 1,
        "research": 1,
        "planning": 1,
    }
    result = JudgeResult(scores, True, "direct answer")
    assert not decide(result)[0]  # applicable to a coding task: rejected
    result.ignored = ("delegation", "tool_use", "research", "planning")
    assert decide(result)[0] and result.mean == 5

    sample = make_sample()
    sample["task_class"] = "quick_answer"
    client = RecordingJudge([json.dumps({**GOOD, "scores": scores})])
    assert decide(judge_sample(sample, client))[0]
    assert "Task class: quick_answer" in client.calls[0][0][0]["content"]
    assert "Task class" not in judge_system("coding")

    clarify = make_sample()
    clarify["task_class"] = "clarify"
    low = {**dict.fromkeys(CRITERIA, 5), "delegation": 1, "research": 1}
    answer = json.dumps({**GOOD, "scores": low})
    assert decide(judge_sample(clarify, RecordingJudge([answer])))[0]
    bad = json.dumps({**GOOD, "scores": {**dict.fromkeys(CRITERIA, 5), "tool_use": 1}})
    assert not decide(judge_sample(clarify, RecordingJudge([bad])))[0]  # tool use still counts


# -- retry rounds and extra data rounds ---------------------------------------------------------


def write_prompts(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(record(id=i).to_dict()) + "\n" for i in ids)
    path.write_text(lines, encoding="utf-8")


def judged(directory: Path, sample_id: str, keep: bool):
    sample = make_sample(sample_id)
    verdict = {"keep": keep, "rejection": [] if keep else ["judge_fail"]}
    sample["meta"] = {"seed": 1, "judge": verdict}
    ShardWriter(directory / "judged", "judged").write(sample)


def test_prepare_round_takes_only_failures_with_a_new_seed_and_is_idempotent(tmp_path):
    write_prompts(tmp_path / "prompts.jsonl", ["p-0", "p-1", "p-2"])
    assert prepare_round(tmp_path, 1) == 0  # the first pass is not finished yet
    judged(tmp_path, "p-0", True)
    judged(tmp_path, "p-1", False)
    ShardWriter(tmp_path / "rejects", "rejects").write({"id": "p-2", "reason": "turn_limit"})
    assert prepare_round(tmp_path, 1) == 2
    text = (tmp_path / "retry-1" / "prompts.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    assert [r["id"] for r in rows] == ["p-1", "p-2"] and all(r["seed"] != 11 for r in rows)
    judged(tmp_path, "p-9", True)  # later changes never rewrite a prepared round
    assert prepare_round(tmp_path, 1) == 2


def test_filter_keeps_the_first_pass_and_adds_retry_wins(tmp_path):
    judged(tmp_path, "p-0", True)
    judged(tmp_path, "p-1", False)
    judged(tmp_path / "retry-1", "p-1", True)
    judged(tmp_path / "retry-1", "p-0", True)  # a duplicate of a first-pass keep is ignored
    assert [p.name for p in retry_dirs(tmp_path)] == ["retry-1"]
    assert filter_kept(tmp_path)["kept"] == 2
    assert {s["id"] for s in iter_records(tmp_path / "final")} == {"p-0", "p-1"}
    report = build_report(tmp_path)
    assert report["kept"] == 1 and report["kept_with_retries"] == 3
    assert report["retry_rounds"]["retry-1"] == {"judged": 2, "kept": 2}


class RoundClient(ScriptedTeacher):
    """Teacher for tool calls, judge for transcripts, environment for everything else."""

    def chat(self, messages, *, tools=None, **kwargs):
        if tools is None:
            if "Transcript" in messages[-1]["content"]:
                return {"content": json.dumps(GOOD)}
            return {"content": "report"}
        return super().chat(messages, tools=tools, **kwargs)


def test_retry_rounds_regenerate_rejected_prompts_and_resume(tmp_path):
    write_prompts(tmp_path / "prompts.jsonl", ["p-0", "p-1"])
    judged(tmp_path, "p-0", True)
    judged(tmp_path, "p-1", False)
    summary = run_retry_rounds(tmp_path, RoundClient(good_script()), rounds=2)
    assert not summary["stopped"]
    assert list(summary["rounds"]) == ["retry-1"]  # round 2 had nothing left to retry
    (kept,) = iter_records(tmp_path / "retry-1" / "judged")
    assert kept["id"] == "p-1" and kept["meta"]["attempt"] == 1 and kept["meta"]["judge"]["keep"]
    again = run_retry_rounds(tmp_path, RoundClient([]), rounds=2)  # nothing to redo
    assert again["rounds"]["retry-1"]["generate"]["todo"] == 0
    assert done_ids(tmp_path / "retry-1" / "judged") == {"p-1"}


def test_judging_a_retry_round_directory_works_like_the_main_one(tmp_path):
    ShardWriter(tmp_path / "retry-1" / "traces", "traces").write(make_sample("p-1"))
    counts = run_judging(tmp_path / "retry-1", RecordingJudge([json.dumps(GOOD)]))
    assert counts["kept"] == 1


def test_load_combined_dedupes_by_sample_id(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"id": "p-1", "v": "a"}\n{"id": "p-2"}\n', encoding="utf-8")
    b.write_text('{"id": "p-1", "v": "b"}\n{"id": "p-3"}\n', encoding="utf-8")
    combined = load_combined([a, b])
    assert [r["id"] for r in combined] == ["p-1", "p-2", "p-3"]
    assert combined[0]["v"] == "a"


def test_round_two_config_trains_fresh_on_both_rounds():
    root = Path(__file__).parent.parent / "configs"
    one, two = load_config(root / "sft.toml"), load_config(root / "sft-r2.toml")
    assert two.data.path == "data/r2/final/final.jsonl"
    assert two.data.extra_paths == [one.data.path]
    assert one.data.extra_paths == []
    assert two.train.output_dir != one.train.output_dir
    assert two.merge.adapter_dir.startswith(two.train.output_dir)
    assert two.merge.output_dir != one.merge.output_dir
    assert parse_config({"data": {"extra_paths": ["x"]}}).data.extra_paths == ["x"]


def read_texts(path: Path) -> set[str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    return {normalise(json.loads(row)["text"]) for row in rows}


def test_pipeline_second_round_uses_its_own_dir_and_avoids_round_one_prompts(tmp_path):
    from test_release import make_pipeline

    from emperor.pipeline.cli import build_parser, settings_from

    argv = ["--root", str(tmp_path), "--data-dir", "data/r2", "--previous-data", "data"]
    settings = settings_from(build_parser().parse_args(argv), tmp_path / "runs" / "r2")
    assert settings.data_dir == tmp_path.resolve() / "data" / "r2"
    assert settings.previous_data == [tmp_path.resolve() / "data"]
    assert settings.retries == 2 and settings.force_first_tool

    first = tmp_path / "one"
    first.mkdir()
    make_pipeline(first, prompts=30).phase_prompts()
    second_root = tmp_path / "two"
    second_root.mkdir()
    second = make_pipeline(second_root, prompts=30, previous_data=[first / "data"])
    second.s.data_dir = second_root / "data" / "r2"
    second.phase_prompts()
    one = read_texts(first / "data" / "prompts.jsonl")
    two = read_texts(second.s.data_dir / "prompts.jsonl")
    assert len(two) == 30 and not one & two
    assert (first / "data" / "prompts.jsonl").exists()


def test_retry_phase_is_listed_between_judge_and_filter():
    from emperor.pipeline.run import PHASES

    assert PHASES.index("judge") + 1 == PHASES.index("retry") == PHASES.index("filter") - 1
