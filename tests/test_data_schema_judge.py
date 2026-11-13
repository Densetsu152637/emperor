import json

import pytest
from conftest import BRIEF, call, make_sample

from emperor.data.build import build_report, filter_kept
from emperor.data.judge import (
    JudgeParseError,
    decide,
    heuristic_flags,
    parse_judge_output,
    render_transcript,
    run_judging,
)
from emperor.data.schema import validate_sample
from emperor.data.shards import ShardWriter, done_ids, iter_records

CRITERIA = "concision delegation tool_use research safety style planning honesty".split()
GOOD = {"scores": dict.fromkeys(CRITERIA, 5), "pass": True, "reason": "clean delegation"}


def reasons(sample):
    return validate_sample(sample).reasons


def test_valid_sample_passes(sample):
    assert validate_sample(sample).ok


def test_roles_must_alternate_and_end_with_an_answer(sample):
    sample["messages"].insert(3, {"role": "assistant", "content": "oops"})
    assert "tool_results_do_not_match_calls" in reasons(sample)
    broken = make_sample()
    broken["messages"].pop()
    assert "must_end_with_final_answer" in reasons(broken)
    second = make_sample()
    second["messages"][1]["role"] = "assistant"
    assert "no_user_second" in reasons(second)


def test_tool_calls_are_checked_against_the_schemas(sample):
    sample["messages"][2]["tool_calls"][0] = call("call_1", "spawn_agent", {"tier": "standard"})
    assert "tool_call_invalid" in reasons(sample)
    unknown = make_sample()
    unknown["messages"][2]["tool_calls"][0] = call("call_1", "rm_rf", {})
    assert "tool_call_invalid" in reasons(unknown)
    malformed = make_sample()
    malformed["messages"][2]["tool_calls"][0] = call("call_1", "spawn_agent", "{not json")
    assert "tool_call_invalid" in reasons(malformed)
    not_offered = make_sample()
    not_offered["tools"] = ["spawn_agent", "await_tasks"]
    not_offered["messages"][2]["tool_calls"][0] = call("call_1", "web_search", {"query": "x"})
    assert "tool_call_invalid" in reasons(not_offered)


def test_brief_sections_and_size_are_enforced(sample):
    sample["messages"][2]["tool_calls"][0] = call("call_1", "spawn_agent", {"brief": "do it"})
    assert "brief_missing_sections" in reasons(sample)
    big = make_sample()
    big["messages"][2]["tool_calls"][0] = call(
        "call_1", "spawn_agent", {"brief": BRIEF + "x" * 9000}
    )
    assert "brief_over_limit" in reasons(big)


def test_secrets_are_rejected(sample):
    sample["messages"][-1]["content"] = "Use sk-abcdefghijklmnopqrstuvwx to log in."
    assert "secret_in_assistant_text" in reasons(sample)
    leaked = make_sample()
    leaked["messages"][2]["tool_calls"][0] = call(
        "call_1", "spawn_agent", {"brief": BRIEF + "\napi_key = abcdef123456"}
    )
    assert "secret_in_tool_arguments" in reasons(leaked)


def test_tool_results_must_follow_their_calls(sample):
    sample["messages"][3]["tool_call_id"] = "call_9"
    assert "tool_results_do_not_match_calls" in reasons(sample)


def test_parse_judge_output_variants():
    assert parse_judge_output(json.dumps(GOOD)).passed
    fenced = parse_judge_output("Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nDone.")
    assert fenced.mean == 5
    for bad in ("no json", "{", '{"scores": {}, "pass": true}'):
        with pytest.raises(JudgeParseError):
            parse_judge_output(bad)
    out_of_range = {**GOOD, "scores": {**GOOD["scores"], "style": 6}}
    with pytest.raises(JudgeParseError):
        parse_judge_output(json.dumps(out_of_range))
    with pytest.raises(JudgeParseError):
        parse_judge_output(json.dumps({**GOOD, "pass": "yes"}))


def test_decide_applies_threshold_floor_and_flags(sample):
    result = parse_judge_output(json.dumps(GOOD))
    assert decide(result) == (True, [])
    low = parse_judge_output(json.dumps({**GOOD, "scores": {**GOOD["scores"], "style": 2}}))
    keep, why = decide(low)
    assert not keep and "low_style" in why
    failed = parse_judge_output(json.dumps({**GOOD, "pass": False}))
    assert "judge_fail" in decide(failed)[1]
    mid = parse_judge_output(json.dumps({**GOOD, "scores": dict.fromkeys(CRITERIA, 3)}))
    assert "below_threshold" in decide(mid)[1]
    sample["messages"][-1]["content"] = "Done \U0001f600\U0001f680\U0001f389!"
    flagged = heuristic_flags(sample)
    assert flagged == ["emojis"]
    result.flags = flagged
    assert not decide(result)[0]


def test_heuristics_flag_tables_and_pasted_code(sample):
    sample["messages"][-1]["content"] = "\n".join(["| a | b |"] * 6)
    assert "tables" in heuristic_flags(sample)
    sample["messages"][-1]["content"] = "```py\n" + "x = 1\n" * 40 + "```"
    assert "pasted_code" in heuristic_flags(sample)


class FakeJudge:
    def __init__(self, answers):
        self.answers = list(answers)

    def chat(self, messages, **kwargs):
        return {"content": self.answers.pop(0)}


def test_judging_is_resumable_and_records_scores(tmp_path):
    traces = ShardWriter(tmp_path / "traces", "traces")
    for i in range(3):
        traces.write(make_sample(f"p-{i}"))
    client = FakeJudge([json.dumps(GOOD), "garbage", "still garbage", json.dumps(GOOD)])
    counts = run_judging(tmp_path, client, limit=2)
    assert counts["judged"] == 2
    assert done_ids(tmp_path / "judged") == {"p-0", "p-1"}
    run_judging(tmp_path, client)  # resumes with p-2 only
    judged = {s["id"]: s for s in iter_records(tmp_path / "judged")}
    assert set(judged) == {"p-0", "p-1", "p-2"}
    assert judged["p-0"]["meta"]["judge"]["keep"] is True
    assert judged["p-1"]["meta"]["judge"]["rejection"] == ["judge_unparseable"]
    assert judged["p-0"]["meta"]["judge"]["scores"]["style"] == 5

    assert filter_kept(tmp_path)["kept"] == 2
    report = build_report(tmp_path)
    assert report["judged"] == 3 and report["kept"] == 2
    assert report["pass_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert report["score_histograms"]["style"] == {"5": 2}
    assert report["judge_rejection_reasons"] == {"judge_unparseable": 1}


def test_reasoning_is_checked_for_secrets_and_hidden_from_the_judge(sample):
    sample["messages"][-1]["reasoning_content"] = "The user wants pagination."
    assert validate_sample(sample).ok
    assert "The user wants" not in render_transcript(sample)
    sample["messages"][-1]["reasoning_content"] = "Log in with sk-abcdefghijklmnopqrstuvwx."
    assert "secret_in_reasoning" in reasons(sample)
    sample["messages"][-1]["reasoning_content"] = ["not", "text"]
    assert "reasoning_not_text" in reasons(sample)


def test_a_thinking_judge_is_parsed_from_its_answer_and_named_in_the_verdict(tmp_path):
    from emperor.data.judge import judge_answer

    thought = 'Scores like {"scores": 1} are wrong here. Let me decide.'
    assert judge_answer({"content": f"<think>{thought}</think>\n{json.dumps(GOOD)}"}) == (
        "\n" + json.dumps(GOOD)
    )
    assert judge_answer({"content": f"{thought}</think>{json.dumps(GOOD)}"}) == json.dumps(GOOD)
    assert judge_answer({"content": f"<think>{thought}"}) == ""  # out of tokens mid-thought
    assert judge_answer({"content": "{}", "reasoning_content": thought}) == "{}"

    ShardWriter(tmp_path / "traces", "traces").write(make_sample("p-0"))
    reply = f"<think>{thought}</think>{json.dumps(GOOD)}"
    run_judging(tmp_path, FakeJudge([reply]))
    (judged,) = iter_records(tmp_path / "judged")
    assert judged["meta"]["judge"]["keep"] is True
    assert judged["meta"]["judge"]["model"] == "Qwen3.8-27B"


def test_the_judge_never_sees_the_teachers_reasoning(sample):
    from emperor.data.judge import render_transcript

    for message in sample["messages"]:
        if message["role"] == "assistant":
            message["reasoning_content"] = "SECRET-CHAIN-OF-THOUGHT"
    assert "SECRET-CHAIN-OF-THOUGHT" not in render_transcript(sample)
