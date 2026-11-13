"""Consul: tasks, schema, scoring, dataset steps, eval and pipeline wiring.

No GPU, no model and no network: the teacher is a stub that answers with the reference critique.
"""

import json
from pathlib import Path

import pytest
from emperor.data.shards import ShardWriter, iter_records
from emperor.pipeline.idle import IdleGate
from emperor.pipeline.processes import OwnProcesses
from emperor.pipeline.run import Settings
from emperor.train.config import load_config

from consul import data, prompts
from consul.eval.cli import main as eval_main
from consul.overlap import OverlapIndex, overlap_report
from consul.pipeline.cli import ConsulPipeline
from consul.schema import (
    CritiqueParseError,
    citation_errors,
    extract_answer,
    extract_json_object,
    schema_errors,
    strip_thinking,
)
from consul.score import anchor_hit, reference_critique, score_critique
from consul.tasks import CritiqueTask, dumps, generate_tasks, make_task

ROOT = Path(__file__).parent.parent


def test_tasks_are_deterministic_diverse_and_carry_ground_truth():
    assert make_task(7).to_dict() == make_task(7).to_dict()
    tasks = generate_tasks(300, 1)
    assert len({t.id for t in tasks}) == 300
    assert {t.kind for t in tasks} == {"plan", "diff", "report"}
    assert 0.05 < sum(t.clean for t in tasks) / 300 < 0.35
    for task in tasks:
        assert task.artifact and task.urls() >= {s.url for s in task.sources}
        assert all(d.where and d.keywords for d in task.defects)
        assert task.clean == (not [d for d in task.defects if d.acceptance_changing])
    assert len({d.kind for t in tasks for d in t.defects}) >= 15


def test_reference_critique_scores_perfectly_on_every_task():
    for task in generate_tasks(200, 5):
        critique = reference_critique(task)
        assert schema_errors(critique) == []
        score = score_critique(task, critique)
        assert score.format_ok and not score.missed and not score.false_findings


def test_task_roundtrips_through_json():
    task = make_task(11, kind="plan", clean=False)
    again = CritiqueTask.from_dict(json.loads(json.dumps(task.to_dict())))
    assert again == task


def test_schema_rejects_inconsistent_results():
    task = make_task(3, kind="plan", clean=False)
    good = reference_critique(task)
    assert schema_errors(good) == []
    accept = {**good, "verdict": "accept"}
    assert any("accept" in e for e in schema_errors(accept))
    assert any("verdict" in e for e in schema_errors({**good, "findings": [], "verdict": "revise"}))
    assert schema_errors({"verdict": "accept"})  # missing keys
    dup = {**good, "findings": [good["findings"][0], good["findings"][0]]}
    assert any("duplicate" in e for e in schema_errors(dup))
    sourced = json.loads(json.dumps(good))
    sourced["findings"][0].update(basis="source", sources=[])
    assert any("source" in e for e in schema_errors(sourced))


def test_extract_json_object_handles_fences_and_prose():
    assert extract_json_object('text ```json\n{"a": {"b": "}"}}\n``` more') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json_object("no object here")


def test_citations_must_come_from_the_context():
    task = next(t for t in generate_tasks(50, 1) if t.sources)
    critique = reference_critique(task)
    critique["findings"][0]["sources"] = [task.sources[0].url, "https://invented.example/x"]
    assert citation_errors(critique, task.urls()) == [
        "cited URL not in the context: https://invented.example/x"
    ]
    score = score_critique(task, critique)
    assert score.cited == 2 and len(score.bad_citations) == 1


def test_scoring_counts_missed_defects_and_false_blocking_findings():
    task = next(t for t in generate_tasks(100, 1) if len(t.defects) >= 1 and not t.clean)
    empty = {"verdict": "accept", "summary": "fine", "findings": []}
    score = score_critique(task, empty)
    assert score.recall == 0.0 and score.missed
    bogus = {
        "verdict": "revise",
        "summary": "x",
        "findings": [
            {
                "id": "f1",
                "severity": "acceptance_changing",
                "location": "Step 99",
                "impact": "nothing relevant",
                "confidence": "low",
                "basis": "judgement",
                "fix_or_check": "none",
            }
        ],
    }
    score = score_critique(task, bogus)
    assert score.false_rate == 1.0
    assert score_critique(task, None).parsed is False
    clean = make_task(9, kind="plan", clean=True)
    assert score_critique(clean, bogus).false_rate == 1.0
    assert score_critique(clean, empty).false_rate is None


def test_anchor_matching():
    assert anchor_hit("Step 4", "Step 4|Step 3")
    assert anchor_hit("step 3 and 4", "Step 4")
    assert not anchor_hit("Step 5", "Step 4")
    assert anchor_hit("Plan", "Step 4")  # a general location is allowed; keywords decide
    assert anchor_hit("paging.py:L12", "paging.py:L13")
    assert not anchor_hit("paging.py:L40", "paging.py:L13")
    assert anchor_hit("Acceptance criteria", "Acceptance")
    assert anchor_hit("Claim 2", "Claim 2")


def test_heldout_uses_other_vocabulary_and_does_not_overlap_training(tmp_path):
    data.write_train_tasks(tmp_path, 400, 1)
    report = data.write_heldout(tmp_path, 25)
    assert report["ok"] and report["exact_duplicates"] == []
    assert report["max_overlap"] < data.DEFAULT_THRESHOLD
    train = data.load_tasks(tmp_path / "prompts.jsonl")
    held = data.load_tasks(tmp_path / "eval" / "heldout.jsonl")
    assert len(held) == 25 and not {t.id for t in train} & {t.id for t in held}
    assert min(t.seed for t in held) >= data.HELDOUT_SEED_OFFSET > max(t.seed for t in train)
    # A copy of a training task is caught by both checks.
    assert not overlap_report(train, [train[0]], 0.6)["ok"]
    assert OverlapIndex(train).closest(train[3])[0] == 1.0


def test_train_tasks_append_and_resume(tmp_path):
    assert data.write_train_tasks(tmp_path, 30, 1)["added"] == 30
    assert data.write_train_tasks(tmp_path, 50, 1)["added"] == 20
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    assert len(tasks) == 50 and len({t.id for t in tasks}) == 50


class StubTeacher:
    """Answers critiques with the reference critique (good), judge calls with a pass."""

    def __init__(self, tasks, garble=(), silent=()):
        self.by_text = {prompts.user_message(t): t for t in tasks}
        self.garble = set(garble)
        self.silent = set(silent)  # answer well, but with no reasoning
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == prompts.JUDGE_SYSTEM:
            scores = {c: 5 for c in prompts.JUDGE_CRITERIA}
            return {"content": json.dumps({"scores": scores, "pass": True, "reason": "ok"})}
        task = self.by_text[user]
        if task.id in self.garble:
            return {"content": "I think it is fine.", "reasoning_content": "Looks fine."}
        reasoning = "" if task.id in self.silent else f"Check {task.id} step by step."
        return {"content": json.dumps(reference_critique(task)), "reasoning_content": reasoning}


def test_generate_judge_filter_end_to_end(tmp_path):
    data.write_train_tasks(tmp_path, 40, 1)
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    broken, silent = tasks[0].id, tasks[1].id
    teacher = StubTeacher(tasks, garble=[broken], silent=[silent])
    counts = data.run_generation(tmp_path, teacher)
    assert counts["accepted"] == 38 and counts["rejected"] == 2
    assert data.run_generation(tmp_path, teacher)["todo"] == 0  # resumable
    judged = data.run_judging(tmp_path, teacher)
    assert judged["judged"] == 38 and judged["kept"] == 38
    final = data.filter_kept(tmp_path)
    assert final["kept"] == 38 and final["no_reasoning"] == 0
    report = data.write_report(tmp_path)
    assert report["generation_rejection_reasons"] == {"unparsable": 1, "empty_reasoning": 1}
    sample = json.loads((tmp_path / "final" / "final.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    answer = sample["messages"][-1]
    assert answer["reasoning_content"] == f"Check {sample['id']} step by step."
    assert "step by step" not in answer["content"]  # the critique alone, as the judge sees it
    assert sample["messages"][0]["content"] == prompts.CONSUL_SYSTEM  # no teacher guidance
    assert "keywords" not in json.dumps(sample["messages"])  # ground truth stays out


def test_critiques_that_miss_a_defect_are_dropped_before_judging(tmp_path):
    data.write_train_tasks(tmp_path, 40, 1)
    tasks = data.load_tasks(tmp_path / "prompts.jsonl")
    target = next(t for t in tasks if not t.clean)

    class Lazy(StubTeacher):
        def chat(self, messages, **kwargs):
            if messages[0]["content"] != prompts.JUDGE_SYSTEM and (
                self.by_text[messages[1]["content"]].id == target.id
            ):
                return {
                    "content": json.dumps({"verdict": "accept", "summary": "ok", "findings": []}),
                    "reasoning_content": "Nothing stands out.",
                }
            return super().chat(messages, **kwargs)

    teacher = Lazy(tasks)
    counts = data.run_generation(tmp_path, teacher)
    assert counts["rejected"] == 1
    (reject,) = iter_records(tmp_path / "rejects")
    assert (reject["id"], reject["reason"]) == (target.id, "missed_defect")
    data.run_judging(tmp_path, teacher)
    kept = {s["id"] for s in iter_records(tmp_path / "judged") if s["meta"]["judge"]["keep"]}
    assert target.id not in kept and len(kept) == 39
    assert data.build_report(tmp_path)["judge_rejection_reasons"] == {}


def test_eval_dry_run_scores_perfectly_and_refuses_overlap(tmp_path, capsys):
    data.write_train_tasks(tmp_path, 60, 1)
    data.write_heldout(tmp_path, 10)
    args = [
        "--tasks", str(tmp_path / "eval" / "heldout.jsonl"),
        "--train-data", str(tmp_path / "prompts.jsonl"),
        "--dry-run", "--output", str(tmp_path / "report.json"),
    ]  # fmt: skip
    assert eval_main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert metrics["defect_recall"]["score"] == 1.0 and metrics["defect_recall"]["passed"]
    assert metrics["false_findings"]["score"] == 0.0 and metrics["false_findings"]["passed"]
    assert metrics["format"]["score"] == 1.0
    assert metrics["loop"]["note"] == "TODO" and report["overlap"]["ok"]
    # Evaluating on the training tasks themselves must be refused.
    copy = ["--tasks", str(tmp_path / "prompts.jsonl")] + args[2:4]
    assert eval_main(copy + ["--dry-run"]) == 2
    assert "overlaps" in capsys.readouterr().err


def test_consul_config_is_emperors_recipe_with_consul_names():
    config = load_config(ROOT / "configs" / "sft.toml")
    assert config.merge.output_dir == "weights/consul-4b-bf16"
    assert config.release.name == "consul-4b" and config.release.eval_alias == "consul"
    assert config.model.base.endswith("Qwen3.5-4B")
    assert config.data.max_length == 8192
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= set(config.lora.target_modules)


def make_pipeline(root: Path, **kwargs) -> ConsulPipeline:
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "sft.toml").write_text(
        (ROOT / "configs" / "sft.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_dir = root / "runs" / "r1"
    run_dir.mkdir(parents=True)
    settings = Settings(root=root, run_dir=run_dir, data_dir=root / "data", prompts=30, **kwargs)
    return ConsulPipeline(settings, IdleGate(asleep=True), OwnProcesses(), eval_tasks=6)


def test_pipeline_prompts_filter_and_eval_command(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = pipeline.phase_prompts()
    assert result["status"] == "done" and result["tasks"] == 30 and result["heldout"]["ok"]
    assert pipeline._pending("prompts.jsonl", "traces") == 30
    assert pipeline.phase_filter()["kept"] == 0
    command = pipeline.eval_command(Path("eval.json"))
    assert command[1:3] == ["-m", "consul.eval.cli"] and "llamacpp:consul" in command


def test_pipeline_runs_without_gpu_phases(tmp_path):
    pipeline = make_pipeline(
        tmp_path, skip={"generate", "judge", "train", "merge", "export", "eval", "card"}
    )
    summary = pipeline.run()
    assert summary["status"] == "done" and summary["phases"]["prompts"]["status"] == "done"
    assert summary["teacher"] == {
        "model": "deepseek-v4-pro",
        "licence": "DeepSeek Open Platform Terms of Service",
    }
    assert summary["phases"]["retry"]["status"] == "skipped"


def test_pipeline_cli_dispatch_and_extra_options():
    from consul.pipeline.cli import split_extra

    known, rest = split_extra(["--detach", "--eval-tasks", "9", "--asleep", "--eval-seed=5"])
    assert (known.eval_tasks, known.eval_seed, rest) == (9, 5, ["--detach", "--asleep"])
    assert known.generate_reasoning_effort == "medium"
    known, rest = split_extra(["--generate-reasoning-effort", "high", "--reasoning-effort", "low"])
    assert known.generate_reasoning_effort == "high" and rest == ["--reasoning-effort", "low"]


def test_prompt_example_is_a_valid_critique():
    from consul.schema import CRITIQUE_EXAMPLE

    assert schema_errors(CRITIQUE_EXAMPLE) == []
    assert '"verdict": "revise"' in prompts.CONSUL_SYSTEM


class ClientRecorder:
    """Records how each client was made, and answers nothing (no tasks are pending)."""

    made: list = []

    def __init__(self, base_url, reasoning_effort=None, **kwargs):
        ClientRecorder.made.append({"reasoning_effort": reasoning_effort, **kwargs})

    def close(self):
        pass


def test_generation_runs_on_the_teacher_and_judging_on_the_thinking_judge(tmp_path, monkeypatch):
    import consul.pipeline.cli as cli

    monkeypatch.setattr(cli, "ChatClient", ClientRecorder)
    ClientRecorder.made = []
    pipeline = make_pipeline(tmp_path, judge_reasoning_effort="high")
    pipeline._generate_work(pipeline.s.data_dir)(None, lambda: False)
    pipeline._judge_work(pipeline.s.data_dir)(None, lambda: False)
    generate, judge = ClientRecorder.made
    assert generate == {"reasoning_effort": "medium", "slot": 0}
    assert judge["model"] == "Qwen3.8-27B-UD-Q4_K_XL"  # named from the GGUF it serves
    assert judge["reasoning_effort"] == "high"
    assert judge["template_kwargs"] == {"enable_thinking": False}


def test_judge_thinking_needs_the_flag_and_obeys_no_judge_thinking(tmp_path):
    for name in "abc":
        (tmp_path / name).mkdir()
    on = make_pipeline(tmp_path / "a")
    on.judge_thinking = True  # --judge-thinking
    assert on.thinks()
    off = make_pipeline(tmp_path / "b", judge_thinking=False)  # Emperor's --no-judge-thinking
    off.judge_thinking = True
    assert not off.thinks()
    assert not make_pipeline(tmp_path / "c").thinks()  # Consul's default: no thinking


def test_the_judges_thinking_is_cut_before_its_verdict_is_parsed(tmp_path):
    task = generate_tasks(1, 7)[0]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prompts.jsonl").write_text(dumps(task) + "\n", encoding="utf-8")
    critique = reference_critique(task)
    sample = data.make_sample(task, critique, "Weigh step 3.")
    ShardWriter(data_dir / "traces", "traces").write(sample)
    verdict = {"scores": dict.fromkeys(prompts.JUDGE_CRITERIA, 5), "pass": True, "reason": "ok"}

    class Judge:
        def chat(self, messages, **kwargs):
            return {"content": '<think>{"scores": 1} is not it.</think>' + json.dumps(verdict)}

    counts = data.run_judging(data_dir, Judge())
    (judged,) = iter_records(data_dir / "judged")
    assert counts["judged"] == 1 and judged["meta"]["judge"]["model"] == "Qwen3.8-27B"
    assert judged["meta"]["judge"]["pass"] is True


def test_judge_temperature_follows_thinking(tmp_path):
    task = generate_tasks(1, 7)[0]
    verdict = {"scores": dict.fromkeys(prompts.JUDGE_CRITERIA, 5), "pass": True, "reason": "ok"}

    class Judge:
        def __init__(self, thinking):
            self.template_kwargs = {"enable_thinking": thinking}
            self.seen = []

        def chat(self, messages, **kwargs):
            self.seen.append(kwargs["temperature"])
            return {"content": json.dumps(verdict)}

    for thinking, temperature in ((False, 0.2), (True, 0.6)):
        data_dir = tmp_path / str(thinking)
        data_dir.mkdir()
        (data_dir / "prompts.jsonl").write_text(dumps(task) + "\n", encoding="utf-8")
        sample = data.make_sample(task, reference_critique(task), "Weigh step 3.")
        ShardWriter(data_dir / "traces", "traces").write(sample)
        judge = Judge(thinking)
        data.run_judging(data_dir, judge)
        assert judge.seen == [temperature]


def test_model_names_come_from_the_served_gguf():
    assert data.model_name("D:/m/Qwen3.8-27B-UD-Q4_K_XL.gguf") == "Qwen3.8-27B-UD-Q4_K_XL"
    assert data.model_name(r"F:\m\gpt-oss-120b-F16.gguf") == "gpt-oss-120b-F16"
    assert data.model_name("Qwen3.8-27B") == "Qwen3.8-27B"


def test_teacher_reply_without_reasoning_is_rejected():
    task = generate_tasks(1, 7)[0]

    class Teacher:
        def __init__(self, reasoning):
            self.reasoning = reasoning

        def chat(self, messages, **kwargs):
            critique = json.dumps(reference_critique(task))
            return {"content": critique, "reasoning_content": self.reasoning}

    for empty in (None, "", "  \n"):
        assert data.teach_one(task, Teacher(empty)) == (None, "empty_reasoning")
    sample, reason = data.teach_one(task, Teacher(" Weigh step 3. "))
    assert reason is None
    assert sample["messages"][-1]["reasoning_content"] == "Weigh step 3."


def test_thinking_is_stripped_before_the_answer_is_parsed():
    critique = {"verdict": "accept", "summary": "ok", "findings": []}
    answer = json.dumps(critique)
    thought = 'Maybe {"verdict": "reject"} fits? No: {nested {braces}} are only thoughts.'
    assert extract_answer(f"<think>\n{thought}\n</think>\n\n{answer}") == critique
    # The Thinking template opens the block in the prompt: only the closing tag comes back.
    assert extract_answer(f"{thought}\n</think>\n{answer}") == critique
    assert extract_answer(answer) == critique  # no thinking at all
    # Cut off mid-thought: no answer is left, so nothing from the reasoning is scored.
    assert strip_thinking(f"<think>\n{thought}") == ""
    with pytest.raises(CritiqueParseError):
        extract_answer(f"<think>\n{thought}")
    with pytest.raises(CritiqueParseError):
        extract_answer(f"{thought}\n</think>\n")


def test_eval_scores_a_thinking_models_answer(tmp_path):
    from consul.eval.report import run_task

    task = generate_tasks(1, 3)[0]
    reply = "<think>{not json} weigh it</think>" + json.dumps(reference_critique(task))
    score, error = run_task(task, lambda _: reply)
    assert error is None and score.format_ok and not score.missed


def test_a_clean_report_cites_a_provided_source_that_supports_its_claim():
    for seed in range(40):
        task = make_task(seed, kind="report", clean=True)
        claim3 = next(line for line in task.artifact.splitlines() if "vendor guidance" in line)
        evidence = task.artifact.splitlines()[task.artifact.splitlines().index(claim3) + 1]
        cited = evidence.split("Source: ")[1].split(";")[0].strip()
        source = next(s for s in task.sources if s.url == cited)
        assert "guidance" in source.title.lower() and source.text.startswith("Vendor guidance")
        assert "staging data set" not in task.artifact


def test_every_seeded_diff_defect_has_a_line():
    for seed in range(300):
        task = make_task(seed, kind="diff", clean=False)
        for defect in task.defects:
            assert "LNone" not in defect.where, (seed, task.family)


def test_a_critique_that_misses_the_ground_truth_is_drawn_again():
    task = generate_tasks(40, 1)
    task = next(t for t in task if not t.clean)

    class LazyOnce(StubTeacher):
        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                empty = {"verdict": "accept", "summary": "ok", "findings": []}
                return {"content": json.dumps(empty), "reasoning_content": "Looks fine."}
            self.calls -= 1
            return super().chat(messages, **kwargs)

    teacher = LazyOnce([task])
    sample, reason = data.teach_one(task, teacher)
    assert reason is None and sample["messages"][-1]["content"]
    always_lazy = LazyOnce([task])
    always_lazy.calls = -10  # every draw is the lazy one
    always_lazy.chat = lambda messages, **kw: {
        "content": json.dumps({"verdict": "accept", "summary": "ok", "findings": []}),
        "reasoning_content": "Looks fine.",
    }
    assert data.teach_one(task, always_lazy) == (None, "missed_defect")


def test_an_unparsable_output_costs_one_draw_not_the_task():
    from emperor.data.teacher import TeacherOutputError

    task = next(t for t in generate_tasks(40, 1) if not t.clean)

    class BadOnce(StubTeacher):
        def chat(self, messages, **kwargs):
            if not getattr(self, "failed", False):
                self.failed = True
                raise TeacherOutputError("server rejected the output 3 times")
            return super().chat(messages, **kwargs)

    sample, reason = data.teach_one(task, BadOnce([task]))
    assert reason is None and sample is not None


def test_research_plans_do_not_offer_an_embedded_store():
    for seed in range(200):
        task = make_task(seed, kind="plan")
        if task.family == "research":
            assert "SQLite" not in task.artifact
