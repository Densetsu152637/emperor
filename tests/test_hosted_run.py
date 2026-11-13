"""The hosted runner end to end, with scripted clients in place of the APIs."""

import json
from pathlib import Path

from conftest import BRIEF

from emperor.data.hosted import BudgetExceeded, usage_tokens
from emperor.data.shards import iter_records
from emperor.pipeline import hosted

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = "concision delegation tool_use research safety style planning honesty".split()
VERDICT = {"scores": dict.fromkeys(CRITERIA, 5), "pass": True, "reason": "ok"}


def call(name, arguments):
    return {
        "content": "",
        "tool_calls": [{"id": "x", "function": {"name": name, "arguments": json.dumps(arguments)}}],
    }


class Fake:
    """Teacher, simulator or judge by role; records a cost per request in the ledger."""

    def __init__(self, role, ledger, budget_after=None):
        self.role, self.ledger, self.model = role, ledger, f"{role}-model"
        self.served_model = f"{role}-served"
        self.template_kwargs = {"enable_thinking": True}
        self.budget_after = budget_after
        self.requests = 0

    def chat(self, messages, *, tools=None, **kwargs):
        self.requests += 1
        if self.budget_after is not None and self.requests > self.budget_after:
            raise BudgetExceeded("deepseek: spend $25.00 reached the $25.00 cap")
        provider = "kimi" if self.role == "judge" else "deepseek"
        tokens = usage_tokens({"prompt_tokens": 100, "completion_tokens": 10})
        self.ledger.record(provider, self.role, self.model, tokens, 0.001)
        if self.role == "judge":
            return {"content": json.dumps(VERDICT), "reasoning_content": "Weigh it."}
        if self.role == "simulator":
            return {"content": "Report: done. Evidence: all checks pass."}
        if not tools:  # the ping
            return {"content": "OK", "reasoning_content": "Say OK."}
        done = sum(1 for m in messages if m["role"] == "tool")
        script = [
            call("spawn_agent", {"brief": BRIEF, "tier": "standard", "tags": ["code"]}),
            call("await_tasks", {}),
            {"content": "Done; the subagent's checks pass."},
        ]
        return {**script[min(done, 2)], "reasoning_content": "Plan it."}

    def close(self):
        pass


def run(tmp_path, monkeypatch, *extra, budget_after=None):
    made = {}

    def make_clients(config, ledger, roles, should_stop=lambda: False):
        for role in roles:
            made[role] = Fake(role, ledger, budget_after if role == "teacher" else None)
        return {role: made[role] for role in roles}

    monkeypatch.setattr(hosted, "make_clients", make_clients)
    argv = [
        "--root", str(tmp_path), "--config", str(ROOT / "configs" / "hosted.toml"),
        "--data-dir", "data", "--run-dir", "runs/r1", "--prompts", "4", "--workers", "2",
        *extra,
    ]  # fmt: skip
    code = hosted.main(argv)
    status = json.loads((tmp_path / "runs" / "r1" / "status.json").read_text(encoding="utf-8"))
    return code, status, made


def test_a_hosted_run_generates_judges_and_reports_spend(tmp_path, monkeypatch):
    code, status, _ = run(tmp_path, monkeypatch)
    assert code == 0 and status["state"] == "done"
    assert (
        status["main"]["prompts"] == 4 and status["main"]["judged"] == status["main"]["generated"]
    )
    assert status["served_models"]["judge"] == "judge-served"
    judged = list(iter_records(tmp_path / "data" / "judged"))
    assert judged and all(s["meta"]["judge"]["model"] == "judge-served" for s in judged)
    assert all(s["meta"]["teacher"] == "teacher-served" for s in judged)
    assert all(s["meta"]["host"] == "DeepSeek API" for s in judged)
    assert status["spend"]["deepseek"]["requests"] > 0 and status["spend_usd"] > 0
    assert status["usd_per_task"] and status["projected_usd_main_pass"]
    assert (tmp_path / "data" / "spend.jsonl").exists()
    assert (tmp_path / "data" / "report.json").exists()


def test_a_budget_stop_is_resumable(tmp_path, monkeypatch):
    code, status, _ = run(tmp_path, monkeypatch, budget_after=3)
    assert code == 3 and status["state"] == "budget"
    code, status, _ = run(tmp_path, monkeypatch)  # the cap was raised: resume
    assert (
        code == 0
        and status["main"]["generated"] + sum(status["main"]["generation_rejects"].values()) == 4
    )


def test_stop_file_stops_before_the_next_phase(tmp_path, monkeypatch):
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "STOP").write_text("", encoding="utf-8")
    code, status, made = run(tmp_path, monkeypatch)
    assert code == 0 and status["state"] == "stopped" and not made


def task_texts(fake):
    """The user tasks the teacher worked on (not the ping or the reasoning preflight)."""
    from emperor.data.generate import _PREFLIGHT_USER

    return {t for t in fake.tasks if t != _PREFLIGHT_USER}


class CountingFake(Fake):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = set()

    def chat(self, messages, *, tools=None, **kwargs):
        if tools and self.role == "teacher":
            self.tasks.add(messages[1]["content"])
        return super().chat(messages, tools=tools, **kwargs)


def counting_run(tmp_path, monkeypatch, *extra, config=None):
    made = {}

    def make_clients(cfg, ledger, roles, should_stop=lambda: False):
        for role in roles:
            made[role] = CountingFake(role, ledger)
        return {role: made[role] for role in roles}

    monkeypatch.setattr(hosted, "make_clients", make_clients)
    argv = [
        "--root", str(tmp_path), "--config", str(config or ROOT / "configs" / "hosted.toml"),
        "--data-dir", "data", "--run-dir", "runs/full", "--prompts", "6", "--workers", "2",
        "--only", "prompts", "--only", "generate", "--only", "judge", "--only", "filter",
        *extra,
    ]  # fmt: skip
    assert hosted.main(argv) == 0
    return made


def test_a_full_run_resumes_after_pilots_counts_their_samples_and_skips_their_tasks(
    tmp_path, monkeypatch
):
    pilot = counting_run(tmp_path, monkeypatch, "--pilot", "1", "--limit", "2")
    assert len(task_texts(pilot["teacher"])) == 2
    full = counting_run(tmp_path, monkeypatch)
    assert len(task_texts(full["teacher"])) == 4  # only the tasks the pilot did not do
    assert not task_texts(pilot["teacher"]) & task_texts(full["teacher"])
    judged = list(iter_records(tmp_path / "data" / "judged"))
    assert len(judged) == 6 and len({s["id"] for s in judged}) == 6  # no duplicates
    pilots = [s for s in judged if s["meta"].get("pilot") == 1]
    assert len(pilots) == 2
    for sample in judged:
        meta = sample["meta"]
        assert {"pipeline_commit", "config_hash", "guidance", "models"} <= set(meta)
        assert set(meta["models"]) == {"teacher", "simulator", "judge"}
    final = list(iter_records(tmp_path / "data" / "final"))
    assert len(final) == 6 and sum(1 for s in final if s["meta"].get("pilot") == 1) == 2


def test_an_excluded_pilot_is_generated_again_and_never_reaches_the_final_set(
    tmp_path, monkeypatch
):
    counting_run(tmp_path, monkeypatch, "--pilot", "1", "--limit", "2")
    full = counting_run(
        tmp_path, monkeypatch, "--exclude-pilot", "1=teacher prompt bug fixed in pilot 2"
    )
    assert len(task_texts(full["teacher"])) == 6
    exclusions = json.loads((tmp_path / "data" / "excluded-pilots.json").read_text())
    assert exclusions == {"1": "teacher prompt bug fixed in pilot 2"}
    final = list(iter_records(tmp_path / "data" / "final"))
    assert len(final) == 6 and not any(s["meta"].get("pilot") == 1 for s in final)
    report = (
        json.loads((tmp_path / "data" / "report.json").read_text())
        if (tmp_path / "data" / "report.json").exists()
        else None
    )
    assert report is None or report["excluded_pilots"] == exclusions


def test_changed_judge_settings_rejudge_pilot_samples_without_the_teacher(tmp_path, monkeypatch):
    counting_run(tmp_path, monkeypatch, "--pilot", "1", "--limit", "2")
    text = (ROOT / "configs" / "hosted.toml").read_text(encoding="utf-8")
    judge = text.index("[roles.judge]")
    low = tmp_path / "low.toml"
    other = text[judge:].replace('reasoning_effort = "low"', 'reasoning_effort = "high"', 1)
    assert other != text[judge:]
    low.write_text(text[:judge] + other, encoding="utf-8")
    again = counting_run(tmp_path, monkeypatch, "--pilot", "1", "--limit", "0", config=low)
    assert not task_texts(again["teacher"])  # no teacher tokens
    judged = list(iter_records(tmp_path / "data" / "judged"))
    assert len(judged) == 4  # two verdicts per pilot sample; the latest counts
    final = list(iter_records(tmp_path / "data" / "final"))
    assert len(final) == 2


def test_status_shows_the_off_peak_wait_and_when_it_ends(tmp_path):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    args = hosted.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--config",
            str(ROOT / "configs" / "hosted.toml"),
            "--data-dir",
            "data",
            "--run-dir",
            "runs/r1",
        ]  # fmt: skip
    )
    run = hosted.HostedRun(args, hosted.configure(args))
    resume = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
    run.clients = {"teacher": SimpleNamespace(waiting_until=resume)}
    run.phase = "generate"
    run.write_status("running")
    status = json.loads((tmp_path / "runs" / "r1" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "waiting_offpeak"
    assert status["resume_at_utc"] == "2026-11-13T10:00+00:00"


def test_peak_hours_need_an_explicit_flag():
    args = hosted.build_parser().parse_args(["--config", str(ROOT / "configs" / "hosted.toml")])
    assert hosted.configure(args).providers["deepseek"].offpeak_only
    args = hosted.build_parser().parse_args(
        ["--config", str(ROOT / "configs" / "hosted.toml"), "--allow-peak"]
    )
    assert not hosted.configure(args).providers["deepseek"].offpeak_only


def test_a_provider_outage_is_waited_out_and_the_phase_resumed(tmp_path):
    args = hosted.build_parser().parse_args(
        ["--root", str(tmp_path), "--config", str(ROOT / "configs" / "hosted.toml"),
         "--data-dir", "data", "--run-dir", "runs/r1", "--outage-waits", "2"]
    )  # fmt: skip
    run = hosted.HostedRun(args, hosted.configure(args))
    run._sleep = lambda s: None
    calls = []

    def phase_judge():
        calls.append(1)
        if len(calls) < 3:
            return {"stopped": True, "error": "kimi: no answer after 30 tries (HTTP 429)"}
        return {"stopped": False, "judged": 5}

    run.phase_judge = phase_judge
    assert run.run_phase("judge") == {"stopped": False, "judged": 5} and len(calls) == 3
    calls.clear()
    args.outage_waits = 0
    assert run.run_phase("judge")["stopped"]  # no waits allowed: the run stops, resumable


def test_budgets_can_be_set_per_run():
    args = hosted.build_parser().parse_args(
        ["--config", str(ROOT / "configs" / "hosted.toml"), "--budget", "deepseek=120",
         "--budget", "kimi=40"]
    )  # fmt: skip
    config = hosted.configure(args)
    assert config.providers["deepseek"].budget_usd == 120.0
    assert config.providers["kimi"].budget_usd == 40.0


def test_one_run_per_data_dir_and_a_dead_holder_is_taken_over(tmp_path):
    import os

    import pytest

    lock = tmp_path / ".hosted.lock"
    lock.write_text(str(os.getpid() + 999999), encoding="utf-8")  # no such process
    assert hosted.take_lock(lock) == lock and lock.read_text() == str(os.getpid())
    lock.write_text("4", encoding="utf-8")  # pid 4 (System on Windows) is alive
    if hosted.pid_alive(4):
        with pytest.raises(SystemExit):
            hosted.take_lock(lock)


def test_a_torn_state_file_is_rebuilt_and_stale_rejects_do_not_count(tmp_path, monkeypatch):
    from emperor.data.shards import ShardWriter

    data = tmp_path / "counts"
    ShardWriter(data / "traces", "traces").write({"id": "p-1", "meta": {}, "messages": []})
    ShardWriter(data / "rejects", "rejects").write({"id": "p-1", "reason": "too_long"})
    ShardWriter(data / "rejects", "rejects").write({"id": "p-2", "reason": "too_long"})
    counts = hosted.count_dir(data)
    assert counts["generated"] == 1 and counts["generation_rejects"] == {"too_long": 1}
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "state.json").write_text('{"phases": {', encoding="utf-8")
    code, status, _ = run(tmp_path, monkeypatch)
    assert code == 0 and status["state"] == "done"
