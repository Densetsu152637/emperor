"""Consul's hosted run end to end, with scripted clients in place of the APIs."""

import json
from pathlib import Path

from emperor.data.hosted import usage_tokens
from emperor.data.shards import iter_records
from emperor.pipeline import hosted as base

from consul import prompts
from consul.pipeline import hosted
from consul.score import reference_critique
from consul.tasks import CritiqueTask

ROOT = Path(__file__).resolve().parents[1]


class Fake:
    def __init__(self, role, ledger, tasks):
        self.role, self.ledger, self.model = role, ledger, f"{role}-model"
        self.served_model = f"{role}-served"
        self.template_kwargs = {"enable_thinking": True}
        self.tasks = tasks
        self.kwargs = []

    def chat(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        provider = "moonshot" if self.role == "judge" else "deepseek"
        self.ledger.record(provider, self.role, self.model, usage_tokens({}), 0.002)
        user = messages[-1]["content"]
        if self.role == "judge":
            scores = dict.fromkeys(prompts.JUDGE_CRITERIA, 5)
            return {"content": json.dumps({"scores": scores, "pass": True, "reason": "ok"})}
        task = self.tasks().get(user)
        if task is None:  # the ping
            return {"content": "OK", "reasoning_content": "Say OK."}
        critique = reference_critique(task)
        return {"content": json.dumps(critique), "reasoning_content": f"Check {task.id}."}

    def close(self):
        pass


def test_consul_hosted_run_writes_critiques_verdicts_and_spend(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    def tasks():
        rows = (data_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        loaded = [CritiqueTask.from_dict(json.loads(r)) for r in rows if r]
        return {prompts.user_message(t): t for t in loaded}

    made = {}

    def make_clients(config, ledger, roles, should_stop=lambda: False):
        for role in roles:
            made[role] = Fake(role, ledger, tasks)
        return {role: made[role] for role in roles}

    monkeypatch.setattr(base, "make_clients", make_clients)
    code = hosted.main(
        ["--root", str(tmp_path), "--config", str(ROOT / "configs" / "hosted.toml"),
         "--data-dir", "data", "--run-dir", "runs/r1", "--prompts", "6", "--eval-tasks", "4",
         "--workers", "3"]
    )  # fmt: skip
    status = json.loads((tmp_path / "runs" / "r1" / "status.json").read_text(encoding="utf-8"))
    assert code == 0 and status["state"] == "done" and status["model"] == "consul"
    judged = list(iter_records(data_dir / "judged"))
    assert len(judged) == 6 and all(s["meta"]["judge"]["model"] == "judge-served" for s in judged)
    assert all(s["meta"]["teacher"] == "teacher-served" for s in judged)
    assert all(s["meta"]["host"] == "DeepSeek API" for s in judged)
    assert status["spend"]["deepseek"]["requests"] >= 6 and status["usd_per_task"]
    assert "simulator" not in made  # critiques make no tool calls


def test_consul_resumes_after_a_pilot_and_skips_its_tasks(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    seen: list[set] = []

    def tasks():
        rows = (data_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        loaded = [CritiqueTask.from_dict(json.loads(r)) for r in rows if r]
        return {prompts.user_message(t): t for t in loaded}

    class Counting(Fake):
        def chat(self, messages, **kwargs):
            if self.role == "teacher" and messages[-1]["content"] in tasks():
                seen[-1].add(messages[-1]["content"])
            return super().chat(messages, **kwargs)

    def make_clients(config, ledger, roles, should_stop=lambda: False):
        return {role: Counting(role, ledger, tasks) for role in roles}

    monkeypatch.setattr(base, "make_clients", make_clients)

    def run(*extra):
        seen.append(set())
        code = hosted.main(
            ["--root", str(tmp_path), "--config", str(ROOT / "configs" / "hosted.toml"),
             "--data-dir", "data", "--run-dir", "runs/full", "--prompts", "6",
             "--eval-tasks", "4", "--workers", "2", *extra]
        )  # fmt: skip
        assert code == 0

    run("--pilot", "1", "--limit", "2")
    run()
    assert len(seen[0]) == 2 and len(seen[1]) == 4 and not seen[0] & seen[1]
    judged = list(iter_records(data_dir / "judged"))
    assert len({s["id"] for s in judged}) == len(judged) == 6
    assert sum(1 for s in judged if s["meta"].get("pilot") == 1) == 2
    assert all("simulator" not in s["meta"]["models"] for s in judged)
    assert all({"pipeline_commit", "config_hash", "guidance"} <= set(s["meta"]) for s in judged)
    assert len(list(iter_records(data_dir / "final"))) == 6
