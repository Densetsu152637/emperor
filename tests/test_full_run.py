"""The end-to-end scheduler script: its derived training config and stage bookkeeping."""

import importlib.util
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_run", ROOT / "scripts" / "full_run.py")
full_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_run)


def test_the_training_config_points_at_the_hosted_data_with_length_from_the_data(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(full_run, "HOME", tmp_path)
    for repo in (ROOT, ROOT.parent / "consul-hosted"):
        if not (repo / "configs" / "sft.toml").exists():
            continue
        original = tomllib.loads((repo / "configs" / "sft.toml").read_text(encoding="utf-8"))
        derived = tomllib.loads(full_run.train_config("x", repo).read_text(encoding="utf-8"))
        assert derived["data"]["path"] == "data/hosted-1/final/final.jsonl"
        assert derived["data"]["max_length_auto"] is True
        assert derived["train"]["offload_activations"] is True
        for section in original:  # everything else is unchanged
            for key, value in original[section].items():
                if (section, key) != ("data", "path"):
                    assert derived[section][key] == value


def test_stage_completion_is_read_from_the_run_state(tmp_path):
    repo = tmp_path / "emperor"
    state = repo / "runs" / "full-hosted" / "state.json"
    state.parent.mkdir(parents=True)
    phases = {p: {"status": "done"} for p in ("prompts", "generate", "judge", "filter", "report")}
    state.write_text(json.dumps({"phases": phases}), encoding="utf-8")
    assert not full_run.hosted_done(repo)  # Emperor also needs its retry rounds
    phases["retry"] = {"status": "done"}
    state.write_text(json.dumps({"phases": phases}), encoding="utf-8")
    assert full_run.hosted_done(repo)
    assert not full_run.build_done(repo)


def test_budgets_are_one_pool_per_provider_reread_each_time(tmp_path, monkeypatch):
    monkeypatch.setattr(full_run, "BUDGETS", tmp_path / "budgets.json")
    emperor, consul = tmp_path / "emperor", tmp_path / "consul"
    ledger = consul / "data" / "hosted-1" / "spend.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = [{"provider": "deepseek", "usd": 20.0}, {"provider": "kimi", "usd": 7.5}]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    flags = full_run.stage_budgets(emperor, [consul])
    assert flags == ["--budget", "deepseek=75.00", "--budget", "kimi=29.50"]  # 95-20, 37-7.5
    (tmp_path / "budgets.json").write_text('{"deepseek": 120}', encoding="utf-8")
    assert full_run.stage_budgets(emperor, [consul])[1] == "deepseek=100.00"


def test_pilot_data_is_seeded_without_the_held_out_set(tmp_path):
    class Args:
        pilot_data_root = str(tmp_path)

    source = tmp_path / "consul-hosted" / "data" / "hosted-1"
    (source / "eval").mkdir(parents=True)
    (source / "eval" / "heldout.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
    repo = tmp_path / "consul"
    full_run.seed_pilot_data(repo, Args())
    assert (repo / "data" / "hosted-1" / "prompts.jsonl").exists()
    assert not (repo / "data" / "hosted-1" / "eval").exists()
