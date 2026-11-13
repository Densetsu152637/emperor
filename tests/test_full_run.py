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
