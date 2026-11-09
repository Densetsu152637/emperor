import json

import pytest

from emperor.eval.cli import METRICS, main


def run(capsys, *argv) -> tuple[int, dict | None, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out.strip() else None, captured.err


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--dry-run" in capsys.readouterr().out


def test_dry_run_scores_every_metric(capsys):
    code, report, _ = run(capsys, "--model", "llamacpp:emperor", "--dry-run")
    assert code == 0
    assert report["model"] == "llamacpp:emperor"
    assert set(report["metrics"]) == set(METRICS)
    for name, entry in report["metrics"].items():
        assert entry["goal"] == METRICS[name]
        assert isinstance(entry["score"], float | int), name
        assert entry["n"] > 0, name


def test_metric_goals_cover_the_stage_2b_goals():
    assert set(METRICS.values()) == {"G1", "G2", "G3", "G4", "G5", "G6", "G10", "G11"}


def test_dry_run_limits_metrics(capsys):
    _, report, _ = run(capsys, "--dry-run", "--metric", "leakage")
    assert list(report["metrics"]) == ["leakage"]
    assert all(list(t["metrics"]) == ["leakage"] for t in report["tasks"])


def test_dry_run_records_context_tier(capsys):
    _, report, _ = run(capsys, "--dry-run", "--context-tier", "256k")
    assert report["context_tier"] == "256k"
    assert report["context_tokens"] == 262_144


def test_e2e_refuses_without_an_api_ceiling(capsys, monkeypatch):
    monkeypatch.delenv("EMPEROR_EVAL_API_CEILING", raising=False)
    code, report, err = run(capsys, "--suite", "e2e", "--model", "x")
    assert code == 2 and report is None
    assert "api-ceiling" in err


def test_e2e_with_a_ceiling_still_needs_the_daemon(capsys):
    code, _, err = run(capsys, "--suite", "e2e", "--model", "x", "--api-ceiling", "5")
    assert code == 2
    assert "imperiumd" in err


def test_e2e_dry_run_is_free(capsys):
    code, report, _ = run(capsys, "--suite", "e2e", "--dry-run")
    assert code == 0
    assert report["cost"]["usd"] == 0
    assert report["metrics"]["task_success"]["score"] == 1.0


def test_imperiumd_backend_is_a_clear_stub(capsys):
    code, _, err = run(capsys, "--backend", "imperiumd", "--model", "x")
    assert code == 2
    assert "not implemented" in err


def test_model_required_without_dry_run(capsys):
    code, _, err = run(capsys)
    assert code == 2
    assert "--model" in err


def test_output_file(tmp_path, capsys):
    out = tmp_path / "results" / "report.json"
    assert main(["--dry-run", "--output", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text(encoding="utf-8"))["suite"] == "heldout"


def test_bad_task_path(capsys, tmp_path):
    code, _, err = run(capsys, "--dry-run", "--tasks", str(tmp_path))
    assert code == 2
    assert "no tasks" in err
