import json

import pytest

from emperor.eval.cli import METRICS, main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--dry-run" in capsys.readouterr().out


def test_dry_run_reports_every_metric(capsys):
    assert main(["--model", "llamacpp:emperor", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "llamacpp:emperor"
    assert set(report["metrics"]) == set(METRICS)
    assert all(entry["score"] is None for entry in report["metrics"].values())


def test_dry_run_limits_metrics(capsys):
    assert main(["--dry-run", "--metric", "leakage"]) == 0
    assert list(json.loads(capsys.readouterr().out)["metrics"]) == ["leakage"]


def test_real_run_is_refused_until_implemented():
    assert main([]) == 2
