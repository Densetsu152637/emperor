import pytest

from emperor.eval.contract import validate
from emperor.eval.imperiumd import BackendUnavailable, discover
from emperor.eval.models import ScriptedModel
from emperor.eval.report import CeilingExceeded, CostMeter, report_schema, run_suite
from emperor.eval.tasks import load_suite, task_from_dict


def suite_report(tasks, metrics=None, **overrides):
    options = dict(
        model="test:model",
        suite="heldout",
        tier="64k",
        context_tokens=65_536,
        metrics=metrics or ["routing", "leakage", "plan"],
        backend="replay",
        dry_run=True,
        goals="G2",
        meter=CostMeter(0.0),
    )
    options.update(overrides)
    return run_suite(tasks, lambda task: ScriptedModel(task.reference), **options)


def test_report_matches_its_schema():
    report = suite_report(load_suite("heldout"), metrics=None)
    assert validate(report, report_schema()) == []
    assert report["schema_version"] == 1
    assert report["contract"]["imperium_commit"]
    assert report["timestamp"].endswith("+00:00")
    assert report["tasks_run"] == len(report["tasks"]) == len(load_suite("heldout"))


def test_metric_without_tasks_is_reported_with_a_note():
    tasks = [task_from_dict({"id": "x", "prompt": "hi", "reference": [{"content": "hello"}]}, "s")]
    report = suite_report(tasks)
    assert report["metrics"]["plan"]["score"] is None
    assert report["metrics"]["plan"]["n"] == 0
    assert "no task" in report["metrics"]["plan"]["note"]
    assert report["tasks"][0]["metrics"]["plan"] is None
    assert validate(report, report_schema()) == []


def test_overflow_is_counted_as_a_failed_task():
    report = suite_report(load_suite("longctx"), metrics=["retrieval"], context_tokens=4096)
    assert report["tasks_failed"] == 1 and report["overflows"] == 1
    assert report["metrics"]["retrieval"]["score"] == 0.0


def test_cost_meter_enforces_the_ceiling():
    meter = CostMeter(1.0)
    meter.add(0.6)
    with pytest.raises(CeilingExceeded):
        meter.add(0.6)
    CostMeter(0.0).add(0.0)  # free work is always allowed
    with pytest.raises(CeilingExceeded):
        CostMeter(0.0).add(0.01)


def test_imperiumd_discovery(tmp_path):
    with pytest.raises(BackendUnavailable):
        discover(tmp_path)
    (tmp_path / "daemon.json").write_text('{"url": "ws://127.0.0.1:1/rpc"}', encoding="utf-8")
    assert discover(tmp_path)["url"] == "ws://127.0.0.1:1/rpc"
