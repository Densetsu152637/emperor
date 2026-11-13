import json

import pytest

from emperor.eval.contract import estimate_tokens
from emperor.eval.tasks import TaskError, load_suite, load_tasks, render_prompt, task_from_dict


@pytest.mark.parametrize("suite", ["heldout", "e2e", "longctx"])
def test_bundled_suites_load(suite):
    tasks = load_suite(suite)
    assert tasks
    assert all(t.reference for t in tasks), "every seed task carries reference turns"


def test_heldout_seed_covers_every_expectation():
    tasks = load_suite("heldout")
    assert 10 <= len(tasks) <= 20
    for key in ("routing", "brief", "plan", "gaps", "success"):
        assert any(key in t.expect for t in tasks), key
    assert any(t.haystack for t in tasks)
    assert any(t.secrets for t in tasks)
    assert {t.task_class for t in tasks} >= {"menial", "hard"}
    assert any(t.expect.get("routing", {}).get("mode") == "chunks" for t in tasks)


def test_task_validation():
    with pytest.raises(TaskError, match="unknown keys"):
        task_from_dict({"id": "x", "prompt": "p", "promt": "typo"}, "s")
    with pytest.raises(TaskError, match="unknown expect keys"):
        task_from_dict({"id": "x", "prompt": "p", "expect": {"routng": {}}}, "s")
    with pytest.raises(TaskError, match="not in the contract"):
        task_from_dict({"id": "x", "prompt": "p", "tools": ["shell"]}, "s")
    with pytest.raises(TaskError, match="prompt or a haystack"):
        task_from_dict({"id": "x"}, "s")


def test_jsonl_and_duplicates(tmp_path):
    file = tmp_path / "tasks.jsonl"
    file.write_text(
        json.dumps({"id": "a", "prompt": "p"}) + "\n\n" + json.dumps({"id": "b", "prompt": "q"}),
        encoding="utf-8",
    )
    assert [t.id for t in load_tasks(file, "custom")] == ["a", "b"]
    (tmp_path / "dup.json").write_text(json.dumps({"id": "a", "prompt": "r"}), encoding="utf-8")
    with pytest.raises(TaskError, match="duplicate"):
        load_tasks(tmp_path, "custom")


def test_haystack_scales_with_the_tier():
    task = load_suite("longctx")[0]
    small, large = render_prompt(task, 32_768), render_prompt(task, 131_072)
    assert 0.8 * 32_768 < estimate_tokens(small) < 0.9 * 32_768
    assert 0.8 * 131_072 < estimate_tokens(large) < 0.9 * 131_072
    for value in task.needles:
        assert value in small and value in large
    assert render_prompt(task, 32_768) == small, "haystacks are deterministic"
    # The 10 % needle comes before the 90 % one.
    assert small.index("8836") < small.index("Aurelia Vento")
