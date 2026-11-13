"""Running held-out tasks through a model and building the JSON report.

The report has the same shape as Emperor's (`metrics` -> goal, n, score, target, passed), so
Emperor's model-card generator renders it unchanged.

Metrics: `defect_recall` and `false_findings` (C1), `format` (C3), `citations` (C2, simple: a
cited URL must be one that appears in the context), and `loop` (C4, not implemented yet).
"""

from collections.abc import Callable
from datetime import UTC, datetime

from emperor.eval.models import ModelError

from consul.schema import CritiqueParseError, extract_answer
from consul.score import CritiqueScore, score_critique
from consul.tasks import CritiqueTask

SCHEMA_VERSION = 1
TARGETS = {"defect_recall": 0.8, "false_findings": 0.15, "format": 0.98, "citations": 0.9}


def run_task(
    task: CritiqueTask, answer: Callable[[CritiqueTask], str]
) -> tuple[CritiqueScore, str | None]:
    """Score one task; returns the score and a model error message (None when it answered)."""
    try:
        text = answer(task)
    except ModelError as error:
        return score_critique(task, None), str(error)
    try:
        critique = extract_answer(text)
    except CritiqueParseError:
        critique = None
    return score_critique(task, critique), None


def aggregate(scores: list[CritiqueScore]) -> dict:
    defects = sum(s.defects for s in scores)
    found = sum(len(s.found) for s in scores)
    blocking = sum(s.blocking_findings for s in scores)
    false = sum(len(s.false_findings) for s in scores)
    cited = sum(s.cited for s in scores)
    bad = sum(len(s.bad_citations) for s in scores)
    n_defect_tasks = sum(1 for s in scores if s.defects)

    def metric(goal, description, n, score, passed):
        return {"goal": goal, "description": description, "n": n, "score": score, "passed": passed}

    recall = found / defects if defects else None
    false_rate = false / blocking if blocking else None
    fmt = sum(s.format_ok for s in scores) / len(scores) if scores else None
    cite = 1 - bad / cited if cited else None
    out = {
        "defect_recall": metric(
            "C1", "Share of seeded acceptance-changing defects found", n_defect_tasks, recall, None
        ),
        "false_findings": metric(
            "C1",
            "Share of acceptance-changing findings that match no seeded defect (lower is better)",
            blocking,
            false_rate,
            None,
        ),
        "format": metric(
            "C3", "Share of answers that validate against the schema", len(scores), fmt, None
        ),
        "citations": metric(
            "C2", "Share of cited URLs that appear in the provided context", cited, cite, None
        ),
        "loop": {
            "goal": "C4",
            "description": "Delta-review convergence (TODO: loop replays are not implemented)",
            "n": 0,
            "score": None,
            "target": None,
            "passed": None,
            "note": "TODO",
        },
    }
    for name, target in TARGETS.items():
        entry = out[name]
        entry["target"] = target
        if entry["score"] is None:
            entry["passed"] = None
        elif name == "false_findings":
            entry["passed"] = entry["score"] <= target
        else:
            entry["passed"] = entry["score"] >= target
    return out


def run_suite(
    tasks: list[CritiqueTask],
    answer: Callable[[CritiqueTask], str],
    *,
    model: str | None,
    backend: str,
    dry_run: bool,
    context_tier: str,
    context_tokens: int,
    overlap: dict | None,
) -> dict:
    scores, entries = [], []
    for task in tasks:
        score, error = run_task(task, answer)
        scores.append(score)
        entries.append({"id": task.id, "class": task.task_class, "error": error, **score.to_dict()})
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "suite": "heldout",
        "backend": backend,
        "dry_run": dry_run,
        "context_tier": context_tier,
        "context_tokens": context_tokens,
        "goals": "C1, C2, C3 (C4 is not implemented)",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge": "deterministic (seeded ground truth)",
        "overlap": overlap,
        "tasks_run": len(tasks),
        "tasks_failed": sum(1 for e in entries if e["error"]),
        "overflows": 0,
        "metrics": aggregate(scores),
        "tasks": entries,
    }
