"""Run a suite and build the JSON report (schema: `report_schema()`, version `SCHEMA_VERSION`)."""

from collections.abc import Callable
from datetime import UTC, datetime

from emperor.eval.contract import contract_source
from emperor.eval.metrics import METRICS, TaskScore
from emperor.eval.models import ChatModel
from emperor.eval.runner import Transcript, run_task
from emperor.eval.tasks import Task

SCHEMA_VERSION = 1


class CeilingExceeded(RuntimeError):
    """A run spent, or would spend, more than its API ceiling."""


class CostMeter:
    """Tracks hosted spend against the API ceiling (USD; 0 means nothing may be spent)."""

    def __init__(self, ceiling_usd: float):
        self.ceiling_usd = ceiling_usd
        self.spent_usd = 0.0

    def add(self, usd: float) -> None:
        self.spent_usd += usd
        if self.spent_usd > self.ceiling_usd:
            raise CeilingExceeded(
                f"spent ${self.spent_usd:.4f}, over the ${self.ceiling_usd:.2f} API ceiling"
            )


def _task_entry(task: Task, transcript: Transcript, scores: dict[str, TaskScore | None]) -> dict:
    return {
        "id": task.id,
        "class": task.task_class,
        "turns": len(transcript.replies),
        "tool_calls": [c.name for c in transcript.calls],
        "final": transcript.final,
        "error": transcript.error,
        "overflowed": transcript.overflowed,
        "prompt_tokens": transcript.prompt_tokens,
        "completion_tokens": transcript.completion_tokens,
        "metrics": {
            name: None if s is None else {"score": round(s.score, 4), "details": s.details}
            for name, s in scores.items()
        },
    }


def run_suite(
    tasks: list[Task],
    model_for: Callable[[Task], ChatModel],
    *,
    model: str | None,
    suite: str,
    tier: str,
    context_tokens: int,
    metrics: list[str],
    backend: str,
    dry_run: bool,
    goals: str,
    meter: CostMeter,
) -> dict:
    results: dict[str, list[tuple[Task, TaskScore]]] = {name: [] for name in metrics}
    entries = []
    for task in tasks:
        transcript = run_task(task, model_for(task), context_tokens)
        meter.add(0.0)  # replay calls no hosted model; the imperiumd backend adds ledger cost
        scores = {name: METRICS[name].scorer(task, transcript) for name in metrics}
        for name, score in scores.items():
            if score is not None:
                results[name].append((task, score))
        entries.append(_task_entry(task, transcript, scores))

    metric_entries = {}
    for name in metrics:
        metric = METRICS[name]
        aggregate = (
            metric.aggregate(results[name], tier)
            if results[name]
            else {
                "score": None,
                "target": None,
                "passed": None,
                "note": "no task in this suite exercises this metric",
            }
        )
        metric_entries[name] = {
            "goal": metric.goal,
            "description": metric.description,
            "n": len(results[name]),
            **aggregate,
        }

    source = contract_source()
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "suite": suite,
        "backend": backend,
        "dry_run": dry_run,
        "context_tier": tier,
        "context_tokens": context_tokens,
        "goals": goals,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract": {"imperium_commit": source.get("commit"), "branch": source.get("branch")},
        "judge": "heuristic",
        "cost": {
            "usd": round(meter.spent_usd, 4),
            "ceiling_usd": meter.ceiling_usd,
            "prompt_tokens": sum(e["prompt_tokens"] for e in entries),
            "completion_tokens": sum(e["completion_tokens"] for e in entries),
        },
        "tasks_run": len(entries),
        "tasks_failed": sum(e["error"] is not None for e in entries),
        "overflows": sum(e["overflowed"] for e in entries),
        "metrics": metric_entries,
        "tasks": entries,
    }


def report_schema() -> dict:
    """JSON Schema of the report (the subset `contract.validate` understands)."""
    number_or_null = {"type": ["number", "null"]}
    metric = {
        "type": "object",
        "required": ["goal", "description", "n", "score", "target", "passed"],
        "properties": {
            "goal": {"type": "string"},
            "description": {"type": "string"},
            "n": {"type": "integer"},
            "score": number_or_null,
            "target": number_or_null,
            "passed": {"type": ["boolean", "null"]},
        },
    }
    task = {
        "type": "object",
        "required": ["id", "turns", "tool_calls", "final", "error", "overflowed", "metrics"],
        "properties": {
            "id": {"type": "string"},
            "class": {"type": ["string", "null"]},
            "turns": {"type": "integer"},
            "tool_calls": {"type": "array", "items": {"type": "string"}},
            "final": {"type": "string"},
            "error": {"type": ["string", "null"]},
            "overflowed": {"type": "boolean"},
            "prompt_tokens": {"type": "integer"},
            "completion_tokens": {"type": "integer"},
            "metrics": {
                "type": "object",
                "properties": {
                    name: {
                        "type": ["object", "null"],
                        "required": ["score", "details"],
                        "properties": {"score": {"type": "number"}, "details": {"type": "object"}},
                    }
                    for name in METRICS
                },
            },
        },
    }
    return {
        "type": "object",
        "required": [
            "schema_version",
            "model",
            "suite",
            "backend",
            "dry_run",
            "context_tier",
            "context_tokens",
            "goals",
            "timestamp",
            "contract",
            "judge",
            "cost",
            "tasks_run",
            "tasks_failed",
            "overflows",
            "metrics",
            "tasks",
        ],
        "properties": {
            "schema_version": {"type": "integer", "enum": [SCHEMA_VERSION]},
            "model": {"type": ["string", "null"]},
            "suite": {"type": "string"},
            "backend": {"type": "string", "enum": ["replay", "imperiumd"]},
            "dry_run": {"type": "boolean"},
            "context_tier": {"type": "string"},
            "context_tokens": {"type": "integer"},
            "goals": {"type": "string"},
            "timestamp": {"type": "string"},
            "contract": {"type": "object", "required": ["imperium_commit"]},
            "judge": {"type": "string"},
            "cost": {
                "type": "object",
                "required": ["usd", "ceiling_usd", "prompt_tokens", "completion_tokens"],
                "properties": {
                    "usd": {"type": "number"},
                    "ceiling_usd": {"type": "number"},
                    "prompt_tokens": {"type": "integer"},
                    "completion_tokens": {"type": "integer"},
                },
            },
            "tasks_run": {"type": "integer"},
            "tasks_failed": {"type": "integer"},
            "overflows": {"type": "integer"},
            "metrics": {"type": "object", "properties": {name: metric for name in METRICS}},
            "tasks": {"type": "array", "items": task},
        },
    }
