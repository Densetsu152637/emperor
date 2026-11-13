"""`uv run eval`: score a critic model on the held-out seeded-defect tasks (goals C1 to C3).

`--dry-run` answers every task with the reference critique built from the ground truth, so the
harness runs without a model; it must score perfectly and is how the metrics are tested. Without
it the model behind an OpenAI-compatible endpoint (llama-server) is asked, locally, task by task.
Consul is a thinking model: its `<think>` block is removed before the answer is scored, and
`--max-tokens` leaves room for the reasoning as well as the critique.
Before any scoring the held-out set is checked against the training tasks and the run refuses
if they overlap.
"""

import argparse
import json
import sys
from pathlib import Path

from emperor.eval.cli import CONTEXT_TIERS, DEFAULT_BASE_URL
from emperor.eval.models import OpenAICompatModel

from consul import prompts
from consul.data import load_tasks
from consul.eval.report import run_suite
from consul.overlap import DEFAULT_THRESHOLD, overlap_report
from consul.score import reference_critique
from consul.tasks import CritiqueTask

DEFAULT_MAX_TOKENS = 16384  # room to reason first; Emperor's 4096 can cut a critique off


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model", default="llamacpp:consul", help="llamacpp:<served name>")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tasks", type=Path, default=Path("data/eval/heldout.jsonl"))
    parser.add_argument("--train-data", type=Path, default=Path("data/prompts.jsonl"))
    parser.add_argument("--overlap-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--context-tier", choices=list(CONTEXT_TIERS), default="32k")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="completion budget per task, reasoning included",
    )
    parser.add_argument("--dry-run", action="store_true", help="answer with the reference critique")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    return parser


def model_answer(model: OpenAICompatModel):
    def answer(task: CritiqueTask) -> str:
        messages = [
            {"role": "system", "content": prompts.CONSUL_SYSTEM},
            {"role": "user", "content": prompts.user_message(task)},
        ]
        return model.complete(messages, []).content or ""

    return answer


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_tasks(args.tasks)
    if not tasks:
        print(f"eval: no tasks at {args.tasks}; run `uv run data heldout`", file=sys.stderr)
        return 2
    overlap = overlap_report(load_tasks(args.train_data), tasks, args.overlap_threshold)
    if not overlap["ok"]:
        print(
            "eval: the held-out set overlaps the training set "
            f"(max {overlap['max_overlap']}, duplicates {overlap['exact_duplicates']}); "
            "regenerate it with `uv run data heldout`",
            file=sys.stderr,
        )
        return 2
    name = args.model.split(":", 1)[-1]
    if args.dry_run:
        by_id = {t.id: t for t in tasks}

        def answer(task: CritiqueTask) -> str:
            return json.dumps(reference_critique(by_id[task.id]))

        backend = "replay"
    else:
        answer = model_answer(OpenAICompatModel(args.base_url, name, max_tokens=args.max_tokens))
        backend = "llamacpp"
    report = run_suite(
        tasks,
        answer,
        model=args.model,
        backend=backend,
        dry_run=args.dry_run,
        context_tier=args.context_tier,
        context_tokens=CONTEXT_TIERS[args.context_tier],
        overlap=overlap,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for metric, entry in report["metrics"].items():
        score = entry["score"]
        shown = "n/a" if score is None else f"{score:.3f}"
        print(f"{entry['goal']} {metric}: {shown} (target {entry['target']}, n={entry['n']})")
    return 1 if report["tasks_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
