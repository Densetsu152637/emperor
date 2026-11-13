"""`uv run data {tasks,heldout,generate,judge,filter,report}`: build Consul's dataset by steps.

tasks     seeded-defect training tasks -> data/prompts.jsonl
heldout   the held-out evaluation set (other seed range and vocabulary) -> data/eval/heldout.jsonl
generate  teacher critiques from a local llama-server (needs `--group data`)
judge     ground-truth and judge checks of the critiques (needs `--group data`)
filter    the final training set -> data/final/final.jsonl
report    counts and rejection reasons -> data/report.json
"""

import argparse
import json
import sys
from pathlib import Path

from consul import data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    tasks = sub.add_parser("tasks")
    tasks.add_argument("--count", type=int, default=2000)
    tasks.add_argument("--seed", type=int, default=1)
    held = sub.add_parser("heldout")
    held.add_argument("--count", type=int, default=60)
    held.add_argument("--seed", type=int, default=data.HELDOUT_SEED_OFFSET)
    for name in ("generate", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
        p.add_argument("--limit", type=int)
        p.add_argument(
            "--reasoning-effort",
            default=data.GENERATE_EFFORT if name == "generate" else "medium",
            choices=["low", "medium", "high"],
        )
        if name == "judge":
            p.add_argument("--threshold", type=float, default=data.JUDGE_MEAN)
            p.add_argument(
                "--judge-thinking",
                action="store_true",
                default=data.JUDGE_THINKING,
                help="let the judge (Qwen3.8-27B) think before it answers (slower)",
            )
    sub.add_parser("filter")
    sub.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    d = args.data_dir
    if args.command == "tasks":
        result = data.write_train_tasks(d, args.count, args.seed)
    elif args.command == "heldout":
        result = data.write_heldout(d, args.count, args.seed)
    elif args.command in ("generate", "judge"):
        from emperor.data.teacher import JUDGE_MODEL, TEACHER_MODEL, ChatClient

        if args.command == "generate":
            client = ChatClient(args.base_url, reasoning_effort=args.reasoning_effort)
        else:  # the judge's own server, a different model from the teacher
            client = ChatClient(
                args.base_url,
                model=JUDGE_MODEL,
                reasoning_effort=args.reasoning_effort,
                template_kwargs={"enable_thinking": args.judge_thinking},
            )
        try:
            # Recorded as whatever the running server reports.
            served = data.served_model(client)
            if args.command == "generate":
                result = data.run_generation(
                    d, client, limit=args.limit, log=print, teacher_model=served or TEACHER_MODEL
                )
            else:
                result = data.run_judging(
                    d,
                    client,
                    threshold=args.threshold,
                    limit=args.limit,
                    log=print,
                    model=served or JUDGE_MODEL,
                )
        finally:
            client.close()
    elif args.command == "filter":
        result = data.filter_kept(d)
    else:
        result = data.write_report(d)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
