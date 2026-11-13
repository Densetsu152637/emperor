"""`uv run --group data data ...`: build prompts, generate and judge traces, filter, report.

The generate and judge commands need a local llama-server already running the teacher (the
`pipeline` command starts and supervises one, and only while the machine is idle).
"""

import argparse
import json
import sys
from pathlib import Path

from emperor.data.build import filter_kept, write_report
from emperor.data.prompts import HeldoutIndex, generate_prompts, normalise
from emperor.data.retry import DEFAULT_RETRIES
from emperor.data.schema import validate_sample
from emperor.data.shards import iter_records, read_jsonl
from emperor.data.teacher import EFFORTS, ENVIRONMENT_EFFORT, GENERATE_EFFORT, JUDGE_EFFORT

DEFAULT_DATA_DIR = Path("data")
DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    prompts = sub.add_parser("prompts", help="generate deduplicated, held-out-safe user requests")
    prompts.add_argument("--count", type=int, default=2000)
    prompts.add_argument("--seed", type=int, default=1)

    # No standalone retry command: a retry round generates with the teacher and judges with a
    # different model, and only one model may run at a time. `pipeline` runs the rounds as
    # separate teacher and judge phases.
    for name, text in (
        ("generate", "teacher traces"),
        ("judge", "score traces"),
    ):
        p = sub.add_parser(name, help=f"{text} against a local llama-server")
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--limit", type=int)
        if name == "generate":
            p.add_argument(
                "--reasoning-effort",
                default=GENERATE_EFFORT,
                choices=EFFORTS,
                help="teacher effort while generating; its reasoning is stored in the traces",
            )
            p.add_argument(
                "--environment-reasoning-effort",
                default=ENVIRONMENT_EFFORT,
                choices=EFFORTS,
                help="effort of the call that simulates tool results",
            )
            p.add_argument(
                "--retries",
                type=int,
                default=DEFAULT_RETRIES,
                help="extra tries per prompt (new seed) after a failed schema check or judgement",
            )
            p.add_argument(
                "--no-force-tool",
                action="store_true",
                help="do not send tool_choice=required on the first turn",
            )
        if name == "judge":
            p.add_argument("--threshold", type=float, default=4.0)
            p.add_argument("--judge-reasoning-effort", default=JUDGE_EFFORT, choices=EFFORTS)
            p.add_argument(
                "--no-judge-thinking",
                action="store_true",
                help="ask the judge to answer without thinking first",
            )

    sub.add_parser("filter", help="keep judged passes in data/final/final.jsonl")
    sub.add_parser("report", help="write data/report.json")
    validate = sub.add_parser("validate", help="check every sample in a shard directory")
    validate.add_argument("directory", type=Path, nargs="?")
    return parser


def cmd_prompts(args) -> int:
    path = args.data_dir / "prompts.jsonl"
    existing = [normalise(r["text"]) for r in read_jsonl(path)] if path.exists() else []
    index = HeldoutIndex.bundled()
    records, stats = generate_prompts(args.count, args.seed, heldout=index, existing=existing)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    print(json.dumps({"written": len(records), "total": len(existing) + len(records), **stats}))
    return 0


def cmd_generate(args) -> int:
    from emperor.data.generate import run_generation
    from emperor.data.teacher import ChatClient, TeacherSetupError

    client = ChatClient(args.base_url, reasoning_effort=args.reasoning_effort)
    environment = ChatClient(args.base_url, reasoning_effort=args.environment_reasoning_effort)
    try:
        counts = run_generation(
            args.data_dir / "prompts.jsonl",
            args.data_dir,
            client,
            environment,
            config=generation_config(args, client.served_model()),
            limit=args.limit,
            log=lambda line: print(line, file=sys.stderr),
        )
    except TeacherSetupError as error:
        print(f"generate: {error}", file=sys.stderr)
        return 2
    finally:
        client.close()
        environment.close()
    print(json.dumps(counts))
    return 0


def generation_config(args, served: str | None = None):
    from emperor.data.generate import GenerationConfig
    from emperor.data.teacher import TEACHER_MODEL

    return GenerationConfig(
        retries=args.retries,
        force_first_tool=not args.no_force_tool,
        teacher_model=served or TEACHER_MODEL,  # the model the running server reports
    )


def judge_chat_client(args):
    from emperor.data.teacher import ChatClient

    return ChatClient(
        args.base_url,
        reasoning_effort=args.judge_reasoning_effort,
        template_kwargs={"enable_thinking": not args.no_judge_thinking},
    )


def cmd_judge(args) -> int:
    from emperor.data.judge import run_judging
    from emperor.data.teacher import JUDGE_MODEL

    client = judge_chat_client(args)
    try:
        counts = run_judging(
            args.data_dir,
            client,
            # Whatever the running server reports, so the verdict names the real judge.
            model=client.served_model() or JUDGE_MODEL,
            threshold=args.threshold,
            limit=args.limit,
            log=lambda line: print(line, file=sys.stderr),
        )
    finally:
        client.close()
    print(json.dumps(counts))
    return 0


def cmd_validate(args) -> int:
    directory = args.directory or args.data_dir / "traces"
    total = bad = 0
    reasons: dict[str, int] = {}
    for sample in iter_records(directory):
        total += 1
        check = validate_sample(sample)
        if not check.ok:
            bad += 1
            for reason in check.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
    print(json.dumps({"samples": total, "invalid": bad, "reasons": reasons}))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prompts":
        return cmd_prompts(args)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "judge":
        return cmd_judge(args)
    if args.command == "filter":
        print(json.dumps(filter_kept(args.data_dir)))
        return 0
    if args.command == "report":
        print(json.dumps(write_report(args.data_dir), indent=2))
        return 0
    return cmd_validate(args)


if __name__ == "__main__":
    sys.exit(main())
