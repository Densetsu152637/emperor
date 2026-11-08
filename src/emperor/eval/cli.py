"""Command line for the evaluation harness.

Scoring is not implemented yet (plan stage 2b). `--dry-run` prints the report shape so callers
and CI can depend on it now.
"""

import argparse
import json
import sys

# Metric name -> plan goal it verifies.
METRICS = {
    "tool_validity": "G1",
    "routing": "G2",
    "brief": "G4",
    "leakage": "G5",
}
# Suite name -> plan goal it verifies.
SUITES = {
    "heldout": "G1, G2, G4, G5",
    "e2e": "G3",
    "longctx": "G6",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Score a model as the root agent of Imperium sessions.",
    )
    parser.add_argument("--model", help="model spec as Imperium names it, e.g. llamacpp:emperor")
    parser.add_argument("--suite", choices=sorted(SUITES), default="heldout")
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        action="append",
        help="limit scoring to these metrics (repeatable); default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the report shape without running any task",
    )
    return parser


def empty_report(model: str | None, suite: str, metrics: list[str]) -> dict:
    return {
        "model": model,
        "suite": suite,
        "goals": SUITES[suite],
        "metrics": {name: {"goal": METRICS[name], "score": None} for name in metrics},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = args.metric or sorted(METRICS)
    if args.dry_run:
        print(json.dumps(empty_report(args.model, args.suite, metrics), indent=2))
        return 0
    print("eval: scoring is not implemented yet (plan stage 2b); use --dry-run", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
