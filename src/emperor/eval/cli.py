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
    "plan": "G10",
    "clarify": "G11",
}
# Suite name -> plan goal it verifies.
SUITES = {
    "heldout": "G1, G2, G4, G5, G10, G11",
    "e2e": "G3",
    "longctx": "G6",
}
# Context tier name -> active context in tokens; the same weights run at each tier.
CONTEXT_TIERS = {
    "32k": 32_768,
    "64k": 65_536,
    "96k": 98_304,
    "128k": 131_072,
    "256k": 262_144,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Score a model as the root agent of Imperium sessions.",
    )
    parser.add_argument("--model", help="model spec as Imperium names it, e.g. llamacpp:emperor")
    parser.add_argument("--suite", choices=sorted(SUITES), default="heldout")
    parser.add_argument(
        "--context-tier",
        choices=list(CONTEXT_TIERS),
        default="64k",
        help="active context the model runs with (default: 64k)",
    )
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


def empty_report(model: str | None, suite: str, tier: str, metrics: list[str]) -> dict:
    return {
        "model": model,
        "suite": suite,
        "context_tier": tier,
        "context_tokens": CONTEXT_TIERS[tier],
        "goals": SUITES[suite],
        "metrics": {name: {"goal": METRICS[name], "score": None} for name in metrics},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = args.metric or sorted(METRICS)
    if args.dry_run:
        report = empty_report(args.model, args.suite, args.context_tier, metrics)
        print(json.dumps(report, indent=2))
        return 0
    print("eval: scoring is not implemented yet (plan stage 2b); use --dry-run", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
