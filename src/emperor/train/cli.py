"""`uv run --group train train {show,fit,merge} --config configs/sft.toml`."""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from emperor.train.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train", description="Emperor bf16 LoRA fine-tuning.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("show", "print the validated configuration"),
        ("fit", "train (or resume from the last checkpoint)"),
        ("merge", "merge the adapter into the base and save bf16 safetensors"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
        p.add_argument("--root", type=Path, default=Path.cwd(), help="base for relative paths")
        if name == "fit":
            p.add_argument(
                "--pause-file",
                type=Path,
                help="when this file appears, save a checkpoint and exit with code 75",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return 0
    from emperor.train import sft

    if args.command == "fit":
        return sft.train(config, args.root, args.pause_file)
    print(sft.merge(config, args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
