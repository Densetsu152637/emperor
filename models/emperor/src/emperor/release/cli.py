"""`uv run release {export,card,publish} --config configs/sft.toml`: make a trained model ready.

export   bf16 GGUF and quantised variants into weights/<name>-gguf/ (CPU only, resumable)
card     write the Hugging Face README.md into the bf16 and GGUF directories
publish  upload to the Hub; refuses unless --confirm is given (never run by the pipeline)
"""

import argparse
import logging
import sys
from pathlib import Path

from emperor.export.gguf import DEFAULT_CACHE, ExportError, export_gguf, gguf_dir_for
from emperor.release.card import write_cards
from emperor.release.publish import PublishRefused, Upload, publish
from emperor.train.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="release", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "card", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
        p.add_argument("--root", type=Path, default=Path.cwd(), help="base for relative paths")
        if name == "export":
            p.add_argument("--llama-cpp", type=Path, help="llama.cpp checkout (default: clone)")
            p.add_argument("--llama-cache", type=Path, default=DEFAULT_CACHE)
            p.add_argument("--quantize", type=Path, help="llama-quantize binary")
            p.add_argument("--type", action="append", dest="types", help="override gguf_types")
        if name == "card":
            p.add_argument("--run-dir", type=Path, help="take eval.json from this run")
        if name == "publish":
            p.add_argument("--repo-id", help="owner/name for the bf16 weights")
            p.add_argument("--gguf-repo-id", help="owner/name for the GGUF files")
            p.add_argument("--public", action="store_true", help="create public repos")
            p.add_argument("--confirm", action="store_true", help="really upload")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = load_config(config_path)
    except (ConfigError, OSError) as error:
        print(f"release: {error}", file=sys.stderr)
        return 2
    release = config.release
    weights = (root / config.merge.output_dir).resolve()
    gguf_dir = gguf_dir_for(root, release.name, release.gguf_dir)
    try:
        if args.command == "export":
            cache = args.llama_cache if args.llama_cache.is_absolute() else root / args.llama_cache
            result = export_gguf(
                weights,
                gguf_dir,
                release.name,
                args.types or release.gguf_types,
                cache=cache,
                llama_cpp=args.llama_cpp,
                quantize_binary=args.quantize,
                log=logging.getLogger("release").info,
            )
            for item in result.files:
                print(f"{item['file']}  {item['bytes']} bytes")
        elif args.command == "card":
            for path in write_cards(config, root, args.run_dir):
                print(path)
        else:
            uploads = []
            if args.repo_id:
                uploads.append(Upload(args.repo_id, weights))
            if args.gguf_repo_id:
                uploads.append(Upload(args.gguf_repo_id, gguf_dir))
            for url in publish(uploads, confirm=args.confirm, private=not args.public):
                print(url)
    except (ExportError, FileNotFoundError, PublishRefused) as error:
        print(f"release: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # a failed subprocess or upload: report it, never a traceback
        print(f"release: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
