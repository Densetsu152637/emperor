#!/usr/bin/env bash
# What CI runs: lint, format check, tests, and the eval CLI smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run eval --help > /dev/null
uv run eval --model llamacpp:emperor --dry-run > /dev/null
uv run data --help > /dev/null
uv run pipeline --help > /dev/null
uv run train show > /dev/null
uv run release --help > /dev/null
uv run release publish --help > /dev/null
