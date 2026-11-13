#!/usr/bin/env bash
# What CI runs: lint, format check, tests, and CLI smoke tests (no GPU, no models).
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
uv run data --data-dir "$tmp" tasks --count 60 > /dev/null
uv run data --data-dir "$tmp" heldout --count 10 > /dev/null
uv run eval --tasks "$tmp/eval/heldout.jsonl" --train-data "$tmp/prompts.jsonl" --dry-run > /dev/null
uv run pipeline --help > /dev/null
uv run hosted --help > /dev/null
uv run train show > /dev/null
uv run release --help > /dev/null
