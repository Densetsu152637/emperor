#!/usr/bin/env bash
# What CI runs: lint, format check, tests, and the eval CLI smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run eval --help > /dev/null
