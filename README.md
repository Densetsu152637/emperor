# Emperor

Training, evaluation, and export for **Emperor**, an open-weight model built to be the root
agent of an [Imperium](https://github.com/Densetsu152637/Imperium-AI) session on a 16 GB GPU:
it plans, writes compact briefs, and delegates the work to hosted and local subagents through
Imperium's `spawn_agent` and `await_tasks` tools.

> **Status:** stage 0 (scaffold). The plan of record is
> [`docs/plans/emperor.md`](https://github.com/Densetsu152637/Imperium-AI/blob/docs/dispatcher-model-plan/docs/plans/emperor.md)
> in the Imperium repository.

## Layout

| Path | Holds |
| ---- | ----- |
| `src/emperor/eval/` | Evaluation harness (stage 2b) |
| `src/emperor/data/` | Dataset builders and validation (stage 4, not yet) |
| `src/emperor/train/` | Fine-tuning recipes (stages 5 and 6, not yet) |
| `src/emperor/export/` | Merging and GGUF/AWQ export (stage 6, not yet) |

Weights and datasets live on the Hugging Face Hub, not in Git.

## Development

```bash
uv sync
uv run eval --help
uv run eval --model llamacpp:emperor --dry-run
bash scripts/check.sh   # what CI runs
```

## Licence

[Apache-2.0](LICENSE)
