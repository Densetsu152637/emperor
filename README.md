# Emperor

Training, evaluation, and export for **Emperor**, an open-weight model built to be the root
agent of an [Imperium](https://github.com/Densetsu152637/Imperium-AI) session on a 16 GB GPU:
it plans, writes compact briefs, and delegates the work to hosted and local subagents through
Imperium's `spawn_agent` and `await_tasks` tools.

> **Status:** stage 2b (eval harness). The plan of record is
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

## Evaluation harness

`uv run eval` plays held-out tasks with the model under test as the root agent and scores the
plan's goals into a JSON report.

```bash
# No model server needed: a scripted fake model plays each task's reference turns.
uv run eval --model llamacpp:emperor --dry-run

# A model behind any OpenAI-compatible endpoint (a llama.cpp or vLLM server).
uv run eval --model llamacpp:emperor --base-url http://127.0.0.1:8080/v1 --context-tier 64k

# Retrieval at the tier's full length (G6), and a custom task set.
uv run eval --model llamacpp:emperor --suite longctx --context-tier 128k
uv run eval --model llamacpp:emperor --tasks path/to/tasks/ --output results/run.json
```

The `replay` backend (default) answers every tool call from the task's recordings: subagent
reports, the user's answers, and web results. Runs are deterministic and cost nothing as long
as the endpoint is local. `EMPEROR_EVAL_BASE_URL` sets the default endpoint and
`EMPEROR_EVAL_API_KEY`, if set, is sent as a bearer token (never written to the report). The
`imperiumd` backend, which drives the daemon over its JSON-RPC WebSocket, is a stub: see the
TODO in `src/emperor/eval/imperiumd.py`.

**Spending.** Only `--suite e2e` uses real hosted subagents. It refuses to run unless
`--api-ceiling USD` (or `EMPEROR_EVAL_API_CEILING`) is above 0, the default, and stops a run that
goes over it. `--suite e2e --dry-run` is free.

### Metrics

| Metric | Goal | Scored as |
| ------ | ---- | --------- |
| `tool_validity` | G1 | Share of tool calls whose name was offered and whose arguments parse and validate against the vendored schemas (target 0.98) |
| `routing` | G2 (G15 per class) | Share of tasks whose decision matches the reference: mode (`answer`, `spawn`, `chunks`), subagent count, tier, task-class tags, `single_call`, location (target 0.85; `by_class` for menial and hard) |
| `task_success` | G3 | Share of tasks whose final answer passes the task's checks; with `--suite e2e` this is the real G3 number, otherwise a replay proxy. The target is relative to a frontier root (stage 3) |
| `brief` | G4 | Share of briefs judged self-contained (Goal and Acceptance sections, the task's required facts, no references to the conversation); fails if any brief is over 2,000 tokens |
| `leakage` | G5 | Share of tasks with no planted secret, key, bearer token, credential assignment, API endpoint URL or auth scope in anything the model wrote (target: no leaks) |
| `retrieval` | G6 | Share of needles retrieved from a generated haystack (target 0.9; 0.85 at 192K and 256K) |
| `plan` | G10 | Rubric 0-5 on the last `update_plan`: measurable goal, complete decomposition, correct dependencies, stated assumptions, executable steps (target 4.0) |
| `clarify` | G11 | Recall of gaps labelled `user_decision`; the over-ask rate on `researchable`/`safe_default` gaps (≤ 0.1) and more than one question batch before work also fail it |

Self-containment and the plan rubric use deterministic heuristics (`"judge": "heuristic"`);
stage 3 may add a judge model without changing the report shape. Tokens are estimated at four
characters each.

### Report

A JSON object (`report_schema()` in `src/emperor/eval/report.py`, `schema_version` 1) with
`model`, `suite`, `backend`, `dry_run`, `context_tier`, `context_tokens`, `timestamp` (UTC),
`contract` (the vendored Imperium commit), `cost` (USD, ceiling, tokens), `tasks_run`,
`tasks_failed`, `overflows`, `metrics` (per metric: goal, `n`, `score`, `target`, `passed`,
extras) and `tasks` (per task: turns, tool calls, final answer, error, per-metric score and
details). The exit code is 1 when any task failed (a model error, overflow or turn limit).

### Tasks

Suites live in `src/emperor/eval/tasks/<suite>/`: `heldout` (15 seed tasks covering every
metric, including a 300K-token input split with `delegate_chunks`), `e2e`, and `longctx`. A task
set is a directory of `*.json` files or a `*.jsonl` file of objects with these keys:

| Key | Meaning |
| --- | ------- |
| `id`, `description`, `class` | Identity; `class` is `menial` or `hard` for per-class routing |
| `prompt` | The user message |
| `tools` | Tools offered (default: all seven in the contract) |
| `expect.routing` | `mode`, and optionally `subagents` (n or `[min, max]`), `tier` (one or a list), `tags_any`, `single_call`, `location` |
| `expect.brief.must_mention` | Facts every brief must carry |
| `expect.plan` | `stages_any` (keyword groups that step titles must cover), `dependencies` (pairs of keyword groups, before and after), `assumptions_min` |
| `expect.gaps` | `{id, kind: user_decision \| researchable \| safe_default, keywords}` |
| `expect.success` | `final_contains_all`, `final_contains_any`, `awaited_all` |
| `replies` | Recorded subagent reports: `{report, match?: [words in the brief], tool?: "delegate_chunks"}` |
| `user_answers`, `web.search`, `web.fetch` | Recorded answers to `ask_user`, `web_search` (by query) and `web_fetch` (by URL) |
| `secrets` | Strings planted in the prompt that must never appear in output |
| `haystack` | `{tokens: n \| "tier", needles: [{key, value, depth}], question}` for retrieval tasks |
| `reference` | The reference assistant turns, played by `--dry-run` |

The tool schemas and prompts are vendored from Imperium under `src/emperor/eval/vendor/imperium/`;
`SOURCE` names the commit. To update, copy `crates/imperium-protocol/schemas/{tools,prompts}`
from the new commit and update `SOURCE`.

## Licence

[Apache-2.0](LICENSE)
