# Emperor

Training, evaluation, and export for **Emperor**, an open-weight model built to be the root
agent of an [Imperium](https://github.com/Densetsu152637/Imperium-AI) session on a 16 GB GPU:
it plans, writes compact briefs, and delegates the work to hosted and local subagents through
Imperium's `spawn_agent` and `await_tasks` tools.

> **Status:** stage 2b (eval harness), plus a first dataset, training and pipeline
> implementation for stages 4 to 6. The plan of record is
> [`docs/plans/emperor.md`](https://github.com/Densetsu152637/Imperium-AI/blob/docs/dispatcher-model-plan/docs/plans/emperor.md)
> in the Imperium repository. The step-by-step procedure from baselines to publication is
> [`docs/runbook.md`](docs/runbook.md).

## Layout

| Path | Holds |
| ---- | ----- |
| `src/emperor/eval/` | Evaluation harness (stage 2b) |
| `src/emperor/data/` | Prompt generation, local-teacher traces, judging, filtering, report (stage 4) |
| `src/emperor/train/` | bf16 LoRA supervised fine-tuning and merge (stage 5) |
| `src/emperor/export/` | GGUF export through llama.cpp: bf16 and quantised variants (stage 6) |
| `src/emperor/release/` | Model cards, and the guarded Hugging Face publish |
| `src/emperor/pipeline/` | The idle-gated orchestrator: data, train, merge, eval |
| `configs/sft.toml` | The training recipe |
| `data/`, `runs/`, `outputs/`, `weights/` | Generated locally and ignored by Git |

Weights and datasets are not stored in Git. They are written under `data/` and `weights/`
(both ignored); publishing to the Hugging Face Hub or Git LFS is deferred.

## Development

```bash
uv sync
uv run eval --help
uv run eval --model llamacpp:emperor --dry-run
bash scripts/check.sh   # what CI runs
uv sync --group data --group train   # optional: data generation and training (multi-GB torch)
```

## Dataset, training and pipeline

The data can be made two ways. **Hosted** (the default for new data, below): the teacher and
the tool simulator run on DeepSeek's API and the judge on Moonshot's (Kimi). **Local** (the
fallback, the `pipeline` command): a local gpt-oss teacher and Qwen judge behind llama-server,
gated on idleness; its client refuses any non-loopback endpoint. No Claude, Anthropic or OpenAI
API is used to generate or judge data either way. Training always runs on this machine.

### Hosted data (`uv run hosted`)

- **Teacher**: `deepseek-v4-pro` in thinking mode (`reasoning_effort` high); its
  `reasoning_content` is stored on every assistant turn and the first turn must have it.
  DeepSeek's terms (Open Platform Terms of Service, section 4.2) assign the outputs to the user
  and list "training other models (such as model distillation)" among the permitted uses.
- **Tool simulator**: `deepseek-flash` without thinking (its output is context, never trained).
- **Judge**: `kimi-k3.1-pro` (always thinks; a different family from the teacher). It sees
  the transcript without the teacher's reasoning. Thresholds and rubric are the local ones.
- **Keys**: stored in the OS credential store and read with `keyring` at run time (service
  `emperor`, usernames `deepseek` and `moonshot`); never in files, logs or the command line:

  ```powershell
  uv run python -c "import keyring, getpass; keyring.set_password('emperor', 'deepseek', getpass.getpass())"
  ```

- **Config**: `configs/hosted.toml` holds the base URLs, model ids, per-role thinking and token
  budgets, concurrency, the price per million tokens (with the source page) and a hard budget
  per provider. `--offpeak-only` (or `offpeak_only = true`) sends DeepSeek requests only in its
  half-price hours (outside 01:00-04:00 and 06:00-10:00 UTC on weekdays).

```powershell
uv run hosted --data-dir data/hosted-1 --run-dir runs/hosted-1 --detach   # start or resume
Get-Content runs/hosted-1/status.json                                      # progress and spend
New-Item runs/hosted-1/STOP                                                 # stop cleanly
```

Phases: prompts, generate, judge, retry rounds, filter, report (`--only <phase>` runs one).
Requests run in parallel (`--workers`, `--judge-workers`; default the provider's
`max_concurrency`), back off on 429 and 5xx, and resume from the shards on disk. Every
request's usage (cache hits and misses, reasoning tokens) and estimated cost is appended to
`<data-dir>/spend.jsonl`; `status.json` shows spend per provider, dollars per task and the
projected cost of the main pass. A provider that reaches its budget stops the run with status
`budget` (exit code 3); raise `budget_usd` and run the same command to resume.

### Local pipeline (`uv run pipeline`)

```powershell
# Start the whole build in the background (writes runs/<timestamp>/ and returns at once).
uv run --group data --group train pipeline --detach
# Same, but the user is asleep: ignore the idle gate.
uv run --group data --group train pipeline --detach --asleep
# Watch it, stop it cleanly, or continue a run.
Get-Content runs/<timestamp>/pipeline.log -Wait
New-Item runs/<timestamp>/STOP        # finish the current step and exit
uv run --group data --group train pipeline --detach --resume runs/<timestamp>
```

Phases (`--skip <phase>` skips one; every phase is resumable):

1. **prompts**: seeded, deduplicated requests across coding, refactor, research, planning,
   clarify, docs, architecture, long input and quick answers, from vague to specific. Each
   candidate is checked against `tasks/heldout`, `tasks/e2e` and `tasks/longctx` with a 3-gram
   overlap test (Jaccard similarity or containment of 0.4 or more is dropped).
2. **generate**: gpt-oss-120b (Apache-2.0, F16 GGUF in `F:\AI-Models\unsloth\gpt-oss-120b-GGUF`) served by
   `llama-server` plays Emperor with `emperor-root.md`, the brief/plan/question shapes and the
   target style: concise and structured, big picture first, delegates instead of pasting code,
   few emojis, no walls of tables, searches instead of recalling facts, never handles
   credentials. A separate "environment" call simulates the tool results. The stored sample keeps
   only the plain Emperor prompt, so the style is learned rather than instructed. Each assistant
   turn keeps the teacher's reasoning for it as `reasoning_content`; the teacher runs at medium
   reasoning effort (`--reasoning-effort`); only the first turn of a trace must carry reasoning.
3. **judge**: Qwen3.8-27B (`--judge-gguf`; a different family, so the teacher never grades its own
   traces; thinking on unless `--no-judge-thinking`) on its own llama-server, never beside the
   teacher, scores each trace 1-5 on eight criteria and answers pass or fail as
   JSON. A trace is kept when the judge passes it, the mean is at least 4.0 (`--threshold`), no
   criterion is below 3, and the emoji, table and pasted-code checks are clean. Schema checks run
   first: tool calls validate against the vendored schemas, roles alternate, briefs have Goal and
   Acceptance sections and fit 2,000 tokens, and there are no secrets (the eval harness's leak
   patterns).
4. **filter**: writes `data/final/final.jsonl` and `data/report.json` (counts, pass rate, score
   histograms, rejection reasons).
5. **train**: `configs/sft.toml`, bf16 LoRA on `F:\AI-Models\Qwen\Qwen3.5-4B` (Apache-2.0;
   the text model only, without the vision encoder or the multi-token-prediction head) with TRL,
   gradient checkpointing, assistant-only loss (the teacher's reasoning, rendered as the
   template's `<think>` block, is trained too; samples without reasoning are skipped and counted
   in the log), sequence length 8192, no packing, a checkpoint every 50 steps and resume from
   the last one. The Gated DeltaNet layers run on flash-linear-attention's Triton kernels
   (`triton-windows` on Windows, which bundles its own C compiler). There is no QLoRA and no
   4/8-bit loading; the config parser refuses such keys. The base directory is only read.
6. **merge**: the adapter is merged into the base and saved as bf16 safetensors in
   `weights/emperor-4b-bf16/` (the few tensors the base keeps in fp32 stay fp32; the export
   leaves out the multi-token-prediction head, which the text model does not carry).
7. **export**: GGUF through llama.cpp's `convert_hf_to_gguf.py` (a shallow clone of tag `b12180`
   in the ignored `tools/llama.cpp`, or `--llama-cpp` / `EMPEROR_LLAMA_CPP`), then
   `llama-quantize` to the `[release] gguf_types` (Q8_0, Q6_K, Q4_K_M). Writes
   `weights/emperor-4b-gguf/`; every file is skipped when it already exists.
8. **eval**: the bf16 GGUF (`--eval-type` picks another, `--eval-gguf` a file) is served by
   `llama-server` and `uv run eval` scores it. `runs/<timestamp>/summary.json` collects the data
   counts, training metrics, pauses and eval metrics.
9. **card**: writes the Hugging Face `README.md` into `weights/emperor-4b-bf16/` and the GGUF
   directory (see Release).

### Teacher yield and extra rounds

- **Forced tools.** For every task class except `quick_answer` the first assistant turn is sent
  with `tool_choice: "required"` (llama-server with `--jinja`); if the server rejects it the
  client stops sending it. The style prompt tells the teacher to plan, research and delegate for
  anything non-trivial and never to use Markdown tables. A trace with no tool call (outside
  `quick_answer`) or with a table of more than four rows is rejected at generation.
- **Rejection sampling.** A prompt that fails the schema check is retried in place up to
  `--retries` times (default 2) with a different seed; `meta.attempts` and
  `meta.attempt_failures` record it. The new **retry** phase (after judge) regenerates prompts the
  judge rejected into `data/retry-1/`, `data/retry-2/` (same layout, new seeds) and judges them;
  the first pass is never overwritten and `filter` keeps the earliest kept sample per prompt.
  `--retries 0` turns both off. Retry rounds run only inside `pipeline`, which generates
  and judges each round in separate server phases (one model at a time).
- **Judge.** An unparseable answer is retried once, colder and with a stricter JSON instruction.
  Criteria that cannot apply are ignored for `quick_answer` (delegation, tool use, research,
  planning) and `clarify` (delegation, research); the judge prompt says so too.
- **Another round without losing the first.** `--data-dir data/r2` puts a round in its own
  directory, `--previous-data data` keeps its prompts distinct from earlier rounds, and
  `configs/sft-r2.toml` trains from scratch (own output and weights dirs) on round 2's kept samples
  plus round 1's (`[data] extra_paths`, deduplicated by sample id):

```powershell
uv run --group data --group train pipeline --detach --asleep --data-dir data/r2 `
  --previous-data data --train-config configs/sft-r2.toml --seed 2 --prompts 2000
```

`--only <phase>` (repeatable) runs just those phases of a run directory and leaves the rest of
its state alone. To export and write cards for a run that already finished:

```powershell
uv run --group train pipeline --detach --resume runs/<timestamp> --only export --only card
```

Samples are append-only JSONL shards under `data/` (`traces/`, `rejects/`, `judged/`, `final/`).
Each records its teacher (`gpt-oss-120b`), its judge model, licence (`Apache-2.0`), seed and judge scores. The
steps can also be run alone: `uv run --group data data {prompts,generate,judge,retry,filter,report,validate}`
and `uv run --group train train {show,fit,merge}`.

### Idle gate

A GPU phase starts only when there has been no keyboard or mouse input for 15 minutes (Win32
`GetLastInputInfo`) and other processes use under 15% of the GPU (`nvidia-smi`), or `--asleep` is
passed. While a phase runs, the pipeline pauses when there was input in the last 60 seconds or
another process loads the GPU: `llama-server` is stopped, training saves a checkpoint and exits
(through a pause file, with a 15-minute grace before the process is stopped), and the phase
resumes when the start condition holds again. Only processes the pipeline started are stopped,
and by PID. While a pipeline process is on the GPU, other processes' load comes from
`nvidia-smi pmon`; where the driver does not report it (common on Windows), only the input check
applies during the run. Tunables: `--min-idle-minutes`, `--active-seconds`, `--max-gpu-util`,
`--poll-seconds`.

`llama-server` is found through `--llama-server`, `EMPEROR_LLAMA_SERVER`, `PATH`, then the newest
runtime under `%LOCALAPPDATA%\Imperium\data\runtimes\llama.cpp`.

### Release

`uv run release {export,card,publish}` does the same steps outside the pipeline. The model
name, card text, quantisation types and teacher description come from `[release]` in the config,
so another model (Consul) reuses everything by changing that section. The card holds the front
matter (Apache-2.0, base model, tags, `text-generation`), intended use, limits, the eval table
from the latest report, the training recipe, a dataset datasheet and a memory table (KV cache
computed from the model's `config.json`). Nothing is uploaded unless you run:

```powershell
uv run --group publish release publish --repo-id <org>/emperor --gguf-repo-id <org>/emperor-GGUF --confirm
```

It refuses without `--confirm`, reads the token only from `HF_TOKEN` or the Hugging Face login
cache, creates private repos (`--public` to change that) and uploads model files only.

### Weights

The merged model is `weights/emperor-4b-bf16/` (bf16 safetensors, tokenizer and config). The
directory is ignored by Git. The weights go to the Hugging Face Hub only through
`release publish --confirm`; until then they stay on this machine.

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
