# Runbook: from baselines to publication

The step-by-step procedure for stages 3–8 of the
[Emperor plan](https://github.com/Densetsu152637/Imperium-AI/blob/docs/dispatcher-model-plan/docs/plans/emperor.md)
on one Windows machine: an RTX 5080 (16 GB VRAM) and 64 GB of RAM. The plan holds the goals
(G1–G17), the context tiers and the teacher licence table; this file only says how to carry the
stages out. Record each stage's outcome in the plan when it is done.

## Ground rules

- **When to run heavy jobs.** Baseline evals, teacher dataset generation and training run only
  while the machine is otherwise nearly idle, ideally overnight. Check first: `nvidia-smi`
  shows no other GPU process and low utilisation, and Task Manager shows the CPU mostly idle.
- **Where models live.** Downloads go to `D:/AI-Models/<owner>/<repo>/` (for example
  `D:/AI-Models/unsloth/Qwen3-4B-Instruct-2507/`). Prefer Unsloth's Hugging Face repositories.
- **Precision.** Train only from bf16 (unquantised) weights: bf16 LoRA or full fine-tuning,
  never QLoRA. Quantise only at export. Quantised GGUFs are fine for eval baselines and for the
  local teacher.
- **Offload.** A model too large for 16 GB may train with CPU/RAM offload. Prefer GPU-only when
  it fits.
- **Teachers.** Only the allowed open-licence teachers in the plan's table produce training
  data. A closed frontier model may be *evaluated* (stage 3 reference) but its outputs never
  enter the dataset.
- **Imperium checks.** Imperium-AI's full Rust checks run in PR CI; locally run only
  `cargo fmt --all` and `cargo check`.
- **Publishing** (stage 8) needs the user's explicit go-ahead.

## Already in place

| Item | Where |
| --- | --- |
| Eval harness (stage 2b) | this repository, `uv run eval` |
| Tool contract and prompts | vendored under `src/emperor/eval/vendor/` (pin in `SOURCE`) |
| Lead base, bf16 (Qwen3.5-4B, thinking on by default, Apache-2.0) | `F:/AI-Models/Qwen/Qwen3.5-4B/` |
| Round 1 base, bf16 (non-thinking) | `D:/AI-Models/unsloth/Qwen3-4B-Instruct-2507/` |
| Lead base, Q4_K_M GGUF | `D:/AI-Models/unsloth/Qwen3-4B-Instruct-2507-GGUF/` |
| Fallback base, Q4_K_M GGUF | `D:/AI-Models/unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` |
| Local teacher, F16 GGUF | `F:/AI-Models/unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-F16.gguf` |
| Local judge, Q4_K_XL GGUF | `D:/AI-Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Round 1 teacher and judge | `D:/AI-Models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf` |
| llama.cpp runtime | Imperium's pinned build (`runtimes/llama-cpp.env` in Imperium-AI) |

Still missing: the Hugging Face organisation (create it on the website; `Imperium.AI` is a
placeholder name), and the eval harness's `imperiumd` backend, which the `e2e` suite (G3) and
teacher traces through real Imperium sessions need.

## One-time setup

1. Install the training stack in a separate uv environment (for example a `train` dependency
   group): PyTorch with CUDA for the RTX 5080 (Blackwell needs a CUDA 12.8+ build), Unsloth,
   TRL, PEFT, `datasets` and Trackio. Pin the exact versions and record them in the recipe.
2. Log in to Hugging Face with `hf auth login` (needed for private datasets and for
   publishing).
3. Check the GPU from Python: `torch.cuda.is_available()` and `torch.cuda.get_device_name()`.

## Stage 3: baselines and base choice

1. For each candidate (Qwen3-4B-Instruct-2507, Qwen3-8B and up to three newer Apache-2.0 or MIT
   models), start llama.cpp's server on its GGUF:

   ```bash
   llama-server -m <model.gguf> --jinja -c <tier tokens> -ngl 99 --port 8080
   ```

2. Run the harness at each context tier the candidate supports, writing one report per run:

   ```bash
   uv run eval --model llamacpp:<name> --suite heldout --context-tier 64k \
     --output eval/results/<name>-64k-heldout.json
   uv run eval --model llamacpp:<name> --suite longctx --context-tier 128k \
     --output eval/results/<name>-128k-longctx.json
   ```

3. Measure the frontier reference for G3 once the `imperiumd` backend exists, with
   `--suite e2e --backend imperiumd --api-ceiling <USD>`. This is evaluation only.
4. Commit a table of every candidate × metric (`eval/results/README.md`), pick the base, set
   the final numeric targets for G1–G4, G10 and G11, and record both in the plan with reasons.

If the 8B model wins, download its bf16 weights (`unsloth/Qwen3-8B`) into `D:/AI-Models/` and
plan for offload in stage 5, because 8B bf16 weights alone take about 16 GB.

## Stage 4: dataset

Target: 20,000–50,000 trajectories, disjoint from the held-out set by task family. Code lives
in `data/`; the dataset goes to a private Hugging Face repository (`<org>/emperor-traces`).

1. **Teacher server.** Run gpt-oss-20b as the teacher with `llama-server --jinja`, overnight.
   Only allowed teachers from the plan's licence table may be used.
2. **Sources** (record the teacher and host for every sample, so a licence problem can be
   removed later):
   1. Teacher traces: the teacher plays root in real Imperium sessions on generated tasks
      (code, research, data, docs, ops). Export the journals and keep only tasks that succeeded
      under budget.
   2. Synthetic routing cases built from the router rules and policy, including "answer
      directly" and "never send `exec` off-machine".
   3. Long-input cases built on `delegate_chunks`.
   4. Negative examples (leaked secrets, over-delegation, thin briefs, needless questions,
      plans missing goals or dependencies) rewritten into correct turns.
   5. Vague-prompt planning: one-line versions of detailed specs, paired with the teacher's full
      plan and its successful execution.
   6. Clarify-or-research cases: gaps labelled "user's decision", "researchable" or "safe
      default"; the reference asks only about the first kind, in one batch with
      recommendations.
3. **Tiers and personalities.** Give every trajectory a context budget from the tiers. Re-run or
   rewrite those that overflow small budgets with earlier delegation and chunking. Vary the
   system prompt's personality (Marcus Aurelius most often, others, and none).
4. **Validate.** `uv run data validate` must report 100 % of samples parsing against the
   vendored schemas, 0 secret-pattern hits, and clean dedup and held-out overlap checks.
5. **Datasheet.** List every source with its licence and provider terms.

### Full run on hosted models (the default)

Teacher and simulator on DeepSeek, judge on Kimi (`configs/hosted.toml`); nothing runs on the
GPU, so the idle gate does not apply. Training still waits for an idle machine.

1. Put the keys in `~/.emperor/.env` (`Deepseek_API_Key=...` and `Kimi_API_Key=...`), or in
   the environment or keyring, and check they resolve (prints the provider and found, never
   the value; a missing key raises an error naming the variable and the file):

   ```powershell
   uv run python -c "from emperor.data.hosted import load_config, read_key; c = load_config(); [print(n, bool(read_key(p, c.env_file))) for n, p in c.providers.items()]"
   ```

2. Check the rates and budgets in `configs/hosted.toml` against the provider pages it cites,
   and set `budget_usd` per provider to what the account may spend on this run.
3. Emperor's data, from the emperor checkout. Rule: DeepSeek generation runs only in its
   discounted off-peak hours (outside 01:00-04:00 and 06:00-10:00 UTC on weekdays; all of the
   weekend). This is the default: at peak the run lets requests in flight finish, starts no
   new DeepSeek call and resumes on its own (`status.json`: phase `waiting_offpeak`,
   `resume_at_utc`). Never pass `--allow-peak` for the full run.

   ```powershell
   uv run hosted --data-dir data/hosted-1 --run-dir runs/hosted-1 --detach
   ```

4. Consul's data, from the consul checkout, the same command. The two use different accounts'
   rate limits only through the shared keys, so they may run together; run them one after the
   other when the budget is tight.
5. Watch `runs/hosted-1/status.json` (spend per provider, dollars per task, projection). On
   status `budget`, raise `budget_usd` and run the same command to resume; `New-Item
   runs/hosted-1/STOP` stops cleanly.
6. Train each model locally, one at a time, on the hosted data: set `[data] path =
   "data/hosted-1/final/final.jsonl"` in `configs/sft.toml`, then run the local pipeline's
   filter (it records the kept count the train phase checks) and training phases:
   `uv run --group data --group train pipeline --detach --data-dir data/hosted-1 --only filter
   --only train --only merge --only export --only eval --only card`.

### Full run on local models (fallback): both datasets first, then training

Generate and judge both models' data before any training. Only one model may run at a time
(teacher, judge, training or eval), and both pipelines serve on port 8081, so **never run two
pipelines at once**: start the next only after the previous one's summary says it finished.
The pipeline itself never runs two models together (the teacher stops before the judge starts).

1. Emperor's data, from the emperor checkout (about 2,000 prompts; generation and judging take
   days and pause whenever the machine is in use):

   ```powershell
   uv run --group data --group train pipeline --detach --only prompts --only generate `
     --only judge --only retry --only filter
   ```

   Wait until `runs/<timestamp>/summary.json` reports `done` (or resume the same run with
   `--resume runs/<timestamp>` and the same `--only` flags after a stop).
2. Consul's data, from the consul checkout, the same way (Consul has no retry rounds):

   ```powershell
   uv run --group data --group train pipeline --detach --only prompts --only generate `
     --only judge --only filter
   ```

3. Training, one model at a time and only while the machine is idle: resume each data run with
   the remaining phases, Emperor first, then Consul once Emperor's summary is `done`:

   ```powershell
   uv run --group data --group train pipeline --detach --resume runs/<timestamp> `
     --only train --only merge --only export --only eval --only card
   ```

`--skip train --skip merge --skip export --skip eval --skip card` also stops after the data,
but records those phases as skipped; `--only` leaves them untouched for the later resume.

## Stage 5: supervised fine-tuning (bf16 LoRA)

The plan says QLoRA; per the user, use bf16 LoRA instead and update the plan's stage 5 text
when starting. Code lives in `train/`.

1. Load the base in bf16 with Unsloth (`load_in_4bit=False`). Add LoRA adapters on the attention
   and MLP projections (rank 16–64 to start). Enable gradient checkpointing.
2. Find the longest sequence that fits: step the length up on a few batches until it runs out
   of memory, then bucket the dataset to that length. Offload to RAM only if the base does not
   fit on the GPU.
3. Train with loss on assistant turns only, logging to Trackio. Save checkpoints regularly.
4. Sweep the checkpoints on the dev split with the harness (serve each merged checkpoint or
   adapter through llama.cpp or a Python OpenAI-compatible server).
5. Accept the best checkpoint when it meets G1 and G2 on the held-out set, loses no more than
   2 points on a small general instruction-following set, and G9 holds.
6. Write the recipe: base, data revision, hyperparameters and exact library versions.

## Stage 6: preference tuning, export and measurement

1. **DPO**, only if stage 5 misses G1–G5: build pairs ranked by verifiable signals (schema
   validity, routing match, brief judge, plan cost, leakage) and train in bf16. Try GRPO only if
   DPO falls short and a local run fits.
2. **Merge** the adapter into the bf16 base.
3. **Export** GGUF with llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`: Q4_K_M,
   Q5_K_M, Q8_0, IQ4_XS and Q3_K_M (an importance matrix helps IQ4_XS and Q3_K_M). Also
   export AWQ.
4. **Measure** every context tier × quant × KV type (f16, q8_0, q4_0): peak VRAM from
   `nvidia-smi` and long-context quality with `--suite longctx`. Emulate smaller cards on the
   16 GB GPU; the peak usage is the measurement.
5. **Accept** when G1, G2, G4, G5, G6, G10, G11 and G12 are met and G3 is measured on `e2e`.
   Replace the plan's estimated VRAM table with the measured one.

## Stage 7: Imperium integration

In Imperium-AI, on a branch off `experimental`:

1. Add the bundled model-table entry and the "Emperor root" preset (session model Emperor,
   `routing.prefer = "cheapest"`, cloud rows for `strong` and `frontier`, `marcus_aurelius`
   personality by default).
2. Teach the memory planner the context tiers and q8_0/q4_0 KV caches.
3. Make `imperium models pull <org>/emperor-GGUF` work and show the model in the desktop Models
   panel.
4. Accept per the plan: G8 with a fresh `IMPERIUM_HOME`, and the `context_tier` tests picking
   64K, 128K and 256K at 8, 12 and 16 GB. CI runs the full checks.

## Stage 8: publication

Ask the user before publishing anything.

1. **Export** (the pipeline does this after the merge; to run it alone on a finished run):
   `uv run --group train release export` writes `weights/emperor-4b-gguf/` (bf16 plus Q8_0,
   Q6_K, Q4_K_M, set by `[release] gguf_types`). `convert_hf_to_gguf.py` comes from a shallow
   clone of llama.cpp `b12180` in `tools/llama.cpp` (ignored by Git); `llama-quantize` is
   Imperium's runtime binary.
2. **Model card**: `uv run release card` writes `README.md` into the bf16 and GGUF directories
   from the config, the latest `runs/*/eval.json`, `data/report.json` and the export listing:
   intended use, limits, the eval table, training recipe, dataset datasheet and a VRAM table.
   Read it and fix the limits text in `[release]` before publishing.
3. **Repositories** in the Hugging Face organisation: `emperor` (bf16) and `emperor-GGUF`.
   Log in with `hf auth login` (or set `HF_TOKEN`); the tool never takes a token argument.
4. **Upload**, after the user's go-ahead:
   `uv run --group publish release publish --repo-id <org>/emperor --gguf-repo-id <org>/emperor-GGUF --confirm`.
   Without `--confirm` it refuses and uploads nothing. Repos are created private; `--public`
   makes them public.
5. **Recipe**: publish the training recipe and data pipeline from this repository.
6. **Fold the plan** into Imperium's docs 03, 04, 12 (roadmap row) and 13 (decision record,
   including the NCM decision), then delete the plan. `markdownlint-cli2` must be clean.

## Consul

Consul (the critic model) follows the same pipeline once Emperor's is proven. Its own plan is
in the [consul repository](https://github.com/Densetsu152637/consul/blob/main/docs/plans/consul.md).
Use the same ground rules, harness pattern, training and export steps, and publication
go-ahead.
