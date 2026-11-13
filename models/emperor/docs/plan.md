# Plan: Emperor, Consul and their Lite models, v1 to v2

Revision 2026-11-13. This is the working plan for the four models up to their release on Hugging Face,
and for the retrieval work that follows. [runbook.md](runbook.md) covers how to run each step.

## Goal

The goal is a lean root agent for Imperium that runs natively on a laptop at a usable speed and
uses a long context window on better hardware. The models learn procedure and decomposition in
their weights. Facts, project details and precedents come from specialised retrieval. Hard cases
escalate to a frontier model.

| Model | Base (Qwen3.5, Apache-2.0, thinking) | Role | Target hardware |
| --- | --- | --- | --- |
| Emperor | 4B | Root agent: plans, delegates, checks | any 8 GB laptop (Q4_K_M ≈ 2.7 GB) |
| Emperor Lite | 2B | Root agent for the smallest machines | ≈ 1.4 GB GGUF |
| Consul | 9B | Critic of plans, diffs and agent reports | 16 GB laptops and desktops |
| Consul Lite | 4B | Critic for laptops | any 8 GB laptop |

Consul uses a larger base than Emperor, so it does not share all of the planner's blind spots.
A zero-shot comparison against Phi-4-mini-reasoning and the DeepSeek-R1 distills (1.5B and 7B) can
still change any slot (phase B1).

## Fixed rules

- No training data is generated or judged by Claude or other closed models whose terms forbid it.
  Open datasets are curated only after their licence and generator have been checked.
- Training is bf16 LoRA only, with no QLoRA. Partial-GPU loading (RAM offload) is allowed.
  Quantisation happens only after training.
- Every dataset is complete before any training starts. One model runs at a time. Training runs
  only while the user is idle.
- Hosted teacher calls run only off-peak (DeepSeek's discount hours) unless `--allow-peak` is given.
- Nothing is published to Hugging Face until the user asks and supplies a write token.

## Training data mix

| Share | Source | Licence |
| --- | --- | --- |
| ≈ 45% | Our hosted data. Teacher: DeepSeek V4 Pro (simulator V4 Flash). Judge: Kimi K3.1 Pro. | DeepSeek Open Platform ToS §4.2 permits distillation |
| ≈ 30% | **Research planning from papers** (below) | papers: arXiv CC-BY, PMC OA commercial subset; tasks written by the teacher |
| ≈ 15% | nvidia/Nemotron-Agentic-v1, nvidia/Nemotron-SFT-Agentic-v2, keeping only subsets made by open models | CC-BY-4.0 / Apache-2.0 / MIT |
| ≈ 10% | open-thoughts/OpenThoughts3-1.2M sample, with thinking length capped | Apache-2.0 |

Excluded: Salesforce/APIGen-MT-5k (CC-BY-NC-4.0), SWE-Gym trajectories (no licence, made by
closed models), SWE-smith trajectories (made by Claude), MegaScience/TextbookReasoning
(CC-BY-NC-SA). Every sample records its source, licence and generator. The model cards credit the
CC-BY sources.

### Research planning (a core backbone)

Landmark papers record how a hard problem was broken down. Each paper becomes two tasks:

- **Emperor** sees only what was known before the paper (the problem and prior work) and plans the
  investigation as very small, atomic steps. Each step has its own check and states its
  dependencies, ordered hypothesis → experiment → verification. The paper's actual path is the
  reference, and the judge scores granularity, order and checkability against it.
- **Consul** critiques such a plan with seeded defects (a missing step, a wrong order, a step too
  coarse to check, an unverifiable claim, a broken dependency) against the paper as ground truth.

Papers are selected for impact and licence. Claude or a retrieval agent only selects them; the
teacher writes every task. A held-out set of papers forms an eval split for both models.

### Pattern library

The library holds strategy cards distilled from the papers and from teacher trajectories, both
successes and failures. Each card has its trigger signals, its steps (small and checkable),
pitfalls, when not to use it, and its provenance. Emperor learns to apply, adapt or ignore cards.
Consul learns to flag a misapplied card or a missed obvious one. Precedents: Buffer of Thoughts,
Agent Workflow Memory, ReasoningBank, Self-Discover (to be verified before citing in the cards).

## Retrieval: one specialised lane per weakness

| Small-model weakness | Lane | Not fixed by retrieval, covered instead by |
| --- | --- | --- |
| Invented paths, commands, project structure | **1. Code intelligence:** repo map, LSP symbols, grep, discovered build and test commands (exact lookups) | – |
| Invented APIs | **2. Versioned library docs** from the lockfiles | – |
| Dropping rules when there are many | **3. Workflow rule cards:** atomic, trigger-tagged, 3–8 per step | – |
| Unfamiliar judgment | **4. Pattern library** | novel problems: escalation |
| Repeated mistakes, calibration | **5. Experience:** distilled from runs, failures and escalations | partly Consul |
| Long sessions, lost middle | **6. Session:** task-scoped context, summaries of ejected material | error buildup: checks, Consul |
| Decomposition depth | **7. Research precedents** | – |
| Reasoning depth within one step | – | RL with verifiable rewards, Consul, escalation |
| Tool-call reliability, overthinking | – | training (tool-use data, thinking-length cap) |

**Router.** Deterministic triggers come first (task class, file types, tools in play). Then a small
CPU reranker, a relevance threshold and per-lane token budgets, about 1.5–3K tokens in total for a
4B model. It needs no heavy embedding model, so it works on a laptop. The models train with
retrieval in the loop and with irrelevant passages mixed in (RAFT), so they learn to use the
context, cite it and ignore the rest.

## Phases

**A. Teacher data (running).** Hosted generation and judging for Emperor (2,000 tasks), then
Consul (2,000), with retry rounds. `scripts/full_run.py` holds before training.

**B. Preparation**
1. Zero-shot comparison on the held-out evals: Qwen3.5 2B/4B/9B, Phi-4-mini-reasoning,
   DeepSeek-R1-Distill 1.5B/7B and a frontier root (DeepSeek V4 Pro in the same harness). Pick the
   bases, and keep the scores as baselines.
2. Training pipeline:
   - partial-GPU weight offload for the 9B;
   - the card base and KV layout taken from config;
   - four model configs;
   - the training length measured on the GPU.
3. Open data: convert it, check each generator, record provenance.
4. Research planning, then the pattern library:
   - select the papers;
   - generate a small batch to measure cost, then the user tops up;
   - generate off-peak;
   - in the same pass, add rule cards and pattern cards in context, with distractors.
5. Independent review and merge.

**C. Training** (local, idle only, one model at a time). For each model: train → merge → export
(bf16 + GGUF) → eval. Order: Emperor 4B, Emperor Lite 2B, Consul Lite 4B, Consul 9B. An ablation
on Emperor 4B compares training with and without the grafted data.

**D. Release preparation**
- Model cards: evals against the baselines and the frontier root, licences, attributions,
  DeepSeek terms.
- Measured laptop tok/s on the CPU for each GGUF.
- A smoke test in Imperium (CPU lane, Consul reviewing Emperor).
- Independent review.

**E. Publish.** The user supplies a write token and approves. Four repositories, each with bf16
weights and GGUF files.

**F. v2**
- Imperium v1.5 retrieval: the code-intelligence and docs lanes, the router and reranker,
  task-scoped context.
- Regenerate the core data with all lanes live.
- RL with verifiable rewards: the schema is valid, every goal has a runnable check, subtasks touch
  disjoint files, delegated work completes.
- The experience lane, which distils new cards after each run and escalation, so the system
  improves between training rounds.
- In Imperium: escalate to a frontier model after two Consul rejections or on low confidence.

## Evaluation

The held-out evals cover each model's role, plus:
- the research-planning split;
- an unfamiliar-task split scored with and without the pattern library;
- an ablation that turns off one retrieval lane at a time;
- the rate of invented paths, commands and APIs;
- the gap to the frontier root.

Results are reported in each model card.
