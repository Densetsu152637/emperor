# Plan: Consul, an open-weight critic model

Revision: 2026-11-12 r4 (base changed to Qwen3.5-4B, a hybrid Gated DeltaNet model with thinking on by default, trained on the teacher's reasoning so it thinks before it critiques; teacher gpt-oss-120b, judge Qwen3.8-27B; the data, training and eval pipeline is implemented in this repository, nothing has been trained yet)

**Consul** is Emperor's counterweight. Emperor (Imperium's orchestrator model, plan in the
Imperium repository at `docs/plans/emperor.md`) turns vague prompts into elaborate plans and
delegates them. Consul's job is to find what is wrong with those plans and their results
before the user pays for it: unsupported assumptions, missed risks, steps that failed when
others tried them, and claims without evidence. Rome had two consuls so that neither could act
unchecked; Consul exists so the Emperor never does.

## Character

- **Speaks as Niccolò Machiavelli by default**, the realist counterpart to Emperor's Marcus
  Aurelius: plans are judged by how things actually go, not how they ought to. The
  personality comes from the Agent-Custom-Workflow catalogue (`personalities/niccolo_machiavelli.md`)
  and, as the workflow requires, shapes only user-facing tone: findings, tool calls, and
  reports stay neutral and precise. It lives in the system prompt, not the weights.
- **Evidence over opinion.** Consul tests a plan against precedent: it searches the web, reads
  documentation, issue trackers, post-mortems, and papers, and reports what happened when
  others tried the same approach, with citations. A finding without a check or a source is
  labelled as judgement.
- **Findings first, ranked.** Each finding has a location, impact, confidence, and the smallest
  fix or check, matching the workflow's critic profile. It separates acceptance-changing
  findings from improvements and out-of-scope remarks.
- **Hard to anchor, easy to resume.** On a delta review it re-checks the revision rather than
  defending its earlier findings, and says when a finding is resolved.

## Goals

Provisional until baselines are measured (stage 3). Each is checked by the eval harness.

| ID | Goal | Verified by |
| -- | ---- | ----------- |
| C1 | On a held-out set of plans, diffs and reports with seeded defects, Consul finds ≥ 80 % of the acceptance-changing defects, with ≤ 15 % of its acceptance-changing findings being false | `eval --metric defects` |
| C2 | ≥ 90 % of findings that cite precedent cite a source that exists and supports the claim | `eval --metric citations` (judge plus link check) |
| C3 | Every finding has location, impact, confidence, and a fix or check, in the critic result format; output validates against the schema | `eval --metric format` |
| C4 | In multi-round critic loops (triage then improvement pass), Consul's delta reviews converge: no repeated or reversed findings on ≥ 90 % of loops | `eval --suite loop` |
| C5 | Fits the machine: the default build (Qwen3.5-4B, Q4_K_M or Q8_0 GGUF) runs on the CPU from system RAM, or on the GPU or split between them, beside Emperor, and produces a typical critique within its time budget | measured run with Imperium's CPU lane (Emperor plan stage 2f) and a GPU or partial-GPU run |
| C6 | Weights, quants, card, eval report, and recipe on Hugging Face under Apache-2.0, every teacher permitted (Emperor plan's teacher table) | Hub pages; datasheet |

## Fit with the Agent-Custom-Workflow critic loop

The workflow's root review (`workflows/root-review.md`) already defines the loop Consul is
trained for:

- **Gates.** A plan gate after a plan is written or revised, and an implementation gate on the
  final diff. The critic is a fresh-context leaf with the critic profile, never the author.
- **Triage.** Findings that could change acceptance are fixed or rebutted; the revised
  artifact goes back to the *same* critic for a delta review, repeated while rounds keep
  producing accepted findings.
- **Improvement pass.** After convergence, at most three ranked in-scope improvements per round
  until none are worth adopting.

Consul is trained on exactly these turns: first review, delta review after revisions, rebuttal
handling (concede or maintain with evidence), and improvement rounds. In Imperium it is the
default model for `critique`, `review`, and `verify` tasks on the CPU lane, so Emperor's plan
gate dispatches it automatically.

## When Consul does not fit: Emperor critiques itself in a separate context

If the machine cannot hold Consul (too little RAM, or the CPU lane is busy past the task's
deadline), Emperor still does not just "think harder" in its own context. It spawns a subagent
of **its own model** with a fresh context, the critic profile, and only the artifact and its
goals, never Emperor's reasoning or transcript. A separate context cannot see the author's
justifications, so it has to challenge the plan from the outside, as the workflow's "never the
author" rule intends. Only if no subagent can run at all does it fall back to the workflow's
disclosed self-check. This fallback is Imperium work and is recorded in the Emperor plan.

## Implementation status

The pipeline lives in this repository and reuses Emperor's package (a pinned git dependency)
for the idle gate, local teacher client, shards, bf16 LoRA training and merge, GGUF export and
model cards. Consul adds only what is critic-specific:

- the critique result schema (`consul.schema`) and Consul's system prompt (`consul.prompts`);
- seeded-defect tasks (`consul.tasks`): plans, diffs and agent reports from several task
  classes with injected defects whose ground truth is recorded; about one task in six is clean;
- the teacher critique step, the schema, citation and ground-truth checks, and the local judge
  (`consul.data`); a critique is kept only if it finds every seeded defect, raises no false
  blocking finding, cites only provided URLs, the judge approves, and the teacher's reasoning
  came with it. The reasoning is stored as `reasoning_content` and trained as the student's
  thinking; the judge sees only the critique;
- the held-out harness (`uv run eval`) for C1 (recall and false findings), C3 (format) and a
  simple C2 (a cited URL must appear in the provided context), scored on the answer after the
  model's thinking is removed. C4 is a TODO. The held-out set
  uses another seed range and held-out vocabulary and is checked for overlap with the
  training tasks before every run.

Not yet done: baselines, any training run, the CPU-lane measurement for C5, delta-review and
rebuttal rounds (C4), real (non-synthetic) plans and diffs, and web-research tool calls.

## Stages

1. **Contract.** Reuse Imperium's exported tool schemas (Emperor plan stage 1) and add the
   critic result schema and Consul's root prompt, including the Machiavelli personality slot.
2. **Eval harness.** Seeded-defect plans and diffs, a citation checker, and loop replays
   (C1–C4), sharing infrastructure with the Emperor harness where possible.
3. **Baselines.** Zero-shot Qwen3.5-4B (and other small Apache-2.0/MIT dense
   models) measured on C1–C5, on the CPU and the GPU.
4. **Dataset.** Critique trajectories from permitted teachers running the workflow's critic
   loop on real Imperium plans and diffs, with web research tool calls; seeded-defect
   variants; rebuttal and delta-review rounds; negative examples (unsupported claims,
   anchoring, invented citations) rewritten correctly.
5. **Training.** bf16 LoRA on the full-precision 4B weights, exactly Emperor's recipe (no
   QLoRA, no 4/8-bit loading); quantise only after training. The 4B trains locally on the
   16 GB GPU, with partial offload to system RAM when another process holds some VRAM.
6. **Packaging and integration.** GGUF quants for CPU inference, the Imperium CPU-lane catalog
   entry, and the default `critique` routing.
7. **Release.** Hugging Face (`Imperium.AI/consul`, placeholder organisation) and docs.

## Open questions

| Question | Owner |
| -------- | ----- |
| ~~Can a 30B MoE be QLoRA-trained locally?~~ Decided: no. A 30B MoE cannot be bf16-LoRA-trained on 16 GB, so the base is a 4B model (now Qwen3.5-4B). Revisit only if baselines show 4B is too weak for C1. | Stage 3 measurement |
| Which search providers Consul's training sessions use, and their terms | Maintainers |
