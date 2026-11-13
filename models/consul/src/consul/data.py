"""Consul's dataset: tasks, teacher critiques, ground-truth and judge checks, the final set.

Reuses Emperor's shard writer, local teacher client and loopback guard. Nothing here calls a
hosted API: the teacher and the judge are the local open-weight model behind llama-server.
"""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from emperor.data.build import provenance
from emperor.data.generate import Chat
from emperor.data.judge import judge_answer
from emperor.data.parallel import run_items
from emperor.data.records import current, exclusions, fingerprint
from emperor.data.shards import ShardWriter, read_jsonl
from emperor.data.teacher import (
    JUDGE_MODEL,
    TEACHER_LICENCE,
    TEACHER_MODEL,
    TeacherError,
    TeacherOutputError,
)

from consul import prompts
from consul.overlap import DEFAULT_THRESHOLD, OverlapIndex, overlap_report
from consul.schema import CritiqueParseError, extract_json_object, schema_errors
from consul.score import score_critique
from consul.tasks import CritiqueTask, dumps, generate_tasks

HELDOUT_SEED_OFFSET = 1_000_003  # held-out seeds start far from the training range
MIN_RECALL = 1.0
MAX_FALSE_RATE = 0.0
JUDGE_MEAN = 4.0
JUDGE_MIN = 3
# The teacher's reasoning is stored and becomes the student's thinking. Low effort gave about 65
# tokens of it in the pilot; medium about 330, and critiques stayed under 2.4k tokens.
GENERATE_EFFORT = "medium"
# The judge answers without thinking first: on the pilot's ten critiques it kept and rejected
# exactly the same ones as with thinking (scores within 0.6 a criterion) in 10 s instead
# of 135 s a critique. The ground-truth check carries most of the decision.
JUDGE_THINKING = False
# Judge sampling: Qwen's recommended 0.6 while it thinks (greedy decoding can loop inside a long
# think block); near-greedy when the verdict is the whole answer, for repeatable scores.
JUDGE_TEMPERATURE = {True: 0.6, False: 0.2}


def model_name(gguf: str | Path) -> str:
    """The name recorded for a served GGUF: its file name without `.gguf`, so a sample names the
    exact weights and quantisation that produced or judged it."""
    name = str(gguf).replace("\\", "/").rsplit("/", 1)[-1]
    return name[: -len(".gguf")] if name.lower().endswith(".gguf") else name


def served_model(client) -> str | None:
    """The model a running llama-server reports at /v1/models, as `model_name` records it."""
    try:
        response = client._http.get(f"{client.base_url}/models")
        response.raise_for_status()
        return model_name(response.json()["data"][0]["id"])
    except Exception:
        return None


# -- tasks ---------------------------------------------------------------------------------


def load_tasks(path: Path) -> list[CritiqueTask]:
    return [CritiqueTask.from_dict(r) for r in read_jsonl(path)] if path.exists() else []


def write_train_tasks(data_dir: Path, count: int, seed: int) -> dict:
    """Append training tasks (training vocabulary) until `prompts.jsonl` holds `count`."""
    path = data_dir / "prompts.jsonl"
    existing = load_tasks(path)
    need = count - len(existing)
    if need > 0:
        tasks = generate_tasks(need, seed + len(existing), seen_ids={t.id for t in existing})
        data_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                handle.write(dumps(task) + "\n")
    return {"tasks": max(count, len(existing)), "added": max(need, 0)}


def write_heldout(
    data_dir: Path,
    count: int,
    seed: int = HELDOUT_SEED_OFFSET,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """The held-out set: another seed range and vocabulary, never overlapping the training set.

    Candidates that duplicate or closely copy a training task are skipped; the overlap report
    (`data/eval/overlap.json`) is written for the record. An existing set is kept as it is.
    """
    out = data_dir / "eval"
    target = out / "heldout.jsonl"
    train = load_tasks(data_dir / "prompts.jsonl")
    if not target.exists():
        index = OverlapIndex(train)
        tasks = generate_tasks(
            count, seed, held=True, avoid=lambda task: index.closest(task)[0] >= threshold
        )
        out.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(dumps(t) + "\n" for t in tasks), encoding="utf-8", newline="\n")
    report = overlap_report(train, load_tasks(target), threshold)
    (out / "overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


# -- teacher critiques ---------------------------------------------------------------------


def critique_text(critique: dict) -> str:
    """The stored assistant answer: the critique as indented JSON."""
    return json.dumps(critique, indent=1, ensure_ascii=False)


def make_sample(
    task: CritiqueTask,
    critique: dict,
    reasoning: str,
    teacher_model: str = TEACHER_MODEL,
    provenance: dict | None = None,
) -> dict:
    """A training sample. The assistant turn carries the teacher's reasoning as
    `reasoning_content` beside the critique; Emperor's trainer renders it as the thinking
    model's `<think>` block, so the student learns to reason before it critiques."""
    return {
        "id": task.id,
        "task_class": task.task_class,
        "kind": task.kind,
        "messages": [
            {"role": "system", "content": prompts.CONSUL_SYSTEM},
            {"role": "user", "content": prompts.user_message(task)},
            {
                "role": "assistant",
                "content": critique_text(critique),
                "reasoning_content": reasoning,
            },
        ],
        "meta": {
            "teacher": teacher_model,
            "licence": TEACHER_LICENCE,
            "seed": task.seed,
            "host": "local llama.cpp",
            **(provenance or {}),  # a hosted run's host and terms
            "clean": task.clean,
        },
    }


# Draws per task (new seed each) until one meets the ground truth. Two: in hosted pilot 4 the
# first draw passed for 6 tasks and the second for 1; the third rescued none, and every task
# that failed all three paid for it.
GENERATE_ATTEMPTS = 2
ATTEMPT_SEED_STEP = 1_000_003


def teach_one(
    task: CritiqueTask,
    teacher: Chat,
    *,
    temperature: float = 0.4,
    attempts: int = GENERATE_ATTEMPTS,
    notes: list | None = None,
    teacher_model: str = TEACHER_MODEL,
    provenance: dict | None = None,
) -> tuple[dict | None, str | None]:
    """One critique from the teacher with its reasoning, or (None, reason).

    A draw must parse, validate, cite only provided sources, carry reasoning and meet the
    task's ground truth (every seeded acceptance-changing defect found, no false blocking
    finding). Otherwise the teacher is asked again with another seed, up to `attempts` draws,
    and the last reason is returned. Consul has no retry rounds, and in the pilots about half
    of the critiques failed the ground truth only at the judge step, after the teacher time
    was spent; the judge still decides on the rest. `notes`, when given, collects what each
    failed draw got wrong (finding locations and impacts), for the reject record.
    """
    reason = "no_attempt"
    for attempt in range(attempts):
        try:
            sample, reason = _draw(
                task,
                teacher,
                temperature,
                task.seed + attempt * ATTEMPT_SEED_STEP,
                notes,
                teacher_model,
                provenance,
            )
        except TeacherOutputError:  # one unparsable output costs this draw, not the task's rest
            sample, reason = None, "teacher_output_unparsable"
        if sample is not None:
            sample["meta"]["draws"] = attempt + 1  # which draw passed (what the extra draws buy)
            return sample, None
    return None, reason


def _draw(
    task: CritiqueTask,
    teacher: Chat,
    temperature: float,
    seed: int,
    notes: list | None = None,
    teacher_model: str = TEACHER_MODEL,
    provenance: dict | None = None,
) -> tuple[dict | None, str]:
    reply = teacher.chat(
        [
            {"role": "system", "content": prompts.teacher_system()},
            {"role": "user", "content": prompts.user_message(task)},
        ],
        temperature=temperature,
        seed=seed,
        max_tokens=8192,  # gpt-oss reasons first; 3000 often left no room for the answer
    )
    try:
        critique = extract_json_object(reply.get("content") or "")
    except CritiqueParseError:
        return None, "unparsable"
    if schema_errors(critique):
        return None, "schema_invalid"
    score = score_critique(task, critique)
    if score.bad_citations:
        return None, "invented_citation"
    reasoning = (reply.get("reasoning_content") or "").strip()
    if not reasoning:
        return None, "empty_reasoning"
    if notes is not None and (score.missed or score.false_findings):
        notes.append(
            {
                "missed": score.missed,
                "false": [
                    f"{f['location']}: {f['impact'][:160]}"
                    for f in critique["findings"]
                    if f["id"] in score.false_findings
                ],
            }
        )
    if score.recall is not None and score.recall < MIN_RECALL:
        return None, "missed_defect"
    if score.false_rate is not None and score.false_rate > MAX_FALSE_RATE:
        return None, "false_finding"
    return make_sample(task, critique, reasoning, teacher_model, provenance), ""


def run_generation(
    data_dir: Path,
    teacher: Chat,
    *,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
    teacher_model: str = TEACHER_MODEL,
    workers: int = 1,
    provenance: dict | None = None,
) -> dict:
    """Critiques for every task not yet done; `workers` tasks at once (hosted teachers)."""
    traces = ShardWriter(data_dir / "traces", "traces")
    rejects = ShardWriter(data_dir / "rejects", "rejects")
    # The latest record per id counts, and an excluded pilot's tasks are done again.
    finished = set(current(data_dir / "traces")) | set(current(data_dir / "rejects"))
    todo = [t for t in load_tasks(data_dir / "prompts.jsonl") if t.id not in finished]
    if limit is not None:
        todo = todo[:limit]
    counts = {"todo": len(todo), "accepted": 0, "rejected": 0, "stopped": False}

    def work(task: CritiqueTask):
        notes: list = []
        try:
            sample, reason = teach_one(
                task, teacher, notes=notes, teacher_model=teacher_model, provenance=provenance
            )
        except TeacherError:
            raise  # the server is gone or the budget is spent: stop, the id stays to do
        except TeacherOutputError:
            sample, reason = None, "teacher_output_unparsable"
        except Exception as error:  # one bad sample must not end an overnight run
            sample, reason = None, f"error:{type(error).__name__}"
        return sample, reason, notes

    def handle(task: CritiqueTask, result) -> None:
        sample, reason, notes = result
        if sample:
            traces.write(sample)
            counts["accepted"] += 1
        else:
            record = {
                "id": task.id,
                "task_class": task.task_class,
                "reason": reason,
                **(provenance or {}),
            }
            if notes:
                record["draws"] = notes
            rejects.write(record)
            counts["rejected"] += 1
        log(f"{task.id} {'ok' if sample else reason}")

    outcome = run_items(todo, work, handle, workers=workers, should_stop=should_stop)
    counts["stopped"] = outcome["stopped"]
    if "error" in outcome:
        counts["error"] = outcome["error"]
    return counts


# -- judging -------------------------------------------------------------------------------


def parse_judge(text: str) -> dict:
    data = extract_json_object(text)
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise CritiqueParseError("missing scores")
    clean = {}
    for name in prompts.JUDGE_CRITERIA:
        value = scores.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not 1 <= value <= 5:
            raise CritiqueParseError(f"bad score for {name}")
        clean[name] = int(value)
    if not isinstance(data.get("pass"), bool):
        raise CritiqueParseError("pass must be true or false")
    return {"scores": clean, "pass": data["pass"], "reason": str(data.get("reason", ""))}


def decide(
    truth: dict,
    judge: dict,
    *,
    threshold: float = JUDGE_MEAN,
    min_recall: float = MIN_RECALL,
    max_false_rate: float = MAX_FALSE_RATE,
) -> tuple[bool, list[str]]:
    """Keep a critique only if it matches the ground truth and the judge approves."""
    reasons = []
    recall, false_rate = truth["recall"], truth["false_rate"]
    if recall is not None and recall < min_recall:
        reasons.append("missed_defect")
    if false_rate is not None and false_rate > max_false_rate:
        reasons.append("false_finding")
    if not judge["pass"]:
        reasons.append("judge_fail")
    scores = judge["scores"]
    if sum(scores.values()) / len(scores) < threshold:
        reasons.append("below_threshold")
    reasons += [f"low_{n}" for n, v in scores.items() if v < JUDGE_MIN]
    return not reasons, reasons


def run_judging(
    data_dir: Path,
    client: Chat,
    *,
    threshold: float = JUDGE_MEAN,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
    model: str = JUDGE_MODEL,
    thinking: bool | None = None,
    workers: int = 1,
    settings: dict | None = None,
) -> dict:
    """`thinking` defaults to the client's `enable_thinking` template option; `workers`
    critiques are judged at once (hosted judges)."""
    if thinking is None:
        thinking = bool(getattr(client, "template_kwargs", {}).get("enable_thinking", True))
    writer = ShardWriter(data_dir / "judged", "judged")
    # With `settings`, a critique judged under other settings is judged again (judge tokens
    # only) and the latest verdict counts.
    stamp = fingerprint(settings) if settings else None
    finished = {
        i
        for i, r in current(data_dir / "judged").items()
        if stamp is None or r.get("meta", {}).get("judge", {}).get("settings") == stamp
    }
    tasks = {t.id: t for t in load_tasks(data_dir / "prompts.jsonl")}
    counts = {"judged": 0, "kept": 0, "stopped": False}
    todo = [s for i, s in current(data_dir / "traces").items() if i not in finished and i in tasks]
    if limit is not None:
        todo = todo[:limit]

    def work(sample: dict) -> tuple[dict, dict]:
        task = tasks[sample["id"]]
        answer = sample["messages"][-1]["content"]
        score = score_critique(task, extract_json_object(answer))
        truth = {
            "recall": score.recall,
            "false_rate": score.false_rate,
            "missed": score.missed,
            "false_findings": score.false_findings,
        }
        try:
            reply = client.chat(
                [
                    {"role": "system", "content": prompts.JUDGE_SYSTEM},
                    {"role": "user", "content": prompts.judge_user(task, answer)},
                ],
                temperature=JUDGE_TEMPERATURE[thinking],
                seed=task.seed,
                max_tokens=12288,  # room for any thinking before the verdict
            )
            judge = parse_judge(judge_answer(reply))
            keep, reasons = decide(truth, judge, threshold=threshold)
            verdict = {**judge, "model": model, "keep": keep, "rejection": reasons}
            if stamp:
                verdict["settings"] = stamp
        except (TeacherOutputError, CritiqueParseError) as error:
            verdict = {"keep": False, "rejection": ["judge_unparseable"], "reason": str(error)}
        return truth, verdict  # a TeacherError stops the phase; the id stays to do

    def handle(sample: dict, result: tuple[dict, dict]) -> None:
        truth, verdict = result
        sample["meta"]["truth"] = truth
        sample["meta"]["judge"] = verdict
        writer.write(sample)
        counts["judged"] += 1
        counts["kept"] += bool(verdict["keep"])
        log(f"{sample['id']} {'keep' if verdict['keep'] else verdict['rejection']}")

    outcome = run_items(todo, work, handle, workers=workers, should_stop=should_stop)
    counts["stopped"] = outcome["stopped"]
    if "error" in outcome:
        counts["error"] = outcome["error"]
    return counts


# -- final set and report ------------------------------------------------------------------


def filter_kept(data_dir: Path) -> dict:
    out = data_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "final.jsonl"
    seen: set[str] = set()
    kept = dropped = no_reasoning = 0
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in current(data_dir / "judged").values():
            judge = sample.get("meta", {}).get("judge", {})
            try:
                valid = not schema_errors(extract_json_object(sample["messages"][-1]["content"]))
            except CritiqueParseError:
                valid = False
            if not judge.get("keep") or sample["id"] in seen or not valid:
                dropped += 1
                continue
            if not (sample["messages"][-1].get("reasoning_content") or "").strip():
                no_reasoning += 1  # from before reasoning was kept; a thinking model needs it
                dropped += 1
                continue
            seen.add(sample["id"])
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    temporary.replace(target)
    return {"kept": kept, "dropped": dropped, "no_reasoning": no_reasoning, "path": str(target)}


def build_report(data_dir: Path) -> dict:
    traces = list(current(data_dir / "traces").values())
    rejects = list(current(data_dir / "rejects").values())
    judged = list(current(data_dir / "judged").values())
    kept_by_class: Counter = Counter()
    judged_by_class: Counter = Counter()
    rejection: Counter = Counter()
    for sample in judged:
        verdict = sample["meta"].get("judge", {})
        judged_by_class[sample["task_class"]] += 1
        if verdict.get("keep"):
            kept_by_class[sample["task_class"]] += 1
        rejection.update(verdict.get("rejection") or [])
    kept = sum(kept_by_class.values())
    return {
        "prompts": len(load_tasks(data_dir / "prompts.jsonl")),
        "generated": len(traces),
        "generation_rejected": len(rejects),
        "generation_rejection_reasons": dict(Counter(r.get("reason", "?") for r in rejects)),
        "judged": len(judged),
        "kept": kept,
        "pass_rate": round(kept / len(judged), 4) if judged else None,
        "kept_by_class": dict(sorted(kept_by_class.items())),
        "judged_by_class": dict(sorted(judged_by_class.items())),
        "judge_rejection_reasons": dict(rejection.most_common()),
        # Read from the samples, so the card names the models that actually served.
        **provenance([*traces, *judged]),
        "excluded_pilots": exclusions(data_dir),
    }


def write_report(data_dir: Path) -> dict:
    report = build_report(data_dir)
    (data_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
