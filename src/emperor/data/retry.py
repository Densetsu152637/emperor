"""Retry rounds: prompts whose trace was rejected get another go with a different seed.

Round `k` lives in `<data_dir>/retry-k/` with the same layout as the main directory (prompts,
traces, rejects, judged), so generation, judging and resuming are the same code. The first pass
is never touched: a prompt is kept from the earliest round that produced a kept sample.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from emperor.data.generate import (
    ROUND_SEED_STEP,
    SEED_MASK,
    Chat,
    GenerationConfig,
    run_generation,
)
from emperor.data.judge import MEAN_THRESHOLD, run_judging
from emperor.data.shards import done_ids, iter_records, read_jsonl

DEFAULT_RETRIES = 2


def round_dir(data_dir: Path, k: int) -> Path:
    return data_dir if k == 0 else data_dir / f"retry-{k}"


def retry_dirs(data_dir: Path) -> list[Path]:
    """Existing retry round directories, in round order."""
    found = [p for p in data_dir.glob("retry-*") if p.is_dir() and p.name[6:].isdigit()]
    return sorted(found, key=lambda p: int(p.name[6:]))


def kept_ids(directory: Path) -> set[str]:
    return {
        s["id"]
        for s in iter_records(directory / "judged")
        if s.get("meta", {}).get("judge", {}).get("keep")
    }


def _round_complete(directory: Path) -> bool:
    prompts = directory / "prompts.jsonl"
    if not prompts.exists():
        return False
    ids = {r["id"] for r in read_jsonl(prompts)}
    finished = done_ids(directory / "rejects") | done_ids(directory / "judged")
    return ids <= finished


def prepare_round(data_dir: Path, k: int) -> int:
    """Write `retry-k/prompts.jsonl` from round k-1's failures; returns how many prompts it holds.

    Idempotent: once the file exists it is left alone, so a resumed run keeps the same prompts.
    Round k-1 must be complete (every prompt either judged or rejected); otherwise nothing is
    written and 0 is returned.
    """
    directory = round_dir(data_dir, k)
    target = directory / "prompts.jsonl"
    if target.exists():
        return sum(1 for _ in read_jsonl(target))
    previous = round_dir(data_dir, k - 1)
    if not _round_complete(previous):
        return 0
    won = set()
    for j in range(k):
        won |= kept_ids(round_dir(data_dir, j))
    lines = []
    for record in read_jsonl(previous / "prompts.jsonl"):
        if record["id"] in won:
            continue
        record = {**record, "seed": (record["seed"] + ROUND_SEED_STEP) & SEED_MASK}
        lines.append(json.dumps(record, ensure_ascii=False))
    directory.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    tmp.replace(target)
    return len(lines)


def run_retry_rounds(
    data_dir: Path,
    client: Chat,
    *,
    judge_client: Chat | None = None,
    environment_client: Chat | None = None,
    rounds: int = DEFAULT_RETRIES,
    config: GenerationConfig | None = None,
    threshold: float = MEAN_THRESHOLD,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Regenerate and rejudge the prompts that failed, up to `rounds` times. Resumable.

    `judge_client` and `environment_client` (default: `client`) judge and simulate tool
    results, so each can run at its own reasoning effort.

    Returns counts per round plus `stopped` (and `error`) when the server went away or the
    caller asked to stop; calling again continues where it left off.
    """
    base = config or GenerationConfig()
    summary: dict = {"rounds": {}, "stopped": False}
    for k in range(1, rounds + 1):
        count = prepare_round(data_dir, k)
        if count == 0:
            break
        directory = round_dir(data_dir, k)
        generated = run_generation(
            directory / "prompts.jsonl",
            directory,
            client,
            environment_client,
            config=dataclasses.replace(base, attempt=k),
            should_stop=should_stop,
            log=log,
        )
        judged = {"stopped": False}
        if not generated["stopped"]:
            judged = run_judging(
                directory,
                judge_client or client,
                threshold=threshold,
                should_stop=should_stop,
                log=log,
            )
        summary["rounds"][f"retry-{k}"] = {"prompts": count, "generate": generated, "judge": judged}
        for part in (generated, judged):
            if part.get("stopped"):
                summary["stopped"] = True
                if "error" in part:
                    summary["error"] = part["error"]
        if summary["stopped"]:
            break
    return summary
