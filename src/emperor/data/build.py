"""Filtering the judged traces into the final set, and the dataset report."""

import json
from collections import Counter
from pathlib import Path

from emperor.data.records import current, exclusions
from emperor.data.retry import retry_dirs
from emperor.data.schema import validate_sample
from emperor.data.shards import shard_files


def iter_judged(data_dir: Path):
    """Judged samples of the first pass, then of each retry round (a kept id is used once):
    the latest verdict per id, without excluded pilots."""
    yield from current(data_dir / "judged").values()
    for directory in retry_dirs(data_dir):
        yield from current(directory / "judged").values()


def filter_kept(data_dir: Path) -> dict:
    """Write `final/final.jsonl` from judged samples that were kept and still validate."""
    out = data_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "final.jsonl"
    seen: set[str] = set()
    kept = dropped = 0
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in iter_judged(data_dir):
            judge = sample.get("meta", {}).get("judge", {})
            if not judge.get("keep") or sample["id"] in seen or not validate_sample(sample).ok:
                dropped += 1
                continue
            seen.add(sample["id"])
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    temporary.replace(target)
    return {"kept": kept, "dropped": dropped, "path": str(target)}


def provenance(samples: list[dict]) -> dict:
    """Which teacher, judge, host and terms produced the samples, with counts."""
    teachers, judges, hosts, licences = Counter(), Counter(), Counter(), Counter()
    seen = set()
    for sample in samples:
        meta = sample.get("meta", {})
        if sample.get("id") not in seen:
            seen.add(sample.get("id"))
            teachers[meta.get("teacher", "unknown")] += 1
            hosts[meta.get("host", "unknown")] += 1
            licences[meta.get("licence", "unknown")] += 1
        if meta.get("judge", {}).get("model"):
            judges[meta["judge"]["model"]] += 1
    top = teachers.most_common(1)
    return {
        "teacher": top[0][0] if top else None,
        "teachers": dict(teachers.most_common()),
        "judges": dict(judges.most_common()),
        "hosts": dict(hosts.most_common()),
        "licences": dict(licences.most_common()),
    }


def build_report(data_dir: Path) -> dict:
    """Counts, pass rate, score histograms and rejection reasons across every stage."""
    prompts = 0
    prompts_file = data_dir / "prompts.jsonl"
    if prompts_file.exists():
        prompts = sum(1 for line in prompts_file.read_text(encoding="utf-8").splitlines() if line)
    traces = list(current(data_dir / "traces").values())
    rejects = list(current(data_dir / "rejects").values())
    judged = list(current(data_dir / "judged").values())

    histograms: dict[str, Counter] = {}
    rejection: Counter = Counter()
    kept_by_class: Counter = Counter()
    judged_by_class: Counter = Counter()
    for sample in judged:
        judge = sample["meta"].get("judge", {})
        judged_by_class[sample["task_class"]] += 1
        for name, score in (judge.get("scores") or {}).items():
            histograms.setdefault(name, Counter())[score] += 1
        if judge.get("keep"):
            kept_by_class[sample["task_class"]] += 1
        for reason in judge.get("rejection") or []:
            rejection[reason] += 1
    generation_rejects = Counter(r.get("reason", "unknown") for r in rejects)
    kept = sum(kept_by_class.values())
    retries = {}
    for directory in retry_dirs(data_dir):
        rows = list(current(directory / "judged").values())
        retries[directory.name] = {
            "judged": len(rows),
            "kept": sum(1 for r in rows if r["meta"].get("judge", {}).get("keep")),
        }
    kept_total = kept + sum(r["kept"] for r in retries.values())
    return {
        "prompts": prompts,
        "generated": len(traces),
        "generation_rejected": len(rejects),
        "generation_rejection_reasons": dict(generation_rejects.most_common()),
        "judged": len(judged),
        "kept": kept,
        "pass_rate": round(kept / len(judged), 4) if judged else None,
        "kept_by_class": dict(sorted(kept_by_class.items())),
        "judged_by_class": dict(sorted(judged_by_class.items())),
        "score_histograms": {
            name: {str(score): histograms[name][score] for score in sorted(histograms[name])}
            for name in sorted(histograms)
        },
        "judge_rejection_reasons": dict(rejection.most_common()),
        "retry_rounds": retries,
        "kept_with_retries": kept_total,
        # Read from the samples, so the card names the models that actually served.
        **provenance([*traces, *judged]),
        "excluded_pilots": exclusions(data_dir),
        "shards": {
            stage: len(shard_files(data_dir / stage))
            for stage in ("traces", "rejects", "judged", "final")
        },
    }


def write_report(data_dir: Path) -> dict:
    report = build_report(data_dir)
    (data_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
