"""Filtering the judged traces into the final set, and the dataset report."""

import json
from collections import Counter
from pathlib import Path

from emperor.data.schema import validate_sample
from emperor.data.shards import iter_records, shard_files


def filter_kept(data_dir: Path) -> dict:
    """Write `final/final.jsonl` from judged samples that were kept and still validate."""
    out = data_dir / "final"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "final.jsonl"
    seen: set[str] = set()
    kept = dropped = 0
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in iter_records(data_dir / "judged"):
            judge = sample.get("meta", {}).get("judge", {})
            if not judge.get("keep") or sample["id"] in seen or not validate_sample(sample).ok:
                dropped += 1
                continue
            seen.add(sample["id"])
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
    temporary.replace(target)
    return {"kept": kept, "dropped": dropped, "path": str(target)}


def build_report(data_dir: Path) -> dict:
    """Counts, pass rate, score histograms and rejection reasons across every stage."""
    prompts = 0
    prompts_file = data_dir / "prompts.jsonl"
    if prompts_file.exists():
        prompts = sum(1 for line in prompts_file.read_text(encoding="utf-8").splitlines() if line)
    traces = list(iter_records(data_dir / "traces"))
    rejects = list(iter_records(data_dir / "rejects"))
    judged = list(iter_records(data_dir / "judged"))

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
        "teacher": "gpt-oss-20b",
        "licence": "Apache-2.0",
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
