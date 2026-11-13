"""Scoring a critique against the ground truth recorded with a task.

A finding identifies a seeded defect when its location points at the defect's anchor and its
text uses one of the defect's keywords. This is a heuristic (a critique that words the problem
very differently is missed), applied identically to training-data filtering and to evaluation.
"""

import re
from dataclasses import dataclass, field

from consul.schema import citation_errors, schema_errors
from consul.tasks import CritiqueTask, Defect

_GENERIC = ("plan", "overall", "whole", "goal", "steps")


def _numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def anchor_hit(location: str, where: str) -> bool:
    loc = location.lower()
    loc_numbers = _numbers(loc)
    for alt in where.split("|"):
        alt_l = alt.lower()
        numbers = _numbers(alt_l)
        if not numbers:  # a named section: "Acceptance", "Summary", "Plan"
            if alt_l in loc or any(word in loc for word in _GENERIC):
                return True
            continue
        if ":l" in alt_l:  # file:Lnn - the file name, or a line within two of the defect
            file_name = alt_l.split(":")[0]
            if file_name in loc and not loc_numbers:
                return True
            if any(abs(n - numbers[-1]) <= 2 for n in loc_numbers):
                return True
        elif not loc_numbers:
            if any(word in loc for word in _GENERIC):
                return True
        elif numbers[0] in loc_numbers:
            return True
    return False


def keyword_hit(finding: dict, defect: Defect) -> bool:
    text = f"{finding.get('impact', '')} {finding.get('fix_or_check', '')}".lower()
    return any(k.lower() in text for k in defect.keywords)


def matches(finding: dict, defect: Defect) -> bool:
    return anchor_hit(finding.get("location", ""), defect.where) and keyword_hit(finding, defect)


@dataclass
class CritiqueScore:
    parsed: bool = False
    format_ok: bool = False
    format_errors: list[str] = field(default_factory=list)
    defects: int = 0  # acceptance-changing defects seeded
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    blocking_findings: int = 0
    false_findings: list[str] = field(default_factory=list)
    cited: int = 0
    bad_citations: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return len(self.found) / self.defects if self.defects else None

    @property
    def false_rate(self) -> float | None:
        return len(self.false_findings) / self.blocking_findings if self.blocking_findings else None

    def to_dict(self) -> dict:
        return {
            "parsed": self.parsed,
            "format_ok": self.format_ok,
            "format_errors": self.format_errors,
            "defects": self.defects,
            "found": self.found,
            "missed": self.missed,
            "blocking_findings": self.blocking_findings,
            "false_findings": self.false_findings,
            "cited": self.cited,
            "bad_citations": self.bad_citations,
        }


def score_critique(task: CritiqueTask, critique: dict | None) -> CritiqueScore:
    """`critique` is the parsed JSON object, or None when the answer held no JSON."""
    seeded = [d for d in task.defects if d.acceptance_changing]
    score = CritiqueScore(defects=len(seeded), missed=[d.id for d in seeded])
    if critique is None:
        return score
    score.parsed = True
    score.format_errors = schema_errors(critique)
    score.format_ok = not score.format_errors
    if not score.format_ok:
        return score
    findings = critique["findings"]
    score.bad_citations = citation_errors(critique, task.urls())
    score.cited = sum(len(f.get("sources", [])) for f in findings)
    found: set[str] = set()
    for finding in findings:
        hits = [d for d in task.defects if matches(finding, d)]
        found.update(d.id for d in hits if d.acceptance_changing)
        if finding["severity"] == "acceptance_changing":
            score.blocking_findings += 1
            if not any(d.acceptance_changing for d in hits):
                score.false_findings.append(finding["id"])
    score.found = sorted(found)
    score.missed = [d.id for d in seeded if d.id not in found]
    return score


def reference_critique(task: CritiqueTask) -> dict:
    """A critique built from the ground truth: what a perfect model would say (dry runs, tests)."""
    findings = []
    for i, d in enumerate(task.defects, 1):
        findings.append(
            {
                "id": f"f{i}",
                "severity": "acceptance_changing" if d.acceptance_changing else "improvement",
                "location": d.where.split("|")[0],
                "impact": d.description,
                "confidence": "high",
                "basis": "check",
                "fix_or_check": f"Address: {d.keywords[0]}. {d.description}",
            }
        )
    blocking = any(f["severity"] == "acceptance_changing" for f in findings)
    return {
        "verdict": "revise" if blocking else "accept",
        "summary": f"{len(findings)} finding(s)." if findings else "No defects found.",
        "findings": findings,
    }
