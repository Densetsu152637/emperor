"""The critic result format: what Consul returns for a plan, a diff or an agent report.

One JSON object. The schema uses the subset of JSON Schema that Emperor's validator understands
(type, properties, required, items, enum, minItems), and `check_critique` adds the rules that a
schema cannot state: the verdict follows the findings, ids are unique, and a finding that rests
on a source cites only URLs that appear in the context it was given.
"""

import json
import re

from emperor.eval.contract import validate

SEVERITIES = ["acceptance_changing", "improvement", "out_of_scope"]
CONFIDENCES = ["high", "medium", "low"]
BASES = ["check", "source", "judgement"]
VERDICTS = ["accept", "revise", "reject"]
MAX_FINDINGS = 12

FINDING_SCHEMA = {
    "type": "object",
    "required": ["id", "severity", "location", "impact", "confidence", "basis", "fix_or_check"],
    "properties": {
        "id": {"type": "string", "description": "f1, f2, ... unique within the result"},
        "severity": {"type": "string", "enum": SEVERITIES},
        "location": {
            "type": "string",
            "description": "Where: 'Step 4', 'orders.py:L13', 'Claim 2', 'Acceptance'",
        },
        "impact": {"type": "string", "description": "What goes wrong if this stands"},
        "confidence": {"type": "string", "enum": CONFIDENCES},
        "basis": {
            "type": "string",
            "enum": BASES,
            "description": "check: reasoned from the artifact; source: backed by a cited "
            "source; judgement: opinion with no check or source",
        },
        "fix_or_check": {"type": "string", "description": "The smallest fix or check"},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs from the provided context that support the finding",
        },
    },
}

CRITIQUE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "summary", "findings"],
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        "resolved": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Delta reviews: ids of earlier findings that the revision fixed",
        },
    },
}

_URL = re.compile(r"https?://[^\s)\]>\"']+")


class CritiqueParseError(ValueError):
    """The model's answer holds no JSON object."""


def extract_json_object(text: str) -> dict:
    """The first balanced JSON object in `text` (fences and surrounding prose are ignored)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    if start < 0:
        raise CritiqueParseError("no JSON object in the answer")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : i + 1])
                except json.JSONDecodeError as error:
                    raise CritiqueParseError(f"invalid JSON: {error.msg}") from error
                if not isinstance(value, dict):
                    raise CritiqueParseError("the JSON value is not an object")
                return value
    raise CritiqueParseError("unterminated JSON object")


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def strip_thinking(text: str) -> str:
    """The answer without the model's reasoning.

    A thinking model writes `<think>...</think>` before its answer; the Qwen3 Thinking template
    opens the block in the prompt, so the reply may carry only the closing tag. Everything up to
    the last `</think>` is reasoning. A block opened but never closed (the model was cut off
    mid-thought) is reasoning to the end, so no answer is left.
    """
    text = text or ""
    close = text.rfind(_THINK_CLOSE)
    if close >= 0:
        text = text[close + len(_THINK_CLOSE) :]
    opened = text.find(_THINK_OPEN)
    if opened >= 0:
        text = text[:opened]
    return text.strip()


def extract_answer(text: str) -> dict:
    """The critique in a model's reply: its reasoning removed, then the first JSON object."""
    return extract_json_object(strip_thinking(text))


def schema_errors(critique: dict) -> list[str]:
    """Schema violations plus the rules the schema cannot express (empty means valid)."""
    errors = validate(critique, CRITIQUE_SCHEMA)
    if errors:
        return errors
    findings = critique["findings"]
    if len(findings) > MAX_FINDINGS:
        errors.append(f"$.findings: more than {MAX_FINDINGS} findings")
    ids = [f["id"] for f in findings]
    if len(set(ids)) != len(ids):
        errors.append("$.findings: duplicate ids")
    for i, finding in enumerate(findings):
        for key in ("location", "impact", "fix_or_check"):
            if not finding[key].strip():
                errors.append(f"$.findings[{i}].{key}: empty")
        errors.extend(validate(finding.get("sources", []), FINDING_SCHEMA["properties"]["sources"]))
        if finding["basis"] == "source" and not finding.get("sources"):
            errors.append(f"$.findings[{i}]: basis 'source' needs at least one source")
    blocking = any(f["severity"] == "acceptance_changing" for f in findings)
    if blocking and critique["verdict"] == "accept":
        errors.append("$.verdict: 'accept' with acceptance-changing findings")
    if not blocking and critique["verdict"] != "accept":
        errors.append("$.verdict: no acceptance-changing finding, so the verdict must be 'accept'")
    for rid in critique.get("resolved", []):
        if rid in ids:
            errors.append(f"$.resolved: {rid} is also reported as a finding")
    return errors


def cited_urls(critique: dict) -> list[str]:
    """Every URL the findings cite, in order."""
    urls: list[str] = []
    for finding in critique.get("findings", []):
        sources = finding.get("sources")
        if isinstance(sources, list):
            urls.extend(s for s in sources if isinstance(s, str))
    return urls


def citation_errors(critique: dict, allowed_urls: set[str]) -> list[str]:
    """Citations that are not URLs from the provided context (invented or altered sources)."""
    return [
        f"cited URL not in the context: {u}" for u in cited_urls(critique) if u not in allowed_urls
    ]


def urls_in(text: str) -> set[str]:
    return {u.rstrip(".,;") for u in _URL.findall(text)}


CRITIQUE_EXAMPLE = {
    "verdict": "revise",
    "summary": "Step 3 deletes the backup before Step 5 checks that the restore worked.",
    "findings": [
        {
            "id": "f1",
            "severity": "acceptance_changing",
            "location": "Step 3",
            "impact": "If the restore fails there is no backup left to retry from.",
            "confidence": "high",
            "basis": "check",
            "fix_or_check": "Delete the backup only after Step 5 confirms the restore.",
        }
    ],
}


def schema_prompt() -> str:
    """The format, spelled out for the system prompt: an example answer, then the schema. Shown
    the schema alone, the teacher often answered with the schema itself."""
    return (
        f"Example answer:\n{json.dumps(CRITIQUE_EXAMPLE, indent=1)}\n\n"
        "Every field, as a JSON Schema (it describes your answer; never repeat it):\n"
        f"{json.dumps(CRITIQUE_SCHEMA, indent=1)}"
    )
