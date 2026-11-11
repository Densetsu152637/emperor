"""Judging: the same local model scores each trace against the rubric; heuristics add hard fails."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from emperor.data.generate import Chat
from emperor.data.schema import MAX_TABLE_ROWS, count_emojis, table_rows
from emperor.data.shards import ShardWriter, done_ids, iter_records
from emperor.data.style import CRITERIA, NOT_APPLICABLE, judge_system
from emperor.data.teacher import TEACHER_MODEL, TeacherError, TeacherOutputError

MEAN_THRESHOLD = 4.0
MIN_CRITERION = 3
MAX_EMOJIS = 2
MAX_CODE_LINES = 25
_TOOL_RESULT_CHARS = 1200
_TRANSCRIPT_CHARS = 28_000


class JudgeParseError(ValueError):
    """The judge's answer is not the expected JSON."""


@dataclass
class JudgeResult:
    scores: dict[str, int]
    passed: bool
    reason: str
    flags: list[str] = field(default_factory=list)
    ignored: tuple[str, ...] = ()  # criteria that do not apply to this task class

    @property
    def applicable(self) -> dict[str, int]:
        return {k: v for k, v in self.scores.items() if k not in self.ignored} or self.scores

    @property
    def mean(self) -> float:
        scores = self.applicable
        return sum(scores.values()) / len(scores)

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "mean": round(self.mean, 3),
            "pass": self.passed,
            "reason": self.reason,
            "flags": self.flags,
            "ignored": list(self.ignored),
        }


def _extract_object(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise JudgeParseError("no JSON object in the judge's answer")
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
                return text[start : i + 1]
    raise JudgeParseError("unterminated JSON object in the judge's answer")


def parse_judge_output(text: str) -> JudgeResult:
    """Parse the judge's answer; raises JudgeParseError unless every criterion has a 1-5 score."""
    try:
        data = json.loads(_extract_object(text))
    except json.JSONDecodeError as error:
        raise JudgeParseError(f"invalid JSON: {error.msg}") from error
    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise JudgeParseError("missing 'scores' object")
    clean: dict[str, int] = {}
    for name in CRITERIA:
        value = scores.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
            raise JudgeParseError(f"score for {name!r} is not an integer")
        if not 1 <= int(value) <= 5:
            raise JudgeParseError(f"score for {name!r} is outside 1-5")
        clean[name] = int(value)
    passed = data.get("pass")
    if not isinstance(passed, bool):
        raise JudgeParseError("'pass' must be true or false")
    reason = data.get("reason", "")
    return JudgeResult(clean, passed, reason if isinstance(reason, str) else str(reason))


def heuristic_flags(sample: dict) -> list[str]:
    """Deterministic style violations, applied on top of the model's scores."""
    flags = []
    texts = [m.get("content") or "" for m in sample["messages"] if m["role"] == "assistant"]
    joined = "\n".join(texts)
    if count_emojis(joined) > MAX_EMOJIS:
        flags.append("emojis")
    if table_rows(joined) > MAX_TABLE_ROWS:
        flags.append("tables")
    final = texts[-1] if texts else ""
    code = 0
    for block in re.findall(r"```.*?\n(.*?)```", final, flags=re.S):
        code += len(block.splitlines())
    if code > MAX_CODE_LINES:
        flags.append("pasted_code")
    return flags


def decide(result: JudgeResult, threshold: float = MEAN_THRESHOLD) -> tuple[bool, list[str]]:
    """Final keep/reject decision and the reasons for a rejection."""
    reasons = list(result.flags)
    if not result.passed:
        reasons.append("judge_fail")
    if result.mean < threshold:
        reasons.append("below_threshold")
    low = [name for name, score in result.applicable.items() if score < MIN_CRITERION]
    reasons += [f"low_{name}" for name in low]
    return not reasons, reasons


def render_transcript(sample: dict) -> str:
    lines = []
    for message in sample["messages"][1:]:  # the system prompt is in the judge's own prompt
        role = message["role"]
        if role == "assistant":
            if message.get("content"):
                lines.append(f"EMPEROR: {message['content']}")
            for call in message.get("tool_calls") or []:
                function = call["function"]
                lines.append(f"EMPEROR calls {function['name']}: {function['arguments']}")
        elif role == "tool":
            body = message["content"]
            if len(body) > _TOOL_RESULT_CHARS:
                body = body[:_TOOL_RESULT_CHARS] + " [truncated]"
            lines.append(f"TOOL RESULT: {body}")
        else:
            lines.append(f"USER: {message['content']}")
    text = "\n\n".join(lines)
    if len(text) > _TRANSCRIPT_CHARS:
        text = text[:_TRANSCRIPT_CHARS] + "\n[transcript truncated]"
    return text


_STRICT_JSON = (
    "\n\nYour previous answer was not valid JSON. Reply with exactly one JSON object and "
    "nothing else: no prose before or after, no code fence, no thinking out loud. Use the keys "
    '"scores" (all eight criteria, integers 1-5), "pass" (true or false) and "reason".'
)
# First try warm and short; the retry is cold, stricter about the format and has room to finish.
_JUDGE_TRIES = ((0.2, 2048, ""), (0.0, 3072, _STRICT_JSON))


def judge_sample(sample: dict, client: Chat, *, seed: int = 0, attempts: int = 2) -> JudgeResult:
    """Score one trace; an unparseable answer is retried with a lower temperature and a
    stricter JSON instruction before giving up."""
    task_class = sample.get("task_class")
    system = judge_system(task_class)
    transcript = "Transcript to score:\n\n" + render_transcript(sample)
    last: Exception | None = None
    for attempt in range(attempts):
        temperature, max_tokens, suffix = _JUDGE_TRIES[min(attempt, len(_JUDGE_TRIES) - 1)]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript + suffix},
        ]
        reply = client.chat(
            messages, temperature=temperature, seed=seed + attempt, max_tokens=max_tokens
        )
        try:
            result = parse_judge_output(reply.get("content") or "")
        except JudgeParseError as error:
            last = error
            continue
        result.flags = heuristic_flags(sample)
        result.ignored = NOT_APPLICABLE.get(task_class or "", ())
        return result
    raise JudgeParseError(str(last))


def run_judging(
    data_dir: Path,
    client: Chat,
    *,
    threshold: float = MEAN_THRESHOLD,
    limit: int | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Judge every trace not yet judged. Resumable; failures to parse are recorded as rejects."""
    writer = ShardWriter(data_dir / "judged", "judged")
    finished = done_ids(data_dir / "judged")
    counts = {"judged": 0, "kept": 0, "stopped": False}
    for sample in iter_records(data_dir / "traces"):
        if sample["id"] in finished:
            continue
        if limit is not None and counts["judged"] >= limit:
            break
        if should_stop():
            counts["stopped"] = True
            break
        try:
            result = judge_sample(sample, client, seed=sample["meta"]["seed"])
            keep, reasons = decide(result, threshold)
            judge = {**result.to_dict(), "model": TEACHER_MODEL, "keep": keep, "rejection": reasons}
        except TeacherOutputError as error:
            judge = {"keep": False, "rejection": ["judge_unparseable"], "reason": str(error)}
            keep = False
        except TeacherError as error:
            counts["stopped"] = True
            counts["error"] = str(error)
            break
        except JudgeParseError as error:
            judge = {"keep": False, "rejection": ["judge_unparseable"], "reason": str(error)}
            keep = False
        sample["meta"]["judge"] = judge
        writer.write(sample)
        counts["judged"] += 1
        counts["kept"] += int(keep)
        log(f"{sample['id']} {'keep' if keep else ','.join(judge['rejection'])}")
    return counts
