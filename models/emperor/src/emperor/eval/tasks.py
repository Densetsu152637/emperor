"""Held-out task sets: loading, validation, and long-context prompt generation.

A task set is a directory of `*.json` files (one task each) and/or `*.jsonl` files (one task
per line). The bundled suites live under `tasks/<suite>/`. See the README for the format.
"""

import json
import random
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from emperor.eval.contract import ALL_TOOLS, CHARS_PER_TOKEN

BUNDLED = resources.files("emperor.eval") / "tasks"

_EXPECT_KEYS = {"routing", "brief", "plan", "gaps", "success"}
_TASK_KEYS = {
    "id",
    "description",
    "class",
    "prompt",
    "tools",
    "expect",
    "replies",
    "user_answers",
    "web",
    "secrets",
    "haystack",
    "reference",
}


class TaskError(ValueError):
    """A task file is malformed."""


@dataclass
class Task:
    id: str
    suite: str
    prompt: str = ""
    description: str = ""
    task_class: str | None = None  # "menial" or "hard", for routing by class (G15)
    tools: list[str] = field(default_factory=lambda: list(ALL_TOOLS))
    expect: dict = field(default_factory=dict)
    replies: list[dict] = field(default_factory=list)
    user_answers: list[str] = field(default_factory=list)
    web: dict = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    haystack: dict | None = None
    reference: list[dict] = field(default_factory=list)

    @property
    def needles(self) -> list[str]:
        """Values the model must retrieve from the haystack (G6)."""
        return [n["value"] for n in (self.haystack or {}).get("needles", [])]


def task_from_dict(data: dict, suite: str) -> Task:
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        raise TaskError("every task needs a string id")
    where = f"task {data['id']!r}"
    unknown = set(data) - _TASK_KEYS
    if unknown:
        raise TaskError(f"{where}: unknown keys {sorted(unknown)}")
    unknown = set(data.get("expect", {})) - _EXPECT_KEYS
    if unknown:
        raise TaskError(f"{where}: unknown expect keys {sorted(unknown)}")
    tools = data.get("tools", list(ALL_TOOLS))
    bad = [t for t in tools if t not in ALL_TOOLS]
    if bad:
        raise TaskError(f"{where}: tools not in the contract: {bad}")
    if not data.get("prompt") and not data.get("haystack"):
        raise TaskError(f"{where}: needs a prompt or a haystack")
    if data.get("class") not in (None, "menial", "hard"):
        raise TaskError(f"{where}: class must be menial or hard")
    return Task(
        id=data["id"],
        suite=suite,
        prompt=data.get("prompt", ""),
        description=data.get("description", ""),
        task_class=data.get("class"),
        tools=tools,
        expect=data.get("expect", {}),
        replies=data.get("replies", []),
        user_answers=data.get("user_answers", []),
        web=data.get("web", {}),
        secrets=data.get("secrets", []),
        haystack=data.get("haystack"),
        reference=data.get("reference", []),
    )


def load_tasks(path: Path | str, suite: str) -> list[Task]:
    """Load every task under `path` (a directory, a .json file, or a .jsonl file)."""
    path = Path(path)
    files = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    tasks = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        try:
            if file.suffix == ".jsonl":
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                records = [json.loads(text)]
        except json.JSONDecodeError as error:
            raise TaskError(f"{file}: {error}") from error
        tasks.extend(task_from_dict(record, suite) for record in records)
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise TaskError(f"duplicate task ids: {duplicates}")
    if not tasks:
        raise TaskError(f"no tasks under {path}")
    return tasks


def load_suite(suite: str) -> list[Task]:
    with resources.as_file(BUNDLED / suite) as path:
        return load_tasks(path, suite)


_FILLER = [
    "The archive room keeps its ledgers in alphabetical order by district.",
    "Each courier signs the arrival log before the morning bell.",
    "Grain shipments from the river ports are weighed twice before storage.",
    "The east gate closes at dusk and reopens when the watch changes.",
    "Scribes copy every decree into three bound volumes for the provinces.",
    "Old maps of the northern passes are stored beside the weather records.",
    "The treasury counts its coins in sealed bags of one hundred.",
    "Messengers rest at the waystations that line the imperial road.",
]


def render_prompt(task: Task, context_tokens: int) -> str:
    """The user message: the task prompt, or a generated needle-in-a-haystack document.

    A haystack's `tokens` is a count, or "tier" for 85 % of the active context (the rest is left
    for the system prompt, the question, and the answer). Needles sit at fractional `depth`s.
    """
    if not task.haystack:
        return task.prompt
    spec = task.haystack
    tokens = spec.get("tokens", 4096)
    if tokens == "tier":
        tokens = int(context_tokens * 0.85)
    budget = tokens * CHARS_PER_TOKEN
    rng = random.Random(task.id)
    sentences: list[str] = []
    size = 0
    while size < budget:
        sentence = rng.choice(_FILLER)
        sentences.append(sentence)
        size += len(sentence) + 1
    for needle in sorted(spec["needles"], key=lambda n: n["depth"], reverse=True):
        at = round(needle["depth"] * len(sentences))
        sentences.insert(at, f"The {needle['key']} is {needle['value']}.")
    document = " ".join(sentences)
    return f"{document}\n\n{spec['question']}"
