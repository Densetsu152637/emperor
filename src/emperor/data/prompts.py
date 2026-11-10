"""Seeded, deduplicated user requests across the task classes, guarded against the eval tasks.

Requests run from vague ("make it faster") to specific (paths, constraints, edge cases). The
generator is combinatorial and deterministic for a seed; every candidate is checked against the
held-out and e2e tasks with an n-gram overlap test so no evaluation prompt leaks into training.
"""

import hashlib
import json
import random
import re
import string
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from emperor.eval.contract import ALL_TOOLS

CLASSES = {
    "coding": 20,
    "refactor": 12,
    "research": 14,
    "planning": 14,
    "clarify": 10,
    "docs": 8,
    "architecture": 10,
    "long_input": 8,
    "quick_answer": 4,
}
# Context budgets the traces are written for (K tokens), with weights; see the plan's tiers.
CONTEXT_BUDGETS_K = {32: 2, 64: 5, 96: 2, 128: 3, 192: 1, 256: 1}

SLOTS: dict[str, list[str]] = {
    "project": [
        "invoice generator",
        "recipe app",
        "inventory tracker",
        "chat bot",
        "booking system",
        "log shipper",
        "metrics exporter",
        "feature-flag service",
        "image resizer",
        "markdown renderer",
        "cron scheduler",
        "rate limiter",
        "url shortener",
        "todo CLI",
        "plugin loader",
        "CSV importer",
        "search index",
        "notification service",
        "billing API",
        "game server",
        "file sync tool",
        "weather dashboard",
        "podcast player",
        "budget tracker",
        "newsletter tool",
        "support-ticket queue",
        "photo gallery",
        "fitness tracker",
        "job board",
        "quiz app",
    ],
    "lang": [
        "Python",
        "Rust",
        "TypeScript",
        "Go",
        "Java",
        "C#",
        "Kotlin",
        "Ruby",
        "Swift",
        "PHP",
    ],
    "feature": [
        "pagination",
        "retry with backoff",
        "a dark mode",
        "CSV export",
        "request logging",
        "a --dry-run flag",
        "response caching",
        "input validation",
        "graceful shutdown",
        "a health-check endpoint",
        "rate limiting",
        "pluggable storage backends",
        "undo support",
        "a config file",
        "progress bars",
        "search",
        "audit logging",
        "bulk import",
        "keyboard shortcuts",
        "timezone support",
        "email notifications",
        "a plugin API",
        "offline mode",
        "soft deletes",
    ],
    "constraint": [
        "not break the public API",
        "stay under 50 ms per call",
        "work offline",
        "avoid new dependencies",
        "handle unicode",
        "be thread-safe",
        "run on Windows and Linux",
        "keep memory under 200 MB",
        "stay backwards compatible",
        "log nothing sensitive",
    ],
    "edge": [
        "empty input",
        "very large files",
        "concurrent writers",
        "time zones around DST changes",
        "malformed records",
        "a missing config file",
        "network timeouts",
        "duplicate entries",
        "non-ASCII names",
        "interrupted runs",
    ],
    "file": [
        "src/main.py",
        "src/app.ts",
        "lib/core.rs",
        "cmd/server/main.go",
        "src/Service.java",
        "src/utils.py",
        "src/index.ts",
        "pkg/store/store.go",
        "app/models.py",
        "src/handler.rs",
    ],
    "proj_dir": [
        "~/code/{p}",
        "~/work/{p}",
        "~/projects/{p}",
        "/srv/{p}",
        "C:\\dev\\{p}",
        "~/src/{p}",
    ],
    "slug": [
        "pantry",
        "harbor",
        "quill",
        "meridian",
        "lantern",
        "kestrel",
        "ember",
        "tidepool",
        "orchard",
        "foundry",
        "beacon",
        "sundial",
        "loom",
        "cairn",
        "ferrous",
        "juniper",
        "nimbus",
        "trellis",
        "vesper",
        "willow",
    ],
    "sym": [
        "get_user",
        "process_batch",
        "calcTotal",
        "parse_row",
        "fetchOrders",
        "handle_event",
        "build_index",
        "sendMail",
        "load_config",
        "render_page",
        "Tracker",
        "SessionStore",
    ],
    "sym2": [
        "fetch_user",
        "run_batch",
        "compute_total",
        "read_row",
        "loadOrders",
        "dispatch_event",
        "make_index",
        "deliverMail",
        "read_config",
        "draw_page",
        "Monitor",
        "SessionCache",
    ],
    "tech_a": [
        "PostgreSQL",
        "Redis",
        "Kafka",
        "RabbitMQ",
        "SQLite",
        "MongoDB",
        "gRPC",
        "GraphQL",
        "Terraform",
        "Docker Compose",
        "Nginx",
        "Celery",
        "FastAPI",
        "Django",
        "Tokio",
    ],
    "tech_b": [
        "DynamoDB",
        "Memcached",
        "NATS",
        "SQS",
        "DuckDB",
        "CockroachDB",
        "REST",
        "tRPC",
        "Pulumi",
        "Kubernetes",
        "Caddy",
        "Dramatiq",
        "Flask",
        "Rails",
        "async-std",
    ],
    "use_case": [
        "a small team's internal tools",
        "a mobile app backend",
        "an event pipeline",
        "a read-heavy dashboard",
        "background job processing",
        "a public API",
        "a hobby project that may grow",
        "a data-science workflow",
        "a multi-tenant SaaS",
    ],
    "topic": [
        "WebAssembly runtimes",
        "vector databases",
        "static site generators",
        "open-weight code models",
        "passkey adoption",
        "Rust web frameworks",
        "CRDT libraries",
        "self-hosted analytics",
        "local-first software",
        "observability stacks",
        "SBOM tooling",
        "edge computing platforms",
    ],
    "category": [
        "Python linters",
        "JavaScript bundlers",
        "Rust async runtimes",
        "ORMs for Go",
        "terminal emulators",
        "static analysis tools",
        "headless CMSs",
        "queue libraries",
    ],
    "business": [
        "flower shop",
        "bike repair shop",
        "dental clinic",
        "yoga studio",
        "pottery class",
        "dog-walking service",
        "bookshop",
        "photography studio",
        "food truck",
        "car-wash chain",
        "tutoring service",
        "board-game cafe",
    ],
    "audience": [
        "my running club",
        "a neighbourhood tool library",
        "my students",
        "our volunteer team",
        "a small law firm",
        "my D&D group",
        "a school choir",
        "our family",
    ],
    "thing": [
        "an app",
        "a tool",
        "a dashboard",
        "a bot",
        "a site",
        "a system",
    ],
    "doc_kind": [
        "README",
        "CONTRIBUTING guide",
        "API reference",
        "changelog",
        "migration guide",
        "architecture overview",
        "onboarding doc",
        "runbook",
    ],
    "scale": [
        "10k users",
        "about 500 requests per second",
        "a million events a day",
        "a team of three",
        "100 GB of data",
        "a single region",
        "spiky traffic on weekends",
    ],
    "system": [
        "notification system",
        "job queue",
        "search service",
        "file upload pipeline",
        "audit trail",
        "multi-tenant billing",
        "feature-flag rollout",
        "session store",
        "analytics ingestion",
        "offline sync layer",
    ],
    "goal": [
        "faster",
        "cheaper to run",
        "easier to maintain",
        "more reliable",
        "ready for launch",
        "ready for more users",
        "safer to change",
        "easier for new contributors",
    ],
    "artifact": [
        "a 200-page PDF export",
        "a 1 GB access log",
        "a folder of 400 meeting transcripts",
        "an exported chat history",
        "a large CSV of survey answers",
        "a 600-file monorepo dump",
        "a year of support tickets",
        "a book-length manuscript",
    ],
    "artifact_task": [
        "List every action item with its owner.",
        "Summarise the main themes with evidence.",
        "Find every contradiction between sections.",
        "Extract all dates and what happened.",
        "Find the five most common problems and one example each.",
        "Produce a glossary of the terms it uses.",
    ],
    "concept": [
        "idempotency",
        "backpressure",
        "a monad",
        "eventual consistency",
        "a mutex",
        "dependency injection",
        "a bloom filter",
        "CORS",
        "a closure",
        "tail recursion",
    ],
    "n": ["120", "250", "400", "600", "900"],
}

# (class, specificity) -> templates. {slots} are filled by the generator.
TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "coding": [
        ("vague", "Add {feature} to my {project}."),
        ("vague", "my {project} needs {feature}, can you sort that out"),
        ("vague", "Can you build {feature} for the {project}?"),
        (
            "partial",
            "In my {lang} {project} at {proj_dir}, add {feature}. Keep the existing tests green.",
        ),
        (
            "partial",
            "Implement {feature} in the {lang} {project} ({proj_dir}); it should {constraint}.",
        ),
        (
            "specific",
            "In {proj_dir}/{file}, implement {feature}. It must {constraint} and cope with {edge}. Add tests for {edge}.",
        ),
        (
            "specific",
            "Fix the bug where {edge} crashes the {project} in {proj_dir} ({lang}); write a failing test first, then the fix, and make sure it will {constraint}.",
        ),
        ("vague", "The {project} sometimes breaks when there is {edge}. Fix it."),
    ],
    "refactor": [
        ("vague", "The {project} code is a mess, clean it up."),
        ("partial", "Rename {sym} to {sym2} everywhere in {proj_dir} and keep the build passing."),
        ("partial", "Split {file} in {proj_dir} into smaller modules without changing behaviour."),
        ("partial", "Remove dead code from {proj_dir}; tell me what you deleted."),
        (
            "specific",
            "Convert the callbacks in {proj_dir}/{file} to async/await, keep every public signature, and run the tests after each step.",
        ),
        (
            "specific",
            "Extract the duplicated validation logic in {proj_dir} into one helper, make sure it will {constraint}, and show me the diff summary.",
        ),
        ("vague", "Tidy up the {lang} in my {project}."),
    ],
    "research": [
        ("vague", "What's going on with {topic} these days?"),
        ("partial", "Compare {tech_a} and {tech_b} for {use_case}. Cite sources."),
        ("partial", "Which {category} are actively maintained right now? I need links."),
        (
            "specific",
            "I'm choosing between {tech_a} and {tech_b} for {use_case}. Check the latest release, licence and open-issue trends of each and give me a recommendation with sources.",
        ),
        ("vague", "Is {topic} worth learning?"),
        (
            "specific",
            "Find out the current best practice for {topic} in production, what the main risks are, and who is using it. Cite every claim.",
        ),
    ],
    "planning": [
        ("vague", "Get my {project} {goal}."),
        ("vague", "I want the {project} to be {goal}, where do we start?"),
        ("partial", "Plan the migration of my {project} from {tech_a} to {tech_b}."),
        ("partial", "Plan the next quarter for my {project} so that it is {goal}."),
        (
            "specific",
            "Our {project} ({lang}, {scale}) must be {goal} by the end of the month. Give me an executable plan with owners, checks and risks, then start on it.",
        ),
        ("vague", "Help me launch the {project}."),
    ],
    "clarify": [
        ("vague", "I need {thing} for my {business}."),
        ("vague", "Make {thing} for {audience}."),
        ("partial", "I run a {business} and want to take bookings online. Sort it out."),
        ("vague", "Can you set up something so {audience} can share files?"),
        (
            "partial",
            "Build {thing} for my {business} so customers can find us; I do not know what I need.",
        ),
        ("vague", "I want to sell stuff from my {business} online."),
    ],
    "docs": [
        ("vague", "Write docs for my {project}."),
        ("partial", "Write a {doc_kind} for {proj_dir}."),
        ("partial", "Add docstrings to {proj_dir}/{file}."),
        (
            "specific",
            "Write the {doc_kind} for the {lang} {project} in {proj_dir}: what it does, how to run it, how to test it. Under two pages.",
        ),
        (
            "specific",
            "Fix the spelling mistakes in the {doc_kind} of {proj_dir} and make the headings consistent.",
        ),
    ],
    "architecture": [
        ("vague", "How should I structure my {project}?"),
        ("partial", "Design the {system} for {scale}."),
        ("partial", "Should we use {tech_a} or {tech_b} for {use_case}? I need a decision."),
        (
            "specific",
            "Design the {system} for {scale} on {tech_a}. Give the data model, failure modes, a migration path from what we have, and the trade-offs you rejected.",
        ),
        (
            "specific",
            "Review the design of our {system} ({scale}, {tech_a}); find the three biggest risks and propose changes ranked by value.",
        ),
    ],
    "long_input": [
        (
            "partial",
            "{artifact} is stored as artefact art_{hex} (about {n}K tokens). {artifact_task}",
        ),
        (
            "specific",
            "Here is {artifact}: artefact art_{hex}, roughly {n}K tokens, one section per heading. {artifact_task} Quote line numbers or headings as evidence.",
        ),
        (
            "vague",
            "Go through artefact art_{hex} ({artifact}, ~{n}K tokens) for me. {artifact_task}",
        ),
    ],
    "quick_answer": [
        ("vague", "In one or two sentences, what is {concept}?"),
        ("vague", "Quick question: when would I actually use {concept}?"),
        ("vague", "Explain {concept} to a new hire, briefly."),
    ],
}


@dataclass
class PromptRecord:
    id: str
    seed: int
    task_class: str
    specificity: str
    text: str
    tools: list[str]
    context_budget_k: int

    def to_dict(self) -> dict:
        return asdict(self)


_OPENERS = ["", "", "", "", "Hey, ", "Hi - ", "Okay so ", "Quick one: "]
_PAIRS = [
    ("PostgreSQL", "DynamoDB"),
    ("Redis", "Memcached"),
    ("Kafka", "NATS"),
    ("RabbitMQ", "SQS"),
    ("SQLite", "DuckDB"),
    ("MongoDB", "CockroachDB"),
    ("gRPC", "REST"),
    ("GraphQL", "tRPC"),
    ("Terraform", "Pulumi"),
    ("Docker Compose", "Kubernetes"),
    ("Nginx", "Caddy"),
    ("Celery", "Dramatiq"),
    ("FastAPI", "Flask"),
    ("Django", "Rails"),
    ("Tokio", "async-std"),
]
_TOKEN = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _TOKEN.findall(text.lower())
    if not words:
        return set()
    n = min(n, len(words))
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlap(candidate: str, reference: str) -> float:
    """Largest of the Jaccard similarity and the share of `reference` that `candidate` contains."""
    a, b = shingles(candidate), shingles(reference)
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return max(shared / len(a | b), shared / len(b))


class HeldoutIndex:
    """The user-facing text of the held-out and e2e tasks, for the n-gram overlap check."""

    def __init__(self, texts: dict[str, str], threshold: float = 0.4):
        self.texts = texts
        self.threshold = threshold

    @classmethod
    def from_task_dirs(cls, paths: Iterable[Path], threshold: float = 0.4) -> "HeldoutIndex":
        texts: dict[str, str] = {}
        for path in paths:
            for file in sorted(Path(path).glob("*.json*")):
                raw = file.read_text(encoding="utf-8")
                records = (
                    [json.loads(line) for line in raw.splitlines() if line.strip()]
                    if file.suffix == ".jsonl"
                    else [json.loads(raw)]
                )
                for record in records:
                    parts = [record.get("prompt", "")]
                    parts.append((record.get("haystack") or {}).get("question", ""))
                    text = " ".join(p for p in parts if p).strip()
                    if text:
                        texts[f"{Path(path).name}/{record.get('id', file.stem)}"] = text
        return cls(texts, threshold)

    @classmethod
    def bundled(cls, threshold: float = 0.4) -> "HeldoutIndex":
        from emperor.eval.tasks import BUNDLED

        base = Path(str(BUNDLED))
        return cls.from_task_dirs([base / "heldout", base / "e2e", base / "longctx"], threshold)

    def closest(self, text: str) -> tuple[float, str | None]:
        best, best_id = 0.0, None
        for task_id, reference in self.texts.items():
            score = overlap(text, reference)
            if score > best:
                best, best_id = score, task_id
        return best, best_id

    def leaks(self, text: str) -> bool:
        return self.closest(text)[0] >= self.threshold


def _fill(template: str, rng: random.Random) -> str:
    values: dict[str, str] = {}
    names = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    if {"tech_a", "tech_b"} <= names:
        first, second = rng.choice(_PAIRS)
        values["tech_a"], values["tech_b"] = (
            (first, second) if rng.random() < 0.5 else (second, first)
        )
    for field in sorted(names):
        if field in values:
            continue
        if field == "hex":
            values[field] = f"{rng.getrandbits(16):04x}"
        elif field == "p":
            values[field] = rng.choice(SLOTS["slug"])
        else:
            values[field] = rng.choice(SLOTS[field])
    # proj_dir itself contains a {p} placeholder.
    if "proj_dir" in values:
        values["proj_dir"] = values["proj_dir"].format(p=rng.choice(SLOTS["slug"]))
    return template.format(**values)


def _weighted(rng: random.Random, weights: dict) -> object:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _tools_for(task_class: str, rng: random.Random) -> list[str]:
    tools = list(ALL_TOOLS)
    if task_class != "long_input" and rng.random() < 0.5:
        tools.remove("delegate_chunks")
    if rng.random() < 0.25:
        tools.remove("ask_user")
    return tools


def generate_prompts(
    count: int,
    seed: int,
    *,
    heldout: HeldoutIndex | None = None,
    existing: Iterable[str] = (),
    classes: dict[str, int] | None = None,
    max_attempts_factor: int = 40,
) -> tuple[list[PromptRecord], dict[str, int]]:
    """Return up to `count` new prompts and counters of what was dropped and why.

    `existing` holds normalised texts already used (for resuming a prompt file). The result is a
    pure function of `seed`, `count`, `classes` and `existing`.
    """
    rng = random.Random(seed)
    heldout = heldout or HeldoutIndex({})
    classes = classes or CLASSES
    seen = set(existing)
    stats = {"duplicate": 0, "heldout_overlap": 0}
    out: list[PromptRecord] = []
    attempts = 0
    while len(out) < count and attempts < count * max_attempts_factor:
        attempts += 1
        task_class = str(_weighted(rng, classes))
        specificity, template = rng.choice(TEMPLATES[task_class])
        text = _fill(template, rng)
        opener = rng.choice(_OPENERS)
        if opener:
            text = opener + text[0].lower() + text[1:]
        key = normalise(text)
        if key in seen:
            stats["duplicate"] += 1
            continue
        if heldout.leaks(text):
            stats["heldout_overlap"] += 1
            continue
        seen.add(key)
        sample_seed = rng.getrandbits(31)
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        budget = 256 if task_class == "long_input" and rng.random() < 0.5 else None
        out.append(
            PromptRecord(
                id=f"p-{digest}",
                seed=sample_seed,
                task_class=task_class,
                specificity=specificity,
                text=text,
                tools=_tools_for(task_class, rng),
                context_budget_k=budget or int(_weighted(rng, CONTEXT_BUDGETS_K)),
            )
        )
    return out, stats
