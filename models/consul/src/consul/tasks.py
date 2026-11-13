"""Seeded-defect critique tasks: plans, diffs and agent reports with injected defects.

Each task is built from a seeded RNG: an artifact written from templates and vocabularies, a few
source documents (the only URLs a critique may cite), and the defects that were injected, with
their ground truth (where, what kind, whether it changes acceptance, and keywords that a finding
about it would use). About one task in six is clean, to measure false findings.

The ground truth never reaches the model or the teacher; it is stored beside the task and used
to score critiques (`score.py`) and to filter training data.
"""

import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

KINDS = ("plan", "diff", "report")
CLEAN_SHARE = 1 / 6


@dataclass
class Defect:
    id: str
    kind: str
    where: str  # "Step 4", "Acceptance", "orders.py:L13", "Claim 2", "Summary"
    acceptance_changing: bool
    keywords: list[str]  # lower-case stems; a finding about this defect uses at least one
    description: str


@dataclass
class Source:
    url: str
    title: str
    text: str


@dataclass
class CritiqueTask:
    id: str
    seed: int
    kind: str  # plan | diff | report
    family: str  # the task class inside the kind
    title: str
    goal: str
    artifact: str
    sources: list[Source] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not [d for d in self.defects if d.acceptance_changing]

    @property
    def task_class(self) -> str:
        return f"{self.kind}/{self.family}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CritiqueTask":
        return cls(
            **{
                **data,
                "sources": [Source(**s) for s in data.get("sources", [])],
                "defects": [Defect(**d) for d in data.get("defects", [])],
            }
        )

    def urls(self) -> set[str]:
        return {s.url for s in self.sources}


# -- vocabularies --------------------------------------------------------------------------

SYSTEMS = [
    "invoice service", "order service", "billing ledger", "notification worker", "search index",
    "user profile API", "report generator", "inventory sync", "payment gateway adapter",
    "audit log store", "image thumbnailer", "email digest job", "session store", "rate limiter",
    "feature-flag service", "booking engine", "metrics collector", "document converter",
]  # fmt: skip
STORES = ["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Elasticsearch", "DynamoDB"]
VENDORS = ["northwind", "contoso", "fabrikam", "tailspin", "wingtip", "litware", "adatum"]
TOKENS = ["payments API token", "storage access key", "webhook signing secret", "SMTP password"]
UNRELATED = [
    "logging framework", "CSS build tooling", "date-formatting helpers", "CI cache layout",
    "package naming scheme", "internal wiki theme",
]  # fmt: skip
TEAMS = ["platform", "payments", "growth", "data", "mobile", "support-tools", "security"]
UNITS = ["rows", "events", "messages", "documents"]


def _pick(rng: random.Random, items):
    return items[rng.randrange(len(items))]


def _part(rng: random.Random, items: list) -> list:
    """The vocabulary a task may draw from: the first three quarters for training tasks, the
    last quarter for held-out ones, so held-out tasks carry names the training set never has."""
    cut = max(1, len(items) * 3 // 4)
    return items[cut:] if getattr(rng, "held", False) else items[:cut]


def _source(rng: random.Random, topic: str, body: str) -> Source:
    vendor = _pick(rng, _part(rng, VENDORS))
    slug = topic.lower().replace(" ", "-")
    return Source(
        url=f"https://docs.{vendor}.example/{slug}-{rng.randrange(100, 999)}",
        title=f"{topic.title()} reference",
        text=body,
    )


def _filler_sources(rng: random.Random, count: int) -> list[Source]:
    topics = [
        ("backups", "Backups run nightly and are kept for {n} days; restores are tested monthly."),
        # Background facts, not obligations: as rules ("deploys only before 11:00", "N approving
        # reviews", "tell on-call N hours ahead") no artefact mentioned them, and the critic
        # flagged the gap as blocking in most plan and report tasks (pilots 6-12) while the ground
        # truth counted it false. The sources still test evidence discipline as distractors.
        (
            "deploy pipeline",
            "The deploy pipeline takes about {n} minutes from merge to production.",
        ),
        ("code review", "The team reviews about {n} pull requests a week."),
        ("secrets", "Secrets belong in the vault; the repository must never contain credentials."),
        ("on-call", "The on-call rotation has {n} engineers and hands over every Monday."),
    ]
    chosen = rng.sample(topics, count)
    return [_source(rng, t, text.format(n=rng.randrange(2, 30))) for t, text in chosen]


def _id(kind: str, artifact: str) -> str:
    return f"ct-{kind}-{hashlib.sha1(artifact.encode()).hexdigest()[:10]}"


# -- plans ---------------------------------------------------------------------------------

# role -> step text. Roles: prep backup change1 change2 verify release rollback cleanup.
PLAN_FAMILIES: dict[str, dict] = {
    "migration": {
        "title": "Move the {system} from {a} to {b}",
        "goal": "Move the {system} data from {a} to {b} with no data loss and under {n} minutes "
        "of downtime.",
        "acceptance": [
            "Row counts and checksums match between {a} and {b} for every table",
            "Downtime stays under {n} minutes, measured from the load balancer logs",
            "The {system} passes its integration suite against {b}",
        ],
        "steps": {
            "prep": "Inventory the tables and consumers of the {system} and freeze schema changes.",
            "backup": "Take a full backup of {a} and test a restore into a scratch instance.",
            # Dual-write starts before the backfill: the other order loses writes made during
            # the backfill, which every hosted-pilot-2 critic rightly flagged.
            "change1": "Create the {b} schema and turn on dual-write of new changes to both "
            "stores behind a feature flag.",
            "change2": "Backfill the existing rows from {a} to {b} in batches.",
            "verify": "Compare row counts and checksums, then run the integration suite on {b}.",
            "release": "Cut reads over to {b} during the agreed window and watch error rates.",
            "rollback": "If error rate exceeds 1% for 5 minutes, flip the flag back to {a}.",
            "cleanup": "After a week of clean running, remove the dual-write code and old data.",
        },
    },
    "feature": {
        "title": "Add export to the {system}",
        "goal": "Let users export their {system} data as CSV, finished within {n} seconds for "
        "the largest account.",
        "acceptance": [
            "An export of the largest account finishes within {n} seconds",
            "The CSV opens in a spreadsheet with correct headers and encoding",
            "Users can only export their own records, checked by an authorisation test",
        ],
        "steps": {
            "prep": "Agree the CSV columns with support and write them into the spec.",
            "backup": "Snapshot the current API responses as fixtures for regression tests.",
            "change1": "Add a streaming export endpoint to the {system} using {a} cursors.",
            "change2": "Add an Export button to the settings page that calls the endpoint.",
            # The timing acceptance is tested too (hosted pilot 2: "never measured").
            "verify": "Write tests for the columns, the encoding and cross-account access, and "
            "time an export of the largest account against the {n}-second limit.",
            "release": "Ship behind a flag to the {team} team, then to everyone.",
            "rollback": "If exports slow {a}, turn the flag off; no data changes are involved.",
            "cleanup": "Remove the flag and document the endpoint in the API reference.",
        },
    },
    "rollout": {
        "title": "Roll out {system} v2 to production",
        "goal": "Replace {system} v1 with v2 in production without dropping requests, within "
        "{n} hours.",
        "acceptance": [
            "No dropped requests: 5xx responses and connection errors stay under 0.1%",
            "p95 latency of v2 is within 10% of v1 on the same traffic",
            "All {n} dependent clients keep working, verified by their smoke tests",
        ],
        "steps": {
            "prep": "List the clients of the {system} and tell their owners about the window.",
            "backup": "Export the current {system} configuration (secrets excluded) and "
            "store it with the release.",
            "change1": "Deploy v2 beside v1 and shadow a copy of 5% of read-only traffic to it "
            "(responses discarded).",
            "change2": "Shift live traffic to v2 at 25%, then 50%, then 75%; after each step, "
            "check the error rate, p95 latency and the clients' smoke tests before going on.",
            "verify": "Run each client's smoke test and compare latency and error dashboards.",
            "release": "Send 100% of traffic to v2 and keep v1 running idle.",
            # Hosted pilots 1-2: a rollback on the error budget alone left latency and smoke
            # failures without one, a real gap the critics flagged.
            "rollback": "If a step breaches the error budget, raises p95 latency more than 10% "
            "or fails a smoke test, send traffic back to v1 at once.",
            "cleanup": "Retire v1 after the {team} team signs off.",
        },
    },
    "research": {
        # Candidates are data stores ({a}, {b}), so the task is about a store: as a queue choice
        # the critic rightly objected that relational databases are not queues. Running cost is
        # measured, as the acceptance requires; without it every task carried an unrecorded
        # acceptance-changing gap.
        "title": "Choose a data store for the {system}",
        "goal": "Pick a data store for the {system} that sustains {n}k writes a second "
        "and can be run by the {team} team.",
        "acceptance": [
            "Each candidate is benchmarked on the same {n}k writes/s workload",
            "The recommendation lists operating cost and failure behaviour per candidate",
            "A decision record names the choice and what would change it",
            # Hosted pilot 1: without these the plan could meet its acceptance and miss its goal.
            "The chosen store met the {n}k writes/s load test and the {team} team confirmed "
            "they can run it",
        ],
        "steps": {
            "prep": "Write down the workload: record size, consistency needs and retention.",
            "backup": "Record the current store's metrics for a baseline.",
            "change1": "Stand up three candidates ({a}, {b} and one more) in a test cluster.",
            "change2": "Run the same load test against each and record throughput, errors and "
            "the cluster's running cost.",
            "verify": "Kill a node during the test and note recovery time and data loss.",
            "release": "Write the decision record and confirm with the {team} team that they "
            "can operate the choice.",
            "rollback": "Keep the current store in place until the decision is approved.",
            "cleanup": "Archive the test cluster configuration.",
        },
    },
}  # fmt: skip

IMPROVEMENT_PLAN = {
    "no_owner": ("Plan", ["owner", "assign", "responsib"], "No step has an owner."),
    "no_timeline": ("Plan", ["timeline", "schedule", "date", "estimate"], "No timeline."),
}


def _plan(rng: random.Random, seed: int, clean: bool) -> CritiqueTask:
    name = _pick(rng, sorted(PLAN_FAMILIES))
    family = PLAN_FAMILIES[name]
    # An embedded store is no candidate for thousands of writes a second across nodes: in pilot 14
    # critics rightly objected to SQLite in every research plan that listed it.
    stores = [s for s in _part(rng, STORES) if not (name == "research" and s == "SQLite")]
    fill = {
        "system": _pick(rng, _part(rng, SYSTEMS)),
        "a": _pick(rng, stores),
        "n": rng.randrange(3, 60),
        "team": _pick(rng, _part(rng, TEAMS)),
    }
    fill["b"] = _pick(rng, [s for s in stores if s != fill["a"]])
    steps = [[role, text.format(**fill)] for role, text in family["steps"].items()]
    acceptance = [line.format(**fill) for line in family["acceptance"]]
    sources = _filler_sources(rng, 2)
    wanted = [] if clean else rng.sample(PLAN_DEFECTS, rng.choice([1, 1, 2, 3]))
    injected: list[tuple[str, Callable]] = []
    for kind, apply in wanted:
        injected.append((kind, apply(rng, steps, acceptance, sources, fill)))

    lines = [
        f"# Plan: {family['title'].format(**fill)}",
        "",
        f"Goal: {family['goal'].format(**fill)}",
    ]
    lines += ["", "Acceptance:"] + [f"- {a}" for a in acceptance] + ["", "Steps:"]
    for number, (_, text) in enumerate(steps, 1):
        lines.append(f"{number}. {text}")
    index = {role: n for n, (role, _) in enumerate(steps, 1)}

    defects = []
    for i, (kind, marker) in enumerate(injected, 1):
        where = "|".join(
            f"Step {index[alt[1:]]}" if alt.startswith("@") else alt
            for alt in marker["where"].split("|")
            if not alt.startswith("@") or alt[1:] in index
        )
        defects.append(
            Defect(f"d{i}", kind, where, True, marker["keywords"], marker["description"])
        )
    if not clean and rng.random() < 0.35:
        kind = _pick(rng, sorted(IMPROVEMENT_PLAN))
        where, keywords, description = IMPROVEMENT_PLAN[kind]
        defects.append(Defect(f"d{len(defects) + 1}", kind, where, False, keywords, description))
    artifact = "\n".join(lines)
    return CritiqueTask(
        id=_id("plan", artifact),
        seed=seed,
        kind="plan",
        family=name,
        title=family["title"].format(**fill),
        goal=family["goal"].format(**fill),
        artifact=artifact,
        sources=sources,
        defects=defects,
    )


def _find(steps: list[list[str]], role: str) -> int:
    return next(i for i, (r, _) in enumerate(steps) if r == role)


def _d_missing_rollback(rng, steps, acceptance, sources, fill):
    del steps[_find(steps, "rollback")]
    return {
        "where": "@release|@change2",
        "keywords": ["rollback", "revert", "back out", "restore", "undo", "fallback"],
        "description": "No rollback step for the release.",
    }


def _d_wrong_order(rng, steps, acceptance, sources, fill):
    steps.insert(_find(steps, "change1") + 1, steps.pop(_find(steps, "backup")))
    return {
        "where": "@change1|@backup",
        "keywords": ["backup", "snapshot", "order", "before", "restore point"],
        "description": "The backup happens after the first destructive change.",
    }


def _d_no_test(rng, steps, acceptance, sources, fill):
    del steps[_find(steps, "verify")]
    return {
        "where": "@change2|@release",
        "keywords": ["test", "verif", "validat", "check", "smoke"],
        "description": "Nothing verifies the change before release.",
    }


def _d_assumption(rng, steps, acceptance, sources, fill):
    limit = rng.randrange(2, 9) * 100
    claim = rng.randrange(3, 12) * 1000
    unit = _pick(rng, UNITS)
    steps[_find(steps, "change1")][1] += f" This assumes {fill['b']} can ingest {claim} {unit}/s."
    sources.append(
        _source(
            rng,
            f"{fill['b']} ingest limits",
            f"A single {fill['b']} node sustains about {limit} {unit}/s on the standard tier; "
            "higher rates need sharding.",
        )
    )
    return {
        "where": "@change1",
        "keywords": ["assum", "ingest", "throughput", "limit", "rate", "benchmark", "unverified"],
        "description": f"Assumes {claim} {unit}/s; the documentation says about {limit}.",
    }


def _d_secret(rng, steps, acceptance, sources, fill):
    token = _pick(rng, _part(rng, TOKENS))
    steps[_find(steps, "prep")][1] += (
        f" Keep the {token} in config/secrets.yaml in the repository so every job can read it."
    )
    return {
        "where": "@prep",
        "keywords": ["secret", "token", "credential", "vault", "plaintext", "repository", "commit"],
        "description": "A credential is stored in the repository.",
    }


def _d_dangling(rng, steps, acceptance, sources, fill):
    missing = len(steps) + rng.randrange(2, 5)
    steps[_find(steps, "release")][1] += f" Use the list produced in step {missing}."
    return {
        "where": "@release",
        "keywords": [f"step {missing}", "exist", "dangling", "reference", "missing", "no such"],
        "description": f"Refers to step {missing}, which does not exist.",
    }


def _d_scope(rng, steps, acceptance, sources, fill):
    steps.insert(
        _find(steps, "change2") + 1,
        [
            "extra",
            f"While there, also rewrite the {_pick(rng, _part(rng, UNRELATED))} for consistency.",
        ],
    )
    return {
        "where": "@extra",
        # Pilot 14: a critic called this step "unnecessary complexity" at the right location and
        # was scored as missing it; the words a critic uses for scope creep are now accepted.
        "keywords": [
            "scope",
            "unrelated",
            "separate",
            "creep",
            "beyond",
            "out of",
            "unnecessary",
            "not needed",
            "not required",
            "outside",
            "not part of",
            "irrelevant",
            "distract",
        ],
        "description": "A step unrelated to the goal.",
    }


def _d_vague_acceptance(rng, steps, acceptance, sources, fill):
    acceptance[:] = ["It works as expected", "Nobody complains afterwards"]
    return {
        "where": "Acceptance",
        "keywords": ["measur", "vague", "criteria", "testable", "verifiable", "acceptance"],
        "description": "Acceptance criteria are not measurable.",
    }


PLAN_DEFECTS = [
    ("missing_rollback", _d_missing_rollback),
    ("wrong_order", _d_wrong_order),
    ("no_test", _d_no_test),
    ("unsupported_assumption", _d_assumption),
    ("secret_in_repo", _d_secret),
    ("dangling_step", _d_dangling),
    ("scope_creep", _d_scope),
    ("vague_acceptance", _d_vague_acceptance),
]


# -- diffs ---------------------------------------------------------------------------------

# Each builder returns (lines, bug_index or None, keywords, description, intent, filename).


def _t_paginate(rng, bug):
    page = _pick(rng, ["page", "pg"])
    start = f"({page} - 1) * size" if not bug else f"{page} * size"
    return {
        "file": "paging.py",
        "intent": "Add a helper that returns one page of results (pages start at 1)",
        "lines": [
            f"def page_of(items, {page}, size):",
            '    """Return the items on page `page`; pages are numbered from 1."""',
            f"    start = {start}",
            "    return items[start : start + size]",
        ],
        "bug": 2,
        "keywords": ["off-by-one", "off by one", "first page", "skip", "page 1", "start", "index"],
        "description": "Page numbers start at 1 but the offset skips the first page.",
    }


def _t_average(rng, bug):
    guard = [] if bug else ["    if not values:", "        return 0.0"]
    return {
        "file": "stats.py",
        "intent": "Add an average helper used by the dashboard",
        "lines": [
            "def average(values):",
            '    """Mean of a list; an empty list averages to 0.0."""',
            *guard,
            "    return sum(values) / len(values)",
        ],
        "bug": 2,  # the return line once the guard is gone (3 pointed past the end: L<None>)
        "keywords": ["empty", "zerodivision", "division by zero", "divide", "no values", "guard"],
        "description": "An empty list raises ZeroDivisionError instead of returning 0.0.",
    }


def _t_lookup(rng, bug):
    query = (
        "    sql = f\"SELECT id, email FROM users WHERE name = '{name}'\""
        if bug
        else '    sql = "SELECT id, email FROM users WHERE name = ?"'
    )
    run = (
        "    return conn.execute(sql).fetchone()"
        if bug
        else ("    return conn.execute(sql, (name,)).fetchone()")
    )
    return {
        "file": "users.py",
        "intent": "Look a user up by name for the admin search box",
        "lines": ["def find_user(conn, name):", query, run],
        "bug": 1,
        "keywords": ["injection", "sql", "parameter", "bind", "escape", "format"],
        "description": "SQL built with string formatting from user input.",
    }


def _t_retry(rng, bug):
    record = "            pass" if bug else "            last = error"
    return {
        "file": "retry.py",
        "intent": "Retry a flaky call a few times before giving up",
        "lines": [
            "def retry(fn, attempts=3):",
            "    last = None",
            "    for _ in range(attempts):",
            "        try:",
            "            return fn()",
            "        except Exception as error:",
            record,
            "    raise last",
        ],
        "bug": 6,
        "keywords": ["swallow", "silent", "raise none", "typeerror", "last", "none", "exception"],
        "description": "The error is dropped, so the final `raise last` raises None.",
    }


def _t_expired(rng, bug):
    op = "<" if bug else "<="
    return {
        "file": "tokens.py",
        "intent": "Treat a token as expired once its lifetime has fully elapsed",
        "lines": [
            "def is_expired(issued_at, ttl, now):",
            '    """True once `ttl` seconds have elapsed since `issued_at`."""',
            f"    return ttl {op} now - issued_at",
        ],
        "bug": 2,
        "keywords": ["boundary", "equal", "<=", ">=", "exactly", "inclusive", "off-by-one"],
        "description": "A token is still valid at exactly ttl seconds.",
    }


def _t_chunk(rng, bug):
    step = "n + 1" if bug else "n"
    return {
        "file": "batching.py",
        "intent": "Split a list into batches of n for bulk inserts",
        "lines": [
            "def chunk(seq, n):",
            "    out = []",
            f"    for i in range(0, len(seq), {step}):",
            "        out.append(seq[i : i + n])",
            "    return out",
        ],
        "bug": 2,
        "keywords": ["skip", "drop", "lost", "n + 1", "step", "stride", "missing item"],
        "description": "The stride is n + 1, so one item per batch is skipped.",
    }


def _t_default(rng, bug):
    sig = "def add_tag(item, tags=[]):" if bug else "def add_tag(item, tags=None):"
    guard = [] if bug else ["    tags = [] if tags is None else tags"]
    return {
        "file": "tags.py",
        # Says what the code does (adds the item to the list): "attach a tag to an item" read
        # as reversed parameters, a real defect in every task built from this template.
        "intent": "Add an item to a tag list, creating the list when none is given",
        "lines": [sig, *guard, "    tags.append(item)", "    return tags"],
        "bug": 0,
        "keywords": ["mutable", "default", "shared", "state", "none"],
        "description": "A mutable default argument is shared across calls.",
    }


def _t_await(rng, bug):
    call = "    result = client.fetch(url)" if bug else "    result = await client.fetch(url)"
    return {
        "file": "fetcher.py",
        "intent": "Fetch a page asynchronously and return its length",
        "lines": ["async def page_length(client, url):", call, "    return len(result.body)"],
        "bug": 1,
        "keywords": ["await", "coroutine", "async", "never awaited", "attribute"],
        "description": "The coroutine is not awaited, so `result.body` fails.",
    }


DIFF_TEMPLATES = [
    ("pure", _t_paginate), ("pure", _t_average), ("data", _t_lookup), ("resilience", _t_retry),
    ("pure", _t_expired), ("pure", _t_chunk), ("state", _t_default), ("async", _t_await),
]  # fmt: skip

RENAMES = {
    "items": ["items", "rows", "entries", "records", "results"],
    "values": ["values", "samples", "readings", "amounts"],
    "size": ["size", "per_page", "limit"],
    "seq": ["seq", "batch", "queue"],
    "fn": ["fn", "action", "call"],
    "attempts": ["attempts", "tries", "max_tries"],
    "client": ["client", "session", "http"],
    "url": ["url", "link", "endpoint"],
    "tags": ["tags", "labels", "marks"],
    "item": ["item", "entry", "label"],
    "page_of": ["page_of", "slice_page", "get_page"],
    "average": ["average", "mean_of", "avg"],
    "find_user": ["find_user", "lookup_user", "user_by_name"],
    "is_expired": ["is_expired", "has_expired", "expired"],
    "chunk": ["chunk", "batched", "split_into"],
    "add_tag": ["add_tag", "attach_tag", "tag_item"],
    "retry": ["retry", "with_retries", "attempt"],
    "page_length": ["page_length", "body_size", "fetch_len"],
}
FILE_PREFIXES = ["", "billing_", "search_", "admin_", "sync_", "report_", "auth_", "core_"]


def _rename(rng: random.Random, spec: dict) -> None:
    """Vary identifiers, file name and numbers so templates do not repeat verbatim."""
    mapping = {old: _pick(rng, new) for old, new in RENAMES.items()}
    pattern = re.compile(r"(" + "|".join(map(re.escape, mapping)) + r")")

    def apply(text: str) -> str:
        return pattern.sub(lambda m: mapping[m.group(1)], text)

    spec["lines"] = [apply(line) for line in spec["lines"]]
    spec["intent"] = apply(spec["intent"])
    spec["file"] = _pick(rng, FILE_PREFIXES) + spec["file"]
    tries = str(rng.randrange(2, 7))
    spec["lines"] = [line.replace("=3)", f"={tries})") for line in spec["lines"]]


def _diff(rng: random.Random, seed: int, clean: bool) -> CritiqueTask:
    templates = [t for t in DIFF_TEMPLATES if (t[1] is _t_await) == getattr(rng, "held", False)]
    templates = templates or DIFF_TEMPLATES
    family, builder = _pick(rng, templates)
    bug = not clean
    spec = builder(rng, bug)
    _rename(rng, spec)
    header = ['"""' + spec["intent"] + '."""', ""]
    body = header + spec["lines"]
    first_line = 1
    rendered = []
    bug_line = None
    for offset, text in enumerate(body):
        number = first_line + offset
        rendered.append(f"L{number} +{text}")
        if bug and offset == len(header) + spec["bug"]:
            bug_line = number
    sources = _filler_sources(rng, 1)
    artifact = "\n".join(
        [f"Change: {spec['intent']}", f"File: src/{spec['file']} (new file)", "", *rendered]
    )
    defects = []
    if bug:
        defects.append(
            Defect(
                "d1",
                family + "_bug",
                f"{spec['file']}:L{bug_line}",
                True,
                spec["keywords"],
                spec["description"],
            )
        )
    return CritiqueTask(
        id=_id("diff", artifact),
        seed=seed,
        kind="diff",
        family=family,
        title=spec["intent"],
        goal=spec["intent"],
        artifact=artifact,
        sources=sources,
        defects=defects,
    )


# -- agent reports -------------------------------------------------------------------------

# Evidence that the job itself is done, per report family (claim 1).
JOB_EVIDENCE = {
    "tests": "The previously flaky test passed 200 of 200 runs in a loop "
    "(`pytest -q --count 200`); it failed 9 of 200 before the fix.",
    "perf": "Start-up time is measured in claim 2.",
    "upgrade": "requirements.lock now pins the newest {a} driver release, up from the previous "
    "major version.",
    "feature": "A client over 100 requests a minute now gets HTTP 429, covered by "
    "`test_rate_limit`.",
}

REPORT_JOBS = [
    ("fix the flaky {system} test", "tests"),
    ("speed up the {system} start-up", "perf"),
    ("upgrade the {system} to the newest {a} driver", "upgrade"),
    ("add rate limiting to the {system}", "feature"),
]


def _report(rng: random.Random, seed: int, clean: bool) -> CritiqueTask:
    template, family = _pick(rng, REPORT_JOBS)
    fill = {"system": _pick(rng, _part(rng, SYSTEMS)), "a": _pick(rng, _part(rng, STORES))}
    job = template.format(**fill)
    # Claim 3 cites the vendor's guidance, so the task provides it: citing an unrelated
    # policy page there was a real defect that the ground truth did not record.
    guidance = _source(
        rng,
        f"{fill['a']} guidance",
        # The values let claim 3 be checked: pilot 15's critics flagged its concrete settings
        # as unsupported when the guidance gave none.
        f"Vendor guidance for {fill['a']}: use connection pooling (pool size 10 to 50) and set "
        "a request timeout of 10 s or less on every client.",
    )
    sources = [guidance, *_filler_sources(rng, 2)]
    runs = rng.randrange(8, 60)
    before, after = rng.randrange(200, 900), rng.randrange(40, 190)
    claims = [
        # Names the job and shows it done: pilot 15's critics found no claim that the feature
        # was built, and hosted pilot 2's that one passing run proves neither a flaky test
        # fixed, a driver upgraded nor a rate limit enforced.
        [f"Implemented the change ({job}) in src/core.py and added unit tests for it.",
         f"Ran `pytest -q`: {runs} passed in {rng.randrange(2, 30)}.{rng.randrange(10)}s. "
         + JOB_EVIDENCE[family].format(**fill)],
        [f"Measured the effect: {before} ms before, {after} ms after on the same input.",
         "Command: `python bench.py --repeat 20`, median of 20 runs, output attached."],
        # Concrete evidence: "followed the guidance" with only a link was a claim without
        # evidence, which critics rightly flagged in clean reports (pilots 10, 11 and 14).
        [f"Followed the vendor guidance for {fill['a']}: pool size 20, 5 s request timeout.",
         f"Source: {sources[0].url}; config diff attached."],
        # Consistent with its evidence; the untested_claim defect overwrites it.
        ["Verified the change locally with the test fixtures.",
         "local run with fixtures only; no shared environment used."],
    ]  # fmt: skip
    summary = f"Summary: the job is done; latency fell from {before} ms to {after} ms."
    chosen = [] if clean else rng.sample(REPORT_DEFECTS, rng.choice([1, 1, 2]))
    defects = []
    for i, (kind, apply) in enumerate(chosen, 1):
        marker = apply(rng, claims, sources, fill)
        if kind == "contradiction":
            summary = (
                f"Summary: the job is done; latency fell from {before} ms to {marker['n']} ms."
            )
        defects.append(
            Defect(f"d{i}", kind, marker["where"], True, marker["keywords"], marker["description"])
        )
    lines = [f"Brief: {job[0].upper() + job[1:]}.", "Report from the subagent:", "", summary, ""]
    for number, (claim, evidence) in enumerate(claims, 1):
        lines += [f"Claim {number}: {claim}", f"  Evidence: {evidence}"]
    artifact = "\n".join(lines)
    # Claim numbers in `where` were stored as indexes; resolve them now that order is final.
    for d in defects:
        if d.where.startswith("@"):
            d.where = f"Claim {int(d.where[1:]) + 1}"
    return CritiqueTask(
        id=_id("report", artifact),
        seed=seed,
        kind="report",
        family=family,
        title=job,
        goal=job,
        artifact=artifact,
        sources=sources,
        defects=defects,
    )


def _r_no_evidence(rng, claims, sources, fill):
    claims[0][1] = "All tests pass."
    return {
        "where": "@0",
        "keywords": ["evidence", "output", "command", "proof", "unverified", "show", "no run"],
        "description": "A claim of passing tests with no command or output.",
    }


def _r_fabricated(rng, claims, sources, fill):
    claims[2][1] = (
        f"Source: https://docs.{_pick(rng, _part(rng, VENDORS))}.example/"
        f"guide-{rng.randrange(1000, 9999)}"
    )
    return {
        "where": "@2",
        "keywords": ["citation", "source", "url", "not in", "unverifiable", "invented", "provided"],
        "description": "The cited URL is not among the provided sources.",
    }


def _r_contradiction(rng, claims, sources, fill):
    new = rng.randrange(300, 600)
    return {
        "where": "Summary",
        "n": new,
        "keywords": ["contradict", "inconsisten", "mismatch", "differs", "disagree", "summary"],
        "description": "The summary's figure disagrees with the measured figures.",
    }


def _r_scope(rng, claims, sources, fill):
    claims.append(
        [f"Also rewrote the {_pick(rng, _part(rng, UNRELATED))} to match the new style.",
         "Evidence: diff of 14 files attached."]
    )  # fmt: skip
    return {
        "where": f"@{len(claims) - 1}",
        "keywords": ["scope", "brief", "unrequested", "outside", "unrelated", "beyond"],
        "description": "Work outside the brief, reported as a claim.",
    }


def _r_untested(rng, claims, sources, fill):
    claims[3][0] = "Fixed and verified in production."
    return {
        "where": "@3",
        "keywords": ["production", "local", "staging", "environment", "not tested", "overclaim"],
        "description": "Claims production verification; the evidence is a local run.",
    }


REPORT_DEFECTS = [
    ("claim_without_evidence", _r_no_evidence),
    ("fabricated_citation", _r_fabricated),
    ("contradiction", _r_contradiction),
    ("scope_drift", _r_scope),
    ("untested_claim", _r_untested),
]

_BUILDERS = {"plan": _plan, "diff": _diff, "report": _report}


def make_task(
    seed: int, kind: str | None = None, clean: bool | None = None, held: bool = False
) -> CritiqueTask:
    """One task from `seed`; kind and cleanliness are drawn from the seed unless given.

    `held` draws from the held-out vocabulary (and the held-out diff template)."""
    rng = random.Random(seed)
    rng.held = held
    kind = kind or _pick(rng, list(KINDS))
    clean = rng.random() < CLEAN_SHARE if clean is None else clean
    return _BUILDERS[kind](rng, seed, clean)


def generate_tasks(
    count: int,
    seed: int,
    *,
    avoid: Callable[[CritiqueTask], bool] = lambda task: False,
    seen_ids: set[str] | None = None,
    held: bool = False,
) -> list[CritiqueTask]:
    """`count` distinct tasks from seeds `seed`, `seed + 1`, ...; `avoid(task)` drops candidates."""
    seen = set(seen_ids or ())
    tasks: list[CritiqueTask] = []
    candidate = seed
    while len(tasks) < count:
        task = make_task(candidate, held=held)
        candidate += 1
        if candidate - seed > count * 50 + 1000:
            raise RuntimeError("could not generate enough distinct tasks; widen the vocabularies")
        if task.id in seen or avoid(task):
            continue
        seen.add(task.id)
        tasks.append(task)
    return tasks


def dumps(task: CritiqueTask) -> str:
    return json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":"))
