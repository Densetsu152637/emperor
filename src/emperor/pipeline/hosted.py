"""`uv run hosted`: the data phases on hosted models (no llama-server, no idle gate).

prompts -> generate -> judge -> retry rounds -> filter -> report, with the teacher and the tool
simulator on DeepSeek and the judge on Kimi (configs/hosted.toml). Every phase resumes from the
data on disk, so a stopped run continues where it left off:

    uv run hosted --run-dir runs/hosted-1 --data-dir data/hosted-1          # start or resume
    uv run hosted --run-dir runs/hosted-1 --data-dir data/hosted-1 --detach # in the background

Create `<run-dir>/STOP` to stop after the requests in flight. A provider whose spend estimate
reaches its `budget_usd` stops the run too (status `budget`); raise the cap and resume.
`<run-dir>/status.json` is rewritten every 30 seconds: phase, progress, rejects by reason,
spend per provider and a cost and time projection. Every request's usage is appended to
`<data-dir>/spend.jsonl`.

Training is not part of this command: train with `uv run pipeline --only train ...` after the
data is in, one model at a time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from emperor.data.hosted import (
    DEFAULT_CONFIG,
    HostedClient,
    HostedConfig,
    Ledger,
    load_config,
    make_clients,
)
from emperor.data.records import current, exclude_pilot, exclusions, fingerprint
from emperor.data.teacher import TeacherError, TeacherSetupError

log = logging.getLogger("emperor.hosted")
# Long multi-subagent traces are what Emperor exists to learn, so they are kept: training may
# offload to RAM. Two too-long pilot traces measured 8.7K and 9.7K tokens without the old 8,192
# cap; 16,384 keeps them with room, and still rejects a runaway trace.
MAX_SAMPLE_TOKENS = 16384
OUTAGE_WAIT = 900  # seconds between tries of a phase whose provider stopped answering
STATUS_SECONDS = 30.0


def count_dir(directory: Path) -> dict:
    """Progress of one data directory (main pass or a retry round)."""
    prompts_file = directory / "prompts.jsonl"
    prompts = 0
    if prompts_file.exists():
        prompts = sum(1 for line in prompts_file.read_text(encoding="utf-8").splitlines() if line)
    # The latest record per id, without excluded pilots: what a resume and the filter count.
    rejects = Counter(r.get("reason", "unknown") for r in current(directory / "rejects").values())
    traces = len(current(directory / "traces"))
    judged = kept = 0
    judge_rejects: Counter = Counter()
    for sample in current(directory / "judged").values():
        judged += 1
        verdict = sample.get("meta", {}).get("judge", {})
        if verdict.get("keep"):
            kept += 1
        else:
            judge_rejects.update(verdict.get("rejection") or ["unknown"])
    return {
        "prompts": prompts,
        "generated": traces,
        "generation_rejects": dict(rejects.most_common()),
        "judged": judged,
        "kept": kept,
        "judge_rejects": dict(judge_rejects.most_common()),
    }


def ping(client: HostedClient) -> str:
    """One tiny request: proves the key and model work, and returns the served model id."""
    client.chat([{"role": "user", "content": "Reply with the word OK."}], max_tokens=256)
    return client.served_model or client.model


class HostedRun:
    """The data phases of one model on hosted APIs. Consul subclasses the phase methods."""

    name = "emperor"
    phases = ("prompts", "generate", "judge", "retry", "filter", "report")
    host = "DeepSeek API"
    # DeepSeek Open Platform Terms of Service 4.2: outputs are assigned to the user, who may
    # use them for "training other models (such as model distillation)".
    licence = "DeepSeek Open Platform Terms of Service 4.2"

    def __init__(self, args, config: HostedConfig):
        self.args = args
        self.config = config
        self.root: Path = args.root.resolve()
        self.data_dir: Path = (self.root / args.data_dir).resolve()
        self.run_dir: Path = (self.root / args.run_dir).resolve()
        self.ledger = Ledger(
            self.data_dir / "spend.jsonl",
            {name: p.budget_usd for name, p in config.providers.items()},
        )
        self.clients: dict[str, HostedClient] = {}
        self.names: dict[str, str] = {}
        self.phase = "starting"
        self.state: dict = {"phases": {}}
        self.started = time.time()
        self.progress: list[tuple[float, int, float]] = []  # (time, items done, usd)
        self._sleep = time.sleep

    # -- plumbing ------------------------------------------------------------------------------

    def should_stop(self) -> bool:
        return (self.run_dir / "STOP").exists()

    def client(self, role: str) -> HostedClient:
        if role not in self.clients:
            made = make_clients(self.config, self.ledger, (role,), should_stop=self.should_stop)
            self.names[role] = ping(made[role])  # kept only once the provider has answered
            self.clients.update(made)
            log.info("%s: %s", role, self.names[role])
        return self.clients[role]

    def run_phase(self, name: str) -> dict:
        """One phase; a provider outage (no answer after every retry, as when Kimi answered
        "engine overloaded" for over an hour in hosted pilot 4) is waited out and the phase
        resumed (by default for as long as it takes) instead of ending the run."""
        waits = 0
        while True:
            try:
                result = getattr(self, f"phase_{name}")()
            except TeacherSetupError:
                raise
            except TeacherError as error:
                result = {"stopped": True, "error": str(error)}
            outage = "no answer after" in str(result.get("error", ""))
            limit = self.args.outage_waits
            if not (result.get("stopped") and outage) or (limit >= 0 and waits >= limit):
                return result
            waits += 1
            self.phase = "waiting_provider"
            log.warning("%s: %s; waiting %ds (wait %d)", name, result["error"], OUTAGE_WAIT, waits)
            for _ in range(OUTAGE_WAIT // 10):
                if self.should_stop():
                    return result
                self._sleep(10)
            self.phase = name

    def workers(self, role: str) -> int:
        override = self.args.judge_workers if role == "judge" else self.args.workers
        return override or self.config.role(role)[1].max_concurrency

    def counts(self) -> dict:
        main = count_dir(self.data_dir)
        rounds = {
            p.name: count_dir(p)
            for p in sorted(self.data_dir.glob("retry-*"))
            if (p / "prompts.jsonl").exists()
        }
        return {"main": main, "retry_rounds": rounds}

    def write_status(self, state: str) -> None:
        counts = self.counts()
        everything = [counts["main"], *counts["retry_rounds"].values()]
        generated = sum(c["generated"] + sum(c["generation_rejects"].values()) for c in everything)
        judged = sum(c["judged"] for c in everything)
        kept = sum(c["kept"] for c in everything)
        spend = self.ledger.summary()
        usd = sum(p["usd"] for p in spend.values())
        now = time.time()
        self.progress.append((now, generated + judged, usd))
        self.progress = [p for p in self.progress if now - p[0] <= 3 * 3600]
        first, last = self.progress[0], self.progress[-1]
        rate = (last[1] - first[1]) / (last[0] - first[0]) if last[0] - first[0] >= 120 else None
        prompts = counts["main"]["prompts"]
        tasks_done = counts["main"]["generated"] + sum(
            counts["main"]["generation_rejects"].values()
        )
        usd_per_task = usd / tasks_done if tasks_done else None
        waiting = [w for c in self.clients.values() if (w := getattr(c, "waiting_until", None))]
        status = {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "model": self.name,
            "state": state,
            # Every new DeepSeek call is held back for the off-peak window.
            "phase": "waiting_offpeak" if waiting else self.phase,
            "resume_at_utc": min(waiting).isoformat(timespec="minutes") if waiting else None,
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "data_dir": str(self.data_dir),
            "served_models": self.names,
            "progress": {"generated_or_rejected": generated, "judged": judged, "kept": kept},
            "items_per_hour": round(rate * 3600, 1) if rate else None,
            "phase_eta_hours": self._eta(counts, rate),
            **counts,
            "spend": spend,
            "spend_usd": round(usd, 4),
            "usd_per_task": round(usd_per_task, 5) if usd_per_task else None,
            "projected_usd_main_pass": round(usd_per_task * prompts, 2) if usd_per_task else None,
            "phases_done": self.state["phases"],
            "pilot": self.args.pilot,
            "excluded_pilots": exclusions(self.data_dir),
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.run_dir / "status.tmp"
        tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
        tmp.replace(self.run_dir / "status.json")

    def _eta(self, counts: dict, rate: float | None) -> float | None:
        if not rate:
            return None
        main = counts["main"]
        done = main["generated"] + sum(main["generation_rejects"].values())
        left = (main["prompts"] - done) + (main["generated"] - main["judged"])
        return round(max(left, 0) / rate / 3600, 1)

    def _status_loop(self, stop: threading.Event) -> None:
        while not stop.wait(STATUS_SECONDS):
            try:
                self.write_status("running")
            except Exception as error:  # the status file must never end the run
                log.warning("status: %r", error)

    # -- phases ---------------------------------------------------------------------------------

    def phase_prompts(self) -> dict:
        from emperor.data.prompts import HeldoutIndex, generate_prompts, normalise
        from emperor.data.shards import read_jsonl

        path = self.data_dir / "prompts.jsonl"
        own = [normalise(r["text"]) for r in read_jsonl(path)] if path.exists() else []
        need = self.args.prompts - len(own)
        if need <= 0:
            return {"status": "done", "prompts": len(own)}
        records, stats = generate_prompts(
            need, self.args.seed + len(own), heldout=HeldoutIndex.bundled(), existing=own
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return {"status": "done", "prompts": len(own) + len(records), **stats}

    def generation_config(self, attempt: int = 0):
        from emperor.data.generate import GenerationConfig

        role = self.config.roles["teacher"]
        return GenerationConfig(
            retries=self.args.retries,
            # DeepSeek cannot force a tool call in thinking mode (the client drops it); the
            # plan-only redraws still apply.
            force_first_tool=True,
            teacher_model=self.names.get("teacher", role.model),
            host=self.host,
            licence=self.licence,
            attempt=attempt,
            max_sample_tokens=self.args.max_sample_tokens,
            provenance=self.provenance(),
        )

    # -- provenance ----------------------------------------------------------------------------

    def guidance(self) -> str:
        """A version of everything the teacher and simulator are told (prompts, style)."""
        from emperor.data.generate import _ENV_SYSTEM, TABLE_REWRITE, teacher_system

        return fingerprint(teacher_system(65536) + _ENV_SYSTEM + TABLE_REWRITE)

    def rubric(self) -> str:
        """A version of what the judge is told."""
        from emperor.data.style import CLASS_NOTES, judge_system

        return fingerprint(judge_system() + json.dumps(CLASS_NOTES, sort_keys=True))

    def commit(self) -> str:
        try:
            head = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return head + ("+dirty" if dirty else "")

    def provenance(self) -> dict:
        """Recorded with every sample and reject, so a pilot's samples can be told apart and,
        if a later fix shows its setup was defective, excluded."""
        if not hasattr(self, "_provenance"):
            roles = {
                name: {
                    "model": role.model,
                    "base_url": self.config.providers[role.provider].base_url,
                    "thinking": role.thinking,
                    "reasoning_effort": role.reasoning_effort,
                    "max_tokens": role.max_tokens,
                }
                for name, role in self.config.roles.items()
            }
            self._provenance = {
                "pilot": self.args.pilot,
                "pipeline_commit": self.commit(),
                "config_hash": fingerprint(roles),
                "guidance": self.guidance(),
                "models": {
                    "teacher": self.names.get("teacher", self.config.roles["teacher"].model),
                    "simulator": self.names.get("simulator", self.config.roles["simulator"].model),
                    "judge": self.config.roles["judge"].model,
                },
            }
        return self._provenance

    def judge_settings(self) -> dict:
        """What a verdict depends on; a sample judged under other settings is judged again."""
        role = self.config.roles["judge"]
        return {
            "model": role.model,
            "reasoning_effort": role.reasoning_effort,
            "threshold": self.args.threshold,
            "rubric": self.rubric(),
        }

    def _generate(self, directory: Path, attempt: int = 0) -> dict:
        from emperor.data.generate import run_generation

        teacher, simulator = self.client("teacher"), self.client("simulator")
        return run_generation(
            directory / "prompts.jsonl",
            directory,
            teacher,
            simulator,
            config=self.generation_config(attempt),
            should_stop=self.should_stop,
            log=log.debug,
            workers=self.workers("teacher"),
            limit=self.args.limit,
        )

    def _judge(self, directory: Path) -> dict:
        from emperor.data.judge import run_judging

        judge = self.client("judge")
        return run_judging(
            directory,
            judge,
            threshold=self.args.threshold,
            model=self.names["judge"],
            should_stop=self.should_stop,
            log=log.debug,
            workers=self.workers("judge"),
            settings=self.judge_settings(),
        )

    def phase_generate(self) -> dict:
        return self._generate(self.data_dir)

    def phase_judge(self) -> dict:
        return self._judge(self.data_dir)

    def phase_retry(self) -> dict:
        from emperor.data.retry import prepare_round, round_dir

        rounds = {}
        for k in range(1, self.args.retries + 1):
            if not prepare_round(self.data_dir, k):
                break
            directory = round_dir(self.data_dir, k)
            generated = self._generate(directory, attempt=k)
            if generated.get("stopped"):
                return {**generated, "rounds": rounds}
            judged = self._judge(directory)
            rounds[directory.name] = {"generate": generated, "judge": judged}
            if judged.get("stopped"):
                return {**judged, "rounds": rounds}
        return {"stopped": False, "rounds": rounds}

    def phase_filter(self) -> dict:
        from emperor.data.build import filter_kept

        return filter_kept(self.data_dir)

    def phase_report(self) -> dict:
        from emperor.data.build import write_report

        report = write_report(self.data_dir)
        return {"kept": report.get("kept_with_retries", report.get("kept"))}

    # -- the run -------------------------------------------------------------------------------

    def run(self) -> str:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.run_dir / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
        stop = threading.Event()
        threading.Thread(target=self._status_loop, args=(stop,), daemon=True).start()
        outcome = "done"
        try:
            for name in self.phases:
                if self.args.only and name not in self.args.only:
                    continue
                if self.should_stop():
                    outcome = "stopped"
                    break
                self.phase = name
                log.info("phase %s", name)
                try:
                    result = self.run_phase(name)
                except TeacherSetupError as error:
                    log.error("%s: %s", name, error)
                    outcome = "failed"
                    self.state["phases"][name] = {"status": "failed", "error": str(error)}
                    break
                stopped = result.get("stopped")
                status = "stopped" if stopped else "done"
                self.state["phases"][name] = {"status": status, **result}
                state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
                log.info("phase %s: %s", name, result)
                if stopped:
                    error = str(result.get("error", ""))
                    outcome = "budget" if "cap" in error or "402" in error else "stopped"
                    break
        finally:
            stop.set()
            self.phase = "finished"
            self.write_status(outcome)
            state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            for client in self.clients.values():
                client.close()
        return outcome


def build_parser(prog: str = "hosted") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.strip().splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hosted"))
    parser.add_argument(
        "--run-dir", type=Path, help="default: runs/hosted-<timestamp>; reuse it to resume"
    )
    parser.add_argument("--prompts", type=int, default=2000, help="tasks in the main pass")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2, help="extra tries and retry rounds")
    parser.add_argument("--threshold", type=float, default=4.0, help="minimum mean judge score")
    parser.add_argument(
        "--max-sample-tokens",
        type=int,
        default=MAX_SAMPLE_TOKENS,
        help="reject a trace once it passes this length (the training length; default %(default)s)",
    )
    parser.add_argument("--workers", type=int, help="parallel tasks (default: the provider's)")
    parser.add_argument("--judge-workers", type=int, help="parallel judgements")
    hours = parser.add_mutually_exclusive_group()
    hours.add_argument(
        "--offpeak-only", action="store_true", help="send DeepSeek requests only off-peak"
    )
    hours.add_argument(
        "--allow-peak",
        action="store_true",
        help="also send DeepSeek requests in peak hours (the default waits for off-peak)",
    )
    parser.add_argument(
        "--only", action="append", default=[], help="run just this phase (repeatable)"
    )
    parser.add_argument("--detach", action="store_true", help="run in the background")
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="PROVIDER=USD",
        help="override a provider's budget_usd for this run (repeatable)",
    )
    parser.add_argument(
        "--outage-waits",
        type=int,
        default=-1,
        help="times a phase waits 15 minutes for a provider that stopped answering "
        "(default -1: it never gives the phase up)",
    )
    pilots = parser.add_argument_group("pilots (they write into the full run's data dir)")
    pilots.add_argument("--pilot", type=int, help="pilot number, recorded with every sample")
    pilots.add_argument(
        "--limit", type=int, help="generate only the next N undone tasks of the pool"
    )
    pilots.add_argument(
        "--exclude-pilot",
        action="append",
        default=[],
        metavar="N=REASON",
        help="exclude a pilot's samples (its tasks are generated again); recorded in the data dir",
    )
    return parser


def configure(args) -> HostedConfig:
    config = load_config(args.root / args.config if not args.config.is_absolute() else args.config)
    for item in getattr(args, "budget", []):
        name, _, usd = item.partition("=")
        if name not in config.providers:
            raise SystemExit(f"--budget: unknown provider {name!r}")
        config.providers[name].budget_usd = float(usd)
    for provider in config.providers.values():
        if args.offpeak_only and provider.offpeak_factor < 1.0:
            provider.offpeak_only = True
        if args.allow_peak:
            provider.offpeak_only = False
    return config


def detach(argv: list[str], run_dir: Path, module: str) -> int:
    child = [a for a in argv if a != "--detach"]
    if "--run-dir" not in child:
        child += ["--run-dir", str(run_dir)]
    run_dir.mkdir(parents=True, exist_ok=True)
    out = (run_dir / "stdout.log").open("ab")
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *child],
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        start_new_session=os.name != "nt",
    )
    (run_dir / "hosted.pid").write_text(str(proc.pid), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "pid": proc.pid}))
    return 0


def main(
    argv: list[str] | None = None,
    *,
    run_class: type[HostedRun] = HostedRun,
    module: str = "emperor.pipeline.hosted",
    prog: str = "hosted",
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Another model's package passes its own `run_class`, `module` and `parser` (built on
    `build_parser`) to reuse this runner with its own phases."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = (parser or build_parser(prog)).parse_args(argv)
    if args.run_dir is None:
        args.run_dir = Path("runs") / f"hosted-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    run_dir = (args.root / args.run_dir).resolve()
    if args.detach:
        return detach(argv, run_dir, module)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "hosted.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "httpcore"):  # request lines would only repeat the status file
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for item in args.exclude_pilot:
        pilot, _, reason = item.partition("=")
        if not pilot.strip().isdigit() or not reason.strip():
            raise SystemExit(f"--exclude-pilot needs N=REASON, got {item!r}")
        exclude_pilot((args.root / args.data_dir).resolve(), int(pilot), reason.strip())
    outcome = run_class(args, configure(args)).run()
    print(json.dumps({"status": outcome, "status_file": str(run_dir / "status.json")}))
    return {"done": 0, "stopped": 0, "budget": 3}.get(outcome, 1)


if __name__ == "__main__":
    sys.exit(main())
