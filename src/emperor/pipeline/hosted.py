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
from emperor.data.shards import iter_records
from emperor.data.teacher import TeacherError, TeacherSetupError

log = logging.getLogger("emperor.hosted")
STATUS_SECONDS = 30.0


def count_dir(directory: Path) -> dict:
    """Progress of one data directory (main pass or a retry round)."""
    prompts_file = directory / "prompts.jsonl"
    prompts = 0
    if prompts_file.exists():
        prompts = sum(1 for line in prompts_file.read_text(encoding="utf-8").splitlines() if line)
    rejects = Counter(r.get("reason", "unknown") for r in iter_records(directory / "rejects"))
    traces = sum(1 for _ in iter_records(directory / "traces"))
    judged = kept = 0
    judge_rejects: Counter = Counter()
    for sample in iter_records(directory / "judged"):
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

    # -- plumbing ------------------------------------------------------------------------------

    def should_stop(self) -> bool:
        return (self.run_dir / "STOP").exists()

    def client(self, role: str) -> HostedClient:
        if role not in self.clients:
            self.clients.update(
                make_clients(self.config, self.ledger, (role,), should_stop=self.should_stop)
            )
            self.names[role] = ping(self.clients[role])
            log.info("%s: %s", role, self.names[role])
        return self.clients[role]

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
        status = {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "model": self.name,
            "state": state,
            "phase": self.phase,
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
        )

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
                    result = getattr(self, f"phase_{name}")()
                except TeacherSetupError as error:
                    log.error("%s: %s", name, error)
                    outcome = "failed"
                    self.state["phases"][name] = {"status": "failed", "error": str(error)}
                    break
                except TeacherError as error:  # budget reached, provider unreachable
                    result = {"stopped": True, "error": str(error)}
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
    parser.add_argument("--workers", type=int, help="parallel tasks (default: the provider's)")
    parser.add_argument("--judge-workers", type=int, help="parallel judgements")
    parser.add_argument(
        "--offpeak-only", action="store_true", help="send DeepSeek requests only off-peak"
    )
    parser.add_argument(
        "--only", action="append", default=[], help="run just this phase (repeatable)"
    )
    parser.add_argument("--detach", action="store_true", help="run in the background")
    return parser


def configure(args) -> HostedConfig:
    config = load_config(args.root / args.config if not args.config.is_absolute() else args.config)
    if args.offpeak_only:
        for provider in config.providers.values():
            if provider.offpeak_factor < 1.0:
                provider.offpeak_only = True
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
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser(prog).parse_args(argv)
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
    outcome = run_class(args, configure(args)).run()
    print(json.dumps({"status": outcome, "status_file": str(run_dir / "status.json")}))
    return {"done": 0, "stopped": 0, "budget": 3}.get(outcome, 1)


if __name__ == "__main__":
    sys.exit(main())
