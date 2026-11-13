"""The whole build of Emperor and Consul, end to end, without manual steps.

1. Hosted data for Emperor, then for Consul (`hosted`, 2,000 tasks each), resuming from the
   pilot data in `data/hosted-1`; DeepSeek calls only off-peak; provider outages and budget
   caps pause the stage and it resumes on its own.
2. Once both datasets are complete: filter -> train -> merge -> export (bf16 and GGUF) -> eval
   -> card for Emperor, then the same for Consul (the local `pipeline`), one model at a time on
   the GPU and only while the user is idle. Training uses the length the data needs (p99,
   capped at 32K) with activation offload to RAM.

Nothing is published to Hugging Face.

Run it from anywhere with Python 3.12+ (standard library only):

    python scripts/full_run.py start
    python scripts/full_run.py status
    python scripts/full_run.py stop      # the running stage stops cleanly; `start` resumes

Budgets are one pool per provider for the whole run, both models together, in USD, read from
`<emperor>/runs/full/budgets.json` ({"deepseek": 95, "kimi": 37} by default) every time a
stage starts: each model's hosted stage gets the pool minus what the other model's ledger
(`data/hosted-1/spend.jsonl`) has spent. Raise a value in that file while a stage is paused on
its cap and it carries on at the next check (every 30 minutes); no restart is needed.
State, logs and `status.json` live in `<emperor>/runs/full/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

EMPEROR = Path(__file__).resolve().parents[1]
HOME = EMPEROR / "runs" / "full"
STATUS, LOG, PID, STOP = (HOME / n for n in ("status.json", "full_run.log", "full_run.pid", "STOP"))
BUDGETS = HOME / "budgets.json"
DEFAULT_BUDGETS = {"deepseek": 95.0, "kimi": 37.0}  # the user's top-up, both models together
DATA = "data/hosted-1"
BUDGET_WAIT = 1800  # seconds between checks while a provider's budget cap holds a stage
RESTART_WAIT = 300
TRAIN_PHASES = ["filter", "train", "merge", "export", "eval", "card"]


def repos(args) -> dict[str, Path]:
    return {"emperor": EMPEROR, "consul": Path(args.consul).resolve()}


def log(message: str) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_status(state: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    state = {**state, "updated": datetime.now().isoformat(timespec="seconds")}
    for name in ("emperor", "consul"):
        hosted = read_json(Path(state["repos"][name]) / "runs" / "full-hosted" / "status.json")
        if hosted:
            state[f"{name}_data"] = {
                k: hosted.get(k)
                for k in ("state", "phase", "resume_at_utc", "progress", "spend", "spend_usd",
                          "usd_per_task", "projected_usd_main_pass", "phase_eta_hours")
            }  # fmt: skip
        pipeline = read_json(Path(state["repos"][name]) / "runs" / "full-train" / "state.json")
        if pipeline:
            state[f"{name}_build"] = {
                k: v.get("status") for k, v in pipeline.get("phases", {}).items()
            }
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATUS)


def venv_python(repo: Path) -> Path:
    return repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str], repo: Path, run_dir: Path, state: dict, stage: str) -> int:
    """Run one stage's process to the end, keeping the status current and passing STOP on."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    log(f"{stage}: {' '.join(command)}")
    with (run_dir / "stdout.log").open("ab") as out:
        proc = subprocess.Popen(
            command, cwd=repo, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )  # fmt: skip
    state.update(stage=stage, stage_pid=proc.pid)
    while proc.poll() is None:
        if STOP.exists() and not (run_dir / "STOP").exists():
            (run_dir / "STOP").write_text("full run stop\n", encoding="utf-8")
            log(f"{stage}: STOP passed on")
        write_status(state)
        time.sleep(30)
    log(f"{stage}: exit {proc.returncode}")
    return proc.returncode


def seed_pilot_data(repo: Path, args) -> None:
    """The pilots ran in the hosted worktrees; their data counts towards the full run. The
    held-out set (`eval/`) is not copied: the prompts phase builds it from the task pool."""
    target = repo / DATA
    source = Path(args.pilot_data_root) / f"{repo.name}-hosted" / DATA
    if not target.exists() and source.exists():
        shutil.copytree(
            source, target, ignore=shutil.ignore_patterns("eval", ".hosted.lock", "*.tmp")
        )
        log(f"{repo.name}: pilot data copied from {source}")


def budgets() -> dict[str, float]:
    """The pool per provider, re-read every time (edit runs/full/budgets.json to raise it)."""
    if not BUDGETS.exists():
        BUDGETS.parent.mkdir(parents=True, exist_ok=True)
        BUDGETS.write_text(json.dumps(DEFAULT_BUDGETS, indent=1), encoding="utf-8")
    pool = {**DEFAULT_BUDGETS, **read_json(BUDGETS)}
    return {k: float(v) for k, v in pool.items()}


def spent(repo: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    ledger = repo / DATA / "spend.jsonl"
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals[row["provider"]] = totals.get(row["provider"], 0.0) + row["usd"]
    return totals


def stage_budgets(repo: Path, others: list[Path]) -> list[str]:
    """This model's caps: the pool less what the other model has spent."""
    flags = []
    for provider, pool in budgets().items():
        other = sum(spent(o).get(provider, 0.0) for o in others)
        flags += ["--budget", f"{provider}={max(pool - other, 0.0):.2f}"]
    return flags


def hosted_done(repo: Path) -> bool:
    phases = read_json(repo / "runs" / "full-hosted" / "state.json").get("phases", {})
    wanted = ["prompts", "generate", "judge", "filter", "report"]
    if repo.name != "consul":
        wanted.insert(3, "retry")
    return all(phases.get(p, {}).get("status") == "done" for p in wanted)


def data_stage(name: str, repo: Path, args, state: dict) -> bool:
    seed_pilot_data(repo, args)
    others = [r for r in repos(args).values() if r != repo]
    restarts = 0
    while not hosted_done(repo):
        if STOP.exists():
            return False
        command = [
            str(venv_python(repo)), "-m", f"{name}.pipeline.hosted", "--root", str(repo),
            "--data-dir", DATA, "--run-dir", "runs/full-hosted", "--prompts", "2000",
            *stage_budgets(repo, others),
        ]  # fmt: skip
        code = run(command, repo, repo / "runs" / "full-hosted", state, f"{name} data")
        if STOP.exists():
            return False
        if code == 3:  # a budget cap: wait for it to be raised in budgets.json, then go on
            state["note"] = f"{name}: a provider's budget pool is spent; raise it in {BUDGETS}"
            write_status(state)
            log(state["note"])
            time.sleep(BUDGET_WAIT)
            continue
        if not hosted_done(repo):
            # Stopped on an error (exit 1: a setup problem such as a retired model) or a crash:
            # not done. Wait, then try again; the status says why.
            restarts += 1
            state["note"] = (
                f"{name} data stopped (exit {code}); see {repo}/runs/full-hosted/hosted.log"
            )
            write_status(state)
            log(f"{state['note']}; restart {restarts}")
            time.sleep(RESTART_WAIT)
    state.pop("note", None)
    return True


def train_config(name: str, repo: Path) -> Path:
    """The repo's training config, pointed at the hosted data, with the length from the data
    and activation offload (written beside the run, the repo's file is left alone)."""
    text = (repo / "configs" / "sft.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    data["data"]["path"] = f"{DATA}/final/final.jsonl"
    data["data"]["max_length_auto"] = True
    data["train"]["offload_activations"] = True
    out = HOME / f"sft-{name}.toml"
    out.write_text(_toml(data), encoding="utf-8")
    return out


def _toml(data: dict) -> str:
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return json.dumps(v)

    lines = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {value(v)}" for k, v in table.items()]
        lines.append("")
    return "\n".join(lines)


def build_done(repo: Path) -> bool:
    phases = read_json(repo / "runs" / "full-train" / "state.json").get("phases", {})
    return all(phases.get(p, {}).get("status") in ("done", "skipped") for p in TRAIN_PHASES)


def build_stage(name: str, repo: Path, args, state: dict) -> bool:
    if not args.skip_sync:
        subprocess.run(["uv", "sync", "--group", "data", "--group", "train", "--group", "hosted"],
                       cwd=repo, check=False)  # fmt: skip
    config = train_config(name, repo)
    only = [a for p in TRAIN_PHASES for a in ("--only", p)]
    command = [
        str(venv_python(repo)), "-m", f"{name}.pipeline.cli", "--root", str(repo),
        "--resume", str(repo / "runs" / "full-train"), "--data-dir", DATA,
        "--train-config", str(config), *only,
    ]  # fmt: skip
    restarts = 0
    while not build_done(repo):
        if STOP.exists():
            return False
        code = run(command, repo, repo / "runs" / "full-train", state, f"{name} build")
        if STOP.exists():
            return False
        if not build_done(repo):
            restarts += 1
            log(f"{name} build: not finished (exit {code}); restart {restarts}")
            if restarts > 5:
                state["note"] = f"{name} build failed 5 times; see its pipeline.log"
                write_status(state)
                return False
            time.sleep(RESTART_WAIT)
    return True


def orchestrate(args) -> int:
    HOME.mkdir(parents=True, exist_ok=True)
    try:  # one full run at a time
        handle = os.open(PID, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = PID.read_text(encoding="utf-8").strip()
        if holder.isdigit() and _alive(int(holder)):
            print(f"another full run (pid {holder}) is running")
            return 1
        PID.unlink(missing_ok=True)
        handle = os.open(PID, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        out.write(str(os.getpid()))
    state = {"pid": os.getpid(), "repos": {k: str(v) for k, v in repos(args).items()}}
    log(f"full run {os.getpid()} started")
    try:
        for name, repo in repos(args).items():
            if not data_stage(name, repo, args, state):
                state["stage"] = "stopped"
                return 0
        for name, repo in repos(args).items():
            if not build_stage(name, repo, args, state):
                state["stage"] = "stopped" if STOP.exists() else "failed"
                return 0 if STOP.exists() else 1
        state["stage"] = "done"
        log("both models built")
        return 0
    finally:
        write_status(state)
        PID.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start(argv: list[str]) -> int:
    if PID.exists():
        holder = PID.read_text(encoding="utf-8").strip()
        if holder.isdigit() and _alive(int(holder)):
            print(f"already running (pid {holder})")
            return 1
    STOP.unlink(missing_ok=True)
    HOME.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)  # fmt: skip
    out = (HOME / "stdout.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run", *argv], cwd=HOME, stdout=out,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True,
    )  # fmt: skip
    print(json.dumps({"pid": proc.pid, "status": str(STATUS), "log": str(LOG)}))
    return 0


def stop() -> int:
    HOME.mkdir(parents=True, exist_ok=True)
    STOP.write_text("stop\n", encoding="utf-8")
    print("STOP written: the running stage stops cleanly, then the full run exits")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["start", "run", "stop", "status"])
    p.add_argument("--consul", default=str(EMPEROR.parent / "consul"))
    p.add_argument("--pilot-data-root", default=str(EMPEROR.parent),
                   help="where the <repo>-hosted worktrees with the pilot data are")  # fmt: skip
    p.add_argument("--skip-sync", action="store_true", help="do not uv sync before training")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.command == "start":
        return start(argv[1:])
    if args.command == "stop":
        return stop()
    if args.command == "status":
        print(STATUS.read_text(encoding="utf-8") if STATUS.exists() else "no status yet")
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
