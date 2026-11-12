"""The pipeline: prompts, generate, judge, filter, train, merge, export, eval, card, summary.

Every GPU phase runs only while `IdleGate` allows it. When the user returns, the phase is
paused cleanly (llama-server stopped; training saves a checkpoint and exits) and resumed when
the machine is idle again. All work is resumable: shards are append-only and training resumes
from its last checkpoint, so a run directory can be reused with `--resume`.
"""

import dataclasses
import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from emperor.data.build import filter_kept, write_report
from emperor.data.generate import GenerationConfig, run_generation
from emperor.data.judge import run_judging
from emperor.data.prompts import HeldoutIndex, generate_prompts, normalise
from emperor.data.retry import prepare_round, round_dir
from emperor.data.shards import done_ids, iter_records, read_jsonl
from emperor.data.teacher import (
    ENVIRONMENT_EFFORT,
    GENERATE_EFFORT,
    JUDGE_EFFORT,
    JUDGE_MODEL,
    ChatClient,
    TeacherSetupError,
)
from emperor.eval.cli import CONTEXT_TIERS
from emperor.export.gguf import DEFAULT_CACHE, gguf_dir_for, gguf_name
from emperor.pipeline.idle import IdleGate, wait_until_idle
from emperor.pipeline.processes import (
    ManagedProcess,
    OwnProcesses,
    find_llama_server,
    llama_server_command,
    wait_for_health,
)
from emperor.release.card import write_cards
from emperor.train.config import SftConfig, load_config

log = logging.getLogger("emperor.pipeline")
PAUSED = 75
PHASES = [
    "prompts",
    "generate",
    "judge",
    "retry",
    "filter",
    "train",
    "merge",
    "export",
    "eval",
    "card",
]
DEFAULT_TEACHER_GGUF = "D:/AI-Models/unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-F16.gguf"
DEFAULT_JUDGE_GGUF = "D:/AI-Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"
# The teacher server keeps one slot per role, so the teacher's growing conversation and the
# tool simulator's short prompts each stay cached instead of evicting each other.
TEACHER_SLOT, ENVIRONMENT_SLOT = 0, 1
# The 65 GB teacher takes about 9 minutes to load from disk; the old 10-minute limit was too tight.
SERVER_LOAD_SECONDS = 3600.0
# The 120B teacher does not fit in RAM beside Windows: its CPU experts (51 GB) are paged in from
# disk on demand. No host-RAM prompt cache next to it, and no weight repacking, so every CPU
# weight stays file-backed and the OS can drop and re-read pages instead of swapping copies.
# No op offload: by default every prompt-processing batch streams all CPU experts to the GPU,
# which on this machine means paging all 51 GB in from disk for each 512-token batch.
# One KV buffer shared by the two role slots (the simulator's prompts are short), and a
# smaller VRAM margin: every GiB of GPU memory not spent on KV holds experts that would
# otherwise compete for RAM.
TEACHER_SERVER_ARGS = ["--cache-ram", "0", "--no-repack", "--no-op-offload", "--kv-unified"]
TEACHER_VRAM_MARGIN_MIB = 512


@dataclass
class Settings:
    root: Path
    run_dir: Path
    data_dir: Path
    previous_data: list[Path] = field(default_factory=list)  # earlier rounds: no repeated prompts
    retries: int = 2  # extra tries per prompt (schema failures in generation, judge failures)
    force_first_tool: bool = True
    prompts: int = 2000
    seed: int = 1
    threshold: float = 4.0
    min_samples: int = 100
    teacher_gguf: Path = Path(DEFAULT_TEACHER_GGUF)
    teacher_port: int = 8081
    # Per slot; the two slots share one 24K buffer. The longest pilot prompt was about 10K
    # tokens plus a 4K reply.
    teacher_context: int = 12288
    judge_gguf: Path = Path(DEFAULT_JUDGE_GGUF)
    judge_context: int = 32768  # transcript plus the judge's own thinking
    judge_thinking: bool = True
    parallel: int = 1
    reasoning_effort: str = GENERATE_EFFORT  # generation; the reasoning is trained on
    judge_reasoning_effort: str = JUDGE_EFFORT
    environment_reasoning_effort: str = ENVIRONMENT_EFFORT
    llama_server: Path | None = None
    llama_cpp: Path | None = None
    llama_cache: Path = DEFAULT_CACHE
    quantize: Path | None = None
    train_config: Path = Path("configs/sft.toml")
    eval_gguf: Path | None = None
    eval_type: str = "bf16"
    eval_port: int = 8082
    eval_tier: str = "64k"
    poll_seconds: float = 30.0
    pause_grace_seconds: float = 900.0
    skip: set[str] = field(default_factory=set)
    only: set[str] = field(default_factory=set)


class GateWatcher:
    """`should_stop` for workers: true when quitting or when the gate says the user is back.

    The gate is polled at most every `interval` seconds, because it shells out to nvidia-smi.
    """

    def __init__(
        self,
        gate: IdleGate,
        quit_flag: Callable[[], bool],
        interval: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.gate = gate
        self.quit_flag = quit_flag
        self.interval = interval
        self.clock = clock
        self.reason = ""
        self._last = -1e9
        self._stop = False

    def __call__(self) -> bool:
        if self.quit_flag():
            self.reason = "stop requested"
            return True
        if self._stop:
            return True
        now = self.clock()
        if now - self._last >= self.interval:
            self._last = now
            status = self.gate.can_continue()
            if not status.ok:
                self.reason = status.reason
                self._stop = True
        return self._stop


def _monitor(watcher, resource, stop, finished: threading.Event, seconds: float) -> None:
    while not finished.wait(seconds):
        if watcher():
            stop(resource)
            return


def run_gated(
    *,
    gate: IdleGate,
    quit_flag: Callable[[], bool],
    start: Callable[[], object],
    stop: Callable[[object], None],
    work: Callable[[object, Callable[[], bool]], dict],
    poll_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    max_failures: int = 3,
    watch_interval: float = 10.0,
    monitor_seconds: float | None = None,
    name: str = "phase",
) -> dict:
    """Run `work` on a resource that exists only while the machine is idle.

    Waits for the start condition, starts the resource, and lets `work(resource, should_stop)`
    run until it reports `stopped: False` (finished). When the user returns, `should_stop`
    turns true, the worker returns `stopped: True`, the resource is stopped and the loop waits
    for idle again. Repeated failures with the gate still open end the phase as failed.
    """
    pauses, failures, result = 0, 0, {}
    while True:
        if not wait_until_idle(
            gate, poll_seconds=poll_seconds, should_quit=quit_flag, sleep=sleep, log=log.info
        ):
            return {"status": "quit", "pauses": pauses}
        watcher = GateWatcher(gate, quit_flag, watch_interval)
        resource = None
        try:
            resource = start()
            finished = threading.Event()
            if monitor_seconds:
                # Stop the resource as soon as the user returns, so a long request in flight
                # does not hold the GPU until it finishes.
                threading.Thread(
                    target=_monitor,
                    args=(watcher, resource, stop, finished, monitor_seconds),
                    daemon=True,
                ).start()
            result = work(resource, watcher)
        except TeacherSetupError as error:  # retrying cannot help: stop the phase now
            log.error("%s: %s", name, error)
            return {"status": "failed", "error": str(error), "pauses": pauses}
        except Exception as error:  # a crashed server or worker: retry while the gate allows
            log.exception("%s: %s", name, error)
            result = {"stopped": True, "error": str(error)}
        finally:
            if resource is not None:
                finished.set()
                stop(resource)
        if not result.get("stopped"):
            return {
                "status": "done",
                "pauses": pauses,
                **{k: v for k, v in result.items() if k != "stopped"},
            }
        if quit_flag():
            return {"status": "quit", "pauses": pauses}
        if watcher.reason:
            pauses += 1
            log.info("%s paused: %s", name, watcher.reason)
        else:
            failures += 1
            log.warning(
                "%s failed (%s); attempt %d of %d",
                name,
                result.get("error"),
                failures,
                max_failures,
            )
            if failures >= max_failures:
                return {"status": "failed", "error": result.get("error"), "pauses": pauses}


class Pipeline:
    def __init__(self, settings: Settings, gate: IdleGate, own: OwnProcesses):
        self.s = settings
        self.gate = gate
        self.own = own
        self.stop_file = settings.run_dir / "STOP"
        self.interrupted = False
        self.state: dict = {"phases": {}}
        self.state_path = settings.run_dir / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.started = datetime.now(UTC).isoformat()

    # -- helpers ---------------------------------------------------------------------------

    @property
    def sft(self) -> SftConfig:
        return load_config(self.s.root / self.s.train_config)

    def teacher_info(self) -> dict:
        release = self.sft.release
        return {"model": release.teacher, "licence": release.teacher_licence}

    def quit_requested(self) -> bool:
        return self.interrupted or self.stop_file.exists()

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _phase_done(self, name: str) -> bool:
        return self.state["phases"].get(name, {}).get("status") == "done"

    def _record(self, name: str, result: dict) -> dict:
        self.state["phases"][name] = result
        self._save()
        log.info("phase %s: %s", name, result.get("status"))
        return result

    def _server_phase(
        self,
        name: str,
        gguf: Path,
        port: int,
        context: int,
        work,
        parallel: int | None = None,
        extra_args: list[str] | None = None,
        vram_margin_mib: int = 1536,
    ) -> dict:
        binary = find_llama_server(self.s.llama_server)
        if binary is None:
            return {"status": "failed", "error": "llama-server not found; set EMPEROR_LLAMA_SERVER"}
        if not gguf.is_file():
            return {"status": "failed", "error": f"model not found: {gguf}"}
        if parallel is None:
            parallel = self.s.parallel if name != "eval" else 1

        def start():
            command = llama_server_command(
                binary, gguf, port, context, parallel, vram_margin_mib=vram_margin_mib
            )
            command += extra_args or []
            server = ManagedProcess(command, self.s.run_dir / f"llama-{name}.log", self.own).start()
            log.info("%s: llama-server pid %s on port %d", name, server.pid, port)
            if not wait_for_health(f"http://127.0.0.1:{port}", server, SERVER_LOAD_SECONDS):
                server.stop()
                raise RuntimeError("llama-server did not become ready")
            return server

        return run_gated(
            gate=self.gate,
            quit_flag=self.quit_requested,
            start=start,
            stop=lambda server: server.stop(),
            work=work,
            poll_seconds=self.s.poll_seconds,
            monitor_seconds=5.0,
            name=name,
        )

    # -- phases ----------------------------------------------------------------------------

    def phase_prompts(self) -> dict:
        path = self.s.data_dir / "prompts.jsonl"
        own = [normalise(r["text"]) for r in read_jsonl(path)] if path.exists() else []
        used = list(own)
        for earlier in self.s.previous_data:  # an earlier round's prompts are never repeated
            earlier_file = earlier / "prompts.jsonl"
            if earlier_file.exists():
                used += [normalise(r["text"]) for r in read_jsonl(earlier_file)]
        need = self.s.prompts - len(own)
        if need > 0:
            records, stats = generate_prompts(
                need, self.s.seed + len(own), heldout=HeldoutIndex.bundled(), existing=used
            )
            self.s.data_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            return {"status": "done", "prompts": len(own) + len(records), **stats}
        return {"status": "done", "prompts": len(own)}

    def _generation_config(self) -> GenerationConfig:
        return GenerationConfig(retries=self.s.retries, force_first_tool=self.s.force_first_tool)

    def _teacher_phase(self, name: str, work) -> dict:
        """The teacher's server: one slot per role (generation runs one request at a time)."""
        return self._server_phase(
            name,
            self.s.teacher_gguf,
            self.s.teacher_port,
            self.s.teacher_context,
            work,
            parallel=max(self.s.parallel, ENVIRONMENT_SLOT + 1),
            extra_args=TEACHER_SERVER_ARGS,
            vram_margin_mib=TEACHER_VRAM_MARGIN_MIB,
        )

    def _judge_phase(self, name: str, work) -> dict:
        """The judge's own server, a different model; it never runs beside the teacher."""
        return self._server_phase(
            name, self.s.judge_gguf, self.s.teacher_port, self.s.judge_context, work, parallel=1
        )

    def _generate_work(self, directory: Path, attempt: int = 0):
        def work(server, should_stop):
            url = f"http://127.0.0.1:{self.s.teacher_port}/v1"
            client = ChatClient(url, reasoning_effort=self.s.reasoning_effort, slot=TEACHER_SLOT)
            environment = ChatClient(
                url, reasoning_effort=self.s.environment_reasoning_effort, slot=ENVIRONMENT_SLOT
            )
            try:
                counts = run_generation(
                    directory / "prompts.jsonl",
                    directory,
                    client,
                    environment,
                    config=dataclasses.replace(self._generation_config(), attempt=attempt),
                    should_stop=should_stop,
                    log=log.debug,
                )
            finally:
                client.close()
                environment.close()
            log.info("generate %s: %s", directory.name, counts)
            return {**counts, "stopped": counts.get("stopped", False)}

        return work

    def _judge_work(self, directory: Path):
        def work(server, should_stop):
            client = ChatClient(
                f"http://127.0.0.1:{self.s.teacher_port}/v1",
                model=JUDGE_MODEL,
                reasoning_effort=self.s.judge_reasoning_effort,
                template_kwargs={"enable_thinking": self.s.judge_thinking},
            )
            try:
                counts = run_judging(
                    directory,
                    client,
                    threshold=self.s.threshold,
                    should_stop=should_stop,
                    log=log.debug,
                    model=JUDGE_MODEL,
                )
            finally:
                client.close()
            log.info("judge %s: %s", directory.name, counts)
            return {**counts, "stopped": counts.get("stopped", False)}

        return work

    def _pending(self, source: str, done: str, directory: Path | None = None) -> int:
        directory = directory or self.s.data_dir
        finished = done_ids(directory / done)
        if source == "prompts.jsonl":
            finished |= done_ids(directory / "rejects")
            records = read_jsonl(directory / source)
        else:
            records = iter_records(directory / source)
        return sum(1 for r in records if r["id"] not in finished)

    def phase_generate(self) -> dict:
        if self._pending("prompts.jsonl", "traces") == 0:
            return {"status": "done", "note": "nothing left to generate"}
        return self._teacher_phase("generate", self._generate_work(self.s.data_dir))

    def phase_judge(self) -> dict:
        if self._pending("traces", "judged") == 0:
            return {"status": "done", "note": "nothing left to judge"}
        return self._judge_phase("judge", self._judge_work(self.s.data_dir))

    def phase_retry(self) -> dict:
        """Regenerate and rejudge rejected prompts (new seeds), up to `retries` rounds."""
        if self.s.retries <= 0:
            return {"status": "skipped", "reason": "retries is 0"}
        if self._retry_pending() == 0:
            return {"status": "done", "note": "nothing left to retry"}
        rounds: dict = {}
        for k in range(1, self.s.retries + 1):
            if prepare_round(self.s.data_dir, k) == 0:
                break
            directory = round_dir(self.s.data_dir, k)
            # Generate the whole round on the teacher, then judge it on the judge's server.
            steps = [
                ("generate", "prompts.jsonl", "traces", self._teacher_phase),
                ("judge", "traces", "judged", self._judge_phase),
            ]
            for kind, source, done, phase in steps:
                if self._pending(source, done, directory) == 0:
                    continue
                work = (
                    self._generate_work(directory, attempt=k)
                    if kind == "generate"
                    else self._judge_work(directory)
                )
                result = phase(f"retry-{k}-{kind}", work)
                rounds[f"retry-{k}-{kind}"] = result
                if result.get("status") in ("quit", "failed"):
                    return {**result, "rounds": rounds}
        return {"status": "done", "rounds": rounds}

    def _retry_pending(self) -> int:
        """Work left in the retry rounds: unfinished rounds, or failures not yet retried."""
        for k in range(1, self.s.retries + 1):
            directory = round_dir(self.s.data_dir, k)
            prompts = directory / "prompts.jsonl"
            if not prompts.exists():
                # Round k-1 must be complete before round k can be prepared.
                if prepare_round(self.s.data_dir, k) == 0:
                    return 0
                prompts = directory / "prompts.jsonl"
            finished = done_ids(directory / "rejects") | done_ids(directory / "judged")
            if any(r["id"] not in finished for r in read_jsonl(prompts)):
                return 1
        return 0

    def phase_filter(self) -> dict:
        result = filter_kept(self.s.data_dir)
        report = write_report(self.s.data_dir)
        return {"status": "done", **result, "pass_rate": report["pass_rate"]}

    def phase_train(self) -> dict:
        config = load_config(self.s.root / self.s.train_config)
        output_dir = (self.s.root / config.train.output_dir).resolve()
        pause_file = self.s.run_dir / "PAUSE"
        pauses = 0
        command = [
            sys.executable, "-m", "emperor.train.cli", "fit",
            "--config", str(self.s.root / self.s.train_config),
            "--root", str(self.s.root),
            "--pause-file", str(pause_file),
        ]  # fmt: skip
        failures = 0
        while True:
            if not wait_until_idle(
                self.gate,
                poll_seconds=self.s.poll_seconds,
                should_quit=self.quit_requested,
                log=log.info,
            ):
                return {"status": "quit", "pauses": pauses}
            pause_file.unlink(missing_ok=True)
            trainer = ManagedProcess(command, self.s.run_dir / "train.log", self.own).start()
            log.info(
                "train: pid %s (resumes from the last checkpoint in %s)", trainer.pid, output_dir
            )
            watcher = GateWatcher(self.gate, self.quit_requested)
            while trainer.poll() is None:
                time.sleep(5)
                if watcher():
                    pause_file.touch()
                    log.info("train: pause requested (%s)", watcher.reason)
                    if trainer.wait(self.s.pause_grace_seconds) is None:
                        log.warning("train: no checkpoint within the grace period; stopping pid")
                        trainer.stop()
                    break
            trainer.wait(60)
            trainer.stop()
            code = trainer.poll()
            pause_file.unlink(missing_ok=True)
            if code == 0:
                return {"status": "done", "pauses": pauses}
            if self.quit_requested():
                return {"status": "quit", "pauses": pauses}
            if watcher.reason or code == PAUSED:
                pauses += 1
                continue
            failures += 1
            if failures >= 2:
                return {
                    "status": "failed",
                    "error": f"trainer exited with code {code}",
                    "pauses": pauses,
                }

    def _run_module(self, name: str, args: list[str]) -> dict:
        command = [sys.executable, "-m", *args]
        proc = ManagedProcess(command, self.s.run_dir / f"{name}.log", self.own).start()
        code = proc.wait(24 * 3600)
        proc.stop()
        if code != 0:
            return {"status": "failed", "error": f"{name} exited with code {code}"}
        return {"status": "done"}

    def phase_merge(self) -> dict:
        config = load_config(self.s.root / self.s.train_config)
        result = self._run_module(
            "merge",
            [
                "emperor.train.cli", "merge",
                "--config", str(self.s.root / self.s.train_config),
                "--root", str(self.s.root),
            ],
        )  # fmt: skip
        result["weights"] = str((self.s.root / config.merge.output_dir).resolve())
        return result

    def phase_export(self) -> dict:
        """bf16 GGUF and quantised variants into weights/<name>-gguf/ (CPU, resumable)."""
        config = self.sft
        args = [
            "emperor.release.cli", "export",
            "--config", str(self.s.root / self.s.train_config),
            "--root", str(self.s.root),
            "--llama-cache", str(self.s.root / self.s.llama_cache),
        ]  # fmt: skip
        if self.s.llama_cpp:
            args += ["--llama-cpp", str(self.s.llama_cpp)]
        if self.s.quantize:
            args += ["--quantize", str(self.s.quantize)]
        result = self._run_module("export", args)
        directory = gguf_dir_for(self.s.root, config.release.name, config.release.gguf_dir)
        result["gguf_dir"] = str(directory)
        if result["status"] == "done":
            listing = json.loads((directory / "export.json").read_text(encoding="utf-8"))
            result["files"] = [f["file"] for f in listing["files"]]
        return result

    def eval_gguf(self) -> Path | None:
        if self.s.eval_gguf:
            return self.s.eval_gguf
        release = self.sft.release
        directory = gguf_dir_for(self.s.root, release.name, release.gguf_dir)
        path = directory / gguf_name(release.name, self.s.eval_type)
        return path if path.is_file() else None

    def eval_command(self, report: Path) -> list[str]:
        """The process that scores the served model; subclasses swap in their own harness."""
        return [
            sys.executable, "-m", "emperor.eval.cli",
            "--model", f"llamacpp:{self.sft.release.eval_alias}",
            "--base-url", f"http://127.0.0.1:{self.s.eval_port}/v1",
            "--context-tier", self.s.eval_tier,
            "--output", str(report),
        ]  # fmt: skip

    def phase_eval(self) -> dict:
        gguf = self.eval_gguf()
        if gguf is None:
            return {"status": "skipped", "reason": "no GGUF of the merged model (see export)"}
        report = self.s.run_dir / "eval.json"
        context = CONTEXT_TIERS[self.s.eval_tier]

        def work(server, should_stop):
            run = ManagedProcess(
                self.eval_command(report), self.s.run_dir / "eval.log", self.own
            ).start()
            while run.poll() is None:
                time.sleep(5)
                if should_stop():
                    run.stop()
                    return {"stopped": True}
            return {"stopped": False, "exit_code": run.poll()}

        result = self._server_phase("eval", gguf, self.s.eval_port, context, work)
        if result["status"] == "done" and report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            result["metrics"] = {
                name: {"score": m["score"], "target": m.get("target"), "passed": m.get("passed")}
                for name, m in data.get("metrics", {}).items()
            }
            result["tasks_failed"] = data.get("tasks_failed")
        return result

    def phase_card(self) -> dict:
        """Hugging Face README.md for the bf16 weights and the GGUF directory."""
        try:
            written = write_cards(self.sft, self.s.root, self.s.run_dir)
        except FileNotFoundError as error:
            return {"status": "failed", "error": str(error)}
        return {"status": "done", "cards": [str(p) for p in written]}

    # -- driver ----------------------------------------------------------------------------

    def phases(self) -> list[tuple[str, Callable[[], dict]]]:
        """The phases in order; a pipeline for another model overrides the methods or this."""
        return [
            ("prompts", self.phase_prompts),
            ("generate", self.phase_generate),
            ("judge", self.phase_judge),
            ("retry", self.phase_retry),
            ("filter", self.phase_filter),
            ("train", self.phase_train),
            ("merge", self.phase_merge),
            ("export", self.phase_export),
            ("eval", self.phase_eval),
            ("card", self.phase_card),
        ]

    def run(self) -> dict:
        status = "done"
        for name, function in self.phases():
            if self.s.only and name not in self.s.only:
                continue  # leave the recorded state of other phases untouched
            if name in self.s.skip:
                self._record(name, {"status": "skipped", "reason": "--skip"})
                continue
            if self._phase_done(name) and name in ("train", "merge", "export", "eval"):
                continue
            if (
                name == "train"
                and self.state["phases"].get("filter", {}).get("kept", 0) < self.s.min_samples
            ):
                self._record(
                    name,
                    {
                        "status": "skipped",
                        "reason": f"fewer than {self.s.min_samples} kept samples",
                    },
                )
                status = "insufficient_data"
                break
            result = self._record(name, function())
            if result["status"] in ("quit", "failed"):
                status = result["status"]
                break
        return self.write_summary(status)

    def write_summary(self, status: str) -> dict:
        data_report = self.s.data_dir / "report.json"
        train_result = self.s.root / self.sft.train.output_dir / "train_result.json"
        summary = {
            "status": status,
            "started": self.started,
            "finished": datetime.now(UTC).isoformat(),
            "phases": self.state["phases"],
            "pauses": sum(p.get("pauses", 0) for p in self.state["phases"].values()),
            "data_report": json.loads(data_report.read_text(encoding="utf-8"))
            if data_report.exists()
            else None,
            "train": json.loads(train_result.read_text(encoding="utf-8"))
            if train_result.exists()
            else None,
            "eval": self.state["phases"].get("eval"),
            "teacher": self.teacher_info(),
        }
        (self.s.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary


def install_signal_handlers(pipeline: Pipeline) -> None:
    def handler(signum, frame):
        pipeline.interrupted = True
        log.warning("signal %s: stopping after the current step", signum)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)
