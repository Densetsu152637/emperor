"""Processes the pipeline starts, and only those: tracked by PID and stopped by PID."""

import os
import subprocess
import time
from pathlib import Path


def find_llama_server(explicit: str | Path | None = None) -> Path | None:
    """The llama-server binary: option, EMPEROR_LLAMA_SERVER, PATH, then Imperium's runtime."""
    import shutil

    for candidate in (explicit, os.environ.get("EMPEROR_LLAMA_SERVER")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        runtimes = Path(local) / "Imperium" / "data" / "runtimes" / "llama.cpp"
        found = sorted(runtimes.glob("*/*/bin/llama-server*"), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    return None


class OwnProcesses:
    """Every process the pipeline launched; the idle gate subtracts these from GPU load."""

    def __init__(self) -> None:
        self._procs: dict[int, subprocess.Popen] = {}

    def add(self, proc: subprocess.Popen) -> None:
        self._procs[proc.pid] = proc

    def pids(self) -> tuple[int, ...]:
        alive = {pid: p for pid, p in self._procs.items() if p.poll() is None}
        self._procs = alive
        return tuple(alive)

    def terminate_all(self, grace: float = 20.0) -> None:
        """Stop every still-running child we started, by PID, and nothing else."""
        for proc in list(self._procs.values()):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(grace)
                except subprocess.TimeoutExpired:
                    proc.kill()


class ManagedProcess:
    """A child process with a log file; `stop` terminates it by PID (and nothing else)."""

    def __init__(self, command: list[str], log_path: Path, own: OwnProcesses, env=None):
        self.command = command
        self.log_path = log_path
        self.own = own
        self.env = env
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> "ManagedProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.env,
            creationflags=flags,
        )
        self.own.add(self.proc)
        return self

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc else None

    def wait(self, timeout: float) -> int | None:
        try:
            return self.proc.wait(timeout) if self.proc else None
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace: float = 20.0) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(grace)
        if self._log:
            self._log.close()
            self._log = None


def llama_server_command(
    binary: Path, model: Path, port: int, context: int, parallel: int = 1
) -> list[str]:
    """llama-server on loopback with GPU offload and the model's own chat template (tool calls)."""
    return [
        str(binary),
        "-m", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(context * parallel),
        "-np", str(parallel),
        "-ngl", "99",
        "--jinja",
    ]  # fmt: skip


def wait_for_health(
    url_root: str, process: ManagedProcess, seconds: float = 600.0, interval: float = 2.0
) -> bool:
    """Poll `<url_root>/health` until it answers 200; False if the process dies or time runs out."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url_root}/health", timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval)
    return False
