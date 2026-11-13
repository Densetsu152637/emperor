"""The idle gate: GPU work runs only while the machine is idle (or when told the user is asleep).

Starting needs no keyboard or mouse input for `min_idle_seconds` (default 15 minutes) and GPU
utilisation below `max_gpu_util` from processes other than the pipeline's own. Continuing is
looser: the pipeline pauses as soon as there has been input within `active_seconds`, or another
process loads the GPU, then waits for the full start condition again.

Probes are injected so the logic is testable without a GPU or a desktop session.
"""

import ctypes
import subprocess
import sys
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

DEFAULT_MIN_IDLE_SECONDS = 15 * 60
DEFAULT_ACTIVE_SECONDS = 60
DEFAULT_MAX_GPU_UTIL = 15.0


def last_input_idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input (Win32 GetLastInputInfo); None elsewhere."""
    if sys.platform != "win32":
        return None

    class LastInputInfo(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LastInputInfo()
    info.cbSize = ctypes.sizeof(LastInputInfo)
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GetTickCount.restype = ctypes.c_uint
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def _nvidia_smi(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def parse_total_utilisation(output: str) -> float | None:
    """Highest GPU utilisation (%) in `nvidia-smi --query-gpu=utilization.gpu` output."""
    values = []
    for line in output.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def parse_pmon_other(output: str, own_pids: Collection[int]) -> float | None:
    """Summed SM % of processes not in `own_pids` from `nvidia-smi pmon -s u` output.

    None when the driver reports no usable per-process figures (common under WDDM).
    """
    total, seen = 0.0, False
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        sm = fields[3]
        if sm == "-":
            continue
        try:
            value = float(sm)
        except ValueError:
            continue
        seen = True
        if pid not in own_pids:
            total += value
    return total if seen else None


def other_gpu_utilisation(own_pids: Collection[int] = ()) -> float | None:
    """GPU utilisation (%) caused by processes other than `own_pids`; None if not measurable.

    With no pipeline process on the GPU the whole-card figure is exact. While one is running the
    figure comes from `nvidia-smi pmon`; where the driver does not provide it, None is returned
    and the caller falls back to the input-activity check alone.
    """
    if not own_pids:
        output = _nvidia_smi("--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
        return parse_total_utilisation(output) if output else None
    output = _nvidia_smi("pmon", "-c", "1", "-s", "u")
    return parse_pmon_other(output, own_pids) if output else None


@dataclass(frozen=True)
class GateStatus:
    ok: bool
    reason: str
    idle_seconds: float | None = None
    gpu_other: float | None = None


class IdleGate:
    def __init__(
        self,
        *,
        asleep: bool = False,
        min_idle_seconds: float = DEFAULT_MIN_IDLE_SECONDS,
        active_seconds: float = DEFAULT_ACTIVE_SECONDS,
        max_gpu_util: float = DEFAULT_MAX_GPU_UTIL,
        input_probe: Callable[[], float | None] = last_input_idle_seconds,
        gpu_probe: Callable[[Collection[int]], float | None] = other_gpu_utilisation,
        own_pids: Callable[[], Collection[int]] = lambda: (),
    ):
        self.asleep = asleep
        self.min_idle_seconds = min_idle_seconds
        self.active_seconds = active_seconds
        self.max_gpu_util = max_gpu_util
        self.input_probe = input_probe
        self.gpu_probe = gpu_probe
        self.own_pids = own_pids

    def can_start(self) -> GateStatus:
        """May a GPU phase begin now?"""
        if self.asleep:
            return GateStatus(True, "asleep flag set")
        idle = self.input_probe()
        if idle is None:
            return GateStatus(False, "cannot read input idle time")
        gpu = self.gpu_probe(tuple(self.own_pids()))
        if idle < self.min_idle_seconds:
            return GateStatus(
                False, f"input {idle:.0f}s ago, need {self.min_idle_seconds:.0f}s", idle, gpu
            )
        if gpu is None:
            return GateStatus(False, "cannot read GPU utilisation", idle, gpu)
        if gpu >= self.max_gpu_util:
            return GateStatus(False, f"other processes use the GPU ({gpu:.0f}%)", idle, gpu)
        return GateStatus(True, "idle", idle, gpu)

    def can_continue(self) -> GateStatus:
        """May a running GPU phase carry on?"""
        if self.asleep:
            return GateStatus(True, "asleep flag set")
        idle = self.input_probe()
        if idle is None:
            return GateStatus(False, "cannot read input idle time")
        if idle < self.active_seconds:
            return GateStatus(False, f"user active ({idle:.0f}s since input)", idle)
        gpu = self.gpu_probe(tuple(self.own_pids()))
        if gpu is not None and gpu >= self.max_gpu_util:
            return GateStatus(False, f"other processes use the GPU ({gpu:.0f}%)", idle, gpu)
        return GateStatus(True, "idle", idle, gpu)


def wait_until_idle(
    gate: IdleGate,
    *,
    poll_seconds: float = 30.0,
    should_quit: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _: None,
    report_every: int = 20,
) -> bool:
    """Block until the gate allows a start; False if `should_quit` became true first."""
    polls = 0
    while not should_quit():
        status = gate.can_start()
        if status.ok:
            return True
        if polls % report_every == 0:
            log(f"waiting for idle: {status.reason}")
        polls += 1
        sleep(poll_seconds)
    return False
