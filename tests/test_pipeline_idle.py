from emperor.pipeline.idle import (
    IdleGate,
    parse_pmon_other,
    parse_total_utilisation,
    wait_until_idle,
)
from emperor.pipeline.run import GateWatcher, run_gated


def gate(idle=None, gpu=None, **kwargs):
    state = {"idle": idle, "gpu": gpu}
    g = IdleGate(
        input_probe=lambda: state["idle"],
        gpu_probe=lambda pids: state["gpu"],
        **kwargs,
    )
    g.state = state
    return g


def test_start_needs_fifteen_idle_minutes_and_a_quiet_gpu():
    g = gate(idle=14 * 60, gpu=0.0)
    assert not g.can_start().ok
    g.state["idle"] = 15 * 60
    assert g.can_start().ok
    g.state["gpu"] = 15.0
    status = g.can_start()
    assert not status.ok and "GPU" in status.reason
    g.state["gpu"] = 14.9
    assert g.can_start().ok


def test_unknown_probes_block_a_start():
    assert not gate(idle=None, gpu=0.0).can_start().ok
    assert not gate(idle=3600, gpu=None).can_start().ok


def test_asleep_bypasses_the_gate():
    g = gate(idle=0, gpu=99.0, asleep=True)
    assert g.can_start().ok and g.can_continue().ok


def test_running_phase_pauses_on_input_or_foreign_gpu_load():
    g = gate(idle=3600, gpu=0.0)
    assert g.can_continue().ok
    g.state["idle"] = 5
    assert not g.can_continue().ok
    g.state["idle"] = 3600
    g.state["gpu"] = 40.0
    assert not g.can_continue().ok
    g.state["gpu"] = None  # per-process load unavailable: input activity alone decides
    assert g.can_continue().ok


def test_gate_passes_the_pipelines_own_pids_to_the_gpu_probe():
    seen = []
    g = IdleGate(
        input_probe=lambda: 9999,
        gpu_probe=lambda pids: seen.append(pids) or 0.0,
        own_pids=lambda: (123, 456),
    )
    g.can_continue()
    assert seen == [(123, 456)]


def test_nvidia_smi_parsers():
    assert parse_total_utilisation("12\n 40 \n") == 40.0
    assert parse_total_utilisation("") is None
    pmon = (
        "# gpu   pid  type    sm   mem   enc   dec   command\n"
        "# Idx     #   C/G     %     %     %     %   name\n"
        "    0   100     C    80    10     0     0   ours.exe\n"
        "    0   200     G    12     1     0     0   game.exe\n"
    )
    assert parse_pmon_other(pmon, {100}) == 12.0
    assert parse_pmon_other(pmon, {100, 200}) == 0.0
    assert parse_pmon_other("    0   100     C     -     -     -     -   x\n", set()) is None


def test_wait_until_idle_polls_then_returns():
    g = gate(idle=0, gpu=0.0)
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            g.state["idle"] = 20 * 60

    assert wait_until_idle(g, poll_seconds=7, sleep=fake_sleep)
    assert sleeps == [7, 7, 7]


def test_wait_until_idle_can_be_cancelled():
    g = gate(idle=0, gpu=0.0)
    calls = iter([False, False, True])
    assert not wait_until_idle(g, should_quit=lambda: next(calls), sleep=lambda s: None)


def test_watcher_polls_at_an_interval_and_latches():
    g = gate(idle=3600, gpu=0.0)
    now = [0.0]
    watcher = GateWatcher(g, lambda: False, interval=10, clock=lambda: now[0])
    assert not watcher()
    g.state["idle"] = 1
    assert not watcher()  # inside the polling interval, not rechecked
    now[0] = 11
    assert watcher() and "active" in watcher.reason
    g.state["idle"] = 3600
    assert watcher()  # stays stopped until the phase restarts


def test_run_gated_pauses_when_the_user_returns_then_resumes():
    g = gate(idle=3600, gpu=0.0)
    log = []
    remaining = {"steps": 6}

    def start():
        log.append("start")
        return object()

    def stop(resource):
        log.append("stop")

    def work(resource, should_stop):
        while remaining["steps"] > 0:
            if should_stop():
                return {"stopped": True}
            remaining["steps"] -= 1
            if remaining["steps"] == 4 and log.count("start") == 1:
                g.state["idle"] = 2  # the user comes back mid-run
        return {"stopped": False, "done": 6}

    def fake_sleep(seconds):
        g.state["idle"] = 3600  # ... and leaves again

    result = run_gated(
        gate=g,
        quit_flag=lambda: False,
        start=start,
        stop=stop,
        work=work,
        poll_seconds=1,
        sleep=fake_sleep,
        watch_interval=0,
    )
    assert result == {"status": "done", "pauses": 1, "done": 6}
    assert log == ["start", "stop", "start", "stop"]
    assert remaining["steps"] == 0


def test_run_gated_stops_after_repeated_failures():
    g = gate(idle=3600, gpu=0.0)

    def work(resource, should_stop):
        raise RuntimeError("server crashed")

    result = run_gated(
        gate=g,
        quit_flag=lambda: False,
        start=lambda: object(),
        stop=lambda r: None,
        work=work,
        sleep=lambda s: None,
        max_failures=2,
    )
    assert result["status"] == "failed" and "server crashed" in result["error"]


def test_run_gated_forgives_failures_after_progress():
    g = gate(idle=3600, gpu=0.0)
    runs = {"n": 0}

    def work(resource, should_stop):
        # Every run gets work done and then loses its server; the sixth finishes.
        runs["n"] += 1
        if runs["n"] < 6:
            return {"stopped": True, "error": "server died", "accepted": 3}
        return {"stopped": False, "accepted": 1}

    result = run_gated(
        gate=g,
        quit_flag=lambda: False,
        start=lambda: object(),
        stop=lambda r: None,
        work=work,
        sleep=lambda s: None,
        max_failures=2,
    )
    assert result["status"] == "done" and runs["n"] == 6


def test_run_gated_quits_without_starting_when_asked():
    g = gate(idle=0, gpu=0.0)
    started = []
    result = run_gated(
        gate=g,
        quit_flag=lambda: True,
        start=lambda: started.append(1),
        stop=lambda r: None,
        work=lambda r, s: {},
        sleep=lambda s: None,
    )
    assert result["status"] == "quit" and not started


def test_monitor_stops_the_resource_mid_request_when_the_user_returns():
    import threading

    g = gate(idle=3600, gpu=0.0)
    killed = threading.Event()
    attempts = []

    def work(resource, should_stop):
        attempts.append(1)
        if len(attempts) == 1:
            g.state["idle"] = 1  # the user returns while a long request is in flight
            assert killed.wait(5), "the monitor did not stop the server"
            return {"stopped": True}
        return {"stopped": False}

    def fake_sleep(seconds):
        g.state["idle"] = 3600

    result = run_gated(
        gate=g,
        quit_flag=lambda: False,
        start=lambda: object(),
        stop=lambda r: killed.set(),
        work=work,
        sleep=fake_sleep,
        watch_interval=0,
        monitor_seconds=0.01,
    )
    assert result["status"] == "done" and result["pauses"] == 1


def test_teacher_and_judge_run_on_their_own_servers(tmp_path, monkeypatch):
    from pathlib import Path

    from emperor.pipeline.processes import OwnProcesses
    from emperor.pipeline.run import (
        DEFAULT_JUDGE_GGUF,
        DEFAULT_TEACHER_GGUF,
        TEACHER_SERVER_ARGS,
        Pipeline,
        Settings,
    )

    data = tmp_path / "data"
    data.mkdir()
    (data / "prompts.jsonl").write_text('{"id": "p-1"}\n', encoding="utf-8")
    from emperor.data.shards import ShardWriter

    ShardWriter(data / "traces", "traces").write({"id": "p-0", "messages": []})
    pipeline = Pipeline(
        Settings(root=tmp_path, run_dir=tmp_path, data_dir=data), None, OwnProcesses()
    )
    calls = []

    def fake_server_phase(
        name, gguf, port, context, work, parallel=None, extra_args=None, vram_margin_mib=1536
    ):
        calls.append((name, str(gguf), parallel, extra_args))
        return {"status": "done"}

    monkeypatch.setattr(pipeline, "_server_phase", fake_server_phase)
    pipeline.phase_generate()
    pipeline.phase_judge()
    assert calls[0] == ("generate", str(Path(DEFAULT_TEACHER_GGUF)), 2, TEACHER_SERVER_ARGS)
    assert calls[1] == ("judge", str(Path(DEFAULT_JUDGE_GGUF)), 1, None)
