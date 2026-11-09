"""imperiumd backend: run tasks through a real Imperium daemon over its JSON-RPC protocol.

TODO(stage 2b follow-up): not implemented. What exists and what is missing:

- **Discovery and transport (documented, Imperium docs/02 "Protocol").** The daemon writes
  `<data>/daemon.json` with its URL (`ws://127.0.0.1:<port>/rpc`), a per-launch token, PID and
  protocol version. Clients open a WebSocket with `Authorization: Bearer <token>`, call
  `system.handshake` first (protocol 0.2), then `session.create` (the caller is subscribed),
  `session.send` with the prompt, and read `session.event` notifications until the session is
  idle. `discover()` below reads the discovery file; the WebSocket client is not written yet
  (this package has no dependencies, and the standard library has no WebSocket client).
- **Recorded subagent replies (missing in Imperium).** Deterministic replay through the daemon
  needs a way to answer `spawn_agent` from the task's recordings instead of a real model, for
  example a fake model endpoint configured as the only subagent model. Until then the `replay`
  backend in `runner.py` stands in for the daemon.
- **e2e (`--suite e2e`).** Real hosted subagents spend money; task cost must come from the
  session ledger and be checked against the API ceiling after every event (`CostMeter`).
"""

import json
from pathlib import Path

from emperor.eval.tasks import Task


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run (not implemented, or no daemon)."""


def discover(data_dir: Path) -> dict:
    """Read the daemon's discovery file: {"url", "token", "pid", "protocol", ...}."""
    path = Path(data_dir) / "daemon.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BackendUnavailable(f"no running imperiumd ({path} not found)") from error


class ImperiumdBackend:
    def __init__(self, data_dir: Path | None, model: str, context_tokens: int):
        self.data_dir = data_dir
        self.model = model
        self.context_tokens = context_tokens

    def run_task(self, task: Task):
        raise BackendUnavailable(
            "the imperiumd backend is not implemented yet (see emperor/eval/imperiumd.py); "
            "use --backend replay, or --dry-run"
        )
