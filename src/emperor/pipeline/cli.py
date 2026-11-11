"""`uv run --group data --group train pipeline`: the whole Emperor build, gated on idleness.

prompts -> generate -> judge -> filter -> train -> merge -> export -> eval -> card
(`runs/<timestamp>/summary.json`)

GPU phases start only when there has been no keyboard or mouse input for 15 minutes and no other
process is using the GPU, unless `--asleep` is passed. They pause when the user returns.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from emperor.eval.cli import CONTEXT_TIERS
from emperor.export.gguf import DEFAULT_CACHE
from emperor.pipeline.idle import (
    DEFAULT_ACTIVE_SECONDS,
    DEFAULT_MAX_GPU_UTIL,
    DEFAULT_MIN_IDLE_SECONDS,
    IdleGate,
)
from emperor.pipeline.processes import OwnProcesses
from emperor.pipeline.run import (
    DEFAULT_TEACHER_GGUF,
    PHASES,
    Pipeline,
    Settings,
    install_signal_handlers,
)

DETACHED = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW


def build_parser(prog: str = "pipeline", description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description=description or __doc__.strip().splitlines()[0]
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--resume", type=Path, help="continue an existing runs/<timestamp> directory"
    )
    parser.add_argument("--detach", action="store_true", help="run in the background and return")
    parser.add_argument("--skip", action="append", choices=PHASES, default=[], help="skip a phase")
    parser.add_argument(
        "--only",
        action="append",
        choices=PHASES,
        default=[],
        help="run just this phase (repeatable); other phases keep their recorded state",
    )

    idle = parser.add_argument_group("idle gate")
    idle.add_argument("--asleep", action="store_true", help="the user is asleep: ignore the gate")
    idle.add_argument("--min-idle-minutes", type=float, default=DEFAULT_MIN_IDLE_SECONDS / 60)
    idle.add_argument("--active-seconds", type=float, default=DEFAULT_ACTIVE_SECONDS)
    idle.add_argument("--max-gpu-util", type=float, default=DEFAULT_MAX_GPU_UTIL)
    idle.add_argument("--poll-seconds", type=float, default=30.0)

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where this round's prompts, traces and judgements live (relative to --root)",
    )
    data.add_argument(
        "--previous-data",
        type=Path,
        action="append",
        default=[],
        help="an earlier round's data dir whose prompts must not be repeated (repeatable)",
    )
    data.add_argument(
        "--retries",
        type=int,
        default=2,
        help="extra tries per prompt (new seed) after a failed schema check or judgement",
    )
    data.add_argument(
        "--no-force-tool", action="store_true", help="no tool_choice=required on the first turn"
    )
    data.add_argument("--prompts", type=int, default=2000, help="user requests to generate")
    data.add_argument("--seed", type=int, default=1)
    data.add_argument("--threshold", type=float, default=4.0, help="minimum mean judge score")
    data.add_argument("--min-samples", type=int, default=100, help="kept traces needed to train")
    data.add_argument("--teacher-gguf", type=Path, default=Path(DEFAULT_TEACHER_GGUF))
    data.add_argument("--teacher-port", type=int, default=8081)
    data.add_argument("--teacher-context", type=int, default=32768, help="tokens per slot")
    data.add_argument("--parallel", type=int, default=1, help="llama-server slots")
    data.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    data.add_argument(
        "--llama-server", type=Path, help="default: EMPEROR_LLAMA_SERVER, PATH, Imperium"
    )

    train = parser.add_argument_group("train, export, eval")
    train.add_argument("--train-config", type=Path, default=Path("configs/sft.toml"))
    train.add_argument("--llama-cpp", type=Path, help="llama.cpp checkout (default: clone b12180)")
    train.add_argument("--llama-cache", type=Path, default=DEFAULT_CACHE, help="clone location")
    train.add_argument("--quantize", type=Path, help="llama-quantize binary (default: Imperium's)")
    train.add_argument("--eval-gguf", type=Path, help="GGUF to evaluate instead of the export")
    train.add_argument("--eval-type", default="bf16", help="exported GGUF type to evaluate")
    train.add_argument("--eval-port", type=int, default=8082)
    train.add_argument("--eval-tier", choices=list(CONTEXT_TIERS), default="64k")
    return parser


def settings_from(args, run_dir: Path) -> Settings:
    root = args.root.resolve()
    return Settings(
        root=root,
        run_dir=run_dir,
        data_dir=root / args.data_dir,
        previous_data=[root / p for p in args.previous_data],
        retries=args.retries,
        force_first_tool=not args.no_force_tool,
        prompts=args.prompts,
        seed=args.seed,
        threshold=args.threshold,
        min_samples=args.min_samples,
        teacher_gguf=args.teacher_gguf,
        teacher_port=args.teacher_port,
        teacher_context=args.teacher_context,
        parallel=args.parallel,
        reasoning_effort=args.reasoning_effort,
        llama_server=args.llama_server,
        llama_cpp=args.llama_cpp,
        llama_cache=args.llama_cache,
        quantize=args.quantize,
        train_config=args.train_config,
        eval_gguf=args.eval_gguf,
        eval_type=args.eval_type,
        eval_port=args.eval_port,
        eval_tier=args.eval_tier,
        poll_seconds=args.poll_seconds,
        skip=set(args.skip),
        only=set(args.only),
    )


def detach(argv: list[str], run_dir: Path, module: str = "emperor.pipeline.cli") -> int:
    """Relaunch this command without a console, outliving the terminal that started it."""
    child = [a for a in argv if a != "--detach"]
    if "--resume" not in child:
        child += ["--resume", str(run_dir)]
    run_dir.mkdir(parents=True, exist_ok=True)
    out = (run_dir / "stdout.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *child],
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    (run_dir / "pipeline.pid").write_text(str(proc.pid), encoding="utf-8")
    print(
        json.dumps({"run_dir": str(run_dir), "pid": proc.pid, "log": str(run_dir / "pipeline.log")})
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    pipeline_class: type[Pipeline] = Pipeline,
    module: str = "emperor.pipeline.cli",
    prog: str = "pipeline",
    description: str | None = None,
) -> int:
    """Run the pipeline. Another model's package passes its own `pipeline_class` (a subclass
    overriding the data and eval phases) and `module`, so `--detach` relaunches that entry."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser(prog, description).parse_args(argv)
    root = args.root.resolve()
    run_dir = (
        args.resume.resolve()
        if args.resume
        else root / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if args.detach:
        return detach(argv, run_dir, module)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    own = OwnProcesses()
    gate = IdleGate(
        asleep=args.asleep,
        min_idle_seconds=args.min_idle_minutes * 60,
        active_seconds=args.active_seconds,
        max_gpu_util=args.max_gpu_util,
        own_pids=own.pids,
    )
    pipeline = pipeline_class(settings_from(args, run_dir), gate, own)
    install_signal_handlers(pipeline)
    logging.getLogger("emperor.pipeline").info(
        "run %s (asleep=%s); create %s to stop cleanly", run_dir, args.asleep, run_dir / "STOP"
    )
    try:
        summary = pipeline.run()
    finally:
        own.terminate_all()
    print(json.dumps({"status": summary["status"], "summary": str(run_dir / "summary.json")}))
    return 0 if summary["status"] in ("done", "insufficient_data") else 1


if __name__ == "__main__":
    sys.exit(main())
