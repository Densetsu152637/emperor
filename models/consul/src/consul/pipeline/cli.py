"""`uv run pipeline`: the whole Consul build, gated on idleness, on Emperor's orchestrator.

Same command line, idle gate, STOP file, partial-GPU fit, resume and phases as Emperor's:
prompts -> generate -> judge -> filter -> train -> merge -> export -> eval -> card.
Consul replaces what is specific to a critic: the tasks (seeded defects) and a held-out set in
`prompts`, the teacher critiques in `generate`, the ground-truth and judge checks in `judge`,
the final set in `filter`, and the held-out harness in `eval`. Training, merge, export and
cards are Emperor's code, driven by this repository's `configs/sft.toml`.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from emperor.data.shards import done_ids, iter_records
from emperor.data.teacher import ChatClient
from emperor.pipeline import cli as base
from emperor.pipeline.run import TEACHER_SLOT, Pipeline

from consul import data

MODULE = "consul.pipeline.cli"
log = logging.getLogger("consul")


class ConsulPipeline(Pipeline):
    def __init__(
        self,
        settings,
        gate,
        own,
        eval_tasks: int = 60,
        eval_seed: int = data.HELDOUT_SEED_OFFSET,
        generate_effort: str = data.GENERATE_EFFORT,
        judge_thinking: bool = data.JUDGE_THINKING,
    ):
        super().__init__(settings, gate, own)
        self.eval_tasks = eval_tasks
        self.eval_seed = eval_seed
        self.generate_effort = generate_effort
        self.judge_thinking = judge_thinking

    def thinks(self) -> bool:
        return self.judge_thinking and self.s.judge_thinking

    def phase_prompts(self) -> dict:
        """Training tasks, then the held-out set checked against them."""
        result = data.write_train_tasks(self.s.data_dir, self.s.prompts, self.s.seed)
        overlap = data.write_heldout(self.s.data_dir, self.eval_tasks, self.eval_seed)
        status = "done" if overlap["ok"] else "failed"
        return {"status": status, **result, "heldout": overlap}

    def _generate_work(self, directory: Path, attempt: int = 0):
        """Critiques on the teacher's server (gpt-oss-120b). Generation's reasoning becomes the
        student's thinking, so it runs at `--generate-reasoning-effort`."""

        def work(server, should_stop):
            client = ChatClient(
                f"http://127.0.0.1:{self.s.teacher_port}/v1",
                reasoning_effort=self.generate_effort,
                slot=TEACHER_SLOT,
            )
            try:
                counts = data.run_generation(
                    directory,
                    client,
                    should_stop=should_stop,
                    log=log.debug,
                    teacher_model=data.model_name(self.s.teacher_gguf),
                )
            finally:
                client.close()
            return {**counts, "stopped": counts.get("stopped", False)}

        return work

    def _judge_work(self, directory: Path):
        """Ground-truth and judge checks on the judge's own server (Qwen3.8-27B). It thinks only
        with `--judge-thinking`, and never with Emperor's `--no-judge-thinking`."""

        def work(server, should_stop):
            judge = data.model_name(self.s.judge_gguf)  # the weights this phase serves
            client = ChatClient(
                f"http://127.0.0.1:{self.s.teacher_port}/v1",
                model=judge,
                reasoning_effort=self.s.judge_reasoning_effort,
                template_kwargs={"enable_thinking": self.thinks()},
            )
            try:
                counts = data.run_judging(
                    directory,
                    client,
                    threshold=self.s.threshold,
                    should_stop=should_stop,
                    log=log.debug,
                    model=judge,
                )
            finally:
                client.close()
            return {**counts, "stopped": counts.get("stopped", False)}

        return work

    def phase_retry(self) -> dict:
        """Emperor's retry rounds replay agent traces; Consul's critiques are not retried."""
        return {"status": "skipped", "reason": "Consul has no retry rounds"}

    def _pending(self, source: str, done: str, directory: Path | None = None) -> int:
        directory = directory or self.s.data_dir
        finished = done_ids(directory / done)
        if source == "prompts.jsonl":
            finished |= done_ids(directory / "rejects")
            tasks = data.load_tasks(directory / source)
            return sum(1 for t in tasks if t.id not in finished)
        return sum(1 for r in iter_records(directory / source) if r["id"] not in finished)

    def phase_filter(self) -> dict:
        result = data.filter_kept(self.s.data_dir)
        report = data.write_report(self.s.data_dir)
        return {"status": "done", **result, "pass_rate": report["pass_rate"]}

    def eval_command(self, report: Path) -> list[str]:
        return [
            sys.executable, "-m", "consul.eval.cli",
            "--model", f"llamacpp:{self.sft.release.eval_alias}",
            "--base-url", f"http://127.0.0.1:{self.s.eval_port}/v1",
            "--tasks", str(self.s.data_dir / "eval" / "heldout.jsonl"),
            "--train-data", str(self.s.data_dir / "prompts.jsonl"),
            "--context-tier", self.s.eval_tier,
            "--output", str(report),
        ]  # fmt: skip


def split_extra(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Consul's own options (held-out size and seed, the teacher's reasoning effort when it
    writes critiques), and the line without them."""
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--eval-tasks", type=int, default=60, help="held-out tasks to generate")
    extra.add_argument("--eval-seed", type=int, default=data.HELDOUT_SEED_OFFSET)
    extra.add_argument(
        "--generate-reasoning-effort",
        default=data.GENERATE_EFFORT,
        choices=["low", "medium", "high"],
        help="teacher reasoning effort for critiques (its reasoning is what the student learns)",
    )
    extra.add_argument(
        "--judge-thinking",
        action="store_true",
        default=data.JUDGE_THINKING,
        help="let the judge think before it answers (off: same verdicts, 13x faster)",
    )
    known, rest = extra.parse_known_args(argv)
    return known, rest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known, rest = split_extra(argv)
    parser = base.build_parser("pipeline", __doc__.strip().splitlines()[0])
    args, _ = parser.parse_known_args(rest)
    if args.detach:  # relaunch with Consul's own options kept, as this module
        root = args.root.resolve()
        run_dir = (
            args.resume.resolve()
            if args.resume
            else root / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        return base.detach(argv, run_dir, MODULE)

    def factory(settings, gate, own):
        return ConsulPipeline(
            settings,
            gate,
            own,
            known.eval_tasks,
            known.eval_seed,
            known.generate_reasoning_effort,
            known.judge_thinking,
        )

    return base.main(
        rest,
        pipeline_class=factory,
        module=MODULE,
        prog="pipeline",
        description=__doc__.strip().splitlines()[0],
    )


if __name__ == "__main__":
    sys.exit(main())
