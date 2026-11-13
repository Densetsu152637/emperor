"""`uv run hosted`: Consul's data phases on hosted models (Emperor's runner, Consul's phases).

prompts (training tasks and the held-out set) -> generate (DeepSeek critiques, checked against
each task's seeded defects) -> judge (Kimi) -> filter -> report. Same options, status file,
budget caps and resume as Emperor's `hosted`; see its module and `configs/hosted.toml`.
Consul has no retry rounds: each task already gets up to two draws.
"""

import sys

from emperor.data.records import fingerprint
from emperor.pipeline import hosted as base

from consul import data, prompts

MODULE = "consul.pipeline.hosted"


class ConsulHostedRun(base.HostedRun):
    name = "consul"
    phases = ("prompts", "generate", "judge", "filter", "report")

    def phase_prompts(self) -> dict:
        result = data.write_train_tasks(self.data_dir, self.args.prompts, self.args.seed)
        overlap = data.write_heldout(self.data_dir, self.args.eval_tasks, self.args.eval_seed)
        if not overlap["ok"]:
            return {"stopped": True, "error": "held-out overlap check failed", **result}
        return {**result, "heldout": overlap}

    def phase_generate(self) -> dict:
        teacher = self.client("teacher")
        return data.run_generation(
            self.data_dir,
            teacher,
            should_stop=self.should_stop,
            log=base.log.debug,
            teacher_model=self.names["teacher"],
            workers=self.workers("teacher"),
            limit=self.args.limit,
            provenance={"host": self.host, "licence": self.licence, **self.provenance()},
        )

    def phase_judge(self) -> dict:
        judge = self.client("judge")
        return data.run_judging(
            self.data_dir,
            judge,
            threshold=self.args.threshold,
            should_stop=self.should_stop,
            log=base.log.debug,
            model=self.names["judge"],
            workers=self.workers("judge"),
            settings=self.judge_settings(),
        )

    def provenance(self) -> dict:
        record = super().provenance()
        record["models"].pop("simulator", None)  # critiques make no tool calls
        return record

    def guidance(self) -> str:
        return fingerprint(prompts.teacher_system() + prompts.CONSUL_SYSTEM)

    def rubric(self) -> str:
        return fingerprint(prompts.JUDGE_SYSTEM)

    def phase_filter(self) -> dict:
        return data.filter_kept(self.data_dir)

    def phase_report(self) -> dict:
        report = data.write_report(self.data_dir)
        return {"kept": report.get("kept")}


def build_parser():
    parser = base.build_parser("hosted")
    parser.set_defaults(threshold=data.JUDGE_MEAN, retries=0)
    parser.add_argument("--eval-tasks", type=int, default=60, help="held-out tasks to generate")
    parser.add_argument("--eval-seed", type=int, default=data.HELDOUT_SEED_OFFSET)
    return parser


def main(argv: list[str] | None = None) -> int:
    return base.main(
        argv, run_class=ConsulHostedRun, module=MODULE, prog="hosted", parser=build_parser()
    )


if __name__ == "__main__":
    sys.exit(main())
