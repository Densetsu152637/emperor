"""Console-script entry points that let a project built on Emperor take over `eval`, `data`
and `pipeline`.

A project that depends on this package (Consul) installs its own `pipeline`, `eval` and `data`
scripts, and two packages writing the same script name race at install time. So these entry
points check the current directory's `pyproject.toml` first: a project that declares

    [tool.emperor.entrypoints]
    pipeline = "consul.pipeline.cli:main"

gets its own function called whichever script file won the race. Without the table (Emperor
itself), the real command runs.
"""

import importlib
import tomllib
from collections.abc import Callable
from pathlib import Path


def project_entrypoint(name: str, directory: Path | None = None) -> Callable | None:
    path = (directory or Path.cwd()) / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    target = data.get("tool", {}).get("emperor", {}).get("entrypoints", {}).get(name)
    if not isinstance(target, str) or ":" not in target:
        return None
    module, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module), attribute)


def _entry(name: str, module: str) -> Callable:
    def run(argv: list[str] | None = None) -> int:
        override = project_entrypoint(name)
        function = override or importlib.import_module(module).main
        return function(argv)

    run.__name__ = f"{name}_main"
    return run


eval_main = _entry("eval", "emperor.eval.cli")
data_main = _entry("data", "emperor.data.cli")
pipeline_main = _entry("pipeline", "emperor.pipeline.cli")
