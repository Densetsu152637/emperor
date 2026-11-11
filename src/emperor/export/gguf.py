"""GGUF export through llama.cpp: a bf16 GGUF, then quantised variants (Q8_0, Q6_K, Q4_K_M).

`convert_hf_to_gguf.py` ships with the llama.cpp source, not with the prebuilt binaries, so it is
taken from a shallow clone of the pinned tag in a gitignored cache (`tools/llama.cpp`), or from
`--llama-cpp` / `EMPEROR_LLAMA_CPP` (an existing checkout). The converter imports its own
`gguf-py` from that checkout, so the `gguf` pip package is not needed. `llama-quantize` is the
prebuilt binary found next to `llama-server`.

Every step is resumable: a file that already exists is kept, and outputs are written under a
temporary name and renamed when complete. Subprocess calls go through `run` so tests can fake them.
"""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from emperor.pipeline.processes import find_llama_tool

LLAMA_CPP_TAG = "b12180"  # the build of the llama.cpp runtime Imperium ships
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp"
DEFAULT_CACHE = Path("tools") / "llama.cpp"

Runner = Callable[..., object]


class ExportError(RuntimeError):
    """A GGUF export step could not run."""


def find_converter(llama_cpp: Path | str | None = None) -> Path | None:
    root = llama_cpp or os.environ.get("EMPEROR_LLAMA_CPP")
    if not root:
        return None
    script = Path(root) / "convert_hf_to_gguf.py"
    return script if script.is_file() else None


def ensure_llama_cpp(
    cache: Path,
    *,
    llama_cpp: Path | str | None = None,
    tag: str = LLAMA_CPP_TAG,
    run: Runner = subprocess.run,
) -> Path:
    """The converter script: an explicit checkout, the cache, or a shallow clone of `tag`."""
    explicit = find_converter(llama_cpp)
    if explicit is not None:
        return explicit
    script = cache / "convert_hf_to_gguf.py"
    if script.is_file():
        return script
    if cache.exists() and any(cache.iterdir()):
        raise ExportError(f"{cache} exists but has no convert_hf_to_gguf.py; remove it")
    cache.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["git", "clone", "--depth", "1", "--branch", tag, LLAMA_CPP_URL, str(cache)],
        check=True,
    )
    if not script.is_file():
        raise ExportError(f"cloned llama.cpp {tag} but {script} is missing")
    return script


def gguf_dir_for(root: Path, name: str, configured: str = "") -> Path:
    path = Path(configured) if configured else Path("weights") / f"{name}-gguf"
    return path if path.is_absolute() else root / path


def gguf_name(name: str, kind: str) -> str:
    """`emperor-4b` + `Q4_K_M` -> `emperor-4b-Q4_K_M.gguf` (the type is lower-case for bf16)."""
    return f"{name}-{kind}.gguf"


def _tensor_names(model_dir: Path) -> set[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    names: set[str] = set()
    for shard in model_dir.glob("*.safetensors"):
        with shard.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            names |= set(json.loads(handle.read(size))) - {"__metadata__"}
    return names


def converter_flags(model_dir: Path) -> list[str]:
    """Extra converter options for this checkpoint. A merged Qwen3.5 model is text-only: its
    config still declares the multi-token-prediction head, but the head's tensors are not
    saved, so the converter is told to leave it out."""
    path = model_dir / "config.json"
    if not path.is_file():
        return []
    config = json.loads(path.read_text(encoding="utf-8"))
    mtp = config.get("mtp_num_hidden_layers") or config.get("text_config", {}).get(
        "mtp_num_hidden_layers"
    )
    if mtp and not any(n.startswith(("mtp.", "model.mtp.")) for n in _tensor_names(model_dir)):
        return ["--no-mtp"]
    return []


def convert(
    script: Path,
    model_dir: Path,
    outfile: Path,
    *,
    outtype: str = "bf16",
    run: Runner = subprocess.run,
) -> Path:
    """Convert a Hugging Face model directory to one GGUF file (bf16 by default, no quantising)."""
    if outfile.is_file():
        return outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    partial = outfile.with_suffix(".gguf.tmp")
    partial.unlink(missing_ok=True)
    run(
        [sys.executable, str(script), str(model_dir), "--outfile", str(partial)]
        + ["--outtype", outtype]
        + converter_flags(model_dir),
        check=True,
    )
    if not partial.is_file():
        raise ExportError(f"the converter wrote nothing to {partial}")
    partial.replace(outfile)
    return outfile


def quantize(
    binary: Path, source: Path, outfile: Path, kind: str, *, run: Runner = subprocess.run
) -> Path:
    """`llama-quantize <source> <outfile> <kind>` (never requantises a quantised source)."""
    if outfile.is_file():
        return outfile
    partial = outfile.with_suffix(".gguf.tmp")
    partial.unlink(missing_ok=True)
    run([str(binary), str(source), str(partial), kind], check=True)
    if not partial.is_file():
        raise ExportError(f"llama-quantize wrote nothing to {partial}")
    partial.replace(outfile)
    return outfile


@dataclass
class ExportResult:
    directory: Path
    files: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"gguf_dir": str(self.directory), "files": self.files}


def export_gguf(
    weights: Path,
    directory: Path,
    name: str,
    types: list[str],
    *,
    cache: Path,
    llama_cpp: Path | str | None = None,
    quantize_binary: Path | str | None = None,
    run: Runner = subprocess.run,
    log: Callable[[str], None] = lambda message: None,
) -> ExportResult:
    """bf16 GGUF, then each quantised type, into `directory`; writes `export.json`."""
    if not (weights / "config.json").is_file():
        raise ExportError(f"no merged model at {weights}; run the merge first")
    binary = find_llama_tool("llama-quantize", quantize_binary, "EMPEROR_LLAMA_QUANTIZE")
    if binary is None and any(t.lower() != "bf16" for t in types):
        raise ExportError("llama-quantize not found; set EMPEROR_LLAMA_QUANTIZE or --quantize")
    script = ensure_llama_cpp(cache, llama_cpp=llama_cpp, run=run)
    log(f"converting {weights} with {script}")
    base = convert(script, weights, directory / gguf_name(name, "bf16"), run=run)
    outputs = [("bf16", base)]
    for kind in types:
        if kind.lower() == "bf16":
            continue
        log(f"quantizing {kind}")
        outputs.append(
            (kind, quantize(binary, base, directory / gguf_name(name, kind), kind, run=run))
        )
    result = ExportResult(directory)
    for kind, path in outputs:
        size = path.stat().st_size if path.exists() else 0
        result.files.append({"type": kind, "file": path.name, "bytes": size})
    (directory / "export.json").write_text(
        json.dumps({"llama_cpp_tag": LLAMA_CPP_TAG, **result.to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
