"""Optional GGUF export through llama.cpp's `convert_hf_to_gguf.py`.

The converter ships with a llama.cpp source checkout, not with the prebuilt binaries, so it is
located through `--llama-cpp` or `EMPEROR_LLAMA_CPP` (the checkout directory). When it is not
found, `find_converter` returns None and callers skip the step.
"""

import os
import subprocess
import sys
from pathlib import Path


def find_converter(llama_cpp: Path | str | None = None) -> Path | None:
    root = llama_cpp or os.environ.get("EMPEROR_LLAMA_CPP")
    if not root:
        return None
    script = Path(root) / "convert_hf_to_gguf.py"
    return script if script.is_file() else None


def convert(model_dir: Path, outfile: Path, *, outtype: str = "bf16", llama_cpp=None) -> Path:
    """Convert a Hugging Face model directory to one GGUF file (default bf16, no quantising)."""
    script = find_converter(llama_cpp)
    if script is None:
        raise FileNotFoundError("convert_hf_to_gguf.py not found; set EMPEROR_LLAMA_CPP")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    subprocess.run(command, check=True)
    return outfile
