"""Hugging Face model cards (`README.md`), rendered from the config, the latest eval report and
the dataset report. Pure text: no torch, no network, no models.

The same renderer serves any model that uses this package (`[release]` in the config names it),
and any eval report with the shared shape (`metrics` -> goal, n, score, target, passed).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from emperor.export.gguf import gguf_dir_for
from emperor.train.config import SftConfig

BASE_MODEL = "Qwen/Qwen3.5-4B"
# Typical bits per weight of llama.cpp types, used only when the file is not there to measure.
BITS_PER_WEIGHT = {
    "bf16": 16.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.7,
    "Q4_K_M": 4.9,
}
DEFAULT_PARAMS = 4.02e9
# Qwen3.5-4B: layers with a KV cache (8 full-attention layers; the 24 Gated DeltaNet layers keep
# a fixed-size state), KV heads, head size. Overridden by the model's own config.json when present.
DEFAULT_ARCH = {"num_hidden_layers": 8, "num_key_value_heads": 4, "head_dim": 256}
GIB = 2**30


def yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def front_matter(fields: dict) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines += [f"- {yaml_scalar(item)}" for item in value]
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def kv_cache_gib(tokens: int, arch: dict | None = None) -> float:
    """f16 KV cache: 2 (K and V) x KV-cache layers x KV heads x head size x 2 bytes, per token."""
    a = {**DEFAULT_ARCH, **(arch or {})}
    per_token = 2 * a["num_hidden_layers"] * a["num_key_value_heads"] * a["head_dim"] * 2
    return tokens * per_token / GIB


def read_arch(weights: Path) -> dict | None:
    path = weights / "config.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data = {**data, **data.get("text_config", {})}  # multimodal configs nest the language model
    keys = ("num_hidden_layers", "num_key_value_heads", "head_dim")
    arch = {k: data[k] for k in keys if isinstance(data.get(k), int)}
    layer_types = data.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        # Hybrid models: only the full-attention layers grow a KV cache with the context.
        arch["num_hidden_layers"] = sum(1 for t in layer_types if t == "full_attention")
    return arch


def weights_gib(kind: str, file_bytes: int | None = None, params: float = DEFAULT_PARAMS) -> float:
    if file_bytes:
        return file_bytes / GIB
    return params * BITS_PER_WEIGHT.get(kind, 16.0) / 8 / GIB


def eval_table(report: dict | None) -> str:
    if not report:
        return "_No evaluation report was available when this card was written._"
    rows = [
        "| Metric | Goal | Tasks | Score | Target | Passed |",
        "| ------ | ---- | ----- | ----- | ------ | ------ |",
    ]
    for name, m in (report.get("metrics") or {}).items():
        score, target, passed = m.get("score"), m.get("target"), m.get("passed")
        rows.append(
            f"| {name} | {m.get('goal', '')} | {m.get('n', '')} "
            f"| {'n/a' if score is None else f'{score:.3f}'} "
            f"| {'n/a' if target is None else f'{target:g}'} "
            f"| {'n/a' if passed is None else ('yes' if passed else 'no')} |"
        )
    context = report.get("context_tokens")
    notes = [
        f"Suite `{report.get('suite', '?')}`, backend `{report.get('backend', '?')}`, "
        f"context {report.get('context_tier', '?')}"
        + (f" ({context} tokens)" if context else "")
        + f", {report.get('tasks_run', '?')} tasks run, {report.get('tasks_failed', '?')} failed, "
        f"{report.get('timestamp', 'unknown time')}.",
        f"Judge: {report.get('judge', 'unknown')}.",
    ]
    if report.get("dry_run"):
        notes.append(
            "**This report is a dry run: a scripted fake model played the reference turns, so "
            "these numbers are not a measurement of the model.**"
        )
    return "\n".join(rows) + "\n\n" + "\n".join(notes)


def datasheet(release, report: dict | None) -> str:
    lines = [
        f"- Teacher: {release.teacher} ({release.teacher_licence}), run locally; no hosted API "
        "(Claude, OpenAI or any other) produced or judged any sample.",
        f"- Judge: {release.judge}.",
        "- Filters: " + "; ".join(release.filters) + ".",
    ]
    if not report:
        lines.append("- Counts: the dataset report was not available when this card was written.")
        return "\n".join(lines)
    for key, label in (
        ("prompts", "Prompts"),
        ("generated", "Traces generated"),
        ("judged", "Traces judged"),
        ("kept", "Samples kept (before the validation split)"),
    ):
        if report.get(key) is not None:
            lines.append(f"- {label}: {report[key]}")
    if report.get("pass_rate") is not None:
        lines.append(f"- Judge pass rate: {report['pass_rate']:.1%}")
    if report.get("kept_by_class"):
        kept = ", ".join(f"{k} {v}" for k, v in report["kept_by_class"].items())
        lines.append(f"- Kept by task class: {kept}")
    return "\n".join(lines)


def recipe(config: SftConfig, train_result: dict | None) -> str:
    t, lo, d = config.train, config.lora, config.data
    lines = [
        f"- Base: `{BASE_MODEL}`, bf16 weights (never quantised during training).",
        "- Method: supervised fine-tuning with a bf16 LoRA adapter (no QLoRA, no 4/8-bit), "
        "merged into the base afterwards; loss on assistant tokens only, including the "
        "teacher's reasoning in each turn's thinking block (samples without it are skipped).",
        f"- LoRA: r={lo.r}, alpha={lo.alpha}, dropout={lo.dropout}, "
        f"targets {', '.join(lo.target_modules)}.",
        f"- Optimiser: AdamW, learning rate {t.learning_rate:g}, {t.lr_scheduler} schedule, "
        f"warmup {t.warmup_ratio:g}, weight decay {t.weight_decay:g}, "
        f"grad-norm clip {t.max_grad_norm:g}.",
        f"- Epochs {t.epochs:g}, batch {t.per_device_batch_size} x {t.gradient_accumulation_steps} "
        f"accumulation, sequence length {d.max_length}, seed {t.seed}, "
        f"gradient checkpointing {'on' if t.gradient_checkpointing else 'off'}.",
    ]
    metrics = (train_result or {}).get("metrics") or {}
    if "train_loss" in metrics:
        lines.append(f"- Final training loss: {metrics['train_loss']:.4f}.")
    if "eval_loss" in metrics:
        lines.append(f"- Validation loss: {metrics['eval_loss']:.4f}.")
    return "\n".join(lines)


def memory_notes(release, files: list[dict], arch: dict | None) -> str:
    if files:
        sizes = {f["type"]: f.get("bytes") for f in files}
        kinds = [(k, sizes.get(k)) for k in dict.fromkeys(["bf16", *sizes])]
    else:
        kinds = [("bf16", None)] + [(t, None) for t in release.gguf_types]
    ctx = release.context_tokens
    kv = kv_cache_gib(ctx, arch)
    rows = [
        f"| Weights | Size (GiB) | f16 KV cache at {ctx} tokens (GiB) | Total (GiB) |",
        "| ------- | ---------- | ---------------------------------- | ----------- |",
    ]
    for kind, size in kinds:
        w = weights_gib(kind, size)
        rows.append(f"| {kind} | {w:.1f} | {kv:.1f} | {w + kv:.1f} |")
    steps = ", ".join(f"{t // 1024}K = {kv_cache_gib(t, arch):.1f} GiB" for t in (8192, 32768, ctx))
    return (
        "\n".join(rows)
        + f"\n\nKV cache (f16) grows with context: {steps}. Compute buffers add roughly 0.5 to "
        "1 GiB. On a 16 GB GPU a Q8_0 or Q6_K file with a 32K to 64K context fits in VRAM; "
        "llama.cpp (`--fit on`) keeps the remainder on the CPU when another process holds part "
        "of the card. Sizes come from the files where they exist and are estimates otherwise."
    )


@dataclass
class CardInputs:
    config: SftConfig
    eval_report: dict | None = None
    data_report: dict | None = None
    train_result: dict | None = None
    gguf_files: list[dict] | None = None
    arch: dict | None = None


def render_card(inputs: CardInputs, kind: str) -> str:
    """The README.md text for the bf16 weights directory (`kind` "bf16") or the GGUF one."""
    r = inputs.config.release
    gguf = kind == "gguf"
    head = {
        "license": r.license,
        "base_model": BASE_MODEL,
        "language": ["en"],
        "pipeline_tag": "text-generation",
        "library_name": "gguf" if gguf else "transformers",
        "tags": [*r.tags, *(["gguf"] if gguf else ["bf16"])],
    }
    title = f"{r.title} 4B" + (" (GGUF)" if gguf else "")
    parts = [front_matter(head), "", f"# {title}", "", r.summary, ""]
    if gguf:
        files = inputs.gguf_files or []
        parts += ["## Files", ""]
        if files:
            parts += ["| File | Type | Size (GiB) |", "| ---- | ---- | ---------- |"]
            parts += [f"| `{f['file']}` | {f['type']} | {f['bytes'] / GIB:.2f} |" for f in files]
        else:
            parts.append("The files are listed here once the export has run.")
        parts += [
            "",
            "Quantised after training from the merged bf16 weights with llama.cpp "
            "`llama-quantize`; nothing was trained on quantised weights. Serve with a llama.cpp "
            f"build and `--jinja` so the chat template (and tool calling) applies, under the "
            f"name `{r.eval_alias}`.",
            "",
        ]
    parts += ["## Intended use", ""] + [f"- {item}" for item in r.intended_use]
    parts += ["", "## Limits", ""] + [f"- {item}" for item in r.limits]
    parts += ["", "## Evaluation", "", eval_table(inputs.eval_report)]
    parts += ["", "## Training recipe", "", recipe(inputs.config, inputs.train_result)]
    parts += ["", "## Dataset datasheet", "", datasheet(r, inputs.data_report)]
    parts += [
        "",
        "## Memory and context",
        "",
        memory_notes(r, inputs.gguf_files or [], inputs.arch),
    ]
    parts += [
        "",
        "## Licence",
        "",
        f"{r.license}, the licence of the base model. The teacher's licence ({r.teacher_licence}) "
        "permits using its outputs for training.",
    ]
    if r.teacher_policy:
        parts.append(r.teacher_policy)
    parts += [
        "",
    ]
    return "\n".join(parts)


def latest_eval_report(root: Path, run_dir: Path | None = None) -> dict | None:
    """`<run_dir>/eval.json`, or the newest `runs/*/eval.json`."""
    if run_dir:
        candidates = [run_dir / "eval.json"]
    else:
        candidates = sorted((root / "runs").glob("*/eval.json"), key=lambda p: p.stat().st_mtime)
    for path in reversed(candidates):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _json_or_none(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_cards(config: SftConfig, root: Path, run_dir: Path | None = None) -> list[Path]:
    """Write `README.md` into the bf16 weights directory and, if it exists, the GGUF one."""
    weights = (root / config.merge.output_dir).resolve()
    if not weights.is_dir():
        raise FileNotFoundError(f"no merged weights at {weights}; run the merge first")
    gguf_dir = gguf_dir_for(root, config.release.name, config.release.gguf_dir)
    export = _json_or_none(gguf_dir / "export.json") or {}
    inputs = CardInputs(
        config=config,
        eval_report=latest_eval_report(root, run_dir),
        data_report=_json_or_none(root / "data" / "report.json"),
        train_result=_json_or_none(root / config.train.output_dir / "train_result.json"),
        gguf_files=export.get("files"),
        arch=read_arch(weights),
    )
    written = []
    targets = [(weights, "bf16")] + ([(gguf_dir, "gguf")] if gguf_dir.is_dir() else [])
    for directory, kind in targets:
        path = directory / "README.md"
        path.write_text(render_card(inputs, kind), encoding="utf-8", newline="\n")
        written.append(path)
    return written
