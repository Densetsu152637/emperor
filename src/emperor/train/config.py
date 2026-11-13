"""Training configuration (TOML under `configs/`), validated without importing torch.

The recipe is bf16 LoRA on full-precision weights. Quantised loading (QLoRA, 4/8-bit) is not
supported and is refused by name so a stray key cannot silently change the recipe.
"""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


class ConfigError(ValueError):
    """The configuration file is invalid."""


@dataclass
class ModelConfig:
    base: str = "F:/AI-Models/Qwen/Qwen3.5-4B"
    attn_implementation: str = "sdpa"
    min_free_vram_gib: float = 13.5  # Qwen3.5 peaks at 13.0 GiB at 8192 tokens, plus CUDA context


@dataclass
class DataConfig:
    path: str = "data/final/final.jsonl"
    extra_paths: list[str] = field(default_factory=list)  # more rounds, merged and deduped by id
    max_length: int = 8192
    # From the data instead: the `max_length_quantile` of the samples' token lengths, rounded
    # up to a multiple of 1,024 and never above `max_length_cap` (long multi-subagent traces are
    # kept; a runaway one is not). Needs train.offload_activations past what the GPU holds.
    max_length_auto: bool = False
    max_length_quantile: float = 0.99
    max_length_cap: int = 32768
    val_fraction: float = 0.05
    split_seed: int = 1337


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",  # Qwen3.5 Gated DeltaNet layers
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    output_dir: str = "outputs/sft"
    epochs: float = 2.0
    learning_rate: float = 1.0e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    # Opt-in: keep the activations saved for the backward pass (with checkpointing, each
    # layer's input) in pinned RAM instead of VRAM, so traces longer than the GPU alone holds
    # can be trained. bf16 LoRA either way; the weights stay on the GPU. Off: training stops
    # with a clear error when the VRAM is not free.
    offload_activations: bool = False
    offload_min_free_vram_gib: float = 10.0  # weights, LoRA, optimiser and one layer's work
    save_steps: int = 50
    save_total_limit: int = 3
    eval_steps: int = 50
    logging_steps: int = 5
    max_grad_norm: float = 1.0
    seed: int = 1337


@dataclass
class MergeConfig:
    output_dir: str = "weights/emperor-4b-bf16"
    adapter_dir: str = "outputs/sft/adapter"


@dataclass
class ReleaseConfig:
    """What the release tooling (GGUF export, model card, publish) needs to name and describe
    the model. Only the name has to differ between models that share this package."""

    name: str = "emperor-4b"  # weights/<name>-gguf/, <name>-<TYPE>.gguf, the card title
    title: str = "Emperor"
    summary: str = (
        "Emperor is an open-weight root agent for Imperium: it plans, writes compact briefs "
        "and delegates the work to subagents through tool calls."
    )
    eval_alias: str = "emperor"  # the model name the eval harness and llama-server see
    license: str = "apache-2.0"
    tags: list[str] = field(default_factory=lambda: ["imperium", "agent", "lora", "qwen3"])
    gguf_dir: str = ""  # default: weights/<name>-gguf
    gguf_types: list[str] = field(default_factory=lambda: ["Q8_0", "Q6_K", "Q4_K_M"])
    context_tokens: int = 65536  # the context the card sizes memory for
    intended_use: list[str] = field(
        default_factory=lambda: [
            "Acting as the root agent of an Imperium session: planning, briefing and delegating.",
        ]
    )
    limits: list[str] = field(
        default_factory=lambda: [
            "A 4B model: it can be wrong, and its answers need the checks Imperium applies.",
            "Trained on synthetic data from one teacher model; coverage outside it is untested.",
        ]
    )
    # Used only when the dataset report does not name them (the card reads the served models
    # from the samples).
    teacher: str = "deepseek-v4-pro"
    teacher_licence: str = "DeepSeek Open Platform Terms of Service"
    teacher_policy: str = (
        "DeepSeek's outputs are used under section 4.2 of the DeepSeek Open Platform Terms of "
        'Service, which assigns the outputs to the user and lists "training other models (such '
        'as model distillation)" among the permitted uses '
        "(https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html)."
    )
    judge: str = (
        "kimi-k3.1-pro (Moonshot API, a different model family from the teacher; scored 1-5 on "
        "eight criteria with a pass or fail)"
    )
    filters: list[str] = field(
        default_factory=lambda: [
            "schema checks (tool calls, roles, brief sections and size)",
            "no secrets or credentials",
            "overlap test against the held-out eval tasks",
            "judge pass and a mean score threshold",
        ]
    )


@dataclass
class SftConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)

    @property
    def dtype(self) -> str:
        return "bfloat16"


_FORBIDDEN = ("4bit", "8bit", "qlora", "quant", "bnb", "load_in")


def _section(cls, raw: dict, name: str):
    known = {f.name for f in fields(cls)}
    for key in raw:
        if any(word in key.lower() for word in _FORBIDDEN):
            raise ConfigError(
                f"[{name}] {key}: quantised training is not supported; Emperor trains bf16 LoRA "
                "on full-precision weights and quantises only after training"
            )
        if key not in known:
            raise ConfigError(f"[{name}] unknown key {key!r}")
    return cls(**raw)


def parse_config(raw: dict) -> SftConfig:
    sections = {
        "model": ModelConfig,
        "data": DataConfig,
        "lora": LoraConfig,
        "train": TrainConfig,
        "merge": MergeConfig,
        "release": ReleaseConfig,
    }
    for name in raw:
        if name not in sections:
            raise ConfigError(f"unknown section [{name}]")
    config = SftConfig(**{n: _section(c, raw.get(n, {}), n) for n, c in sections.items()})
    if config.data.max_length < 256:
        raise ConfigError("[data] max_length must be at least 256")
    if not 0 <= config.data.val_fraction < 0.5:
        raise ConfigError("[data] val_fraction must be in [0, 0.5)")
    if config.train.per_device_batch_size < 1 or config.train.gradient_accumulation_steps < 1:
        raise ConfigError("[train] batch size and accumulation steps must be at least 1")
    if config.train.save_steps < 1 or config.train.eval_steps < 1:
        raise ConfigError("[train] save_steps and eval_steps must be at least 1")
    if config.lora.r < 1:
        raise ConfigError("[lora] r must be at least 1")
    name = config.release.name
    if not name or any(c in name for c in ("/", "\\", " ", ":", "*", "?", '"', "<", ">", "|")):
        raise ConfigError("[release] name must be a plain file name")
    if not config.release.gguf_types:
        raise ConfigError("[release] gguf_types must list at least one type")
    return config


def load_config(path: Path | str) -> SftConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error
    return parse_config(raw)
