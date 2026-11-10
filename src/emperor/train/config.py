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
    base: str = "D:/AI-Models/unsloth/Qwen3-4B-Instruct-2507"
    attn_implementation: str = "sdpa"


@dataclass
class DataConfig:
    path: str = "data/final/final.jsonl"
    max_length: int = 8192
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
class SftConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)

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
    return config


def load_config(path: Path | str) -> SftConfig:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: {error}") from error
    return parse_config(raw)
