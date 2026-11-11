"""bf16 LoRA supervised fine-tuning with TRL, and merging the adapter into the base weights.

Needs the `train` dependency group (torch, transformers, peft, trl). Imports are deferred so the
rest of the package and its tests never load them. The base model directory is only read.
"""

import json
import logging
from pathlib import Path

from emperor.train.config import SftConfig
from emperor.train.data import encode_sample, load_records, pad_batch, split_records

log = logging.getLogger("emperor.train")
PAUSED_EXIT_CODE = 75  # EX_TEMPFAIL: a checkpoint was saved and the run should be resumed


def gpu_split(margin_gib: float = 1.5) -> dict:
    """Free VRAM less a margin for the GPU, the rest from system RAM: training runs partly on the
    GPU when another process holds some of it, and fully on it when the card is free."""
    import torch

    free, _total = torch.cuda.mem_get_info(0)
    gpu_gib = max(free / 2**30 - margin_gib, 1.0)
    return {0: f"{gpu_gib:.1f}GiB", "cpu": "48GiB"}


def _resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def last_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in output_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return max(candidates, key=lambda p: int(p.name.split("-")[-1]), default=None)


def prepare_datasets(config: SftConfig, tokenizer, root: Path):
    records = load_records(_resolve(config.data.path, root))
    train, val = split_records(records, config.data.val_fraction, config.data.split_seed)
    encoded = {}
    for name, items in (("train", train), ("val", val)):
        rows, skipped = [], 0
        for sample in items:
            row = encode_sample(sample, tokenizer, config.data.max_length)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
        log.info(
            "%s: %d samples, %d skipped (too long or no assistant tokens)", name, len(rows), skipped
        )
        encoded[name] = rows
    if not encoded["train"]:
        raise SystemExit("no trainable samples; run the data pipeline first")
    return encoded


def train(config: SftConfig, root: Path, pause_file: Path | None = None) -> int:
    """Run (or resume) training; returns 0 when finished, PAUSED_EXIT_CODE when paused."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; check the torch build (cu128 for RTX 50 series)")
    base = Path(config.model.base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    data = prepare_datasets(config, tokenizer, root)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,  # full-precision (bf16) weights; never quantised
        attn_implementation=config.model.attn_implementation,
        device_map="auto",
        max_memory=gpu_split(),
    )
    model.config.use_cache = False
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    def collate(batch):
        padded = pad_batch(batch, pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}

    output_dir = _resolve(config.train.output_dir, root)
    has_val = bool(data["val"])
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.train.epochs,
        learning_rate=config.train.learning_rate,
        lr_scheduler_type=config.train.lr_scheduler,
        warmup_steps=config.train.warmup_ratio,  # transformers 5: a float below 1 is a ratio
        weight_decay=config.train.weight_decay,
        per_device_train_batch_size=config.train.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        gradient_checkpointing=config.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="adamw_torch_fused",
        max_grad_norm=config.train.max_grad_norm,
        logging_steps=config.train.logging_steps,
        save_strategy="steps",
        save_steps=config.train.save_steps,
        save_total_limit=config.train.save_total_limit,
        eval_strategy="steps" if has_val else "no",
        eval_steps=config.train.eval_steps,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False if has_val else None,
        seed=config.train.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_length=config.data.max_length,
        packing=False,  # one conversation per sequence, assistant tokens only in the loss
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    paused = {"flag": False}

    class PauseOnFile(TrainerCallback):
        """Save a checkpoint and stop at the next step boundary when the pause file appears."""

        def on_step_end(self, args, state, control, **kwargs):
            if pause_file is not None and pause_file.exists():
                log.info("pause requested at step %d; saving a checkpoint", state.global_step)
                paused["flag"] = True
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(data["train"]),
        eval_dataset=Dataset.from_list(data["val"]) if has_val else None,
        data_collator=collate,
        processing_class=tokenizer,
        peft_config=PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        ),
        callbacks=[PauseOnFile()],
    )
    resume = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    log.info("resuming from %s", resume) if resume else log.info("starting a new run")
    result = trainer.train(resume_from_checkpoint=resume)
    if paused["flag"]:
        return PAUSED_EXIT_CODE
    adapter_dir = _resolve(config.merge.adapter_dir, root)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (output_dir / "train_result.json").write_text(
        json.dumps({"metrics": result.metrics, "max_memory_gib": _peak_gib(torch)}, indent=2),
        encoding="utf-8",
    )
    return 0


def _peak_gib(torch) -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def merge(config: SftConfig, root: Path) -> Path:
    """Merge the LoRA adapter into the bf16 base and save bf16 safetensors."""
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = Path(config.model.base).resolve()
    adapter = _resolve(config.merge.adapter_dir, root)
    out = _resolve(config.merge.output_dir, root).resolve()
    if not adapter.exists():
        raise SystemExit(f"no adapter at {adapter}; train first")
    if out == base or base in out.parents:
        raise SystemExit("refusing to write inside the base model directory")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    for shard in sorted(out.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118
                dtype = handle.get_slice(key).get_dtype()
                if dtype != "BF16":
                    raise SystemExit(f"{shard.name}:{key} is {dtype}, expected BF16")
    return out
