"""Multi-GPU image-conditioned SFT entrypoint for task-level BC VLM training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed

try:
    from transformers import AutoModelForImageTextToText as AutoVisionLanguageModel
except ImportError:  # pragma: no cover - compatibility with older transformers
    from transformers import AutoModelForVision2Seq as AutoVisionLanguageModel

from training.bc_task_vlm.dataset import (
    LazyVisionSFTCollator,
    TrajectoryStepDataset,
    build_examples,
    build_split_manifest,
)
from training.bc_task_vlm.evaluation import evaluate_structured_generation
from training.bc_task_vlm.task_registry import resolve_task_name, supported_task_names


@dataclass(frozen=True)
class RunConfiguration:
    dataset_root: str
    model_name_or_path: str
    processor_name_or_path: str
    train_tasks: list[str]
    val_tasks: list[str]
    output_dir: str
    per_device_batch_size: int
    grad_accum: int
    num_epochs: float
    learning_rate: float
    max_length: int | None
    num_workers: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    attn_implementation: str
    trust_remote_code: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    save_steps: int
    eval_steps: int
    logging_steps: int
    save_total_limit: int
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    eval_max_new_tokens: int
    eval_generation_batch_size: int
    eval_generation_max_samples: int | None
    report_to: list[str]
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_tags: list[str]
    wandb_mode: str
    resume_from_checkpoint: str | None
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data_generation/task_level/data/image/20260404T191734Z"),
        help="Root directory of the rendered trajectory dataset.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3.5-9B-Base",
        help="Base multimodal checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--processor-name-or-path",
        default=None,
        help="Optional processor path override. Defaults to the model path.",
    )
    parser.add_argument(
        "--train-tasks",
        default="hot_dog_setup,prepare_sandwich_station",
        help="Comma-separated task names for the training split.",
    )
    parser.add_argument(
        "--val-tasks",
        default="prepare_coffee",
        help="Comma-separated task names for the validation split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, manifests, and metrics.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional sequence truncation length. Leave unset for VLM training.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to target with LoRA.",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-generation-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-generation-max-samples",
        type=int,
        default=None,
        help="Optional cap for structured generation evaluation.",
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Comma-separated Trainer report targets. Use 'none' to disable.",
    )
    parser.add_argument("--wandb-project", default="robocasa-bc-task-vlm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="bc_task_vlm,qwen3.5,sft")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_task_list(raw_value: str) -> list[str]:
    resolved = [resolve_task_name(task_name) for task_name in _split_csv(raw_value)]
    if not resolved:
        supported = ", ".join(supported_task_names())
        raise ValueError(
            f"At least one task is required. Supported tasks: {supported}."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate task names are not allowed: {resolved}.")
    return resolved


def _resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("training/bc_task_vlm/runs") / timestamp


def _resolve_report_targets(raw_value: str) -> list[str]:
    targets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not targets or targets == ["none"]:
        return []
    return targets


def _ensure_optional_dependency(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Missing optional dependency {module_name!r}. "
            f"Install training/bc_task_vlm/requirements.txt first."
        )


def _configure_wandb(
    args: argparse.Namespace, report_targets: list[str], output_dir: Path
) -> None:
    if "wandb" not in report_targets:
        return
    _ensure_optional_dependency("wandb")

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
    if args.wandb_tags:
        os.environ.setdefault("WANDB_TAGS", args.wandb_tags)
    if args.wandb_mode == "disabled":
        os.environ.setdefault("WANDB_DISABLED", "true")
    else:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str((output_dir / "wandb").resolve()))


def _count_parameters(model) -> dict[str, int]:
    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _build_run_configuration(args: argparse.Namespace) -> RunConfiguration:
    if args.bf16 and args.fp16:
        raise ValueError("Enable at most one of --bf16 or --fp16.")
    processor_name_or_path = args.processor_name_or_path or args.model_name_or_path
    report_targets = _resolve_report_targets(args.report_to)
    train_tasks = _resolve_task_list(args.train_tasks)
    val_tasks = _resolve_task_list(args.val_tasks)
    overlapping_tasks = sorted(set(train_tasks).intersection(val_tasks))
    if overlapping_tasks:
        overlap_text = ", ".join(overlapping_tasks)
        raise ValueError(
            f"Train and validation tasks must be disjoint. Overlap: {overlap_text}."
        )
    return RunConfiguration(
        dataset_root=str(args.dataset_root.resolve()),
        model_name_or_path=args.model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        output_dir=str(_resolve_output_dir(args.output_dir).resolve()),
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        num_workers=args.num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_split_csv(args.lora_target_modules),
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_generation_batch_size=args.eval_generation_batch_size,
        eval_generation_max_samples=args.eval_generation_max_samples,
        report_to=report_targets,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=_split_csv(args.wandb_tags),
        wandb_mode=args.wandb_mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_processor(
    processor_name_or_path: str,
    *,
    trust_remote_code: bool,
):
    processor = AutoProcessor.from_pretrained(
        processor_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
    return processor


def _load_model(config: RunConfiguration):
    torch_dtype = None
    if torch.cuda.is_available():
        if config.bf16:
            torch_dtype = torch.bfloat16
        elif config.fp16:
            torch_dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoVisionLanguageModel.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model


def main() -> None:
    args = parse_args()
    config = _build_run_configuration(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_wandb(args, config.report_to, output_dir)
    set_seed(config.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_examples = build_examples(
        dataset_root=Path(config.dataset_root),
        task_names=config.train_tasks,
    )
    val_examples = build_examples(
        dataset_root=Path(config.dataset_root),
        task_names=config.val_tasks,
    )
    train_dataset = TrajectoryStepDataset(train_examples)
    val_dataset = TrajectoryStepDataset(val_examples)

    split_manifest = build_split_manifest(
        dataset_root=Path(config.dataset_root),
        train_examples=train_examples,
        val_examples=val_examples,
    )

    processor = _load_processor(
        config.processor_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    model = _load_model(config)
    parameter_counts = _count_parameters(model)

    run_config_payload = asdict(config) | parameter_counts
    _save_json(output_dir / "run_config.json", run_config_payload)
    _save_json(output_dir / "split_manifest.json", split_manifest)

    processor.save_pretrained(output_dir / "processor")

    data_collator = LazyVisionSFTCollator(
        processor_name_or_path=config.processor_name_or_path,
        max_length=config.max_length,
        trust_remote_code=config.trust_remote_code,
    )

    evaluation_strategy = "steps" if len(val_dataset) > 0 else "no"
    if len(val_dataset) > 0 and config.save_steps % config.eval_steps != 0:
        raise ValueError(
            "--save-steps must be a multiple of --eval-steps when validation is enabled."
        )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        report_to=config.report_to,
        run_name=config.wandb_run_name or output_dir.name,
        dataloader_num_workers=config.num_workers,
        gradient_checkpointing=config.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        seed=config.seed,
        label_names=["labels"],
        load_best_model_at_end=len(val_dataset) > 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
    )

    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics: dict[str, float] = {}
    if len(val_dataset) > 0:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero() and len(val_dataset) > 0:
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        structured_metrics = evaluate_structured_generation(
            model=unwrapped_model,
            eval_dataset=val_dataset,
            processor_name_or_path=config.processor_name_or_path,
            output_dir=output_dir,
            max_length=config.max_length,
            max_new_tokens=config.eval_max_new_tokens,
            batch_size=config.eval_generation_batch_size,
            trust_remote_code=config.trust_remote_code,
            max_samples=config.eval_generation_max_samples,
        )
        trainer.log(structured_metrics)
        _save_json(
            output_dir / "final_metrics.json",
            train_result.metrics | eval_metrics | structured_metrics,
        )
    elif trainer.is_world_process_zero():
        _save_json(output_dir / "final_metrics.json", train_result.metrics)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
