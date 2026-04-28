"""Multi-GPU image-conditioned SFT entrypoint for task-level BC VLM training."""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from accelerate.state import PartialState
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed
from transformers.utils import is_torch_bf16_gpu_available

try:
    from transformers import AutoModelForImageTextToText as AutoVisionLanguageModel
except ImportError:  # pragma: no cover - compatibility with older transformers
    from transformers import AutoModelForVision2Seq as AutoVisionLanguageModel

from training.bc_task_vlm.dataset import (
    CentralizedDataset,
    DecentralizedDataset,
    LazyVisionSFTCollator,
    build_example_cache_fingerprint,
    build_example_cache_path,
    build_centralized_examples,
    build_decentralized_examples,
    build_split_manifest,
    load_examples_from_cache,
    save_examples_to_cache,
)
from data_generation.task_level.runtime.client import load_dotenv_file
from training.bc_task_vlm.evaluation import evaluate_structured_generation
from training.bc_task_vlm.task_registry import resolve_task_name, supported_task_names

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WANDB_PROJECT = "robocasa-bc-task-vlm"


@dataclass(frozen=True)
class RunConfiguration:
    dataset_root: str
    model_name_or_path: str
    processor_name_or_path: str
    train_tasks: list[str]
    val_tasks: list[str]
    train_example_granularity: str
    use_example_cache: bool
    training_samples_cache_dir: str
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
        default="Qwen/Qwen3.5-0.8B",
        help=(
            "Multimodal checkpoint to fine-tune. Defaults to the post-trained "
            "chat model because this pipeline uses chat-formatted JSON supervision."
        ),
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
        "--train-example-granularity",
        choices=("centralized", "decentralized"),
        default="decentralized",
        help=(
            "Training example format. 'centralized' is one next-action "
            "prediction per global step. 'decentralized' turns each joint "
            "two-agent episode into one conversation per agent and only "
            "supervises that agent's assistant tool-call turns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, manifests, and metrics.",
    )
    parser.add_argument(
        "--use-example-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached serialized training samples across training runs.",
    )
    parser.add_argument(
        "--training-samples-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached serialized training samples. Defaults to "
            "<repo>/.cache/bc_task_vlm/examples."
        ),
    )
    parser.add_argument(
        "--example-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", _DEFAULT_WANDB_PROJECT),
    )
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


def _resolve_training_samples_cache_dir(
    training_samples_cache_dir: Path | None,
) -> Path:
    if training_samples_cache_dir is not None:
        return training_samples_cache_dir
    return _REPO_ROOT / ".cache" / "bc_task_vlm" / "examples"


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
        train_example_granularity=args.train_example_granularity,
        use_example_cache=args.use_example_cache,
        training_samples_cache_dir=str(
            _resolve_training_samples_cache_dir(
                args.training_samples_cache_dir
            ).resolve()
        ),
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
        # Model stores fewer intermediate activations during the forward pass and recomputes them during backprop. Lower VRAM use, but slower training.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Inference KV-cache. Not useful for training, so disable it.
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


def _launcher_world_size() -> int:
    raw_value = os.environ.get("WORLD_SIZE", "1")
    try:
        return max(int(raw_value), 1)
    except (TypeError, ValueError):
        return 1


def _is_primary_process() -> bool:
    raw_value = os.environ.get("RANK", "0")
    try:
        return int(raw_value) == 0
    except (TypeError, ValueError):
        return True


def _validate_runtime_environment() -> None:
    if _launcher_world_size() <= 1 and os.environ.get("LOCAL_RANK") is None:
        return
    if torch.cuda.is_available():
        return
    cuda_build = getattr(torch.version, "cuda", None) or "unknown"
    raise RuntimeError(
        "This run was launched for distributed GPU training, but PyTorch cannot "
        "initialize CUDA in the current environment. "
        f"Installed torch={torch.__version__} "
        f"(CUDA build {cuda_build}). Update the NVIDIA driver on the host or "
        "install a torch build compatible with the host driver before retrying."
    )


def _resolve_precision_config(
    config: RunConfiguration,
    *,
    state: PartialState,
) -> RunConfiguration:
    if not config.bf16:
        return config
    if is_torch_bf16_gpu_available():
        return config
    if not torch.cuda.is_available():
        _log_startup(
            "bf16 requested but CUDA is unavailable; falling back to fp32",
            state=state,
        )
        return replace(config, bf16=False, fp16=False)
    _log_startup(
        "bf16 requested but not supported by the current CUDA runtime/GPU; "
        "falling back to fp16",
        state=state,
    )
    return replace(config, bf16=False, fp16=True)


def _selected_mixed_precision(config: RunConfiguration) -> str:
    if config.bf16:
        return "bf16"
    if config.fp16:
        return "fp16"
    return "no"


def _synchronize_accelerate_precision_env(config: RunConfiguration) -> None:
    os.environ["ACCELERATE_MIXED_PRECISION"] = _selected_mixed_precision(config)


def _log_startup(message: str, *, state: PartialState | None = None) -> None:
    if not _is_primary_process():
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _build_examples_for_tasks(
    *,
    dataset_root: Path,
    task_names: list[str],
    split_name: str,
    granularity: str,
    use_example_cache: bool,
    training_samples_cache_dir: Path,
    state: PartialState,
) -> list[Any]:
    if granularity == "decentralized":
        builder = build_decentralized_examples
    elif granularity == "centralized":
        builder = build_centralized_examples
    else:  # pragma: no cover - parser restricts values
        raise ValueError(f"Unsupported granularity: {granularity}")

    examples: list[Any] = []
    total_tasks = len(task_names)
    for task_index, task_name in enumerate(task_names, start=1):
        task_label = (
            f"{split_name} {granularity} examples for task "
            f"{task_name} ({task_index}/{total_tasks})"
        )
        progress_description = f"{split_name} {granularity} {task_name}"
        fingerprint: dict[str, Any] | None = None
        cache_path: Path | None = None
        task_examples: list[Any] | None = None

        if use_example_cache:
            fingerprint = build_example_cache_fingerprint(
                dataset_root=dataset_root,
                task_name=task_name,
                granularity=granularity,
            )
            cache_path = build_example_cache_path(
                cache_dir=training_samples_cache_dir,
                dataset_root=dataset_root,
                task_name=task_name,
                granularity=granularity,
            )
            task_examples = load_examples_from_cache(
                cache_path=cache_path,
                expected_fingerprint=fingerprint,
                granularity=granularity,
            )
            if task_examples is not None:
                _log_startup(
                    f"Loaded {task_label} from cache {cache_path}",
                    state=state,
                )

        if task_examples is None:
            _log_startup(f"Building {task_label}", state=state)
            if use_example_cache and not state.is_main_process:
                state.wait_for_everyone()
                if cache_path is not None and fingerprint is not None:
                    task_examples = load_examples_from_cache(
                        cache_path=cache_path,
                        expected_fingerprint=fingerprint,
                        granularity=granularity,
                    )
            else:
                task_examples = builder(
                    dataset_root=dataset_root,
                    task_names=[task_name],
                    show_progress=state.is_main_process,
                    progress_description=progress_description,
                )
                if (
                    use_example_cache
                    and cache_path is not None
                    and fingerprint is not None
                ):
                    try:
                        save_examples_to_cache(
                            cache_path=cache_path,
                            fingerprint=fingerprint,
                            examples=task_examples,
                        )
                    except OSError as exc:
                        _log_startup(
                            "Warning: failed to save training-samples cache "
                            f"{cache_path}: {exc}",
                            state=state,
                        )
                if use_example_cache:
                    state.wait_for_everyone()

            if task_examples is None:
                task_examples = builder(
                    dataset_root=dataset_root,
                    task_names=[task_name],
                    show_progress=False,
                    progress_description=progress_description,
                )

        examples.extend(task_examples)
        _log_startup(
            f"Finished {split_name} task {task_name}: {len(task_examples)} "
            f"examples ({len(examples)} cumulative)",
            state=state,
        )
    return examples


def _build_training_arguments(
    *,
    config: RunConfiguration,
    output_dir: Path,
    evaluation_strategy: str,
    has_validation: bool,
) -> TrainingArguments:
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": config.save_total_limit,
        "remove_unused_columns": False,
        "report_to": config.report_to,
        "run_name": config.wandb_run_name or output_dir.name,
        "dataloader_num_workers": config.num_workers,
        "gradient_checkpointing": config.gradient_checkpointing,
        "ddp_find_unused_parameters": False,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "seed": config.seed,
        "label_names": ["labels"],
        "load_best_model_at_end": has_validation,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = evaluation_strategy
    else:
        training_kwargs["eval_strategy"] = evaluation_strategy
    return TrainingArguments(**training_kwargs)


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    config = _build_run_configuration(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_wandb(args, config.report_to, output_dir)
    set_seed(config.seed)
    distributed_state = PartialState()
    _validate_runtime_environment()
    config = _resolve_precision_config(config, state=distributed_state)
    _synchronize_accelerate_precision_env(config)
    dataset_root = Path(config.dataset_root)

    _log_startup(
        f"Starting task VLM training run in {output_dir}",
        state=distributed_state,
    )
    _log_startup(
        f"Using precision {_selected_mixed_precision(config)}",
        state=distributed_state,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=config.train_tasks,
        split_name="train",
        granularity=config.train_example_granularity,
        use_example_cache=config.use_example_cache,
        training_samples_cache_dir=Path(config.training_samples_cache_dir),
        state=distributed_state,
    )
    if config.train_example_granularity == "decentralized":
        train_dataset = DecentralizedDataset(train_examples)
    else:
        train_dataset = CentralizedDataset(train_examples)
    val_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=config.val_tasks,
        split_name="validation",
        granularity="centralized",
        use_example_cache=config.use_example_cache,
        training_samples_cache_dir=Path(config.training_samples_cache_dir),
        state=distributed_state,
    )
    val_dataset = CentralizedDataset(val_examples)

    _log_startup("Building split manifest", state=distributed_state)
    split_manifest = build_split_manifest(
        dataset_root=dataset_root,
        train_examples=train_examples,
        val_examples=val_examples,
    )

    _log_startup(
        f"Loading processor from {config.processor_name_or_path}",
        state=distributed_state,
    )
    processor = _load_processor(
        config.processor_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    _log_startup(
        f"Loading model weights from {config.model_name_or_path}",
        state=distributed_state,
    )
    model = _load_model(config)
    parameter_counts = _count_parameters(model)
    _log_startup(
        "Model loaded "
        f"({parameter_counts['trainable_params']:,} trainable / "
        f"{parameter_counts['total_params']:,} total parameters)",
        state=distributed_state,
    )

    run_config_payload = asdict(config) | parameter_counts
    if distributed_state.is_main_process:
        _log_startup("Saving run metadata", state=distributed_state)
        _save_json(output_dir / "run_config.json", run_config_payload)
        _save_json(output_dir / "split_manifest.json", split_manifest)
        processor.save_pretrained(output_dir / "processor")
    distributed_state.wait_for_everyone()

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
    training_args = _build_training_arguments(
        config=config,
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        has_validation=len(val_dataset) > 0,
    )

    _log_startup("Initializing Trainer", state=distributed_state)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
    )

    _log_startup(
        f"Starting trainer.train() with {len(train_dataset)} training samples "
        f"and {len(val_dataset)} validation samples",
        state=distributed_state,
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
        _log_startup(
            "Running structured generation evaluation",
            state=distributed_state,
        )
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
    _log_startup("Run complete", state=distributed_state)


if __name__ == "__main__":
    main()
