"""Estimate average multimodal tokens per trajectory for task-level data."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from training.bc_task_vlm.dataset import (
    build_centralized_examples,
    build_decentralized_examples,
    image_resolution_to_pixels,
)
from training.bc_task_vlm.task_registry import resolve_task_name, supported_task_names


@dataclass(frozen=True)
class TaskTokenSummary:
    task_name: str
    sampled_trajectories: int
    examples: int
    mean_tokens_per_trajectory: float
    mean_image_tokens_per_trajectory: float
    mean_images_per_trajectory: float
    stddev_tokens_per_trajectory: float
    min_tokens_per_trajectory: int
    max_tokens_per_trajectory: int


@dataclass(frozen=True)
class FeatureTokenCounts:
    total_tokens: int
    image_tokens: int
    images: int


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_task_list(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return list(supported_task_names())
    return [resolve_task_name(task_name) for task_name in _split_csv(raw_value)]


def _load_processor(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
    return processor


def _image_sources(feature: dict[str, Any]) -> list[Any]:
    if "image_paths" in feature:
        return list(feature["image_paths"])
    return list(feature.get("images", ()))


def _image_cache_key(image_source: Any) -> str | None:
    if isinstance(image_source, dict):
        if image_source.get("path"):
            return str(image_source["path"])
        return None
    return str(image_source)


def _image_size(
    image_source: Any,
    *,
    image_size_cache: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    from PIL import Image

    cache_key = _image_cache_key(image_source)
    if cache_key is not None and cache_key in image_size_cache:
        return image_size_cache[cache_key]

    if isinstance(image_source, dict):
        if image_source.get("bytes") is not None:
            with Image.open(BytesIO(bytes(image_source["bytes"]))) as image:
                width, height = image.size
        elif image_source.get("path"):
            with Image.open(image_source["path"]) as image:
                width, height = image.size
        else:
            raise ValueError("Image dictionary must contain either bytes or path.")
    else:
        with Image.open(image_source) as image:
            width, height = image.size

    size = (height, width)
    if cache_key is not None:
        image_size_cache[cache_key] = size
    return size


def _image_token_counts(
    processor,
    *,
    image_sources: list[Any],
    image_resolution: int | None,
    image_size_cache: dict[str, tuple[int, int]],
) -> list[int]:
    if not image_sources:
        return []

    if not hasattr(processor, "_get_num_multimodal_tokens"):
        raise TypeError(
            "Processor does not expose _get_num_multimodal_tokens; this estimator "
            "currently supports Qwen VL processors."
        )

    image_sizes = [
        _image_size(image_source, image_size_cache=image_size_cache)
        for image_source in image_sources
    ]
    image_pixels = image_resolution_to_pixels(image_resolution)
    token_data = processor._get_num_multimodal_tokens(
        image_sizes=image_sizes,
        min_pixels=image_pixels,
        max_pixels=image_pixels,
    )
    if isinstance(token_data, dict):
        return [int(value) for value in token_data["num_image_tokens"]]
    return [int(value) for value in token_data.num_image_tokens]


def _expand_image_placeholders(
    processor,
    *,
    text: str,
    image_token_counts: list[int],
) -> str:
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    for count in image_token_counts:
        text = text.replace(image_token, image_token * count, 1)
    return text


def _count_feature_tokens(
    processor,
    feature: dict[str, Any],
    *,
    image_resolution: int | None,
    image_size_cache: dict[str, tuple[int, int]],
) -> FeatureTokenCounts:
    image_sources = _image_sources(feature)
    image_counts = _image_token_counts(
        processor,
        image_sources=image_sources,
        image_resolution=image_resolution,
        image_size_cache=image_size_cache,
    )
    text = processor.apply_chat_template(
        feature["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    text = _expand_image_placeholders(
        processor,
        text=text,
        image_token_counts=image_counts,
    )
    encoded = processor.tokenizer(text, return_attention_mask=True)
    return FeatureTokenCounts(
        total_tokens=int(sum(encoded["attention_mask"])),
        image_tokens=sum(image_counts),
        images=len(image_sources),
    )


def _select_trajectory_ids(
    *,
    dataset_root: Path,
    task_name: str,
    trajectories_per_task: int,
) -> list[str]:
    task_root = dataset_root / task_name
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_root}")
    trajectory_ids = sorted(path.name for path in task_root.iterdir() if path.is_dir())
    return trajectory_ids[:trajectories_per_task]


def _build_examples(
    *,
    dataset_root: Path,
    task_name: str,
    trajectory_ids: list[str],
    granularity: str,
) -> list[Any]:
    builder = (
        build_decentralized_examples
        if granularity == "decentralized"
        else build_centralized_examples
    )
    return builder(
        dataset_root=dataset_root,
        task_names=[task_name],
        trajectory_ids_by_task={task_name: trajectory_ids},
        num_workers=1,
        show_progress=False,
    )


def summarize_task_tokens(
    *,
    processor,
    dataset_root: Path,
    task_name: str,
    trajectories_per_task: int,
    granularity: str,
    image_resolution: int | None,
    image_size_cache: dict[str, tuple[int, int]],
) -> TaskTokenSummary:
    trajectory_ids = _select_trajectory_ids(
        dataset_root=dataset_root,
        task_name=task_name,
        trajectories_per_task=trajectories_per_task,
    )
    examples = _build_examples(
        dataset_root=dataset_root,
        task_name=task_name,
        trajectory_ids=trajectory_ids,
        granularity=granularity,
    )
    tokens_by_trajectory = {trajectory_id: 0 for trajectory_id in trajectory_ids}
    image_tokens_by_trajectory = {trajectory_id: 0 for trajectory_id in trajectory_ids}
    images_by_trajectory = {trajectory_id: 0 for trajectory_id in trajectory_ids}

    for example in examples:
        feature = example.to_feature_dict()
        trajectory_id = feature["trajectory_id"]
        counts = _count_feature_tokens(
            processor,
            feature,
            image_resolution=image_resolution,
            image_size_cache=image_size_cache,
        )
        tokens_by_trajectory[trajectory_id] += counts.total_tokens
        image_tokens_by_trajectory[trajectory_id] += counts.image_tokens
        images_by_trajectory[trajectory_id] += counts.images

    trajectory_totals = list(tokens_by_trajectory.values())
    trajectory_image_totals = list(image_tokens_by_trajectory.values())
    trajectory_image_counts = list(images_by_trajectory.values())
    if not trajectory_totals:
        return TaskTokenSummary(
            task_name=task_name,
            sampled_trajectories=0,
            examples=0,
            mean_tokens_per_trajectory=0.0,
            mean_image_tokens_per_trajectory=0.0,
            mean_images_per_trajectory=0.0,
            stddev_tokens_per_trajectory=0.0,
            min_tokens_per_trajectory=0,
            max_tokens_per_trajectory=0,
        )

    return TaskTokenSummary(
        task_name=task_name,
        sampled_trajectories=len(trajectory_totals),
        examples=len(examples),
        mean_tokens_per_trajectory=mean(trajectory_totals),
        mean_image_tokens_per_trajectory=mean(trajectory_image_totals),
        mean_images_per_trajectory=mean(trajectory_image_counts),
        stddev_tokens_per_trajectory=stdev(trajectory_totals)
        if len(trajectory_totals) > 1
        else 0.0,
        min_tokens_per_trajectory=min(trajectory_totals),
        max_tokens_per_trajectory=max(trajectory_totals),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--trajectories-per-task", type=int, default=20)
    parser.add_argument(
        "--granularity",
        choices=("decentralized", "centralized"),
        default="decentralized",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _print_table(
    *,
    model_name_or_path: str,
    dataset_root: Path,
    trajectories_per_task: int,
    granularity: str,
    summaries: list[TaskTokenSummary],
) -> None:
    print(f"model: {model_name_or_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"granularity: {granularity}")
    print(f"trajectories_per_task: {trajectories_per_task}")
    print()

    headers = [
        "task_name",
        "trajectories",
        "examples",
        "avg_tokens_per_trajectory",
        "avg_image_tokens_per_trajectory",
        "avg_images_per_trajectory",
        "stddev",
        "min",
        "max",
    ]
    rows = [
        [
            summary.task_name,
            str(summary.sampled_trajectories),
            str(summary.examples),
            f"{summary.mean_tokens_per_trajectory:.2f}",
            f"{summary.mean_image_tokens_per_trajectory:.2f}",
            f"{summary.mean_images_per_trajectory:.2f}",
            f"{summary.stddev_tokens_per_trajectory:.2f}",
            str(summary.min_tokens_per_trajectory),
            str(summary.max_tokens_per_trajectory),
        ]
        for summary in summaries
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    if args.trajectories_per_task < 1:
        raise ValueError("--trajectories-per-task must be at least 1.")

    tasks = _resolve_task_list(args.tasks)
    print(f"loading processor: {args.model_name_or_path}", file=sys.stderr, flush=True)
    processor = _load_processor(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print("loaded processor", file=sys.stderr, flush=True)
    image_size_cache: dict[str, tuple[int, int]] = {}
    summaries = []
    for task_name in tasks:
        print(f"processing task: {task_name}", file=sys.stderr, flush=True)
        summaries.append(
            summarize_task_tokens(
                processor=processor,
                dataset_root=args.dataset_root,
                task_name=task_name,
                trajectories_per_task=args.trajectories_per_task,
                granularity=args.granularity,
                image_resolution=args.image_resolution,
                image_size_cache=image_size_cache,
            )
        )
    _print_table(
        model_name_or_path=args.model_name_or_path,
        dataset_root=args.dataset_root,
        trajectories_per_task=args.trajectories_per_task,
        granularity=args.granularity,
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
