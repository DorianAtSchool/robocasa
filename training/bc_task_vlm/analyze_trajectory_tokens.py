"""Count Qwen multimodal tokens by trajectory and task."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from training.bc_task_vlm.preprocessed_data import (
    PreprocessedFeatureDataset,
    load_preprocessed_artifact_from_disk,
)
from training.bc_task_vlm.dataset import image_resolution_to_pixels


@dataclass
class TokenCounts:
    samples: int = 0
    images: int = 0
    total_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    max_sample_tokens: int = 0

    def add(self, *, total_tokens: int, image_tokens: int, image_count: int) -> None:
        self.samples += 1
        self.images += image_count
        self.total_tokens += total_tokens
        self.image_tokens += image_tokens
        self.text_tokens += total_tokens - image_tokens
        self.max_sample_tokens = max(self.max_sample_tokens, total_tokens)

    def row(self, **keys: Any) -> dict[str, Any]:
        avg_sample_tokens = self.total_tokens / self.samples if self.samples else 0.0
        avg_image_tokens = self.image_tokens / self.images if self.images else 0.0
        return {
            **keys,
            **asdict(self),
            "avg_sample_tokens": round(avg_sample_tokens, 2),
            "avg_image_tokens_per_image": round(avg_image_tokens, 2),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-data-dir",
        type=Path,
        default=Path("training/bc_task_vlm/preprocessed/task_vlm_bc_v3"),
    )
    parser.add_argument(
        "--processor-name-or-path",
        default="Qwen/Qwen3.5-0.8B",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="all",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional comma-separated task filter.",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _split_csv(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def _load_processor(args: argparse.Namespace):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.processor_name_or_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
    return processor


def _iter_split_features(args: argparse.Namespace):
    print(f"loading artifact: {args.preprocessed_data_dir}", flush=True)
    artifact = load_preprocessed_artifact_from_disk(args.preprocessed_data_dir)
    print(
        "loaded artifact "
        f"(train={len(artifact.train_split)}, validation={len(artifact.validation_split)})",
        flush=True,
    )
    splits = (
        ("train", artifact.train_split),
        ("validation", artifact.validation_split),
    )
    task_filter = _split_csv(args.tasks)
    seen_by_task: dict[tuple[str, str], int] = defaultdict(int)

    for split_name, split_data in splits:
        if args.split != "all" and split_name != args.split:
            continue
        manifest_tasks = set(
            artifact.split_manifest.get(split_name, {}).get("tasks", {})
        )
        target_tasks = (
            manifest_tasks if task_filter is None else manifest_tasks & task_filter
        )
        dataset = PreprocessedFeatureDataset(
            split_data,
            artifact_root=artifact.artifact_root,
        )
        for index in range(len(dataset)):
            feature = dataset[index]
            task_name = feature["task_name"]
            if task_filter is not None and task_name not in task_filter:
                continue
            counter_key = (split_name, task_name)
            if (
                args.max_samples_per_task is not None
                and seen_by_task[counter_key] >= args.max_samples_per_task
            ):
                continue
            seen_by_task[counter_key] += 1
            yield split_name, feature
            if args.max_samples_per_task is not None and target_tasks:
                if all(
                    seen_by_task[(split_name, target_task)] >= args.max_samples_per_task
                    for target_task in target_tasks
                ):
                    break


def _image_sources(feature: dict[str, Any]) -> list[Any]:
    if "image_paths" in feature:
        return list(feature["image_paths"])
    return list(feature.get("images", ()))


def _image_size(image_source: Any) -> tuple[int, int]:
    from PIL import Image

    if isinstance(image_source, dict):
        if image_source.get("bytes") is not None:
            with Image.open(BytesIO(bytes(image_source["bytes"]))) as image:
                width, height = image.size
                return height, width
        if image_source.get("path"):
            image_source = image_source["path"]

    with Image.open(image_source) as image:
        width, height = image.size
        return height, width


def _image_token_counts(
    processor,
    *,
    image_sources: list[Any],
    image_resolution: int | None,
) -> list[int]:
    if not image_sources:
        return []

    if not hasattr(processor, "_get_num_multimodal_tokens"):
        raise TypeError(
            "Processor does not expose _get_num_multimodal_tokens; "
            "use a Qwen VL processor."
        )

    image_sizes = [_image_size(image_source) for image_source in image_sources]
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
) -> tuple[int, int, int]:
    image_sources = _image_sources(feature)
    image_counts = _image_token_counts(
        processor,
        image_sources=image_sources,
        image_resolution=image_resolution,
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
    encoded = processor.tokenizer(
        text,
        return_attention_mask=True,
    )
    total_tokens = int(sum(encoded["attention_mask"]))
    return total_tokens, sum(image_counts), len(image_sources)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading processor: {args.processor_name_or_path}", flush=True)
    processor = _load_processor(args)
    print("loaded processor", flush=True)
    by_trajectory: dict[tuple[str, str, str], TokenCounts] = defaultdict(TokenCounts)
    by_task: dict[tuple[str, str], TokenCounts] = defaultdict(TokenCounts)

    for index, (split_name, feature) in enumerate(_iter_split_features(args), start=1):
        total_tokens, image_tokens, image_count = _count_feature_tokens(
            processor,
            feature,
            image_resolution=args.image_resolution,
        )
        task_key = (split_name, feature["task_name"])
        trajectory_key = (
            split_name,
            feature["task_name"],
            feature["trajectory_id"],
        )
        by_task[task_key].add(
            total_tokens=total_tokens,
            image_tokens=image_tokens,
            image_count=image_count,
        )
        by_trajectory[trajectory_key].add(
            total_tokens=total_tokens,
            image_tokens=image_tokens,
            image_count=image_count,
        )
        if index % 1000 == 0:
            print(f"processed {index} samples", flush=True)

    task_rows = [
        counts.row(split=split_name, task_name=task_name)
        for (split_name, task_name), counts in sorted(by_task.items())
    ]
    trajectory_rows = [
        counts.row(
            split=split_name,
            task_name=task_name,
            trajectory_id=trajectory_id,
        )
        for (split_name, task_name, trajectory_id), counts in sorted(
            by_trajectory.items()
        )
    ]

    _write_csv(args.output_dir / "task_token_summary.csv", task_rows)
    _write_csv(args.output_dir / "trajectory_token_summary.csv", trajectory_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "preprocessed_data_dir": str(args.preprocessed_data_dir),
                "processor_name_or_path": args.processor_name_or_path,
                "image_resolution": args.image_resolution,
                "split": args.split,
                "tasks": sorted(_split_csv(args.tasks) or []),
                "max_samples_per_task": args.max_samples_per_task,
                "task_summary": task_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"wrote {args.output_dir / 'task_token_summary.csv'}")
    print(f"wrote {args.output_dir / 'trajectory_token_summary.csv'}")
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
