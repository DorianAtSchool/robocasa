"""Merge task-VLM pretokenization shards into a final preprocessed artifact."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional in lightweight test envs
    tqdm = None

from data_generation.task_level.runtime.client import load_dotenv_file
from training.bc_task_vlm.dataset import (
    build_centralized_examples,
    build_decentralized_examples,
    build_same_task_trajectory_split,
    build_split_manifest,
)
from training.bc_task_vlm.preprocess import (
    _build_examples_for_tasks,
    _resolve_task_list,
    _resolve_training_samples_cache_dir,
    _resolve_validation_task_list,
    stable_sample_rank,
)
from training.bc_task_vlm.preprocessed_data import (
    _PREPROCESSED_FORMAT_VERSION,
    StreamedPreprocessedArtifactWriter,
    existing_artifact_image_relpaths_for_examples,
    iter_pretokenized_shard_split_records,
    normalize_pretokenization_metadata,
    pretokenized_shard_paths,
    push_preprocessed_artifact_to_hub,
    write_preprocessed_artifact_sidecars,
    _load_json_if_exists,
    _row_from_example,
)

_DEFAULT_DATASET_ROOT = Path("data_generation/task_level/data/image/20260413T205634Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--train-tasks",
        default="hot_dog_setup,prepare_sandwich_station,prepare_cheese_station,prepare_sausage_cheese",
    )
    parser.add_argument("--val-tasks", default=None)
    parser.add_argument(
        "--validation-split-mode",
        choices=("same-task", "task-holdout"),
        default="same-task",
    )
    parser.add_argument("--validation-trajectory-fraction", type=float, default=0.1)
    parser.add_argument("--validation-min-trajectories-per-task", type=int, default=1)
    parser.add_argument(
        "--use-example-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--training-samples-cache-dir", type=Path, default=None)
    parser.add_argument("--example-build-workers", type=int, default=1)
    parser.add_argument("--artifact-image-size", type=int, default=None)
    parser.add_argument(
        "--merge-flush-interval",
        type=int,
        default=400,
        help="Number of final artifact rows to buffer before flushing to disk.",
    )
    parser.add_argument(
        "--resume-existing-artifact-images",
        nargs="?",
        const="validated",
        choices=("validated", "unchecked"),
        default="unchecked",
        help=(
            "Reuse output images/ during merge. Defaults to unchecked because "
            "array pretokenization usually validates or builds images earlier."
        ),
    )
    parser.add_argument(
        "--skip-existing-artifact-image-validation",
        action="store_true",
        help="Alias for `--resume-existing-artifact-images unchecked`.",
    )
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--hub-revision", default=None)
    parser.add_argument(
        "--hub-private",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--hub-data-dir", default=None)
    return parser.parse_args()


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    if tqdm is not None:
        tqdm.write(formatted_message, file=sys.stderr)
        return
    print(formatted_message, file=sys.stderr, flush=True)


def _validate_output_dir_for_merge(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    unexpected_entries = [
        entry
        for entry in output_dir.iterdir()
        if entry.name not in {"dataset", "images", "pretokenized_shards"}
    ]
    if unexpected_entries:
        preview = ", ".join(
            str(entry.relative_to(output_dir)) for entry in unexpected_entries[:5]
        )
        if len(unexpected_entries) > 5:
            preview += f", ... ({len(unexpected_entries)} entries)"
        raise FileExistsError(
            "Cannot merge pretokenized shards into an output directory that "
            f"already contains final artifact files: {preview}"
        )


def _merge_split_blobs(
    *,
    split_name: str,
    shard_blobs_by_sample_id: list[dict[str, bytes]],
    expected_sample_ids: set[str],
    num_shards: int,
) -> dict[str, bytes]:
    merged: dict[str, bytes] = {}
    for shard_index, blobs_by_sample_id in enumerate(shard_blobs_by_sample_id):
        for sample_id, blob in blobs_by_sample_id.items():
            expected_shard_index = stable_sample_rank(sample_id) % num_shards
            if expected_shard_index != shard_index:
                raise ValueError(
                    f"{split_name} sample {sample_id!r} is in shard {shard_index}, "
                    f"but stable assignment expects shard {expected_shard_index}."
                )
            if sample_id in merged:
                raise ValueError(f"Duplicate {split_name} sample id: {sample_id}")
            merged[sample_id] = blob

    actual_sample_ids = set(merged)
    missing_sample_ids = sorted(expected_sample_ids.difference(actual_sample_ids))
    unexpected_sample_ids = sorted(actual_sample_ids.difference(expected_sample_ids))
    if missing_sample_ids or unexpected_sample_ids:
        details = []
        if missing_sample_ids:
            details.append(
                "missing "
                f"{len(missing_sample_ids)} ids, first: {missing_sample_ids[:5]}"
            )
        if unexpected_sample_ids:
            details.append(
                "unexpected "
                f"{len(unexpected_sample_ids)} ids, first: {unexpected_sample_ids[:5]}"
            )
        raise ValueError(
            f"Pretokenized shard coverage mismatch for {split_name}: "
            + "; ".join(details)
        )
    return merged


def _load_and_validate_shards(
    *,
    shard_dir: Path,
    num_shards: int,
    train_sample_ids: set[str],
    validation_sample_ids: set[str],
) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, Any]]:
    shard_train_blobs: list[dict[str, bytes]] = []
    shard_validation_blobs: list[dict[str, bytes]] = []
    pretokenization: dict[str, Any] | None = None

    for shard_index in range(num_shards):
        shard = load_pretokenized_shard(
            shard_dir=shard_dir,
            shard_index=shard_index,
        )
        if shard.shard_index != shard_index:
            raise ValueError(
                f"Shard manifest index mismatch for {shard_index}: "
                f"{shard.shard_index}"
            )
        if shard.num_shards != num_shards:
            raise ValueError(
                f"Shard {shard_index} reports {shard.num_shards} shards; "
                f"expected {num_shards}."
            )
        if pretokenization is None:
            pretokenization = shard.pretokenization
        elif shard.pretokenization != pretokenization:
            raise ValueError(
                f"Shard {shard_index} pretokenization metadata differs from "
                "previous shards."
            )
        shard_train_blobs.append(shard.train_blobs_by_sample_id)
        shard_validation_blobs.append(shard.validation_blobs_by_sample_id)

    if pretokenization is None:  # pragma: no cover - num_shards validation covers this
        raise ValueError("No pretokenization shards were loaded.")

    return (
        _merge_split_blobs(
            split_name="train",
            shard_blobs_by_sample_id=shard_train_blobs,
            expected_sample_ids=train_sample_ids,
            num_shards=num_shards,
        ),
        _merge_split_blobs(
            split_name="validation",
            shard_blobs_by_sample_id=shard_validation_blobs,
            expected_sample_ids=validation_sample_ids,
            num_shards=num_shards,
        ),
        pretokenization,
    )


def _load_and_validate_shard_manifests(
    *,
    shard_dir: Path,
    num_shards: int,
) -> dict[str, Any]:
    pretokenization: dict[str, Any] | None = None
    for shard_index in range(num_shards):
        manifest_path = pretokenized_shard_paths(
            shard_dir=shard_dir,
            shard_index=shard_index,
        )["manifest"]
        manifest = _load_json_if_exists(manifest_path)
        if manifest is None:
            raise FileNotFoundError(
                f"Missing pretokenized shard manifest: {manifest_path}"
            )
        if int(manifest["shard_index"]) != shard_index:
            raise ValueError(
                f"Shard manifest index mismatch for {shard_index}: "
                f"{manifest['shard_index']}"
            )
        if int(manifest["num_shards"]) != num_shards:
            raise ValueError(
                f"Shard {shard_index} reports {manifest['num_shards']} shards; "
                f"expected {num_shards}."
            )
        shard_pretokenization = normalize_pretokenization_metadata(
            dict(manifest["pretokenization"])
        )
        if pretokenization is None:
            pretokenization = shard_pretokenization
        elif shard_pretokenization != pretokenization:
            raise ValueError(
                f"Shard {shard_index} pretokenization metadata differs from "
                "previous shards."
            )
    if pretokenization is None:  # pragma: no cover - num_shards validation covers this
        raise ValueError("No pretokenization shards were loaded.")
    return pretokenization


def _stream_split_from_shards(
    *,
    split_name: str,
    examples_by_sample_id: dict[str, Any],
    expected_sample_ids: set[str],
    image_relpaths_by_source: dict[str, str],
    shard_dir: Path,
    num_shards: int,
    artifact_writer: StreamedPreprocessedArtifactWriter,
    flush_interval: int,
) -> None:
    seen_sample_ids = set(artifact_writer.completed_sample_ids(split_name))
    unknown_completed_ids = seen_sample_ids.difference(expected_sample_ids)
    if unknown_completed_ids:
        raise ValueError(
            f"Existing streamed {split_name} output contains unexpected sample ids: "
            f"{sorted(unknown_completed_ids)[:5]}"
        )
    pending_rows: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending_rows:
            return
        artifact_writer.write_split_rows(
            split_name=split_name,
            rows=list(pending_rows),
        )
        pending_rows.clear()

    for shard_index in range(num_shards):
        for sample_id, blob in iter_pretokenized_shard_split_records(
            shard_dir=shard_dir,
            shard_index=shard_index,
            split_name=split_name,
        ):
            expected_shard_index = stable_sample_rank(sample_id) % num_shards
            if expected_shard_index != shard_index:
                raise ValueError(
                    f"{split_name} sample {sample_id!r} is in shard {shard_index}, "
                    f"but stable assignment expects shard {expected_shard_index}."
                )
            if sample_id in seen_sample_ids:
                continue
            example = examples_by_sample_id.get(sample_id)
            if example is None:
                raise ValueError(f"Unexpected {split_name} sample id: {sample_id}")
            feature = example.to_feature_dict()
            pending_rows.append(
                _row_from_example(
                    example,
                    feature=feature,
                    image_relpaths_by_source=image_relpaths_by_source,
                    pretokenized_blob=blob,
                )
            )
            seen_sample_ids.add(sample_id)
            if len(pending_rows) >= flush_interval:
                flush_pending()
    flush_pending()

    missing_sample_ids = sorted(expected_sample_ids.difference(seen_sample_ids))
    if missing_sample_ids:
        raise ValueError(
            f"Pretokenized shard coverage mismatch for {split_name}: missing "
            f"{len(missing_sample_ids)} ids, first: {missing_sample_ids[:5]}"
        )


def _build_examples(args: argparse.Namespace, dataset_root: Path):
    train_tasks = _resolve_task_list(args.train_tasks)
    val_tasks = _resolve_validation_task_list(
        args.val_tasks,
        train_tasks=train_tasks,
        validation_split_mode=args.validation_split_mode,
    )
    train_trajectory_ids_by_task: dict[str, list[str]] | None = None
    val_trajectory_ids_by_task: dict[str, list[str]] | None = None
    if args.validation_split_mode == "same-task":
        trajectory_split = build_same_task_trajectory_split(
            dataset_root=dataset_root,
            train_task_names=train_tasks,
            validation_task_names=val_tasks,
            validation_fraction=args.validation_trajectory_fraction,
            min_validation_trajectories_per_task=(
                args.validation_min_trajectories_per_task
            ),
        )
        train_trajectory_ids_by_task = trajectory_split.train_trajectory_ids_by_task
        val_trajectory_ids_by_task = trajectory_split.validation_trajectory_ids_by_task
    else:
        overlapping_tasks = sorted(set(train_tasks).intersection(val_tasks))
        if overlapping_tasks:
            raise ValueError(
                "Train and validation tasks must be disjoint. "
                f"Overlap: {', '.join(overlapping_tasks)}."
            )

    training_samples_cache_dir = _resolve_training_samples_cache_dir(
        args.training_samples_cache_dir
    ).resolve()
    train_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=train_tasks,
        split_name="train",
        cache_granularity="train",
        builder=build_decentralized_examples,
        trajectory_ids_by_task=train_trajectory_ids_by_task,
        use_example_cache=args.use_example_cache,
        training_samples_cache_dir=training_samples_cache_dir,
        example_build_workers=args.example_build_workers,
    )
    validation_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=val_tasks,
        split_name="validation",
        cache_granularity="centralized",
        builder=build_centralized_examples,
        trajectory_ids_by_task=val_trajectory_ids_by_task,
        use_example_cache=args.use_example_cache,
        training_samples_cache_dir=training_samples_cache_dir,
        example_build_workers=args.example_build_workers,
    )
    return (
        train_tasks,
        val_tasks,
        train_trajectory_ids_by_task,
        val_trajectory_ids_by_task,
        training_samples_cache_dir,
        train_examples,
        validation_examples,
    )


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    if args.skip_existing_artifact_image_validation:
        args.resume_existing_artifact_images = "unchecked"
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if args.example_build_workers < 1:
        raise ValueError("--example-build-workers must be at least 1.")
    if args.artifact_image_size is not None and args.artifact_image_size < 1:
        raise ValueError("--artifact-image-size must be at least 1.")
    if args.merge_flush_interval < 1:
        raise ValueError("--merge-flush-interval must be at least 1.")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    shard_dir = args.shard_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    _validate_output_dir_for_merge(output_dir)

    _log(f"Loading examples for shard merge from {dataset_root}")
    (
        train_tasks,
        val_tasks,
        train_trajectory_ids_by_task,
        val_trajectory_ids_by_task,
        training_samples_cache_dir,
        train_examples,
        validation_examples,
    ) = _build_examples(args, dataset_root)
    split_manifest = build_split_manifest(
        dataset_root=dataset_root,
        train_examples=train_examples,
        val_examples=validation_examples,
    )
    _log("Validating " f"{args.num_shards} pretokenization shards from {shard_dir}")
    pretokenization_config = _load_and_validate_shard_manifests(
        shard_dir=shard_dir,
        num_shards=args.num_shards,
    )

    image_relpaths_by_source = None
    if (output_dir / "images").is_dir():
        image_relpaths_by_source = existing_artifact_image_relpaths_for_examples(
            examples=list(train_examples) + list(validation_examples),
            output_dir=output_dir,
            validate=args.resume_existing_artifact_images == "validated",
            progress_callback=_log,
        )
    artifact_image_size = (
        None
        if args.artifact_image_size is None
        else (args.artifact_image_size, args.artifact_image_size)
    )
    if image_relpaths_by_source is None:
        from training.bc_task_vlm.preprocessed_data import (
            stage_artifact_images_for_examples,
        )

        image_relpaths_by_source = stage_artifact_images_for_examples(
            examples=list(train_examples) + list(validation_examples),
            output_dir=output_dir,
            artifact_image_size=artifact_image_size,
            progress_callback=_log,
        )
    preprocess_config = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "command": " ".join(sys.argv),
        "dataset_root": str(dataset_root),
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "validation_split_mode": args.validation_split_mode,
        "validation_trajectory_fraction": args.validation_trajectory_fraction,
        "validation_min_trajectories_per_task": (
            args.validation_min_trajectories_per_task
        ),
        "train_trajectory_ids_by_task": train_trajectory_ids_by_task,
        "val_trajectory_ids_by_task": val_trajectory_ids_by_task,
        "train_example_format": "single_turn",
        "use_example_cache": args.use_example_cache,
        "training_samples_cache_dir": str(training_samples_cache_dir),
        "example_build_workers": args.example_build_workers,
        "preprocessed_format_version": _PREPROCESSED_FORMAT_VERSION,
        "artifact_image_size": args.artifact_image_size,
        "pretokenized": True,
        "pretokenization": pretokenization_config,
        "pretokenize_num_shards": args.num_shards,
        "pretokenize_shard_dir": str(shard_dir),
        "push_to_hub": args.push_to_hub,
        "hub_revision": args.hub_revision,
        "hub_private": args.hub_private,
        "hub_data_dir": args.hub_data_dir,
    }
    _log("Streaming merged preprocessed artifact")
    with StreamedPreprocessedArtifactWriter(
        output_dir=output_dir,
        progress_callback=_log,
    ) as artifact_writer:
        _stream_split_from_shards(
            split_name="train",
            examples_by_sample_id={
                example.sample_id: example for example in train_examples
            },
            expected_sample_ids={example.sample_id for example in train_examples},
            image_relpaths_by_source=image_relpaths_by_source,
            shard_dir=shard_dir,
            num_shards=args.num_shards,
            artifact_writer=artifact_writer,
            flush_interval=args.merge_flush_interval,
        )
        _stream_split_from_shards(
            split_name="validation",
            examples_by_sample_id={
                example.sample_id: example for example in validation_examples
            },
            expected_sample_ids={example.sample_id for example in validation_examples},
            image_relpaths_by_source=image_relpaths_by_source,
            shard_dir=shard_dir,
            num_shards=args.num_shards,
            artifact_writer=artifact_writer,
            flush_interval=args.merge_flush_interval,
        )
    write_preprocessed_artifact_sidecars(
        output_dir=output_dir,
        split_manifest=split_manifest,
        preprocess_config=preprocess_config,
        progress_callback=_log,
    )

    if args.push_to_hub:
        _log(
            f"Pushing preprocessed artifact to https://huggingface.co/datasets/{args.push_to_hub}"
        )
        push_preprocessed_artifact_to_hub(
            output_dir=output_dir,
            repo_id=args.push_to_hub,
            split_manifest=split_manifest,
            preprocess_config=preprocess_config,
            private=args.hub_private,
            revision=args.hub_revision,
            data_dir=args.hub_data_dir,
        )
    _log("Pretokenization shard merge complete")


if __name__ == "__main__":
    main()
