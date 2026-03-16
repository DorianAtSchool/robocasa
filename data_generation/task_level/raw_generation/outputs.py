"""Resolve output paths and build persisted generation payload sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_generation.task_level.raw_generation.config import (
    COST_SUMMARY_OUTPUT_FILENAME,
    DATASET_RUN_TIMESTAMP_FORMAT,
    DEFAULT_OUTPUT_DIR,
    ERROR_SUMMARY_OUTPUT_FILENAME,
    REQUEST_DIRECTORY_NAME,
    RuntimeConfig,
    SUMMARY_OUTPUT_FILENAME,
    TRAJECTORY_DIRECTORY_NAME,
)
from data_generation.task_level.raw_generation.costs import (
    _append_sampling_cost_note,
    _build_cost_summary_from_generation_usages,
)
from data_generation.task_level.raw_generation.errors import (
    _collect_payload_error_events,
    _error_event_key,
)
from data_generation.task_level.raw_generation.runtime_support import (
    _global_trajectory_index,
    format_trajectory_id,
)
from data_generation.task_level.runtime.client import (
    COST_DECIMAL_PLACES,
    TrajectoryGenerationError,
)
from data_generation.utils import camel_to_snake_case, round_cost, write_json_output


@dataclass(frozen=True)
class OutputPaths:
    summary_path: Path
    trajectory_dir: Path
    prompt_dir: Path
    output_dir: Path
    cost_path: Path
    error_summary_path: Path


def resolve_cost_output_path(
    summary_path: Path,
    cost_output_path: Path | None = None,
) -> Path:
    if cost_output_path is not None:
        return cost_output_path
    return summary_path.with_name(COST_SUMMARY_OUTPUT_FILENAME)


def resolve_error_output_path(summary_path: Path) -> Path:
    """Resolves the default error-summary sidecar path for one run."""

    return summary_path.with_name(ERROR_SUMMARY_OUTPUT_FILENAME)


def resolve_dataset_output_path(
    composite_task: str,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Resolves the default single-task summary output path."""

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    task_dir = camel_to_snake_case(composite_task)
    return (
        DEFAULT_OUTPUT_DIR
        / task_dir
        / timestamp.strftime(DATASET_RUN_TIMESTAMP_FORMAT)
        / SUMMARY_OUTPUT_FILENAME
    )


def resolve_request_output_path(
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Resolves the request-level combined summary path for multi-task generation."""

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return (
        DEFAULT_OUTPUT_DIR
        / REQUEST_DIRECTORY_NAME
        / timestamp.strftime(DATASET_RUN_TIMESTAMP_FORMAT)
        / SUMMARY_OUTPUT_FILENAME
    )


def resolve_request_task_output_path(
    request_summary_path: Path,
    composite_task: str,
) -> Path:
    """Resolves one per-task summary path nested under a shared request directory."""

    return (
        request_summary_path.parent
        / camel_to_snake_case(composite_task)
        / SUMMARY_OUTPUT_FILENAME
    )


def _resolve_summary_path(runtime_config: RuntimeConfig) -> Path:
    """Returns one stable summary path for the current generation run."""

    if runtime_config.summary_path is not None:
        return runtime_config.summary_path
    return resolve_dataset_output_path(runtime_config.composite_task)


def resolve_trajectory_output_dir(output_path: Path) -> Path:
    return output_path.parent / TRAJECTORY_DIRECTORY_NAME


def resolve_prompt_output_dir(output_path: Path) -> Path:
    """Resolves the sibling prompt output directory for one dataset summary."""

    return output_path.parent / "prompts"


def resolve_raw_output_dir(output_path: Path) -> Path:
    """Resolves the sibling raw-output directory for one dataset summary."""

    return output_path.parent / "outputs"


def _trajectory_output_filename(trajectory_id: str) -> str:
    return f"{trajectory_id}.json"


def _prompt_output_filename(trajectory_id: str) -> str:
    """Formats one prompt sidecar filename to match its trajectory basename."""

    return f"{trajectory_id}.md"


def _attempt_prompt_output_filename(
    run_id: str,
    attempt_number: int,
) -> str:
    """Formats one per-attempt prompt sidecar filename."""

    return f"{run_id}_{attempt_number}.md"


def format_attempt_prompt_owner_id(
    runtime_config: RuntimeConfig,
    *,
    run_index: int,
) -> str:
    """Maps one run's attempt prompt to the first saved trajectory ID from that run."""

    return format_trajectory_id(
        _global_trajectory_index(
            runtime_config,
            run_index=run_index,
            candidate_index=0,
        )
    )


def _raw_output_filename(trajectory_id: str) -> str:
    """Formats one raw-output sidecar filename using the trajectory basename."""

    return f"{trajectory_id}.txt"


def _payload_run_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    run_metadata = {
        "composite_task": payload["composite_task"],
        "sdk": payload["sdk"],
        "model": payload["model"],
        "num_runs": payload.get("num_runs", payload["num_trajectories"]),
        "num_trajectories": payload["num_trajectories"],
        "generated_at": payload["generated_at"],
    }
    if "model_config" in payload:
        run_metadata["model_config"] = payload["model_config"]
    return run_metadata


def _build_model_config_payload(runtime_config: RuntimeConfig) -> dict[str, Any]:
    """Serializes request-time model settings for dataset metadata."""

    sampling_payload = {
        "temperature": runtime_config.temperature,
        "strategy": runtime_config.sampling,
    }
    if runtime_config.sampling == "verbalized":
        sampling_payload["verbalized_k"] = runtime_config.verbalized_k

    return {
        "reasoning": {
            "thinking_level": runtime_config.thinking_level,
        },
        "sampling": sampling_payload,
    }


def _summary_trajectory_entry(trajectory: dict[str, Any]) -> dict[str, str]:
    trajectory_id = trajectory["trajectory_id"]
    return {
        "trajectory_id": trajectory_id,
        "path": (
            Path(TRAJECTORY_DIRECTORY_NAME) / _trajectory_output_filename(trajectory_id)
        ).as_posix(),
    }


def _summary_trajectory_stats(
    trajectories: list[dict[str, Any]],
) -> dict[str, int | float]:
    """Builds aggregate completion and validation stats for summary metadata."""

    completed_trajectories = len(trajectories)
    invalid_trajectories = 0
    successful_trajectories = 0

    for trajectory in trajectories:
        validation = trajectory.get("validation")

        # Older payloads used in tests may not include validation metadata.
        if not isinstance(validation, dict):
            successful_trajectories += 1
            continue

        if validation.get("is_valid") is False:
            invalid_trajectories += 1
            continue

        successful_trajectories += 1

    successful_trajectory_fraction = 0.0
    if completed_trajectories > 0:
        successful_trajectory_fraction = (
            successful_trajectories / completed_trajectories
        )

    return {
        "completed_trajectories": completed_trajectories,
        "invalid_trajectories": invalid_trajectories,
        "successful_trajectory_fraction": successful_trajectory_fraction,
    }


def _trajectory_cost_entry(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "generation_usage": trajectory["generation_usage"],
    }


def build_summary_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = _payload_run_metadata(payload)
    if "cost_summary" in payload:
        summary_payload["cost_summary"] = payload["cost_summary"]
    summary_payload.update(_summary_trajectory_stats(payload["trajectories"]))
    summary_payload["trajectory_directory"] = TRAJECTORY_DIRECTORY_NAME
    summary_payload["trajectory_files"] = [
        _summary_trajectory_entry(trajectory) for trajectory in payload["trajectories"]
    ]
    return summary_payload


def write_trajectory_output_payloads(
    trajectories: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    written_paths: list[Path] = []
    for trajectory in trajectories:
        output_path = output_dir / _trajectory_output_filename(
            trajectory["trajectory_id"]
        )
        write_json_output(
            {
                key: value
                for key, value in trajectory.items()
                if key not in {"prompt", "raw_output"}
            },
            output_path,
        )
        written_paths.append(output_path)
    return written_paths


def write_prompt_output_payloads(
    trajectory_prompts: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    """Writes one prompt sidecar per trajectory using matching trajectory IDs."""

    written_paths: list[Path] = []
    for prompt_entry in trajectory_prompts:
        output_path = output_dir / _prompt_output_filename(
            prompt_entry["trajectory_id"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(prompt_entry["prompt"], encoding="utf-8")
        written_paths.append(output_path)
    return written_paths


def write_attempt_prompt_output_payloads(
    attempt_prompts: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Writes one prompt sidecar per attempted run prompt."""

    written_paths: list[Path] = []
    for prompt_entry in attempt_prompts:
        if prompt_entry["attempt_number"] <= 1:
            continue
        output_path = output_dir / _attempt_prompt_output_filename(
            prompt_entry["run_id"],
            prompt_entry["attempt_number"],
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(prompt_entry["prompt"], encoding="utf-8")
        written_paths.append(output_path)
    return written_paths


def _raw_output_text(raw_output: Any) -> str:
    """Serializes the stored raw model output into a text sidecar."""

    if isinstance(raw_output, str):
        return raw_output
    return json.dumps(raw_output, indent=2, sort_keys=True)


def write_raw_output_payloads(
    trajectory_outputs: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Writes one raw model output sidecar per trajectory."""

    written_paths: list[Path] = []
    for output_entry in trajectory_outputs:
        output_path = output_dir / _raw_output_filename(output_entry["trajectory_id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _raw_output_text(output_entry["raw_output"]),
            encoding="utf-8",
        )
        written_paths.append(output_path)
    return written_paths


def build_cost_output_payload(
    payload: dict[str, Any],
    *,
    trajectory_output_path: Path | None = None,
) -> dict[str, Any]:
    # Keep the sidecar small enough to inspect without loading full trajectories.
    cost_payload = {
        **_payload_run_metadata(payload),
        "trajectory_output_path": (
            str(trajectory_output_path) if trajectory_output_path is not None else None
        ),
        "trajectory_directory": (
            str(resolve_trajectory_output_dir(trajectory_output_path))
            if trajectory_output_path is not None
            else None
        ),
        "trajectory_costs": [
            _trajectory_cost_entry(trajectory) for trajectory in payload["trajectories"]
        ],
    }
    if "cost_summary" in payload:
        cost_payload["cost_summary"] = payload["cost_summary"]
    return cost_payload


def build_error_summary_output_payload(
    payload: dict[str, Any],
    *,
    trajectory_output_path: Path | None = None,
) -> dict[str, Any]:
    """Builds the sidecar payload that aggregates all observed run errors."""

    error_events = _collect_payload_error_events(payload)
    error_counts_by_type: dict[str, int] = {}
    distinct_error_counts: dict[tuple[str, str], int] = {}

    for error_event in error_events:
        error_type = error_event["error_type"]
        error_counts_by_type[error_type] = error_counts_by_type.get(error_type, 0) + 1
        error_message = error_event.get("message", "")
        distinct_key = (error_type, error_message)
        distinct_error_counts[distinct_key] = (
            distinct_error_counts.get(distinct_key, 0) + 1
        )

    error_payload = {
        **_payload_run_metadata(payload),
        "trajectory_output_path": (
            str(trajectory_output_path) if trajectory_output_path is not None else None
        ),
        "trajectory_directory": (
            str(resolve_trajectory_output_dir(trajectory_output_path))
            if trajectory_output_path is not None
            else None
        ),
        "total_errors": len(error_events),
        "possible_errors": sorted(error_counts_by_type),
        "error_counts_by_type": [
            {
                "error_type": error_type,
                "count": error_counts_by_type[error_type],
            }
            for error_type in sorted(error_counts_by_type)
        ],
        "distinct_errors": [
            {
                "error_type": error_type,
                "message": error_message,
                "summary": (
                    f"{error_type}: {error_message}" if error_message else error_type
                ),
                "count": distinct_error_counts[(error_type, error_message)],
            }
            for error_type, error_message in sorted(distinct_error_counts)
        ],
        "error_events": error_events,
    }
    return error_payload


def _relative_output_path(path: Path, *, root: Path) -> str:
    """Formats one output path relative to the shared request root."""

    return path.relative_to(root).as_posix()


def _aggregate_task_cost_summaries(
    task_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Builds one combined cost summary from per-task payload summaries."""

    combined_notes: list[str] = []
    pricing_payloads: list[dict[str, Any]] = []
    total_trajectories = sum(
        int(task_payload.get("num_trajectories", 0)) for task_payload in task_payloads
    )
    aggregated_cost_summary = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "average_trajectory_cost_usd": None,
        "notes": combined_notes,
    }

    missing_cost_data = False
    for task_payload in task_payloads:
        task_cost_summary = task_payload.get("cost_summary", {})
        if not isinstance(task_cost_summary, dict):
            missing_cost_data = True
            continue
        aggregated_cost_summary["prompt_tokens"] += int(
            task_cost_summary.get("prompt_tokens", 0)
        )
        aggregated_cost_summary["output_tokens"] += int(
            task_cost_summary.get("output_tokens", 0)
        )
        aggregated_cost_summary["reasoning_tokens"] += int(
            task_cost_summary.get("reasoning_tokens", 0)
        )
        aggregated_cost_summary["total_tokens"] += int(
            task_cost_summary.get("total_tokens", 0)
        )

        for cost_key in ("input_cost_usd", "output_cost_usd", "total_cost_usd"):
            task_cost_value = task_cost_summary.get(cost_key)
            if task_cost_value is None:
                missing_cost_data = True
                continue
            aggregated_cost_summary[cost_key] += float(task_cost_value)

        for note in task_cost_summary.get("notes", []):
            if isinstance(note, str) and note not in combined_notes:
                combined_notes.append(note)

        pricing_payload = task_cost_summary.get("pricing")
        if isinstance(pricing_payload, dict):
            pricing_payloads.append(pricing_payload)

    if missing_cost_data:
        aggregated_cost_summary["input_cost_usd"] = None
        aggregated_cost_summary["output_cost_usd"] = None
        aggregated_cost_summary["total_cost_usd"] = None
        aggregated_cost_summary["average_trajectory_cost_usd"] = None
    else:
        aggregated_cost_summary["input_cost_usd"] = round_cost(
            aggregated_cost_summary["input_cost_usd"],
            decimal_places=COST_DECIMAL_PLACES,
        )
        aggregated_cost_summary["output_cost_usd"] = round_cost(
            aggregated_cost_summary["output_cost_usd"],
            decimal_places=COST_DECIMAL_PLACES,
        )
        aggregated_cost_summary["total_cost_usd"] = round_cost(
            aggregated_cost_summary["total_cost_usd"],
            decimal_places=COST_DECIMAL_PLACES,
        )
        aggregated_cost_summary["average_trajectory_cost_usd"] = round_cost(
            aggregated_cost_summary["total_cost_usd"] / total_trajectories
            if total_trajectories > 0
            and aggregated_cost_summary["total_cost_usd"] is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        )

    unique_pricing_payloads = {
        json.dumps(pricing_payload, sort_keys=True)
        for pricing_payload in pricing_payloads
    }
    if len(unique_pricing_payloads) == 1:
        aggregated_cost_summary["pricing"] = pricing_payloads[0]

    return aggregated_cost_summary


def build_request_summary_output_payload(
    runtime_config: RuntimeConfig,
    task_run_entries: list[dict[str, Any]],
    *,
    request_summary_path: Path,
) -> dict[str, Any]:
    """Builds the combined summary payload for one multi-task generation request."""

    task_payloads = [task_run_entry["payload"] for task_run_entry in task_run_entries]
    summary_payload = {
        "composite_tasks": [
            task_run_entry["composite_task"] for task_run_entry in task_run_entries
        ],
        "sdk": runtime_config.sdk,
        "model": runtime_config.model,
        "model_config": _build_model_config_payload(runtime_config),
        "num_tasks": len(task_run_entries),
        "num_runs_per_task": runtime_config.num_runs,
        "total_requested_runs": runtime_config.num_runs * len(task_run_entries),
        "num_trajectories": sum(
            int(task_payload.get("num_trajectories", 0))
            for task_payload in task_payloads
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_summary": _aggregate_task_cost_summaries(task_payloads),
        "task_summaries": [],
    }

    request_root = request_summary_path.parent
    for task_run_entry in task_run_entries:
        task_payload = task_run_entry["payload"]
        output_paths = task_run_entry["output_paths"]
        task_summary_entry = {
            "composite_task": task_run_entry["composite_task"],
            "summary_path": _relative_output_path(
                output_paths.summary_path, root=request_root
            ),
            "cost_summary_path": _relative_output_path(
                output_paths.cost_path, root=request_root
            ),
            "error_summary_path": _relative_output_path(
                output_paths.error_summary_path, root=request_root
            ),
            "trajectory_directory": _relative_output_path(
                output_paths.trajectory_dir, root=request_root
            ),
            "prompt_directory": _relative_output_path(
                output_paths.prompt_dir, root=request_root
            ),
            "raw_output_directory": _relative_output_path(
                output_paths.output_dir, root=request_root
            ),
            "num_runs": task_payload.get("num_runs"),
            "num_trajectories": task_payload.get("num_trajectories"),
            "generated_at": task_payload.get("generated_at"),
            "trajectory_stats": _summary_trajectory_stats(task_payload["trajectories"]),
            "cost_summary": task_payload.get("cost_summary"),
        }
        summary_payload["task_summaries"].append(task_summary_entry)

    return summary_payload


def build_request_cost_output_payload(
    request_summary_payload: dict[str, Any],
    *,
    request_summary_path: Path,
) -> dict[str, Any]:
    """Builds the cost sidecar for one multi-task generation request."""

    return {
        "composite_tasks": list(request_summary_payload["composite_tasks"]),
        "sdk": request_summary_payload["sdk"],
        "model": request_summary_payload["model"],
        "model_config": request_summary_payload["model_config"],
        "num_tasks": request_summary_payload["num_tasks"],
        "num_runs_per_task": request_summary_payload["num_runs_per_task"],
        "total_requested_runs": request_summary_payload["total_requested_runs"],
        "num_trajectories": request_summary_payload["num_trajectories"],
        "generated_at": request_summary_payload["generated_at"],
        "summary_output_path": str(request_summary_path),
        "cost_summary": request_summary_payload["cost_summary"],
        "task_cost_summaries": [
            {
                "composite_task": task_summary["composite_task"],
                "summary_path": task_summary["summary_path"],
                "cost_summary_path": task_summary["cost_summary_path"],
                "cost_summary": task_summary["cost_summary"],
            }
            for task_summary in request_summary_payload["task_summaries"]
        ],
    }


def build_request_error_output_payload(
    request_summary_payload: dict[str, Any],
    task_run_entries: list[dict[str, Any]],
    *,
    request_summary_path: Path,
) -> dict[str, Any]:
    """Builds the error sidecar for one multi-task generation request."""

    error_events: list[dict[str, Any]] = []
    error_counts_by_type: dict[str, int] = {}
    for task_run_entry in task_run_entries:
        task_payload = task_run_entry["payload"]
        for error_event in _collect_payload_error_events(task_payload):
            error_event_with_task = {
                "composite_task": task_run_entry["composite_task"],
                **error_event,
            }
            error_events.append(error_event_with_task)
            error_type = error_event_with_task["error_type"]
            error_counts_by_type[error_type] = (
                error_counts_by_type.get(error_type, 0) + 1
            )

    return {
        "composite_tasks": list(request_summary_payload["composite_tasks"]),
        "sdk": request_summary_payload["sdk"],
        "model": request_summary_payload["model"],
        "num_tasks": request_summary_payload["num_tasks"],
        "num_runs_per_task": request_summary_payload["num_runs_per_task"],
        "total_requested_runs": request_summary_payload["total_requested_runs"],
        "num_trajectories": request_summary_payload["num_trajectories"],
        "generated_at": request_summary_payload["generated_at"],
        "summary_output_path": str(request_summary_path),
        "total_errors": len(error_events),
        "error_counts_by_type": [
            {
                "error_type": error_type,
                "count": error_counts_by_type[error_type],
            }
            for error_type in sorted(error_counts_by_type)
        ],
        "task_error_summaries": [
            {
                "composite_task": task_summary["composite_task"],
                "summary_path": task_summary["summary_path"],
                "error_summary_path": task_summary["error_summary_path"],
            }
            for task_summary in request_summary_payload["task_summaries"]
        ],
        "error_events": error_events,
    }


def _sanitize_generation_usage_for_output(
    generation_usage: dict[str, Any],
    *,
    disable_validation: bool,
) -> dict[str, Any]:
    if not disable_validation:
        return generation_usage
    return {
        key: value
        for key, value in generation_usage.items()
        if key != "successful_attempt_number"
    }


def _sanitize_trajectory_for_output(
    trajectory: dict[str, Any],
    *,
    disable_validation: bool,
) -> dict[str, Any]:
    sanitized_trajectory = {
        key: value
        for key, value in trajectory.items()
        if key not in {"prompt", "raw_output"}
    }
    if not disable_validation:
        return sanitized_trajectory
    return {
        **sanitized_trajectory,
        "generation_usage": _sanitize_generation_usage_for_output(
            trajectory["generation_usage"],
            disable_validation=disable_validation,
        ),
    }


def _build_generation_payload(
    runtime_config: RuntimeConfig,
    ordered_trajectories: list[dict[str, Any]],
    *,
    error_events: list[dict[str, Any]] | None = None,
    attempt_prompts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generation_usages = [
        trajectory["generation_usage"] for trajectory in ordered_trajectories
    ]
    output_trajectories = [
        _sanitize_trajectory_for_output(
            trajectory,
            disable_validation=runtime_config.disable_validation,
        )
        for trajectory in ordered_trajectories
    ]
    trajectory_prompts = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "prompt": trajectory["prompt"],
        }
        for trajectory in ordered_trajectories
        if isinstance(trajectory.get("prompt"), str)
    ]
    trajectory_outputs = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "raw_output": trajectory["raw_output"],
        }
        for trajectory in ordered_trajectories
        if "raw_output" in trajectory
    ]
    cost_summary = _build_cost_summary_from_generation_usages(generation_usages)
    cost_summary = _append_sampling_cost_note(cost_summary, runtime_config)
    return {
        "composite_task": runtime_config.composite_task,
        "sdk": runtime_config.sdk,
        "model": runtime_config.model,
        "model_config": _build_model_config_payload(runtime_config),
        "num_runs": runtime_config.num_runs,
        "num_trajectories": len(output_trajectories),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_summary": cost_summary,
        "error_events": (
            [
                dict(error_event)
                for error_event in sorted(error_events, key=_error_event_key)
            ]
            if error_events is not None
            else []
        ),
        "attempt_prompts": [
            dict(prompt_entry) for prompt_entry in (attempt_prompts or [])
        ],
        "trajectory_prompts": trajectory_prompts,
        "trajectory_outputs": trajectory_outputs,
        "trajectories": output_trajectories,
    }


def _resolve_output_paths(runtime_config: RuntimeConfig) -> OutputPaths:
    summary_path = _resolve_summary_path(runtime_config)
    cost_path = resolve_cost_output_path(
        summary_path,
        runtime_config.cost_output_path,
    )
    if cost_path.resolve() == summary_path.resolve():
        raise TrajectoryGenerationError(
            "--cost-output must differ from the generated summary output path."
        )

    return OutputPaths(
        summary_path=summary_path,
        trajectory_dir=resolve_trajectory_output_dir(summary_path),
        prompt_dir=resolve_prompt_output_dir(summary_path),
        output_dir=resolve_raw_output_dir(summary_path),
        cost_path=cost_path,
        error_summary_path=resolve_error_output_path(summary_path),
    )


def _write_generation_outputs(
    payload: dict[str, Any],
    *,
    output_paths: OutputPaths,
) -> tuple[list[Path], list[Path], list[Path]]:
    summary_payload = build_summary_output_payload(payload)
    cost_payload = build_cost_output_payload(
        payload,
        trajectory_output_path=output_paths.summary_path,
    )
    error_summary_payload = build_error_summary_output_payload(
        payload,
        trajectory_output_path=output_paths.summary_path,
    )
    written_trajectory_paths = write_trajectory_output_payloads(
        payload["trajectories"],
        output_paths.trajectory_dir,
    )
    written_prompt_paths = write_prompt_output_payloads(
        payload.get("trajectory_prompts", []),
        output_paths.prompt_dir,
    )
    written_prompt_paths.extend(
        write_attempt_prompt_output_payloads(
            payload.get("attempt_prompts", []),
            output_paths.prompt_dir,
        )
    )
    written_output_paths = write_raw_output_payloads(
        payload.get("trajectory_outputs", []),
        output_paths.output_dir,
    )
    write_json_output(summary_payload, output_paths.summary_path)
    write_json_output(cost_payload, output_paths.cost_path)
    write_json_output(error_summary_payload, output_paths.error_summary_path)
    return written_trajectory_paths, written_prompt_paths, written_output_paths


def _write_request_outputs(
    runtime_config: RuntimeConfig,
    task_run_entries: list[dict[str, Any]],
    *,
    request_summary_path: Path,
) -> None:
    """Writes the combined summary and sidecars for one multi-task request."""

    request_cost_path = resolve_cost_output_path(
        request_summary_path,
        runtime_config.cost_output_path,
    )
    if request_cost_path.resolve() == request_summary_path.resolve():
        raise TrajectoryGenerationError(
            "--cost-output must differ from the generated combined summary output path."
        )

    request_summary_payload = build_request_summary_output_payload(
        runtime_config,
        task_run_entries,
        request_summary_path=request_summary_path,
    )
    request_cost_payload = build_request_cost_output_payload(
        request_summary_payload,
        request_summary_path=request_summary_path,
    )
    request_error_payload = build_request_error_output_payload(
        request_summary_payload,
        task_run_entries,
        request_summary_path=request_summary_path,
    )
    write_json_output(request_summary_payload, request_summary_path)
    write_json_output(request_cost_payload, request_cost_path)
    write_json_output(
        request_error_payload,
        resolve_error_output_path(request_summary_path),
    )


def _print_written_output_summary(
    output_paths: OutputPaths,
    *,
    written_trajectory_paths: list[Path],
    written_prompt_paths: list[Path],
    written_output_paths: list[Path],
) -> None:
    """Prints the standard per-task output summary for one saved dataset."""

    print(f"Wrote trajectory summary to {output_paths.summary_path}")
    print(
        f"Wrote {len(written_trajectory_paths)} trajectory files to "
        f"{output_paths.trajectory_dir}"
    )
    print(
        f"Wrote {len(written_prompt_paths)} prompt files to "
        f"{output_paths.prompt_dir}"
    )
    print(
        f"Wrote {len(written_output_paths)} raw output files to "
        f"{output_paths.output_dir}"
    )
    print(f"Wrote cost summary to {output_paths.cost_path}")
    print(f"Wrote error summary to {output_paths.error_summary_path}")
