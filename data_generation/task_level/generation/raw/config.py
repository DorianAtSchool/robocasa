"""Define shared generation configuration and package-wide constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from data_generation.task_level.runtime.client import TrajectoryGenerationError
from data_generation.task_level.tasks import supported_task_names

# This module lives under data_generation/task_level/generation/raw, so the
# repository root is four parents above this file.
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_generation" / "task_level" / "data"
DEFAULT_COMPOSITE_TASK = supported_task_names()[0]
ALL_COMPOSITE_TASKS_OPTION = "all"
DATASET_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TRAJECTORY_DIRECTORY_NAME = "trajectories"
SUMMARY_OUTPUT_FILENAME = "summary.json"
COST_SUMMARY_OUTPUT_FILENAME = "cost_summary.json"
ERROR_SUMMARY_OUTPUT_FILENAME = "summary_errors.json"
TRAJECTORY_ID_DIGITS = 6
BATCH_DIRECTORY_NAME = "batch"
REQUEST_DIRECTORY_NAME = "requests"
RETRY_PROGRESS_ERROR_MESSAGE_MAX_LENGTH = 96
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")
GENERATION_ERROR_EXIT_CODE = 1
BATCH_POLL_INTERVAL_SECONDS = 10
GOOGLE_CLOUD_BATCH_GCS_PREFIX_ENV_VAR = "GOOGLE_CLOUD_BATCH_GCS_PREFIX"
INTERRUPTED_EXIT_CODE = 130
INTERRUPTED_MESSAGE = (
    "Interrupted. Exiting immediately. Queued trajectories were cancelled; "
    "requests already in flight may still be billed."
)
BATCH_INTERRUPTED_MESSAGE = (
    "Interrupted. Active remote batch jobs were cancelled. "
    "No local outputs were written."
)


@dataclass(frozen=True)
class RuntimeConfig:
    composite_task: str | None
    num_runs: int
    model: str
    sdk: str
    project: str | None
    location: str
    temperature: float
    max_workers: int
    max_retries: int
    sampling: str = "base"
    verbalized_k: int = 1
    thinking_level: str | None = None
    summary_path: Path | None = None
    cost_output_path: Path | None = None
    disable_validation: bool = False
    batch_processing: bool = False
    batch_gcs_prefix: str | None = None
    composite_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalizes single-task and multi-task config fields to a stable shape."""

        normalized_tasks = self._normalize_composite_tasks(
            self.composite_task,
            self.composite_tasks,
        )
        primary_task = normalized_tasks[0] if normalized_tasks else None
        object.__setattr__(self, "composite_tasks", normalized_tasks)
        object.__setattr__(self, "composite_task", primary_task)

    @staticmethod
    def _normalize_composite_tasks(
        composite_task: str | None,
        composite_tasks: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Builds the ordered task list while preserving explicit request order."""

        requested_tasks: list[str] = []
        if composite_task is not None:
            requested_tasks.append(composite_task)
        requested_tasks.extend(composite_tasks)

        normalized_tasks: list[str] = []
        for task_name in requested_tasks:
            if not isinstance(task_name, str):
                continue
            normalized_task_name = " ".join(task_name.strip().split())
            if normalized_task_name:
                # Expand the CLI sentinel into the full supported task list.
                if normalized_task_name.casefold() == ALL_COMPOSITE_TASKS_OPTION:
                    return supported_task_names()
                normalized_tasks.append(normalized_task_name)
        return tuple(normalized_tasks)

    def for_task(
        self,
        composite_task: str,
        *,
        summary_path: Path | None = None,
        cost_output_path: Path | None = None,
    ) -> RuntimeConfig:
        """Builds one task-scoped runtime config for the shared single-task runtime."""

        return replace(
            self,
            composite_task=composite_task,
            composite_tasks=(),
            summary_path=summary_path,
            cost_output_path=cost_output_path,
        )


def _validate_runtime_config(runtime_config: RuntimeConfig) -> None:
    if not runtime_config.composite_tasks:
        raise TrajectoryGenerationError("At least one task must be selected.")
    duplicate_tasks = sorted(
        {
            composite_task
            for composite_task in runtime_config.composite_tasks
            if runtime_config.composite_tasks.count(composite_task) > 1
        }
    )
    if duplicate_tasks:
        raise TrajectoryGenerationError(
            "Duplicate tasks are not allowed: " + ", ".join(duplicate_tasks) + "."
        )
    if runtime_config.num_runs <= 0:
        raise TrajectoryGenerationError("--num-runs must be greater than 0.")
    if runtime_config.max_workers <= 0:
        raise TrajectoryGenerationError("--max-workers must be greater than 0.")
    if runtime_config.max_retries <= 0:
        raise TrajectoryGenerationError("--max-retries must be greater than 0.")
    if runtime_config.sampling == "verbalized" and runtime_config.verbalized_k <= 0:
        raise TrajectoryGenerationError(
            "--verbalized-k must be greater than 0 when --sampling verbalized is enabled."
        )
    if runtime_config.sampling != "verbalized" and runtime_config.verbalized_k != 1:
        raise TrajectoryGenerationError(
            "--verbalized-k may only be used when --sampling verbalized is enabled."
        )
    if runtime_config.batch_processing and not runtime_config.batch_gcs_prefix:
        raise TrajectoryGenerationError(
            "--batch-gcs-prefix or GOOGLE_CLOUD_BATCH_GCS_PREFIX is required "
            "when --batch-processing is enabled."
        )
