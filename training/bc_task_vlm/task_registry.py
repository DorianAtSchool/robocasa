"""Task metadata used by the task-level VLM SFT pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from data_generation.task_level.tasks.specs import load_task_spec

AGENT_IDS: tuple[str, ...] = ("agent_0", "agent_1")


@dataclass(frozen=True)
class TaskMetadata:
    """Static metadata for one task directory in the rendered dataset."""

    dataset_name: str
    composite_task: str
    allowed_tool_specs: dict[str, dict[str, Any]]


TASK_METADATA_REGISTRY: dict[str, TaskMetadata] = {
    "hot_dog_setup": TaskMetadata(
        dataset_name="hot_dog_setup",
        composite_task="HotDogSetup",
        allowed_tool_specs=deepcopy(load_task_spec("HotDogSetup").allowed_tool_specs),
    ),
    "prepare_coffee": TaskMetadata(
        dataset_name="prepare_coffee",
        composite_task="PrepareCoffee",
        allowed_tool_specs=deepcopy(load_task_spec("PrepareCoffee").allowed_tool_specs),
    ),
    "prepare_sandwich_station": TaskMetadata(
        dataset_name="prepare_sandwich_station",
        composite_task="PrepareSandwichStation",
        allowed_tool_specs=deepcopy(
            load_task_spec("PrepareSandwichStation").allowed_tool_specs
        ),
    ),
}

COMPOSITE_TO_DATASET_NAME: dict[str, str] = {
    metadata.composite_task.lower(): dataset_name
    for dataset_name, metadata in TASK_METADATA_REGISTRY.items()
}


def supported_task_names() -> tuple[str, ...]:
    """Returns the supported snake_case dataset task names."""

    return tuple(sorted(TASK_METADATA_REGISTRY))


def resolve_task_name(task_name: str) -> str:
    """Resolves either snake_case or composite task names to the dataset key."""

    normalized = task_name.strip()
    if not normalized:
        raise ValueError("Task names must be non-empty.")
    snake_case = normalized.lower()
    if snake_case in TASK_METADATA_REGISTRY:
        return snake_case
    composite_match = COMPOSITE_TO_DATASET_NAME.get(snake_case)
    if composite_match is not None:
        return composite_match
    supported = ", ".join(supported_task_names())
    raise ValueError(f"Unsupported task {task_name!r}. Supported tasks: {supported}.")


def get_task_metadata(task_name: str) -> TaskMetadata:
    """Returns immutable metadata for one supported task."""

    return TASK_METADATA_REGISTRY[resolve_task_name(task_name)]
