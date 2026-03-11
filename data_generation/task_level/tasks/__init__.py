from __future__ import annotations

from data_generation.task_level.tasks.base import (
    DuplicateTrajectoryValidationError,
    ResponseFormatValidationError,
    TaskSemanticValidationError,
    TaskDefinition,
    TaskValidator,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
)
from data_generation.task_level.tasks.prepare_coffee import (
    PREPARE_COFFEE_TASK,
    PrepareCoffeeValidator,
    build_prepare_coffee_prompt,
)


TASK_REGISTRY: dict[str, TaskDefinition] = {
    PREPARE_COFFEE_TASK.composite_task: PREPARE_COFFEE_TASK,
}


def get_task_definition(composite_task: str) -> TaskDefinition | None:
    return TASK_REGISTRY.get(composite_task)


def supported_task_names() -> tuple[str, ...]:
    return tuple(sorted(TASK_REGISTRY))
