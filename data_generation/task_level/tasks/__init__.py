"""Register task definitions and expose task-level validation error types."""

from __future__ import annotations

from data_generation.task_level.tasks.base import (
    CommunicationStepSemanticValidationError,
    DuplicateTrajectoryValidationError,
    HeldObjectSemanticValidationError,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    InsufficientValidUniqueTrajectoriesValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    NavigationSemanticValidationError,
    ObjectStateSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    PlacementDestinationSemanticValidationError,
    PostGoalActionSemanticValidationError,
    ResponseFormatValidationError,
    TaskPreconditionSemanticValidationError,
    TaskSemanticValidationError,
    TaskDefinition,
    TaskValidator,
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
    WaitDurationSemanticValidationError,
)
from data_generation.task_level.tasks.prepare_coffee import (
    PREPARE_COFFEE_TASK,
    PrepareCoffeeValidator,
    build_prepare_coffee_prompt,
)


TASK_REGISTRY: dict[str, TaskDefinition] = {
    PREPARE_COFFEE_TASK.composite_task: PREPARE_COFFEE_TASK,
}

__all__ = [
    "CommunicationStepSemanticValidationError",
    "DuplicateTrajectoryValidationError",
    "HeldObjectSemanticValidationError",
    "InsufficientValidUniqueTrajectoriesDuplicateError",
    "InsufficientValidUniqueTrajectoriesInvalidError",
    "InsufficientValidUniqueTrajectoriesMixedError",
    "InsufficientValidUniqueTrajectoriesValidationError",
    "MissingInitialCommunicationSemanticValidationError",
    "MissingTaskActionSemanticValidationError",
    "NavigationSemanticValidationError",
    "ObjectStateSemanticValidationError",
    "ObservationSequenceSemanticValidationError",
    "PREPARE_COFFEE_TASK",
    "PlacementDestinationSemanticValidationError",
    "PostGoalActionSemanticValidationError",
    "PrepareCoffeeValidator",
    "ResponseFormatValidationError",
    "TASK_REGISTRY",
    "TaskPreconditionSemanticValidationError",
    "TaskDefinition",
    "TaskSemanticValidationError",
    "TaskValidator",
    "ToolArgumentSemanticValidationError",
    "TrajectoryStructureValidationError",
    "TrajectoryValidationError",
    "UnexpectedStepIndexSemanticValidationError",
    "UnsupportedToolSemanticValidationError",
    "UnsatisfiedGoalSemanticValidationError",
    "WaitDurationSemanticValidationError",
    "build_prepare_coffee_prompt",
    "get_task_definition",
    "supported_task_names",
]


def get_task_definition(composite_task: str) -> TaskDefinition | None:
    return TASK_REGISTRY.get(composite_task)


def supported_task_names() -> tuple[str, ...]:
    return tuple(sorted(TASK_REGISTRY))
