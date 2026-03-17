"""Backward-compatible facade for shared task-level prompting and validation code."""

from __future__ import annotations

from data_generation.task_level.tasks.shared.constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    PLACE_LOCATION_ARG_NAMES,
    RELEASE_TOOL_NAMES,
)
from data_generation.task_level.tasks.shared.errors import (
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
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
)
from data_generation.task_level.tasks.shared.fsm import FiniteStateTaskValidator
from data_generation.task_level.tasks.shared.instances import (
    build_canonical_agents,
    build_randomized_fixture_task_instance,
    resolve_initial_position_fixture_ids,
)
from data_generation.task_level.tasks.shared.prompting import make_task_prompt_builder
from data_generation.task_level.tasks.shared.schema import build_task_response_schema
from data_generation.task_level.tasks.shared.state import (
    AgentRuntimeState,
    TaskRuntimeState,
)
from data_generation.task_level.tasks.shared.types import (
    PreflightTokenEstimate,
    TaskDefinition,
    TaskInstance,
    TaskPromptBuilder,
    TaskValidator,
)
