"""Define shared task-level protocols and immutable task metadata types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from data_generation.task_level.generation.raw.config import RuntimeConfig


class TaskValidator(Protocol):
    """Protocol implemented by task validators used by generation runtime."""

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Validates one candidate trajectory and returns normalized metadata."""


class TaskPromptBuilder(Protocol):
    """Protocol implemented by task prompt builders used by generation runtime."""

    def __call__(
        self,
        variation_key: str,
        task_instance: TaskInstance | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds one task prompt, optionally including retry feedback."""


@dataclass(frozen=True)
class PreflightTokenEstimate:
    """Stores the manual token estimate used for preflight cost projection."""

    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class TaskInstance:
    """Stores the concrete task state used for one generation run."""

    initial_state: dict[str, Any]
    allowed_tool_specs: dict[str, dict[str, Any]] | None = None
    task_goal: str | None = None
    extra_execution_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskDefinition:
    """Stores the prompt, schema, and validator factory for one composite task."""

    composite_task: str
    response_schema: dict[str, Any]
    preflight_token_estimate: PreflightTokenEstimate
    build_task_instance: Callable[[int, RuntimeConfig | None], TaskInstance]
    build_prompt: TaskPromptBuilder
    build_trajectory_record: Callable[
        [dict[str, Any], dict[str, Any], str, dict[str, Any], TaskInstance],
        dict[str, Any],
    ]
    validator_factory: Callable[[TaskInstance | None], TaskValidator]
