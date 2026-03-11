from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TrajectoryValidationError(ValueError):
    pass


class TaskValidator(Protocol):
    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class TaskDefinition:
    composite_task: str
    response_schema: dict[str, Any]
    preflight_reference_candidate: dict[str, Any]
    build_prompt: Callable[[str], str]
    build_trajectory_record: Callable[
        [dict[str, Any], dict[str, Any], str, dict[str, Any]],
        dict[str, Any],
    ]
    validator_factory: Callable[[], TaskValidator]
