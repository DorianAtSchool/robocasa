"""Load JSON-backed task specs that mirror task-level Python definitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightTokenEstimateSpec:
    """Serializable preflight token estimate for one task spec."""

    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreflightTokenEstimateSpec":
        """Build one token estimate from a JSON object."""

        return cls(
            prompt_tokens=int(payload["prompt_tokens"]),
            output_tokens=int(payload["output_tokens"]),
            reasoning_tokens=int(payload.get("reasoning_tokens", 0)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Structured representation of one JSON-backed task configuration."""

    spec_version: int
    composite_task: str
    source_python_module: str
    agent_ids: tuple[str, ...]
    max_reasoning_chars: int
    validator_checks: tuple[str, ...]
    preflight_token_estimate: PreflightTokenEstimateSpec
    initial_state: dict[str, Any]
    allowed_tool_specs: dict[str, dict[str, Any]]
    task_goal: str
    extra_execution_rules: tuple[str, ...]
    initial_public_state: dict[str, Any]
    task_preconditions: tuple[dict[str, Any], ...]
    goal_conditions: tuple[dict[str, Any], ...]
    task_effects: tuple[dict[str, Any], ...]
    grounding: dict[str, Any]
    example_trajectory: dict[str, Any]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        """Build one validated task spec from a JSON object."""

        required_fields = (
            "spec_version",
            "composite_task",
            "source_python_module",
            "agent_ids",
            "max_reasoning_chars",
            "validator_checks",
            "preflight_token_estimate",
            "initial_state",
            "allowed_tool_specs",
            "task_goal",
            "extra_execution_rules",
            "initial_public_state",
            "task_preconditions",
            "goal_conditions",
            "task_effects",
            "grounding",
            "example_trajectory",
        )
        missing_fields = [field_name for field_name in required_fields if field_name not in payload]
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Task spec is missing required fields: {missing}")

        return cls(
            spec_version=int(payload["spec_version"]),
            composite_task=str(payload["composite_task"]),
            source_python_module=str(payload["source_python_module"]),
            agent_ids=tuple(str(agent_id) for agent_id in payload["agent_ids"]),
            max_reasoning_chars=int(payload["max_reasoning_chars"]),
            validator_checks=tuple(str(check_name) for check_name in payload["validator_checks"]),
            preflight_token_estimate=PreflightTokenEstimateSpec.from_dict(
                dict(payload["preflight_token_estimate"])
            ),
            initial_state=dict(payload["initial_state"]),
            allowed_tool_specs={
                str(tool_name): dict(tool_spec)
                for tool_name, tool_spec in dict(payload["allowed_tool_specs"]).items()
            },
            task_goal=str(payload["task_goal"]),
            extra_execution_rules=tuple(
                str(rule_text) for rule_text in payload["extra_execution_rules"]
            ),
            initial_public_state=dict(payload["initial_public_state"]),
            task_preconditions=tuple(
                dict(condition_spec) for condition_spec in payload["task_preconditions"]
            ),
            goal_conditions=tuple(
                dict(condition_spec) for condition_spec in payload["goal_conditions"]
            ),
            task_effects=tuple(
                dict(effect_spec) for effect_spec in payload["task_effects"]
            ),
            grounding=dict(payload["grounding"]),
            example_trajectory=dict(payload["example_trajectory"]),
            notes=tuple(str(note_text) for note_text in payload.get("notes", ())),
        )


SPEC_DIRECTORY = Path(__file__).resolve().parent
TASK_SPEC_DIRECTORY_OVERRIDE_ENV_VAR = "ROBOCASA_TASK_SPEC_DIR"


def _resolve_spec_directory() -> Path:
    override_directory = os.environ.get(TASK_SPEC_DIRECTORY_OVERRIDE_ENV_VAR)
    if not override_directory:
        return SPEC_DIRECTORY
    return Path(override_directory).expanduser().resolve()


def _task_spec_path(task_name: str) -> Path:
    """Resolve the JSON file path for one composite task name."""

    normalized_name = "".join(
        character.lower() if character.isalnum() else "_" for character in task_name
    )
    collapsed_name = "_".join(part for part in normalized_name.split("_") if part)
    return _resolve_spec_directory() / f"{collapsed_name}.json"


def load_task_spec(task_name: str) -> TaskSpec:
    """Load one JSON-backed task spec by composite task name."""

    spec_path = _task_spec_path(task_name)
    if not spec_path.exists():
        raise FileNotFoundError(f"Unknown task spec {task_name!r}: {spec_path}")
    with spec_path.open("r", encoding="utf-8") as handle:
        return TaskSpec.from_dict(json.load(handle))


def load_all_task_specs() -> tuple[TaskSpec, ...]:
    """Load every JSON-backed task spec shipped in this package."""

    specs: list[TaskSpec] = []
    for spec_path in sorted(_resolve_spec_directory().glob("*.json")):
        with spec_path.open("r", encoding="utf-8") as handle:
            specs.append(TaskSpec.from_dict(json.load(handle)))
    return tuple(specs)
