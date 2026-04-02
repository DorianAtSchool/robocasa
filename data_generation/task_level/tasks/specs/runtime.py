"""Build runtime task definitions from JSON-backed task specs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.shared.errors import (
    TaskPreconditionSemanticValidationError,
)
from data_generation.task_level.tasks.shared.fsm import FiniteStateTaskValidator
from data_generation.task_level.tasks.shared.instances import (
    build_randomized_fixture_task_instance,
    make_symbolic_trajectory_record_builder,
)
from data_generation.task_level.tasks.shared.prompting import make_task_prompt_builder
from data_generation.task_level.tasks.shared.schema import build_task_response_schema
from data_generation.task_level.tasks.shared.state import TaskRuntimeState
from data_generation.task_level.tasks.shared.types import (
    PreflightTokenEstimate,
    TaskDefinition,
    TaskInstance,
)

from . import TaskSpec, load_all_task_specs


def _resolve_machine_path(
    machine_state: dict[str, Any],
    machine_path: list[str] | tuple[str, ...],
) -> Any:
    current_value: Any = machine_state
    for path_part in machine_path:
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(path_part)
    return current_value


def _set_machine_path(
    machine_state: dict[str, Any],
    machine_path: list[str] | tuple[str, ...],
    value: Any,
) -> None:
    current_value = machine_state
    for path_part in machine_path[:-1]:
        current_value = current_value.setdefault(path_part, {})
    current_value[machine_path[-1]] = value


class SpecDrivenTaskValidator(FiniteStateTaskValidator):
    """Generic FSM validator that interprets TaskSpec preconditions and goals."""

    def __init__(self, task_spec: TaskSpec, task_instance: TaskInstance | None = None) -> None:
        self._task_spec = task_spec
        effective_initial_state = (
            task_instance.initial_state
            if task_instance is not None
            else deepcopy(task_spec.initial_state)
        )
        super().__init__(
            composite_task=task_spec.composite_task,
            agent_ids=task_spec.agent_ids,
            initial_state=effective_initial_state,
            allowed_tool_specs=task_spec.allowed_tool_specs,
            checks=task_spec.validator_checks,
            max_reasoning_chars=task_spec.max_reasoning_chars,
            initial_public_state=task_spec.initial_public_state,
        )

    def _refresh_public_state(self, runtime_state: TaskRuntimeState) -> None:
        for public_key in tuple(runtime_state.public_state):
            if public_key.endswith("_location"):
                object_id = public_key[: -len("_location")]
                if object_id in runtime_state.objects:
                    runtime_state.public_state[public_key] = runtime_state.objects[object_id].get(
                        "location"
                    )
            elif public_key.endswith("_started"):
                machine_id = public_key[: -len("_started")]
                machine_entry = runtime_state.machine_state.get(machine_id, {})
                if isinstance(machine_entry, dict) and "started" in machine_entry:
                    runtime_state.public_state[public_key] = machine_entry.get("started")
            else:
                machine_path = public_key.split(".")
                if len(machine_path) > 1:
                    machine_value = _resolve_machine_path(
                        runtime_state.machine_state, machine_path
                    )
                    if machine_value is not None:
                        runtime_state.public_state[public_key] = machine_value

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        for condition in self._task_spec.task_preconditions:
            condition_kind = condition["kind"]
            if condition_kind == "object_must_remain_at_location":
                object_id = condition["object_id"]
                required_location = condition["location"]
                actual_location = runtime_state.objects[object_id]["location"]
                if actual_location != required_location:
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "object_id": object_id,
                            "required_location": required_location,
                            "actual_location": actual_location,
                        },
                    )
            elif condition_kind == "fixture_part_state_required_for_pickup":
                if step["tool"] != condition["tool"]:
                    continue
                if step["args"].get("source_id") != condition["source_id"]:
                    continue
                actual_state = runtime_state.fixtures[condition["fixture_id"]]["parts"][
                    condition["part_id"]
                ]["state"]
                if actual_state != condition["required_state"]:
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "fixture_id": condition["fixture_id"],
                            "part_id": condition["part_id"],
                            "required_state": condition["required_state"],
                            "actual_state": actual_state,
                        },
                    )
            elif condition_kind == "object_location_required_for_action":
                if step["tool"] != condition["tool"]:
                    continue
                actual_location = runtime_state.objects[condition["object_id"]]["location"]
                if actual_location != condition["required_location"]:
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "object_id": condition["object_id"],
                            "required_location": condition["required_location"],
                            "actual_location": actual_location,
                        },
                    )
            else:
                raise ValueError(
                    f"Unsupported TaskSpec precondition kind: {condition_kind}"
                )

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        for effect in self._task_spec.task_effects:
            effect_kind = effect["kind"]
            if effect_kind != "set_machine_flag_on_action":
                raise ValueError(f"Unsupported TaskSpec effect kind: {effect_kind}")
            if step["tool"] != effect["tool"]:
                continue
            if any(step["args"].get(arg_name) != arg_value for arg_name, arg_value in effect["args"].items()):
                continue
            _set_machine_path(
                runtime_state.machine_state,
                tuple(effect["machine_path"]),
                effect["value"],
            )

        self._refresh_public_state(runtime_state)

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        for condition in self._task_spec.goal_conditions:
            condition_kind = condition["kind"]
            if condition_kind == "object_at_location":
                if (
                    runtime_state.objects[condition["object_id"]]["location"]
                    != condition["location"]
                ):
                    return False
            elif condition_kind == "machine_flag_true":
                if not _resolve_machine_path(
                    runtime_state.machine_state,
                    tuple(condition["machine_path"]),
                ):
                    return False
            else:
                raise ValueError(f"Unsupported TaskSpec goal kind: {condition_kind}")
        return True


def build_task_definition_from_spec(task_spec: TaskSpec) -> TaskDefinition:
    """Build one runtime TaskDefinition from a JSON-backed task spec."""

    tool_names = tuple(task_spec.allowed_tool_specs)
    overrides: dict[str, dict[str, Any]] = {}
    for tool_name, tool_spec in task_spec.allowed_tool_specs.items():
        tool_override = {
            key: deepcopy(value)
            for key, value in tool_spec.items()
            if key not in {"description", "tool_args", "tool_arg_types"}
        }
        if tool_override:
            overrides[tool_name] = tool_override

    allowed_tool_specs = build_allowed_tool_specs(tool_names, overrides=overrides)
    response_schema = build_task_response_schema(
        agent_ids=task_spec.agent_ids,
        allowed_tool_specs=allowed_tool_specs,
    )
    non_communicate_tool_names = tuple(
        tool_name for tool_name in allowed_tool_specs if tool_name != "communicate"
    )
    build_prompt = make_task_prompt_builder(
        composite_task=task_spec.composite_task,
        task_goal=task_spec.task_goal,
        initial_state=task_spec.initial_state,
        allowed_tool_specs=allowed_tool_specs,
        non_communicate_tool_names=non_communicate_tool_names,
        extra_execution_rules=task_spec.extra_execution_rules,
    )
    build_trajectory_record = make_symbolic_trajectory_record_builder(
        composite_task=task_spec.composite_task,
        agent_ids=task_spec.agent_ids,
    )

    def _build_task_instance(run_index: int, runtime_config: Any | None = None) -> TaskInstance:
        return build_randomized_fixture_task_instance(
            composite_task=task_spec.composite_task,
            agent_ids=task_spec.agent_ids,
            initial_state=task_spec.initial_state,
            allowed_tool_specs=allowed_tool_specs,
            run_index=run_index,
            runtime_config=runtime_config,
        )

    def _validator_factory(task_instance: TaskInstance | None) -> SpecDrivenTaskValidator:
        return SpecDrivenTaskValidator(task_spec, task_instance)

    return TaskDefinition(
        composite_task=task_spec.composite_task,
        response_schema=response_schema,
        preflight_token_estimate=PreflightTokenEstimate(
            prompt_tokens=task_spec.preflight_token_estimate.prompt_tokens,
            output_tokens=task_spec.preflight_token_estimate.output_tokens,
            reasoning_tokens=task_spec.preflight_token_estimate.reasoning_tokens,
        ),
        build_task_instance=_build_task_instance,
        build_prompt=build_prompt,
        build_trajectory_record=build_trajectory_record,
        validator_factory=_validator_factory,
    )


SPEC_TASK_REGISTRY: dict[str, TaskDefinition] = {
    task_spec.composite_task: build_task_definition_from_spec(task_spec)
    for task_spec in load_all_task_specs()
}

