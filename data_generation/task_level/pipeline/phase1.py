"""Phase 1: LLM-based TaskSpec generation.

Takes the filtered candidates from Phase 0b and generates one TaskSpec JSON
per task. Each generation is independent and parallelized. Failures are
recorded as empty placeholder specs so Phase 2 can report them without
losing per-task context.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.runtime.client import (
    DEFAULT_GENERATION_TIMEOUT_SEC,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_SDK,
    BaseGenerationClient,
    build_generation_client,
)
from data_generation.task_level.tasks.shared.constants import (
    CLOSE_PART_TOOL_NAMES,
    INTERACTION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
)
from data_generation.task_level.tasks.shared.schema import _allowed_ids_key_for_arg_name

from .few_shot import FewShotExample, load_few_shot_examples
from .models import TaskAnalysis
from .prompts.spec_generation import (
    SPEC_GENERATION_RESPONSE_SCHEMA,
    build_spec_generation_prompt,
)
from .sim_normalization import normalize_spec_payload_against_simulation


@dataclass
class SpecGenerationResult:
    """Per-task result of Phase 1."""

    task_name: str
    module_path: str
    batch: str
    spec_payload: dict[str, Any] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpecRepairContext:
    """Optional revision context for regenerating one TaskSpec."""

    previous_spec_payload: dict[str, Any] | None = None
    feedback_lines: tuple[str, ...] = ()


_SHARED_TOOL_METADATA_KEYS = frozenset(
    {"description", "tool_args", "tool_arg_types", "tool_arg_any_of"}
)
_PART_STATE_VALUES = frozenset({"open", "closed", "pulled_out", "pushed_in"})
_PART_STATE_ALIASES = {"pulled_out": "open", "pushed_in": "closed"}
_DIRECT_PLACEMENT_LOCATION_ARG_NAMES = (
    "support_id",
    "receptacle_id",
    "support_object_id",
)
_CONTROL_TOOL_TYPES = {
    "press_button": "button",
    "press_lever": "lever",
    "set_rotary_control": "rotary_control",
}
_EFFECT_CARRYING_TOOL_NAMES = frozenset(
    RELEASE_TOOL_NAMES
    | INTERACTION_TOOL_NAMES
    | OPEN_PART_TOOL_NAMES
    | CLOSE_PART_TOOL_NAMES
)


def _load_task_source(candidate: TaskAnalysis) -> str:
    return Path(candidate.file_path).read_text(encoding="utf-8")


def _parse_spec_payload(payload: Any) -> tuple[dict[str, Any] | None, str]:
    """Parse the model's raw JSON response into a TaskSpec dict.

    Phase 1 runs Gemini in `response_mime_type=application/json` mode
    without a response schema, so the raw `payload` should already be a
    JSON-encoded TaskSpec object. Returns (spec_dict, error_message);
    on success, error_message is empty.
    """

    if payload is None:
        return None, "empty model response"

    text = payload if isinstance(payload, str) else str(payload)
    if not text:
        return None, "empty model response"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from model: {exc.msg}"

    if not isinstance(data, dict):
        return None, "model response was not a JSON object"

    return data, ""


def _normalize_allowed_tool_override(
    raw_tool_spec: dict[str, Any],
) -> dict[str, Any]:
    """Normalize raw per-tool overrides before merging with canonical specs."""

    normalized_tool_spec: dict[str, Any] = {}
    for key, value in raw_tool_spec.items():
        normalized_key = key
        normalized_value = value
        if key.startswith("allowed_") and key.endswith("_id"):
            normalized_key = f"{key}s"
            if isinstance(value, str):
                normalized_value = [value]
        existing_value = normalized_tool_spec.get(normalized_key)
        if isinstance(existing_value, list) and isinstance(normalized_value, list):
            normalized_tool_spec[normalized_key] = list(
                dict.fromkeys(existing_value + normalized_value)
            )
            continue
        normalized_tool_spec[normalized_key] = normalized_value
    return normalized_tool_spec


def _canonicalize_allowed_tool_specs(spec_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Rebuild the task-local tool registry from the shared canonical catalog.

    Phase 1 asks the model for tool names plus task-local constraint overrides
    only. This post-processing step restores the shared `description`,
    `tool_args`, and `tool_arg_types` fields programmatically so later phases
    validate against the runtime's canonical tool definitions instead of
    whatever metadata the model happened to emit.
    """

    raw_allowed_tool_specs = spec_payload.get("allowed_tool_specs")
    if raw_allowed_tool_specs is None:
        raw_allowed_tool_specs = {}
    if not isinstance(raw_allowed_tool_specs, dict):
        raise ValueError("allowed_tool_specs must be a JSON object.")

    requested_tool_names: list[str] = []
    for tool_name in raw_allowed_tool_specs:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("allowed_tool_specs keys must be non-empty strings.")
        normalized_name = tool_name.strip()
        if normalized_name not in requested_tool_names:
            requested_tool_names.append(normalized_name)

    trajectory = spec_payload.get("example_trajectory")
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            normalized_name = tool_name.strip()
            if normalized_name not in requested_tool_names:
                requested_tool_names.append(normalized_name)

    if not requested_tool_names:
        raise ValueError(
            "allowed_tool_specs must list at least one tool or the example_trajectory "
            "must use at least one tool."
        )

    overrides: dict[str, dict[str, Any]] = {}
    for tool_name in requested_tool_names:
        raw_tool_spec = raw_allowed_tool_specs.get(tool_name, {})
        if raw_tool_spec is None:
            raw_tool_spec = {}
        if not isinstance(raw_tool_spec, dict):
            raise ValueError(
                f"allowed_tool_specs.{tool_name} must be a JSON object of task-local overrides."
            )
        normalized_raw_tool_spec = _normalize_allowed_tool_override(raw_tool_spec)
        tool_override = {
            key: value
            for key, value in normalized_raw_tool_spec.items()
            if key not in _SHARED_TOOL_METADATA_KEYS
        }
        if tool_override:
            overrides[tool_name] = tool_override

    try:
        return build_allowed_tool_specs(tuple(requested_tool_names), overrides=overrides)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def _normalize_part_state_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _PART_STATE_ALIASES.get(value, value)


def _normalize_rotary_goal_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return "on" if value > 0 else "off"
    return value


def _normalize_pressed_control_state(
    current_state: Any,
    *,
    tool_name: str,
) -> str:
    if tool_name == "press_button":
        if current_state == "off":
            return "on"
        return "pressed"
    if tool_name == "press_lever":
        if current_state == "up":
            return "down"
        return "pressed"
    return "pressed"


def _normalize_free_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().replace("_", " ").split())


def _symbol_text_forms(symbol_id: str) -> tuple[str, ...]:
    normalized_symbol = " ".join(symbol_id.lower().split("_"))
    return tuple(dict.fromkeys((symbol_id.lower(), normalized_symbol)))


def _text_mentions_symbol(text: Any, symbol_id: str) -> bool:
    normalized_text = _normalize_free_text(text)
    return any(symbol_form in normalized_text for symbol_form in _symbol_text_forms(symbol_id))


def _composite_task_machine_namespace(composite_task: str) -> str:
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", composite_task).lower()
    return "_".join(part for part in snake_case.split("_") if part)


def _primary_machine_namespace(
    *,
    initial_state: dict[str, Any] | None,
    composite_task: str,
) -> str | None:
    if not isinstance(initial_state, dict):
        return None
    machine_state = initial_state.setdefault("machine_state", {})
    if not isinstance(machine_state, dict):
        return None
    fixture_ids = {
        fixture_id
        for fixture_id in (initial_state.get("fixtures") or {})
        if isinstance(fixture_id, str)
    }
    candidate_namespaces = [
        machine_id
        for machine_id in machine_state
        if isinstance(machine_id, str) and machine_id not in fixture_ids
    ]
    if candidate_namespaces:
        namespace = candidate_namespaces[0]
        machine_state.setdefault(namespace, {})
        return namespace
    namespace = _composite_task_machine_namespace(composite_task)
    machine_state.setdefault(namespace, {})
    return namespace


def _ensure_machine_flag_default(
    initial_state: dict[str, Any] | None,
    *,
    machine_path: tuple[str, str],
    default: Any = False,
) -> None:
    if not isinstance(initial_state, dict):
        return
    machine_state = initial_state.setdefault("machine_state", {})
    if not isinstance(machine_state, dict):
        return
    machine_namespace = machine_state.setdefault(machine_path[0], {})
    if not isinstance(machine_namespace, dict):
        return
    machine_namespace.setdefault(machine_path[1], default)


def _ensure_public_state_default(
    initial_public_state: dict[str, Any] | None,
    key: str,
    value: Any = False,
) -> None:
    if isinstance(initial_public_state, dict):
        initial_public_state.setdefault(key, value)


def _merge_unique_dict_list(
    existing_items: Any,
    new_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = [
        dict(item) for item in existing_items or () if isinstance(item, dict)
    ]
    for new_item in new_items:
        if any(existing_item == new_item for existing_item in normalized_items):
            continue
        normalized_items.append(dict(new_item))
    return normalized_items


def _ensure_machine_flag_effect(
    task_effects: list[dict[str, Any]],
    *,
    tool: str,
    args: dict[str, Any],
    machine_path: list[str],
    value: Any,
    required_object_locations: list[dict[str, Any]] | None = None,
    required_machine_values: list[dict[str, Any]] | None = None,
    required_fixture_controls: list[dict[str, Any]] | None = None,
) -> None:
    normalized_args = dict(args)
    normalized_machine_path = list(machine_path)
    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        if (
            effect.get("kind") != "set_machine_flag_on_action"
            or effect.get("tool") != tool
            or effect.get("args") != normalized_args
            or effect.get("machine_path") != normalized_machine_path
            or effect.get("value") != value
        ):
            continue
        if required_object_locations:
            effect["required_object_locations"] = _merge_unique_dict_list(
                effect.get("required_object_locations"),
                required_object_locations,
            )
        if required_machine_values:
            effect["required_machine_values"] = _merge_unique_dict_list(
                effect.get("required_machine_values"),
                required_machine_values,
            )
        if required_fixture_controls:
            effect["required_fixture_controls"] = _merge_unique_dict_list(
                effect.get("required_fixture_controls"),
                required_fixture_controls,
            )
        return

    synthesized_effect = {
        "kind": "set_machine_flag_on_action",
        "tool": tool,
        "args": normalized_args,
        "machine_path": normalized_machine_path,
        "value": value,
    }
    if required_object_locations:
        synthesized_effect["required_object_locations"] = [
            dict(requirement) for requirement in required_object_locations
        ]
    if required_machine_values:
        synthesized_effect["required_machine_values"] = [
            dict(requirement) for requirement in required_machine_values
        ]
    if required_fixture_controls:
        synthesized_effect["required_fixture_controls"] = [
            dict(requirement) for requirement in required_fixture_controls
        ]
    task_effects.append(synthesized_effect)


def _goal_has_location_constraint(
    goal_conditions: list[dict[str, Any]],
    *,
    object_id: str,
) -> bool:
    for condition in goal_conditions:
        if not isinstance(condition, dict):
            continue
        if (
            condition.get("kind") in {"object_at_location", "object_at_location_one_of"}
            and condition.get("object_id") == object_id
        ):
            return True
        if (
            condition.get("kind") == "object_count_at_location"
            and object_id in (condition.get("object_ids") or [])
        ):
            return True
    return False


def _object_group_label(object_ids: list[str]) -> str | None:
    if not object_ids:
        return None
    normalized_labels = [
        re.sub(r"\d+$", "", object_id).rstrip("_")
        for object_id in object_ids
        if isinstance(object_id, str)
    ]
    if not normalized_labels:
        return None
    first_label = normalized_labels[0]
    if all(label == first_label for label in normalized_labels):
        return first_label
    return None


def _normalize_grounding_symbols(
    *,
    initial_state: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
) -> None:
    if not isinstance(initial_state, dict) or not isinstance(grounding, dict):
        return
    grounding_symbols = grounding.get("symbols") or {}
    if not isinstance(grounding_symbols, dict):
        return

    fixtures_by_id = initial_state.get("fixtures") or {}
    objects_by_id = initial_state.get("objects") or {}
    fixture_type_counts = Counter(
        fixture_state.get("fixture_type")
        for fixture_state in fixtures_by_id.values()
        if isinstance(fixture_state, dict)
        and isinstance(fixture_state.get("fixture_type"), str)
    )
    symbols_to_drop: list[str] = []
    for symbol_name, symbol_spec in grounding_symbols.items():
        if not isinstance(symbol_spec, dict):
            continue
        entity_type = symbol_spec.get("entity_type")
        if entity_type not in {"object", "fixture"}:
            symbols_to_drop.append(symbol_name)
            continue
        if entity_type == "object" and symbol_name in objects_by_id:
            object_state = objects_by_id.get(symbol_name, {})
            if isinstance(object_state, dict):
                object_type = object_state.get("object_type")
                if (
                    symbol_spec.get("resolver") == "object_by_type"
                    and isinstance(object_type, str)
                    and object_type
                    and not isinstance(symbol_spec.get("object_type"), str)
                ):
                    symbol_spec["object_type"] = object_type
        if entity_type != "fixture" or symbol_name not in fixtures_by_id:
            continue
        fixture_type = fixtures_by_id.get(symbol_name, {}).get("fixture_type")
        if (
            isinstance(fixture_type, str)
            and fixture_type_counts.get(fixture_type) == 1
            and not symbol_name.endswith("_adjacent_surface")
        ):
            symbol_spec["resolver"] = "unique_fixture_type"
            symbol_spec["fixture_type"] = fixture_type
            symbol_spec.pop("object_symbol", None)
            symbol_spec.pop("fixture_symbol", None)
            symbol_spec.pop("fixture_id", None)

    for symbol_name in symbols_to_drop:
        grounding_symbols.pop(symbol_name, None)


def _infer_final_object_locations(
    initial_state: dict[str, Any] | None,
    trajectory_payload: Any,
) -> dict[str, str]:
    final_locations: dict[str, str] = {}
    if not isinstance(initial_state, dict):
        return final_locations

    objects = initial_state.get("objects") or {}
    for object_id, object_state in objects.items():
        if not isinstance(object_id, str) or not isinstance(object_state, dict):
            continue
        location = object_state.get("location")
        if isinstance(location, str):
            final_locations[object_id] = location

    machine_state = initial_state.get("machine_state") or {}
    if not isinstance(trajectory_payload, dict):
        return final_locations

    for step in trajectory_payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        args = step.get("args")
        if not isinstance(args, dict):
            continue
        object_id = args.get("object_id")
        if not isinstance(object_id, str):
            continue
        placed_location: str | None = None
        for arg_name in _DIRECT_PLACEMENT_LOCATION_ARG_NAMES:
            location = args.get(arg_name)
            if isinstance(location, str):
                placed_location = location
                break
        if placed_location is None and step.get("tool") == "place_next_to":
            reference_object_id = args.get("reference_object_id")
            if isinstance(reference_object_id, str):
                reference_location = final_locations.get(reference_object_id)
                if isinstance(reference_location, str):
                    placed_location = reference_location
            reference_fixture_id = args.get("reference_fixture_id")
            if (
                placed_location is None
                and isinstance(reference_fixture_id, str)
                and isinstance(machine_state, dict)
            ):
                adjacent_location_id = (
                    machine_state.get(reference_fixture_id, {}) or {}
                ).get("adjacent_location_id")
                if isinstance(adjacent_location_id, str):
                    placed_location = adjacent_location_id
        if placed_location is not None:
            final_locations[object_id] = placed_location

    return final_locations


def _insert_missing_navigation_before_give_space(
    *,
    initial_state: dict[str, Any] | None,
    trajectory: Any,
) -> None:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return
    steps = trajectory.get("steps") or []
    if not isinstance(steps, list):
        return

    agent_locations = {
        agent_id: agent_state.get("location")
        for agent_id, agent_state in (initial_state.get("agents") or {}).items()
        if isinstance(agent_id, str) and isinstance(agent_state, dict)
    }
    normalized_steps: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        agent_id = step.get("agent")
        step_args = step.get("args")
        if (
            step.get("tool") == "give_space"
            and isinstance(agent_id, str)
            and isinstance(step_args, dict)
            and isinstance(step_args.get("fixture_id"), str)
            and agent_locations.get(agent_id) != step_args["fixture_id"]
        ):
            normalized_steps.append(
                {
                    "step": -1,
                    "agent": agent_id,
                    "tool": "navigate_to_fixture",
                    "args": {"fixture_id": step_args["fixture_id"]},
                    "reasoning": "Moving to the shared fixture before yielding space.",
                }
            )
            agent_locations[agent_id] = step_args["fixture_id"]
        normalized_steps.append(step)
        if (
            step.get("tool") == "navigate_to_fixture"
            and isinstance(agent_id, str)
            and isinstance(step_args, dict)
            and isinstance(step_args.get("fixture_id"), str)
        ):
            agent_locations[agent_id] = step_args["fixture_id"]
        elif step.get("tool") == "give_space" and isinstance(agent_id, str):
            agent_locations[agent_id] = None

    for index, step in enumerate(normalized_steps):
        step["step"] = index
    trajectory["steps"] = normalized_steps


def _resolve_step_destination(
    step: dict[str, Any],
    current_object_locations: dict[str, str],
    machine_state: dict[str, Any] | None,
) -> str | None:
    step_args = step.get("args")
    if not isinstance(step_args, dict):
        return None
    for arg_name in _DIRECT_PLACEMENT_LOCATION_ARG_NAMES:
        location = step_args.get(arg_name)
        if isinstance(location, str):
            return location
    reference_object_id = step_args.get("reference_object_id")
    if isinstance(reference_object_id, str):
        return current_object_locations.get(reference_object_id)
    reference_fixture_id = step_args.get("reference_fixture_id")
    if not isinstance(reference_fixture_id, str):
        return None
    if step.get("tool") == "place_next_to" and isinstance(machine_state, dict):
        adjacent_location_id = (
            machine_state.get(reference_fixture_id, {}) or {}
        ).get("adjacent_location_id")
        if isinstance(adjacent_location_id, str):
            return adjacent_location_id
    return reference_fixture_id


def _rewrite_shared_location_pool_goals(
    goal_conditions: list[dict[str, Any]],
    *,
    initial_state: dict[str, Any] | None,
    final_object_locations: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return list(goal_conditions)

    object_types = {
        object_id: object_state.get("object_type")
        for object_id, object_state in (initial_state.get("objects") or {}).items()
        if isinstance(object_id, str) and isinstance(object_state, dict)
    }
    grouped_conditions: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for condition in goal_conditions:
        if condition.get("kind") != "object_at_location_one_of":
            continue
        object_id = condition.get("object_id")
        locations = condition.get("locations")
        if not isinstance(object_id, str) or not isinstance(locations, list):
            continue
        normalized_locations = tuple(
            location for location in locations if isinstance(location, str)
        )
        if normalized_locations:
            grouped_conditions[normalized_locations].append(condition)

    replacement_goals_by_group: dict[
        tuple[tuple[str, ...], tuple[str, ...]],
        list[dict[str, Any]],
    ] = {}
    rewrite_group_by_object: dict[tuple[tuple[str, ...], str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for locations, grouped_pool_conditions in grouped_conditions.items():
        object_ids_by_type: dict[str, list[str]] = defaultdict(list)
        for condition in grouped_pool_conditions:
            object_id = condition["object_id"]
            object_type = object_types.get(object_id)
            if not isinstance(object_type, str) or not object_type:
                continue
            object_ids_by_type[object_type].append(object_id)

        for grouped_object_ids in object_ids_by_type.values():
            normalized_object_ids = tuple(sorted(dict.fromkeys(grouped_object_ids)))
            if len(normalized_object_ids) <= 1:
                continue
            counts_by_location: Counter[str] = Counter()
            valid_group = True
            for object_id in normalized_object_ids:
                final_location = final_object_locations.get(object_id)
                if not isinstance(final_location, str) or final_location not in locations:
                    valid_group = False
                    break
                counts_by_location[final_location] += 1
            if not valid_group or not counts_by_location:
                continue
            distinct_object_types = {
                object_types.get(condition["object_id"])
                for condition in grouped_pool_conditions
                if isinstance(condition.get("object_id"), str)
            }
            should_rewrite = (
                len(
                    {
                        object_type
                        for object_type in distinct_object_types
                        if isinstance(object_type, str) and object_type
                    }
                )
                > 1
                or len(normalized_object_ids) > len(locations)
                or any(count > 1 for count in counts_by_location.values())
            )
            if not should_rewrite:
                continue

            rewrite_group = (locations, normalized_object_ids)
            replacement_goals_by_group[rewrite_group] = [
                {
                    "kind": "object_count_at_location",
                    "object_ids": list(normalized_object_ids),
                    "location": location,
                    "count": count,
                }
                for location, count in sorted(counts_by_location.items())
            ]
            for object_id in normalized_object_ids:
                rewrite_group_by_object[(locations, object_id)] = rewrite_group

    normalized_goal_conditions: list[dict[str, Any]] = []
    emitted_rewrite_groups: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for condition in goal_conditions:
        if condition.get("kind") == "object_at_location_one_of":
            object_id = condition.get("object_id")
            locations = condition.get("locations")
            if isinstance(object_id, str) and isinstance(locations, list):
                normalized_locations = tuple(
                    location for location in locations if isinstance(location, str)
                )
                rewrite_group = rewrite_group_by_object.get(
                    (normalized_locations, object_id)
                )
                if rewrite_group is not None:
                    if rewrite_group not in emitted_rewrite_groups:
                        normalized_goal_conditions.extend(
                            replacement_goals_by_group[rewrite_group]
                        )
                        emitted_rewrite_groups.add(rewrite_group)
                    continue
        normalized_goal_conditions.append(condition)

    return normalized_goal_conditions


def _ensure_fixture_control(
    fixtures_by_id: dict[str, Any],
    *,
    fixture_id: str,
    control_id: str,
    tool_name: str,
) -> None:
    fixture_state = fixtures_by_id.get(fixture_id)
    if not isinstance(fixture_state, dict):
        return
    controls = fixture_state.setdefault("controls", {})
    if not isinstance(controls, dict):
        return
    control_state = controls.setdefault(control_id, {})
    if not isinstance(control_state, dict):
        return
    control_type = _CONTROL_TOOL_TYPES.get(tool_name)
    if isinstance(control_type, str):
        control_state.setdefault("control_type", control_type)


def _infer_effect_control_id(
    effect: dict[str, Any],
    trajectory: Any,
    allowed_tool_specs: dict[str, Any],
) -> str | None:
    effect_args = effect.get("args")
    if not isinstance(effect_args, dict):
        return None
    if isinstance(effect_args.get("control_id"), str):
        return effect_args["control_id"]

    tool_name = effect.get("tool")
    if tool_name not in _CONTROL_TOOL_TYPES:
        return None

    matching_control_ids: list[str] = []
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict) or step.get("tool") != tool_name:
                continue
            step_args = step.get("args")
            if not isinstance(step_args, dict):
                continue
            if any(
                step_args.get(arg_name) != arg_value
                for arg_name, arg_value in effect_args.items()
            ):
                continue
            control_id = step_args.get("control_id")
            if isinstance(control_id, str):
                matching_control_ids.append(control_id)
    deduped_control_ids = tuple(dict.fromkeys(matching_control_ids))
    if len(deduped_control_ids) == 1:
        return deduped_control_ids[0]

    tool_spec = allowed_tool_specs.get(tool_name, {})
    allowed_control_ids = tool_spec.get("allowed_control_ids")
    if (
        isinstance(allowed_control_ids, list)
        and len(allowed_control_ids) == 1
        and isinstance(allowed_control_ids[0], str)
    ):
        return allowed_control_ids[0]
    return None


def _backfill_fixture_controls(
    *,
    initial_state: dict[str, Any] | None,
    allowed_tool_specs: dict[str, Any],
    trajectory: Any,
    task_effects: list[dict[str, Any]],
) -> None:
    if not isinstance(initial_state, dict):
        return
    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return

    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        tool_name = effect.get("tool")
        if tool_name not in _CONTROL_TOOL_TYPES:
            continue
        effect_args = effect.get("args")
        if not isinstance(effect_args, dict):
            continue
        if "control_id" not in effect_args:
            inferred_control_id = _infer_effect_control_id(
                effect,
                trajectory,
                allowed_tool_specs,
            )
            if isinstance(inferred_control_id, str):
                effect_args["control_id"] = inferred_control_id

    usage_specs: list[tuple[str, dict[str, Any]]] = []
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            usage_specs.append((str(step.get("tool")), step.get("args") or {}))
    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        usage_specs.append((str(effect.get("tool")), effect.get("args") or {}))

    for tool_name, args in usage_specs:
        if tool_name not in _CONTROL_TOOL_TYPES or not isinstance(args, dict):
            continue
        target_id = args.get("target_id")
        control_id = args.get("control_id")
        if not isinstance(target_id, str) or not isinstance(control_id, str):
            continue
        _ensure_fixture_control(
            fixtures_by_id,
            fixture_id=target_id,
            control_id=control_id,
            tool_name=tool_name,
        )

    for tool_name, tool_spec in allowed_tool_specs.items():
        if tool_name not in _CONTROL_TOOL_TYPES or not isinstance(tool_spec, dict):
            continue
        allowed_target_ids = [
            target_id
            for target_id in tool_spec.get("allowed_target_ids") or []
            if isinstance(target_id, str)
        ]
        allowed_control_ids = [
            control_id
            for control_id in tool_spec.get("allowed_control_ids") or []
            if isinstance(control_id, str)
        ]
        if len(allowed_target_ids) != 1:
            continue
        fixture_id = allowed_target_ids[0]
        for control_id in allowed_control_ids:
            _ensure_fixture_control(
                fixtures_by_id,
                fixture_id=fixture_id,
                control_id=control_id,
                tool_name=tool_name,
            )


def _infer_support_site_parent_fixture(
    *,
    site_id: str,
    object_id: str | None,
    initial_state: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
) -> str | None:
    if not isinstance(initial_state, dict):
        return None
    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return None

    if isinstance(object_id, str) and isinstance(grounding, dict):
        symbol_spec = (grounding.get("symbols") or {}).get(object_id, {})
        preferred_fixture_types = [
            fixture_type
            for fixture_type in symbol_spec.get("preferred_fixture_types") or []
            if isinstance(fixture_type, str)
        ]
        candidate_fixture_ids = [
            fixture_id
            for fixture_id, fixture_state in fixtures_by_id.items()
            if isinstance(fixture_id, str)
            and isinstance(fixture_state, dict)
            and fixture_state.get("fixture_type") in preferred_fixture_types
        ]
        deduped_candidate_fixture_ids = tuple(dict.fromkeys(candidate_fixture_ids))
        if len(deduped_candidate_fixture_ids) == 1:
            return deduped_candidate_fixture_ids[0]

    name_matched_fixture_ids = [
        fixture_id
        for fixture_id, fixture_state in fixtures_by_id.items()
        if isinstance(fixture_id, str)
        and isinstance(fixture_state, dict)
        and (
            fixture_id in site_id
            or (
                isinstance(fixture_state.get("fixture_type"), str)
                and fixture_state["fixture_type"] in site_id
            )
        )
    ]
    deduped_name_matched_fixture_ids = tuple(dict.fromkeys(name_matched_fixture_ids))
    if len(deduped_name_matched_fixture_ids) == 1:
        return deduped_name_matched_fixture_ids[0]
    return None


def _backfill_fixture_support_sites(
    *,
    initial_state: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
    trajectory: Any,
    task_effects: list[dict[str, Any]],
) -> None:
    if not isinstance(initial_state, dict):
        return
    fixtures_by_id = initial_state.get("fixtures") or {}
    objects_by_id = initial_state.get("objects") or {}
    if not isinstance(fixtures_by_id, dict) or not isinstance(objects_by_id, dict):
        return

    known_site_ids: set[str] = set(fixtures_by_id)
    known_site_ids.update(objects_by_id)
    for fixture_state in fixtures_by_id.values():
        if not isinstance(fixture_state, dict):
            continue
        for section_name in ("parts", "controls", "support_sites"):
            section = fixture_state.get(section_name) or {}
            if isinstance(section, dict):
                known_site_ids.update(
                    key for key in section if isinstance(key, str)
                )

    candidate_usages: list[tuple[str, str | None]] = []
    for object_id, object_state in objects_by_id.items():
        if not isinstance(object_id, str) or not isinstance(object_state, dict):
            continue
        location = object_state.get("location")
        if isinstance(location, str):
            candidate_usages.append((location, object_id))

    def _record_step_sites(step_like: Any) -> None:
        if not isinstance(step_like, dict):
            return
        step_args = step_like.get("args")
        if not isinstance(step_args, dict):
            return
        object_id = step_args.get("object_id")
        if not isinstance(object_id, str):
            object_id = None
        for arg_name in ("source_id", "support_id", "receptacle_id"):
            site_id = step_args.get(arg_name)
            if isinstance(site_id, str):
                candidate_usages.append((site_id, object_id))

    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            _record_step_sites(step)
    for effect in task_effects:
        _record_step_sites(effect)

    for site_id, object_id in candidate_usages:
        if site_id in known_site_ids:
            continue
        parent_fixture_id = _infer_support_site_parent_fixture(
            site_id=site_id,
            object_id=object_id,
            initial_state=initial_state,
            grounding=grounding,
        )
        if not isinstance(parent_fixture_id, str):
            continue
        fixture_state = fixtures_by_id.get(parent_fixture_id)
        if not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.setdefault("support_sites", {})
        if not isinstance(support_sites, dict):
            continue
        support_sites.setdefault(site_id, {"site_type": "support"})
        known_site_ids.add(site_id)


def _rewrite_control_state_goals(
    *,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return goal_conditions

    rewritten_machine_paths: set[tuple[str, ...]] = set()
    normalized_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        if not isinstance(condition, dict):
            normalized_goal_conditions.append(condition)
            continue
        condition_kind = condition.get("kind")
        if condition_kind not in {"machine_flag_true", "machine_flag_equals"}:
            normalized_goal_conditions.append(condition)
            continue
        machine_path = condition.get("machine_path")
        if not isinstance(machine_path, list) or len(machine_path) != 2:
            normalized_goal_conditions.append(condition)
            continue
        matching_effects = [
            effect
            for effect in task_effects
            if isinstance(effect, dict)
            and effect.get("tool") in _CONTROL_TOOL_TYPES
            and effect.get("machine_path") == machine_path
            and isinstance((effect.get("args") or {}).get("target_id"), str)
            and isinstance((effect.get("args") or {}).get("control_id"), str)
        ]
        if len(matching_effects) != 1:
            normalized_goal_conditions.append(condition)
            continue
        effect = matching_effects[0]
        effect_args = effect.get("args") or {}
        if machine_path[0] != effect_args.get("target_id"):
            normalized_goal_conditions.append(condition)
            continue
        if effect.get("tool") == "set_rotary_control":
            control_state = effect_args.get("goal")
        else:
            current_state = (
                ((initial_state.get("fixtures") or {}).get(effect_args["target_id"], {}) or {})
                .get("controls", {})
                .get(effect_args["control_id"], {})
                .get("state")
            )
            control_state = _normalize_pressed_control_state(
                current_state,
                tool_name=str(effect.get("tool")),
            )
        if not isinstance(control_state, str):
            normalized_goal_conditions.append(condition)
            continue
        normalized_goal_conditions.append(
            {
                "kind": "fixture_control_state",
                "fixture_id": effect_args["target_id"],
                "control_id": effect_args["control_id"],
                "state": control_state,
            }
        )
        rewritten_machine_paths.add(tuple(machine_path))

    if not rewritten_machine_paths:
        return normalized_goal_conditions

    retained_effects: list[dict[str, Any]] = []
    for effect in task_effects:
        if (
            isinstance(effect, dict)
            and effect.get("tool") in _CONTROL_TOOL_TYPES
            and tuple(effect.get("machine_path") or ()) in rewritten_machine_paths
        ):
            continue
        retained_effects.append(effect)
    task_effects[:] = retained_effects

    machine_state = initial_state.get("machine_state")
    if isinstance(machine_state, dict):
        for fixture_id, state_key in rewritten_machine_paths:
            fixture_machine_state = machine_state.get(fixture_id)
            if not isinstance(fixture_machine_state, dict):
                continue
            fixture_machine_state.pop(state_key, None)
    if isinstance(initial_public_state, dict):
        for _fixture_id, state_key in rewritten_machine_paths:
            initial_public_state.pop(state_key, None)

    return normalized_goal_conditions


def _normalize_guarded_communicate_effects(task_effects: list[dict[str, Any]]) -> None:
    for effect in task_effects:
        if not isinstance(effect, dict) or effect.get("tool") != "communicate":
            continue
        effect_args = effect.get("args")
        if not isinstance(effect_args, dict):
            continue
        if not (
            effect.get("required_object_locations")
            or effect.get("required_machine_values")
        ):
            continue
        # Guarded communicate effects should not depend on exact message text.
        # Keep any semantic routing args like `to`, but drop `message`.
        effect["args"] = {
            arg_name: arg_value
            for arg_name, arg_value in effect_args.items()
            if arg_name != "message"
        }


def _rewrite_container_pickup_preconditions(
    *,
    initial_state: dict[str, Any] | None,
    task_preconditions: list[dict[str, Any]],
    trajectory: Any,
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return task_preconditions

    initial_locations = {
        object_id: object_state.get("location")
        for object_id, object_state in (initial_state.get("objects") or {}).items()
        if isinstance(object_id, str)
        and isinstance(object_state, dict)
        and isinstance(object_state.get("location"), str)
    }
    loaded_object_ids_by_container: dict[str, list[str]] = defaultdict(list)
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_args = step.get("args")
        if not isinstance(step_args, dict):
            continue
        step_tool = step.get("tool")
        if step_tool == "place_in_receptacle":
            inner_object_id = step_args.get("object_id")
            container_id = step_args.get("receptacle_id")
            if isinstance(inner_object_id, str) and isinstance(container_id, str):
                loaded_object_ids_by_container[container_id].append(inner_object_id)
            continue
        if step_tool == "place_on_object":
            inner_object_id = step_args.get("object_id")
            container_id = step_args.get("support_object_id")
            if isinstance(inner_object_id, str) and isinstance(container_id, str):
                loaded_object_ids_by_container[container_id].append(inner_object_id)
            continue
        if step_tool != "pick_up_object":
            continue
        container_id = step_args.get("object_id")
        if not isinstance(container_id, str):
            continue
        candidate_inner_object_ids = tuple(
            dict.fromkeys(loaded_object_ids_by_container.get(container_id) or ())
        )
        if len(candidate_inner_object_ids) != 1:
            continue
        initial_location = initial_locations.get(container_id)
        if not isinstance(initial_location, str):
            continue
        inner_object_id = candidate_inner_object_ids[0]
        for condition in task_preconditions:
            if not isinstance(condition, dict):
                continue
            if condition.get("kind") != "object_location_required_for_action":
                continue
            if condition.get("tool") != "pick_up_object":
                continue
            if condition.get("object_id") != container_id:
                continue
            if condition.get("required_location") != initial_location:
                continue
            condition["object_id"] = inner_object_id
            condition["required_location"] = container_id
            condition["arg_name"] = "object_id"
            condition["arg_value"] = container_id
            condition["message"] = (
                f"{inner_object_id} must already be at {container_id} before "
                f"picking up {container_id}."
            )
    return task_preconditions


def _move_guarded_communicate_effects_to_following_action(
    *,
    task_effects: list[dict[str, Any]],
    trajectory: Any,
) -> None:
    if not isinstance(trajectory, dict):
        return
    steps = [step for step in (trajectory.get("steps") or []) if isinstance(step, dict)]
    for effect in task_effects:
        if not isinstance(effect, dict) or effect.get("tool") != "communicate":
            continue
        if not (
            effect.get("required_object_locations")
            or effect.get("required_machine_values")
        ):
            continue
        effect_args = effect.get("args")
        if not isinstance(effect_args, dict):
            continue
        matching_step_indexes = [
            index
            for index, step in enumerate(steps)
            if step.get("tool") == "communicate" and (step.get("args") or {}) == effect_args
        ]
        if len(matching_step_indexes) != 1:
            continue
        next_action_step = next(
            (
                step
                for step in steps[matching_step_indexes[0] + 1 :]
                if step.get("tool") in _EFFECT_CARRYING_TOOL_NAMES
            ),
            None,
        )
        if not isinstance(next_action_step, dict):
            continue
        next_action_args = next_action_step.get("args")
        if not isinstance(next_action_args, dict):
            continue
        effect["tool"] = next_action_step["tool"]
        effect["args"] = dict(next_action_args)


def _synthesize_timed_completion_goal_and_effect(
    *,
    composite_task: str,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    task_goal: str,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    trajectory: Any,
    final_object_locations: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions
    normalized_goal_text = _normalize_free_text(task_goal)
    if not any(
        token in normalized_goal_text
        for token in ("seconds", "second", "timesteps", "duration")
    ):
        return goal_conditions
    if not any(
        phrase in normalized_goal_text
        for phrase in ("turn the water off", "turn the faucet off", "turn the handle off", "water off")
    ):
        return goal_conditions
    if any(
        isinstance(condition, dict)
        and condition.get("kind") in {"machine_flag_true", "machine_flag_equals"}
        and isinstance(condition.get("machine_path"), list)
        and condition["machine_path"]
        and isinstance(condition["machine_path"][-1], str)
        and "complete" in condition["machine_path"][-1]
        for condition in goal_conditions
    ):
        return goal_conditions

    steps = [step for step in (trajectory.get("steps") or []) if isinstance(step, dict)]
    final_off_step = next(
        (
            step
            for step in reversed(steps)
            if step.get("tool") == "set_rotary_control"
            and (step.get("args") or {}).get("goal") == "off"
        ),
        None,
    )
    if not isinstance(final_off_step, dict):
        return goal_conditions

    machine_namespace = _primary_machine_namespace(
        initial_state=initial_state,
        composite_task=composite_task,
    )
    if not isinstance(machine_namespace, str):
        return goal_conditions
    machine_key = "rinsing_complete" if "rinse" in normalized_goal_text else "timed_complete"
    machine_path = [machine_namespace, machine_key]
    _ensure_machine_flag_default(
        initial_state,
        machine_path=(machine_namespace, machine_key),
        default=False,
    )
    _ensure_public_state_default(initial_public_state, machine_key, False)
    required_object_locations = [
        {"object_id": object_id, "location": location}
        for object_id, location in final_object_locations.items()
        if isinstance(object_id, str)
        and isinstance(location, str)
        and location == (final_off_step.get("args") or {}).get("target_id")
    ]
    _ensure_machine_flag_effect(
        task_effects,
        tool="set_rotary_control",
        args=dict(final_off_step.get("args") or {}),
        machine_path=machine_path,
        value=True,
        required_object_locations=required_object_locations or None,
    )
    return [
        *goal_conditions,
        {
            "kind": "machine_flag_true",
            "machine_path": machine_path,
        },
    ]


def _add_water_state_guards_for_completion_effects(
    *,
    initial_state: dict[str, Any] | None,
    task_goal: str,
    task_effects: list[dict[str, Any]],
    trajectory: Any,
) -> None:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return
    normalized_goal_text = _normalize_free_text(task_goal)
    mentions_hot_water = "hot water" in normalized_goal_text
    steps = [step for step in (trajectory.get("steps") or []) if isinstance(step, dict)]
    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        machine_path = effect.get("machine_path")
        if (
            not isinstance(machine_path, list)
            or not machine_path
            or not isinstance(machine_path[-1], str)
            or "complete" not in machine_path[-1]
        ):
            continue
        effect_args = effect.get("args")
        if (
            effect.get("tool") != "set_rotary_control"
            or not isinstance(effect_args, dict)
            or effect_args.get("goal") != "off"
        ):
            continue
        target_id = effect_args.get("target_id")
        control_id = effect_args.get("control_id")
        if not isinstance(target_id, str) or not isinstance(control_id, str):
            continue
        matching_on_step = next(
            (
                step
                for step in steps
                if step.get("tool") == "set_rotary_control"
                and (step.get("args") or {}).get("target_id") == target_id
                and (step.get("args") or {}).get("control_id") == control_id
                and (step.get("args") or {}).get("goal") != "off"
            ),
            None,
        )
        if not isinstance(matching_on_step, dict):
            continue
        on_args = dict(matching_on_step.get("args") or {})
        activation_goal = on_args.get("goal")
        water_on_path = [target_id, "water_on"]
        _ensure_machine_flag_default(
            initial_state,
            machine_path=(target_id, "water_on"),
            default=False,
        )
        _ensure_machine_flag_effect(
            task_effects,
            tool="set_rotary_control",
            args=on_args,
            machine_path=water_on_path,
            value=True,
        )
        required_machine_values = [
            {
                "machine_path": water_on_path,
                "value": True,
            }
        ]
        if mentions_hot_water or activation_goal == "hot":
            water_hot_path = [target_id, "water_hot"]
            _ensure_machine_flag_default(
                initial_state,
                machine_path=(target_id, "water_hot"),
                default=False,
            )
            _ensure_machine_flag_effect(
                task_effects,
                tool="set_rotary_control",
                args=on_args,
                machine_path=water_hot_path,
                value=True,
            )
            required_machine_values.append(
                {
                    "machine_path": water_hot_path,
                    "value": True,
                }
            )
            effect["required_machine_values"] = _merge_unique_dict_list(
            effect.get("required_machine_values"),
            required_machine_values,
        )


def _add_persistent_initial_on_control_goals(
    *,
    initial_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    trajectory: Any,
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions
    controls_turned_on_by_fixture: dict[str, set[str]] = defaultdict(set)
    controls_turned_off: set[tuple[str, str]] = set()
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict) or step.get("tool") != "set_rotary_control":
            continue
        step_args = step.get("args")
        if not isinstance(step_args, dict):
            continue
        fixture_id = step_args.get("target_id")
        control_id = step_args.get("control_id")
        goal = step_args.get("goal")
        if not isinstance(fixture_id, str) or not isinstance(control_id, str):
            continue
        if goal == "off":
            controls_turned_off.add((fixture_id, control_id))
        else:
            controls_turned_on_by_fixture[fixture_id].add(control_id)

    normalized_goal_conditions = list(goal_conditions)
    existing_control_goals = {
        (condition.get("fixture_id"), condition.get("control_id"))
        for condition in normalized_goal_conditions
        if isinstance(condition, dict) and condition.get("kind") == "fixture_control_state"
    }
    for fixture_id, fixture_state in (initial_state.get("fixtures") or {}).items():
        if (
            not isinstance(fixture_id, str)
            or not isinstance(fixture_state, dict)
            or not controls_turned_on_by_fixture.get(fixture_id)
        ):
            continue
        fixture_controls = fixture_state.get("controls") or {}
        if not isinstance(fixture_controls, dict):
            continue
        for control_id, control_state in fixture_controls.items():
            if (
                not isinstance(control_id, str)
                or not isinstance(control_state, dict)
                or control_state.get("state") != "on"
                or (fixture_id, control_id) in controls_turned_off
                or (fixture_id, control_id) in existing_control_goals
            ):
                continue
            normalized_goal_conditions.append(
                {
                    "kind": "fixture_control_state",
                    "fixture_id": fixture_id,
                    "control_id": control_id,
                    "state": "on",
                }
            )
            existing_control_goals.add((fixture_id, control_id))
    return normalized_goal_conditions


def _drop_redundant_water_off_goals(
    goal_conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    off_control_fixtures = {
        condition.get("fixture_id")
        for condition in goal_conditions
        if isinstance(condition, dict)
        and condition.get("kind") == "fixture_control_state"
        and condition.get("state") == "off"
        and isinstance(condition.get("fixture_id"), str)
    }
    normalized_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        machine_path = condition.get("machine_path") if isinstance(condition, dict) else None
        if (
            isinstance(condition, dict)
            and condition.get("kind") == "machine_flag_equals"
            and condition.get("value") is False
            and isinstance(machine_path, list)
            and machine_path == [machine_path[0], "water_on"]
            and isinstance(machine_path[0], str)
            and machine_path[0] in off_control_fixtures
        ):
            continue
        normalized_goal_conditions.append(condition)
    return normalized_goal_conditions


def _consolidate_upright_machine_flags(
    *,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return goal_conditions
    object_ids = [
        object_id
        for object_id in (initial_state.get("objects") or {})
        if isinstance(object_id, str)
    ]

    def _matching_machine_paths(object_id: str) -> list[tuple[str, str]]:
        object_tokens = set(object_id.lower().split("_"))
        candidate_paths: list[tuple[str, str]] = []
        for item in [*goal_conditions, *task_effects]:
            if not isinstance(item, dict):
                continue
            machine_path = item.get("machine_path")
            if not isinstance(machine_path, list) or len(machine_path) != 2:
                continue
            machine_namespace, machine_key = machine_path
            if not isinstance(machine_namespace, str) or not isinstance(machine_key, str):
                continue
            if not machine_key.endswith("_upright"):
                continue
            key_tokens = set(machine_key[: -len("_upright")].split("_"))
            if object_tokens & key_tokens:
                candidate_paths.append((machine_namespace, machine_key))
        return list(dict.fromkeys(candidate_paths))

    for object_id in object_ids:
        candidate_paths = _matching_machine_paths(object_id)
        if len(candidate_paths) <= 1:
            continue
        canonical_path = min(candidate_paths, key=lambda path: (len(path[1]), path[1]))
        for item in [*goal_conditions, *task_effects]:
            if not isinstance(item, dict):
                continue
            machine_path = item.get("machine_path")
            if not isinstance(machine_path, list) or len(machine_path) != 2:
                continue
            if tuple(machine_path) in candidate_paths:
                item["machine_path"] = list(canonical_path)
        machine_state = initial_state.get("machine_state") or {}
        if isinstance(machine_state, dict):
            canonical_namespace = machine_state.setdefault(canonical_path[0], {})
            if isinstance(canonical_namespace, dict):
                canonical_namespace.setdefault(canonical_path[1], False)
            for machine_namespace, machine_key in candidate_paths:
                if machine_namespace == canonical_path[0] and machine_key == canonical_path[1]:
                    continue
                namespace_state = machine_state.get(machine_namespace)
                if not isinstance(namespace_state, dict):
                    continue
                namespace_state.pop(machine_key, None)
        if isinstance(initial_public_state, dict):
            initial_public_state.setdefault(canonical_path[1], False)
            for _machine_namespace, machine_key in candidate_paths:
                if machine_key != canonical_path[1]:
                    initial_public_state.pop(machine_key, None)

    deduped_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        if any(existing == condition for existing in deduped_goal_conditions):
            continue
        deduped_goal_conditions.append(condition)

    deduped_task_effects: list[dict[str, Any]] = []
    for effect in task_effects:
        if any(existing == effect for existing in deduped_task_effects):
            continue
        deduped_task_effects.append(effect)
    task_effects[:] = deduped_task_effects
    return deduped_goal_conditions


def _drop_impossible_location_preconditions(
    *,
    initial_state: dict[str, Any] | None,
    task_preconditions: list[dict[str, Any]],
    trajectory: Any,
) -> list[dict[str, Any]]:
    current_object_locations: dict[str, str] = {}
    machine_state = {}
    if isinstance(initial_state, dict):
        current_object_locations = {
            object_id: object_state.get("location")
            for object_id, object_state in (initial_state.get("objects") or {}).items()
            if isinstance(object_id, str) and isinstance(object_state, dict)
            and isinstance(object_state.get("location"), str)
        }
        machine_state = initial_state.get("machine_state") or {}

    impossible_condition_ids: set[int] = set()
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            step_tool = step.get("tool")
            step_args = step.get("args")
            if not isinstance(step_args, dict):
                continue
            step_object_id = step_args.get("object_id")
            for index, condition in enumerate(task_preconditions):
                if not isinstance(condition, dict):
                    continue
                if condition.get("kind") != "object_location_required_for_action":
                    continue
                if step_tool != condition.get("tool"):
                    continue
                condition_object_id = condition.get("object_id")
                if (
                    not isinstance(step_object_id, str)
                    or step_object_id != condition_object_id
                ):
                    continue
                actual_location = current_object_locations.get(condition_object_id)
                if actual_location != condition.get("required_location"):
                    impossible_condition_ids.add(index)

            if not isinstance(step_object_id, str):
                continue
            if step_tool == "pick_up_object":
                current_object_locations[step_object_id] = f"held_by_{step.get('agent')}"
            elif step_tool in RELEASE_TOOL_NAMES:
                destination = _resolve_step_destination(
                    step,
                    current_object_locations,
                    machine_state if isinstance(machine_state, dict) else None,
                )
                if isinstance(destination, str):
                    current_object_locations[step_object_id] = destination

    normalized_preconditions: list[dict[str, Any]] = []
    for index, condition in enumerate(task_preconditions):
        if not isinstance(condition, dict):
            normalized_preconditions.append(condition)
            continue
        if index in impossible_condition_ids:
            continue
        normalized_preconditions.append(condition)
    return normalized_preconditions


def _synthesize_fixture_part_action_preconditions(
    *,
    initial_state: dict[str, Any] | None,
    trajectory: Any,
    task_preconditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return task_preconditions
    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return task_preconditions

    existing_condition_keys = {
        (
            condition.get("kind"),
            condition.get("tool"),
            condition.get("fixture_id"),
            condition.get("part_id"),
            condition.get("arg_name"),
            condition.get("arg_value"),
        )
        for condition in task_preconditions
        if isinstance(condition, dict)
    }
    normalized_preconditions = list(task_preconditions)
    if not isinstance(trajectory, dict):
        return normalized_preconditions

    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict) or step.get("tool") != "place_in_receptacle":
            continue
        step_args = step.get("args")
        if not isinstance(step_args, dict):
            continue
        receptacle_id = step_args.get("receptacle_id")
        if not isinstance(receptacle_id, str):
            continue
        fixture_state = fixtures_by_id.get(receptacle_id)
        if not isinstance(fixture_state, dict):
            continue
        fixture_parts = fixture_state.get("parts") or {}
        if not isinstance(fixture_parts, dict) or len(fixture_parts) != 1:
            continue
        part_id, part_state = next(iter(fixture_parts.items()))
        if not isinstance(part_id, str) or not isinstance(part_state, dict):
            continue
        if part_state.get("part_type") not in {"hinged_part", "sliding_part"}:
            continue
        condition_key = (
            "fixture_part_state_required_for_action",
            "place_in_receptacle",
            receptacle_id,
            part_id,
            "receptacle_id",
            receptacle_id,
        )
        if condition_key in existing_condition_keys:
            continue
        normalized_preconditions.append(
            {
                "kind": "fixture_part_state_required_for_action",
                "tool": "place_in_receptacle",
                "fixture_id": receptacle_id,
                "part_id": part_id,
                "required_state": "open",
                "arg_name": "receptacle_id",
                "arg_value": receptacle_id,
                "message": (
                    f"{receptacle_id}.{part_id} must be open before placing objects "
                    f"into {receptacle_id}."
                ),
            }
        )
        existing_condition_keys.add(condition_key)
    return normalized_preconditions


def _normalize_trajectory_symbolic_args(
    *,
    initial_state: dict[str, Any] | None,
    allowed_tool_specs: dict[str, Any],
    trajectory: Any,
    task_effects: list[dict[str, Any]],
) -> None:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return

    current_object_locations = {
        object_id: object_state.get("location")
        for object_id, object_state in (initial_state.get("objects") or {}).items()
        if isinstance(object_id, str)
        and isinstance(object_state, dict)
        and isinstance(object_state.get("location"), str)
    }
    machine_state = initial_state.get("machine_state") or {}
    used_ids_by_tool_arg: dict[tuple[str, str], list[str]] = defaultdict(list)

    def _record_arg_usage(tool_name: str, args: dict[str, Any]) -> None:
        tool_spec = allowed_tool_specs.get(tool_name, {})
        if not isinstance(tool_spec, dict):
            return
        for arg_name, arg_value in args.items():
            if not isinstance(arg_value, str):
                continue
            allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
            if allowed_ids_key is None:
                continue
            if allowed_ids_key not in tool_spec:
                continue
            used_ids_by_tool_arg[(tool_name, arg_name)].append(arg_value)

    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        step_args = step.get("args")
        if not isinstance(tool_name, str) or not isinstance(step_args, dict):
            continue
        if tool_name == "pick_up_object":
            object_id = step_args.get("object_id")
            actual_location = (
                current_object_locations.get(object_id)
                if isinstance(object_id, str)
                else None
            )
            if isinstance(actual_location, str):
                step_args["source_id"] = actual_location
        _record_arg_usage(tool_name, step_args)

        object_id = step_args.get("object_id")
        if not isinstance(object_id, str):
            continue
        if tool_name == "pick_up_object":
            current_object_locations[object_id] = f"held_by_{step.get('agent')}"
        elif tool_name in RELEASE_TOOL_NAMES:
            destination = _resolve_step_destination(
                step,
                current_object_locations,
                machine_state if isinstance(machine_state, dict) else None,
            )
            if isinstance(destination, str):
                current_object_locations[object_id] = destination

    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        tool_name = effect.get("tool")
        effect_args = effect.get("args")
        if not isinstance(tool_name, str) or not isinstance(effect_args, dict):
            continue
        _record_arg_usage(tool_name, effect_args)

    pick_up_tool_spec = allowed_tool_specs.get("pick_up_object")
    if isinstance(pick_up_tool_spec, dict) and "allowed_source_ids" in pick_up_tool_spec:
        for object_id in pick_up_tool_spec.get("allowed_object_ids") or []:
            object_location = (
                (initial_state.get("objects") or {}).get(object_id, {}).get("location")
                if isinstance(object_id, str)
                else None
            )
            if isinstance(object_location, str):
                used_ids_by_tool_arg[("pick_up_object", "source_id")].append(
                    object_location
                )

    for (tool_name, arg_name), used_ids in used_ids_by_tool_arg.items():
        tool_spec = allowed_tool_specs.get(tool_name, {})
        if not isinstance(tool_spec, dict):
            continue
        allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
        if allowed_ids_key is None or allowed_ids_key not in tool_spec:
            continue
        existing_ids = [
            value
            for value in tool_spec.get(allowed_ids_key) or []
            if isinstance(value, str)
        ]
        for used_id in used_ids:
            if used_id not in existing_ids:
                existing_ids.append(used_id)
        tool_spec[allowed_ids_key] = existing_ids


def _rewrite_multi_state_control_goals(
    *,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    trajectory: Any,
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions

    grouped_conditions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for condition in goal_conditions:
        if (
            isinstance(condition, dict)
            and condition.get("kind") == "fixture_control_state"
            and isinstance(condition.get("fixture_id"), str)
            and isinstance(condition.get("control_id"), str)
            and isinstance(condition.get("state"), str)
        ):
            grouped_conditions[
                (condition["fixture_id"], condition["control_id"])
            ].append(condition)

    final_goal_states: dict[tuple[str, str], str] = {}
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        step_args = step.get("args")
        if tool_name not in _CONTROL_TOOL_TYPES or not isinstance(step_args, dict):
            continue
        target_id = step_args.get("target_id")
        control_id = step_args.get("control_id")
        if not isinstance(target_id, str) or not isinstance(control_id, str):
            continue
        if tool_name == "set_rotary_control":
            goal_state = step_args.get("goal")
        else:
            current_state = (
                ((initial_state.get("fixtures") or {}).get(target_id, {}) or {})
                .get("controls", {})
                .get(control_id, {})
                .get("state")
            )
            goal_state = _normalize_pressed_control_state(
                current_state,
                tool_name=tool_name,
            )
        if isinstance(goal_state, str):
            final_goal_states[(target_id, control_id)] = goal_state

    rewritten_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        if not isinstance(condition, dict) or condition.get("kind") != "fixture_control_state":
            rewritten_goal_conditions.append(condition)
            continue
        condition_key = (condition.get("fixture_id"), condition.get("control_id"))
        if not (
            isinstance(condition_key[0], str)
            and isinstance(condition_key[1], str)
            and len(
                {
                    grouped_condition.get("state")
                    for grouped_condition in grouped_conditions[condition_key]
                    if isinstance(grouped_condition.get("state"), str)
                }
            )
            > 1
        ):
            rewritten_goal_conditions.append(condition)
            continue
        final_state = final_goal_states.get(condition_key)
        if condition.get("state") == final_state:
            rewritten_goal_conditions.append(condition)
            continue

        fixture_id, control_id = condition_key
        visited_state = condition["state"]
        machine_key = f"{control_id}_{visited_state}_visited"
        initial_state.setdefault("machine_state", {}).setdefault(fixture_id, {})[
            machine_key
        ] = False
        if isinstance(initial_public_state, dict):
            initial_public_state.setdefault(machine_key, False)

        machine_path = [fixture_id, machine_key]
        rewritten_goal_conditions.append(
            {
                "kind": "machine_flag_true",
                "machine_path": machine_path,
            }
        )
        if any(
            isinstance(effect, dict) and effect.get("machine_path") == machine_path
            for effect in task_effects
        ):
            continue

        matching_step = next(
            (
                step
                for step in trajectory.get("steps") or []
                if isinstance(step, dict)
                and step.get("tool") in _CONTROL_TOOL_TYPES
                and (step.get("args") or {}).get("target_id") == fixture_id
                and (step.get("args") or {}).get("control_id") == control_id
                and (
                    (
                        step.get("tool") == "set_rotary_control"
                        and (step.get("args") or {}).get("goal") == visited_state
                    )
                    or (
                        step.get("tool") in {"press_button", "press_lever"}
                        and final_goal_states.get(condition_key) == visited_state
                    )
                )
            ),
            None,
        )
        if not isinstance(matching_step, dict):
            continue
        effect_args = dict(matching_step.get("args") or {})
        synthesized_effect = {
            "kind": "set_machine_flag_on_action",
            "tool": matching_step["tool"],
            "args": effect_args,
            "machine_path": machine_path,
            "value": True,
        }
        for existing_effect in task_effects:
            if not isinstance(existing_effect, dict):
                continue
            existing_args = existing_effect.get("args") or {}
            if (
                existing_effect.get("tool") == matching_step["tool"]
                and isinstance(existing_args, dict)
                and existing_args.get("target_id") == fixture_id
                and existing_args.get("control_id") == control_id
                and existing_effect.get("required_fixture_controls")
            ):
                synthesized_effect["required_fixture_controls"] = [
                    dict(requirement)
                    for requirement in existing_effect.get("required_fixture_controls")
                    if isinstance(requirement, dict)
                ]
                break
        task_effects.append(synthesized_effect)

    return rewritten_goal_conditions


def _parse_visited_machine_key(machine_key: Any) -> tuple[str, str] | None:
    if not isinstance(machine_key, str) or not machine_key.endswith("_visited"):
        return None
    base_key = machine_key[: -len("_visited")]
    if "_" not in base_key:
        return None
    control_id, visited_state = base_key.rsplit("_", 1)
    if not control_id or not visited_state:
        return None
    return control_id, visited_state


def _rewrite_partial_control_visit_goals(
    *,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    trajectory: Any,
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions

    visited_machine_keys_by_control: dict[tuple[str, str], set[str]] = defaultdict(set)
    for condition in goal_conditions:
        if not isinstance(condition, dict) or condition.get("kind") != "machine_flag_true":
            continue
        machine_path = condition.get("machine_path")
        if not isinstance(machine_path, list) or len(machine_path) != 2:
            continue
        fixture_id, machine_key = machine_path
        parsed_machine_key = _parse_visited_machine_key(machine_key)
        if not isinstance(fixture_id, str) or parsed_machine_key is None:
            continue
        control_id, _visited_state = parsed_machine_key
        visited_machine_keys_by_control[(fixture_id, control_id)].add(machine_key)

    rewritten_goal_conditions: list[dict[str, Any]] = []
    steps = [step for step in (trajectory.get("steps") or []) if isinstance(step, dict)]
    for condition in goal_conditions:
        if not isinstance(condition, dict) or condition.get("kind") != "fixture_control_state":
            rewritten_goal_conditions.append(condition)
            continue
        fixture_id = condition.get("fixture_id")
        control_id = condition.get("control_id")
        visited_state = condition.get("state")
        if (
            not isinstance(fixture_id, str)
            or not isinstance(control_id, str)
            or not isinstance(visited_state, str)
            or (fixture_id, control_id) not in visited_machine_keys_by_control
        ):
            rewritten_goal_conditions.append(condition)
            continue

        machine_key = f"{control_id}_{visited_state}_visited"
        machine_path = [fixture_id, machine_key]
        _ensure_machine_flag_default(
            initial_state,
            machine_path=(fixture_id, machine_key),
            default=False,
        )
        _ensure_public_state_default(initial_public_state, machine_key, False)
        rewritten_goal_conditions.append(
            {
                "kind": "machine_flag_true",
                "machine_path": machine_path,
            }
        )
        if any(
            isinstance(effect, dict) and effect.get("machine_path") == machine_path
            for effect in task_effects
        ):
            continue
        matching_step = next(
            (
                step
                for step in steps
                if step.get("tool") in _CONTROL_TOOL_TYPES
                and (step.get("args") or {}).get("target_id") == fixture_id
                and (step.get("args") or {}).get("control_id") == control_id
                and (
                    step.get("tool") == "set_rotary_control"
                    and (step.get("args") or {}).get("goal") == visited_state
                )
            ),
            None,
        )
        if not isinstance(matching_step, dict):
            continue
        guard_source_effect = next(
            (
                effect
                for effect in task_effects
                if isinstance(effect, dict)
                and effect.get("tool") == matching_step["tool"]
                and isinstance(effect.get("args"), dict)
                and effect["args"].get("target_id") == fixture_id
                and effect["args"].get("control_id") == control_id
                and (
                    effect.get("required_object_locations")
                    or effect.get("required_machine_values")
                    or effect.get("required_fixture_controls")
                )
            ),
            None,
        )
        _ensure_machine_flag_effect(
            task_effects,
            tool=matching_step["tool"],
            args=dict(matching_step.get("args") or {}),
            machine_path=machine_path,
            value=True,
            required_object_locations=[
                dict(requirement)
                for requirement in (guard_source_effect or {}).get(
                    "required_object_locations", ()
                )
                if isinstance(requirement, dict)
            ]
            or None,
            required_machine_values=[
                dict(requirement)
                for requirement in (guard_source_effect or {}).get(
                    "required_machine_values", ()
                )
                if isinstance(requirement, dict)
            ]
            or None,
            required_fixture_controls=[
                dict(requirement)
                for requirement in (guard_source_effect or {}).get(
                    "required_fixture_controls", ()
                )
                if isinstance(requirement, dict)
            ]
            or None,
        )

    return rewritten_goal_conditions


def _guard_visited_effects_with_water_on(
    task_effects: list[dict[str, Any]],
) -> None:
    water_on_paths_by_fixture: dict[str, list[list[str]]] = defaultdict(list)
    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        machine_path = effect.get("machine_path")
        if (
            effect.get("kind") == "set_machine_flag_on_action"
            and isinstance(machine_path, list)
            and len(machine_path) == 2
            and effect.get("value") is True
            and machine_path[1] == "water_on"
            and isinstance(machine_path[0], str)
        ):
            water_on_paths_by_fixture[machine_path[0]].append(machine_path)

    for effect in task_effects:
        if not isinstance(effect, dict):
            continue
        machine_path = effect.get("machine_path")
        if not isinstance(machine_path, list) or len(machine_path) != 2:
            continue
        fixture_id, machine_key = machine_path
        if not isinstance(fixture_id, str) or _parse_visited_machine_key(machine_key) is None:
            continue
        for water_on_path in water_on_paths_by_fixture.get(fixture_id, ()):
            effect["required_machine_values"] = _merge_unique_dict_list(
                effect.get("required_machine_values"),
                [{"machine_path": list(water_on_path), "value": True}],
            )


def _prioritize_guarded_effects(
    task_effects: list[dict[str, Any]],
) -> None:
    def _priority(effect: dict[str, Any]) -> tuple[int, int]:
        has_guards = any(
            effect.get(field_name)
            for field_name in (
                "required_object_locations",
                "required_machine_values",
                "required_fixture_controls",
            )
        )
        if has_guards:
            return (0, 0)
        if effect.get("value") is False:
            return (2, 0)
        return (1, 0)

    task_effects.sort(
        key=lambda effect: _priority(effect) if isinstance(effect, dict) else (3, 0)
    )


def _add_location_goals_for_proxy_placement_effects(
    *,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    final_object_locations: dict[str, str],
) -> list[dict[str, Any]]:
    normalized_goal_conditions = list(goal_conditions)
    for effect in task_effects:
        if not isinstance(effect, dict) or effect.get("tool") not in RELEASE_TOOL_NAMES:
            continue
        effect_args = effect.get("args")
        if not isinstance(effect_args, dict):
            continue
        object_id = effect_args.get("object_id")
        final_location = (
            final_object_locations.get(object_id)
            if isinstance(object_id, str)
            else None
        )
        if not isinstance(object_id, str) or not isinstance(final_location, str):
            continue
        if final_location.startswith("held_by_"):
            continue
        if _goal_has_location_constraint(normalized_goal_conditions, object_id=object_id):
            continue
        normalized_goal_conditions.append(
            {
                "kind": "object_at_location",
                "object_id": object_id,
                "location": final_location,
            }
        )
    return normalized_goal_conditions


def _rewrite_fixture_object_goals_to_support_sites(
    *,
    initial_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    final_object_locations: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return goal_conditions
    support_site_parent_by_id: dict[str, str] = {}
    for fixture_id, fixture_state in (initial_state.get("fixtures") or {}).items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if not isinstance(support_sites, dict):
            continue
        for support_site_id in support_sites:
            if isinstance(support_site_id, str):
                support_site_parent_by_id[support_site_id] = fixture_id

    normalized_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        if (
            isinstance(condition, dict)
            and condition.get("kind") == "object_at_location"
            and isinstance(condition.get("object_id"), str)
            and isinstance(condition.get("location"), str)
        ):
            final_location = final_object_locations.get(condition["object_id"])
            if (
                isinstance(final_location, str)
                and support_site_parent_by_id.get(final_location) == condition["location"]
            ):
                rewritten_condition = dict(condition)
                rewritten_condition["location"] = final_location
                normalized_goal_conditions.append(rewritten_condition)
                continue
        normalized_goal_conditions.append(condition)
    return normalized_goal_conditions


def _add_zero_count_goals_for_either_or_rules(
    *,
    initial_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    extra_execution_rules: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return goal_conditions
    location_ids = {
        symbol_id
        for symbol_id in (
            list((initial_state.get("objects") or {}).keys())
            + list((initial_state.get("fixtures") or {}).keys())
        )
        if isinstance(symbol_id, str)
    }
    normalized_goal_conditions = list(goal_conditions)
    for condition in list(goal_conditions):
        if (
            not isinstance(condition, dict)
            or condition.get("kind") != "object_count_at_location"
            or condition.get("count") != 1
        ):
            continue
        object_ids = [
            object_id
            for object_id in condition.get("object_ids") or []
            if isinstance(object_id, str)
        ]
        location = condition.get("location")
        if not object_ids or not isinstance(location, str):
            continue
        group_label = _object_group_label(object_ids)
        if not isinstance(group_label, str):
            continue
        matching_rule = next(
            (
                rule
                for rule in extra_execution_rules
                if "either" in _normalize_free_text(rule)
                and _text_mentions_symbol(rule, group_label)
            ),
            None,
        )
        if not isinstance(matching_rule, str):
            continue
        alternate_locations = [
            candidate_location
            for candidate_location in sorted(location_ids)
            if candidate_location != location and _text_mentions_symbol(matching_rule, candidate_location)
        ]
        if len(alternate_locations) != 1:
            continue
        alternate_location = alternate_locations[0]
        if any(
            isinstance(existing_condition, dict)
            and existing_condition.get("kind") == "object_count_at_location"
            and existing_condition.get("object_ids") == object_ids
            and existing_condition.get("location") == alternate_location
            for existing_condition in normalized_goal_conditions
        ):
            continue
        normalized_goal_conditions.append(
            {
                "kind": "object_count_at_location",
                "object_ids": list(object_ids),
                "location": alternate_location,
                "count": 0,
            }
        )
    return normalized_goal_conditions


def _add_persistent_control_state_goals_from_rules(
    *,
    initial_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    extra_execution_rules: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict):
        return goal_conditions
    normalized_goal_conditions = list(goal_conditions)
    existing_control_goals = {
        (condition.get("fixture_id"), condition.get("control_id"))
        for condition in normalized_goal_conditions
        if isinstance(condition, dict) and condition.get("kind") == "fixture_control_state"
    }
    fixtures_by_id = initial_state.get("fixtures") or {}
    for fixture_id, fixture_state in fixtures_by_id.items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        fixture_controls = fixture_state.get("controls") or {}
        if not isinstance(fixture_controls, dict):
            continue
        for control_id, control_state in fixture_controls.items():
            if not isinstance(control_id, str) or not isinstance(control_state, dict):
                continue
            if (fixture_id, control_id) in existing_control_goals:
                continue
            desired_state: str | None = None
            for rule in extra_execution_rules:
                normalized_rule = _normalize_free_text(rule)
                if not _text_mentions_symbol(rule, control_id):
                    continue
                if "do not turn off" in normalized_rule or "must remain on" in normalized_rule:
                    desired_state = "on"
                    break
                if "do not turn on" in normalized_rule or "must remain off" in normalized_rule:
                    desired_state = "off"
                    break
            if not isinstance(desired_state, str):
                continue
            normalized_goal_conditions.append(
                {
                    "kind": "fixture_control_state",
                    "fixture_id": fixture_id,
                    "control_id": control_id,
                    "state": desired_state,
                }
            )
            existing_control_goals.add((fixture_id, control_id))
    return normalized_goal_conditions


def _add_activation_machine_goals_from_controls(
    *,
    composite_task: str,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    trajectory: Any,
    task_goal: str,
    extra_execution_rules: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions
    normalized_text = _normalize_free_text(" ".join((task_goal, *extra_execution_rules)))
    if not any(
        phrase in normalized_text
        for phrase in ("turned on", "turn on", "start", "toasting", "toast")
    ):
        return goal_conditions

    steps = [step for step in (trajectory.get("steps") or []) if isinstance(step, dict)]
    if not steps:
        return goal_conditions
    last_step = steps[-1]
    rewritten_goal_conditions: list[dict[str, Any]] = []
    for condition in goal_conditions:
        if not isinstance(condition, dict) or condition.get("kind") != "fixture_control_state":
            rewritten_goal_conditions.append(condition)
            continue
        fixture_id = condition.get("fixture_id")
        control_id = condition.get("control_id")
        state = condition.get("state")
        if not (
            isinstance(fixture_id, str)
            and isinstance(control_id, str)
            and isinstance(state, str)
        ):
            rewritten_goal_conditions.append(condition)
            continue
        if (
            last_step.get("tool") not in {"press_button", "press_lever"}
            or (last_step.get("args") or {}).get("target_id") != fixture_id
            or (last_step.get("args") or {}).get("control_id") != control_id
        ):
            rewritten_goal_conditions.append(condition)
            continue

        machine_path = [fixture_id, "turned_on"]
        _ensure_machine_flag_default(
            initial_state,
            machine_path=(fixture_id, "turned_on"),
            default=False,
        )
        _ensure_public_state_default(initial_public_state, "turned_on", False)
        required_object_locations = [
            {
                "object_id": goal_condition["object_id"],
                "location": goal_condition["location"],
            }
            for goal_condition in goal_conditions
            if isinstance(goal_condition, dict)
            and goal_condition.get("kind") == "object_at_location"
            and isinstance(goal_condition.get("object_id"), str)
            and isinstance(goal_condition.get("location"), str)
        ]
        _ensure_machine_flag_effect(
            task_effects,
            tool=last_step["tool"],
            args=dict(last_step.get("args") or {}),
            machine_path=machine_path,
            value=True,
            required_object_locations=required_object_locations or None,
        )
        rewritten_goal_conditions.append(
            {
                "kind": "machine_flag_true",
                "machine_path": machine_path,
            }
        )

    return rewritten_goal_conditions


def _add_upright_machine_flags_from_rules(
    *,
    composite_task: str,
    initial_state: dict[str, Any] | None,
    initial_public_state: dict[str, Any] | None,
    goal_conditions: list[dict[str, Any]],
    task_effects: list[dict[str, Any]],
    trajectory: Any,
    extra_execution_rules: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not isinstance(initial_state, dict) or not isinstance(trajectory, dict):
        return goal_conditions
    machine_namespace = _primary_machine_namespace(
        initial_state=initial_state,
        composite_task=composite_task,
    )
    if not isinstance(machine_namespace, str):
        return goal_conditions

    object_ids = [
        object_id
        for object_id in (initial_state.get("objects") or {})
        if isinstance(object_id, str)
    ]
    placement_steps_by_object_id = {
        (step.get("args") or {}).get("object_id"): step
        for step in (trajectory.get("steps") or [])
        if isinstance(step, dict)
        and step.get("tool") in RELEASE_TOOL_NAMES
        and isinstance((step.get("args") or {}).get("object_id"), str)
    }

    def _find_existing_upright_path(object_id: str) -> tuple[str, str] | None:
        object_tokens = set(object_id.lower().split("_"))
        candidate_paths: list[tuple[int, tuple[str, str]]] = []
        for item in [*goal_conditions, *task_effects]:
            if not isinstance(item, dict):
                continue
            machine_path = item.get("machine_path")
            if not isinstance(machine_path, list) or len(machine_path) != 2:
                continue
            machine_namespace_value, machine_key = machine_path
            if not isinstance(machine_namespace_value, str) or not isinstance(machine_key, str):
                continue
            if not machine_key.endswith("_upright"):
                continue
            key_tokens = set(machine_key[: -len("_upright")].split("_"))
            overlap = len(object_tokens & key_tokens)
            if overlap > 0:
                candidate_paths.append((overlap, (machine_namespace_value, machine_key)))
        if not candidate_paths:
            return None
        return max(candidate_paths, key=lambda item: (item[0], -len(item[1][1])))[1]

    normalized_goal_conditions = list(goal_conditions)
    for rule in extra_execution_rules:
        if "upright" not in _normalize_free_text(rule):
            continue
        matching_object_ids = [
            object_id for object_id in object_ids if _text_mentions_symbol(rule, object_id)
        ]
        if len(matching_object_ids) != 1:
            continue
        object_id = matching_object_ids[0]
        existing_machine_path = _find_existing_upright_path(object_id)
        if existing_machine_path is not None:
            machine_namespace, machine_key = existing_machine_path
        else:
            machine_key = f"{object_id}_upright"
        machine_path = [machine_namespace, machine_key]
        if not any(
            isinstance(condition, dict)
            and condition.get("kind") == "machine_flag_true"
            and condition.get("machine_path") == machine_path
            for condition in normalized_goal_conditions
        ):
            normalized_goal_conditions.append(
                {
                    "kind": "machine_flag_true",
                    "machine_path": machine_path,
                }
            )
        placement_step = placement_steps_by_object_id.get(object_id)
        if not isinstance(placement_step, dict):
            continue
        _ensure_machine_flag_default(
            initial_state,
            machine_path=(machine_namespace, machine_key),
            default=False,
        )
        _ensure_public_state_default(initial_public_state, machine_key, False)
        _ensure_machine_flag_effect(
            task_effects,
            tool=placement_step["tool"],
            args=dict(placement_step.get("args") or {}),
            machine_path=machine_path,
            value=True,
        )

    return normalized_goal_conditions


def _postprocess_spec_payload(spec_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model JSON into the canonical TaskSpec payload shape."""

    normalized_payload = dict(spec_payload)
    normalized_payload["allowed_tool_specs"] = _canonicalize_allowed_tool_specs(
        normalized_payload
    )
    initial_state = normalized_payload.get("initial_state")
    object_ids = set()
    fixture_ids = set()
    fixtures_by_id: dict[str, Any] = {}
    if isinstance(initial_state, dict):
        object_ids = set((initial_state.get("objects") or {}).keys())
        fixture_ids = set((initial_state.get("fixtures") or {}).keys())
        fixtures_by_id = dict(initial_state.get("fixtures") or {})
        for fixture_state in fixtures_by_id.values():
            if not isinstance(fixture_state, dict):
                continue
            fixture_parts = fixture_state.get("parts") or {}
            if not isinstance(fixture_parts, dict):
                fixture_parts = {}
            for part_state in fixture_parts.values():
                if not isinstance(part_state, dict):
                    continue
                normalized_state = _normalize_part_state_value(part_state.get("state"))
                if isinstance(normalized_state, str):
                    part_state["state"] = normalized_state
            fixture_controls = fixture_state.get("controls") or {}
            if isinstance(fixture_controls, dict):
                for control_state in fixture_controls.values():
                    if not isinstance(control_state, dict):
                        continue
                    control_type = control_state.get("control_type")
                    if control_type in _CONTROL_TOOL_TYPES:
                        control_state["control_type"] = _CONTROL_TOOL_TYPES[control_type]
            raw_support_sites = fixture_state.get("support_sites")
            if isinstance(raw_support_sites, list):
                fixture_state["support_sites"] = {
                    support_site_id: {"site_type": "support"}
                    for support_site_id in raw_support_sites
                    if isinstance(support_site_id, str)
                }
            elif isinstance(raw_support_sites, dict):
                fixture_state["support_sites"] = {
                    support_site_id: (
                        dict(support_site_state)
                        if isinstance(support_site_state, dict)
                        else {"site_type": "support"}
                    )
                    for support_site_id, support_site_state in raw_support_sites.items()
                    if isinstance(support_site_id, str)
                }

    place_next_to_spec = normalized_payload["allowed_tool_specs"].get("place_next_to")
    if isinstance(place_next_to_spec, dict):
        raw_reference_object_ids = place_next_to_spec.get("allowed_reference_object_ids")
        if isinstance(raw_reference_object_ids, list):
            allowed_reference_object_ids = [
                value for value in raw_reference_object_ids if value in object_ids
            ]
            allowed_reference_fixture_ids = [
                value for value in raw_reference_object_ids if value in fixture_ids
            ]
            if allowed_reference_object_ids:
                place_next_to_spec["allowed_reference_object_ids"] = (
                    allowed_reference_object_ids
                )
            else:
                place_next_to_spec.pop("allowed_reference_object_ids", None)
            if allowed_reference_fixture_ids:
                place_next_to_spec["allowed_reference_fixture_ids"] = (
                    allowed_reference_fixture_ids
                )

    def _normalize_place_next_to_args(args: Any) -> None:
        if not isinstance(args, dict):
            return
        reference_object_id = args.get("reference_object_id")
        if (
            isinstance(reference_object_id, str)
            and reference_object_id in fixture_ids
            and "reference_fixture_id" not in args
        ):
            args["reference_fixture_id"] = reference_object_id
            args.pop("reference_object_id", None)

    machine_state = (
        initial_state.setdefault("machine_state", {})
        if isinstance(initial_state, dict)
        else {}
    )
    trajectory = normalized_payload.get("example_trajectory")
    if isinstance(trajectory, dict):
        _insert_missing_navigation_before_give_space(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            trajectory=trajectory,
        )
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if step.get("tool") == "place_next_to":
                _normalize_place_next_to_args(step.get("args"))
            if step.get("tool") == "set_rotary_control":
                args = step.get("args")
                if isinstance(args, dict):
                    args["goal"] = _normalize_rotary_goal_value(args.get("goal"))
        inverse_adjacent_fixture_ids = {
            fixture_state.get("adjacent_location_id"): fixture_id
            for fixture_id, fixture_state in (machine_state if isinstance(machine_state, dict) else {}).items()
            if isinstance(fixture_id, str)
            and isinstance(fixture_state, dict)
            and isinstance(fixture_state.get("adjacent_location_id"), str)
        }
        agent_locations = {
            agent_id: agent_state.get("location")
            for agent_id, agent_state in (initial_state.get("agents") or {}).items()
            if isinstance(agent_id, str) and isinstance(agent_state, dict)
        }
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            agent_id = step.get("agent")
            args = step.get("args")
            if (
                step.get("tool") == "give_space"
                and isinstance(agent_id, str)
                and isinstance(args, dict)
            ):
                fixture_id = args.get("fixture_id")
                anchor_fixture_id = inverse_adjacent_fixture_ids.get(fixture_id)
                if (
                    isinstance(fixture_id, str)
                    and isinstance(anchor_fixture_id, str)
                    and agent_locations.get(agent_id) == anchor_fixture_id
                ):
                    args["fixture_id"] = anchor_fixture_id
            if (
                step.get("tool") == "navigate_to_fixture"
                and isinstance(agent_id, str)
                and isinstance(args, dict)
                and isinstance(args.get("fixture_id"), str)
            ):
                agent_locations[agent_id] = args["fixture_id"]
            elif step.get("tool") == "give_space" and isinstance(agent_id, str):
                agent_locations[agent_id] = None

    for effect in normalized_payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        if effect.get("tool") == "place_next_to":
            _normalize_place_next_to_args(effect.get("args"))
        if effect.get("tool") == "set_rotary_control":
            args = effect.get("args")
            if isinstance(args, dict):
                args["goal"] = _normalize_rotary_goal_value(args.get("goal"))
    task_effects = [
        effect
        for effect in (normalized_payload.get("task_effects") or [])
        if isinstance(effect, dict)
    ]
    normalized_payload["task_effects"] = task_effects
    _backfill_fixture_controls(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        allowed_tool_specs=normalized_payload["allowed_tool_specs"],
        trajectory=trajectory,
        task_effects=task_effects,
    )
    _backfill_fixture_support_sites(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        grounding=normalized_payload.get("grounding")
        if isinstance(normalized_payload.get("grounding"), dict)
        else None,
        trajectory=trajectory,
        task_effects=task_effects,
    )
    _normalize_trajectory_symbolic_args(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        allowed_tool_specs=normalized_payload["allowed_tool_specs"],
        trajectory=trajectory,
        task_effects=task_effects,
    )
    candidate_adjacent_support_ids = [
        fixture_id
        for fixture_id, fixture_state in fixtures_by_id.items()
        if fixture_state.get("fixture_type") == "counter"
        and not fixture_state.get("parts")
    ]
    referenced_fixture_ids: set[str] = set()
    if isinstance(place_next_to_spec, dict):
        for fixture_id in place_next_to_spec.get("allowed_reference_fixture_ids", []) or []:
            if isinstance(fixture_id, str):
                referenced_fixture_ids.add(fixture_id)
    give_space_spec = normalized_payload["allowed_tool_specs"].get("give_space")
    referenced_give_space_fixture_ids: set[str] = set()
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            if step.get("tool") == "place_next_to":
                reference_fixture_id = (step.get("args") or {}).get("reference_fixture_id")
                if isinstance(reference_fixture_id, str):
                    referenced_fixture_ids.add(reference_fixture_id)
            if step.get("tool") == "give_space":
                fixture_id = (step.get("args") or {}).get("fixture_id")
                if isinstance(fixture_id, str):
                    referenced_give_space_fixture_ids.add(fixture_id)
    for effect in normalized_payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        if effect.get("tool") == "place_next_to":
            reference_fixture_id = (effect.get("args") or {}).get("reference_fixture_id")
            if isinstance(reference_fixture_id, str):
                referenced_fixture_ids.add(reference_fixture_id)
    if isinstance(machine_state, dict):
        for fixture_id in referenced_fixture_ids:
            if fixture_id not in fixture_ids:
                continue
            fixture_machine_state = machine_state.setdefault(fixture_id, {})
            if not isinstance(fixture_machine_state, dict):
                continue
            if isinstance(fixture_machine_state.get("adjacent_location_id"), str):
                continue
            eligible_support_ids = [
                support_id
                for support_id in candidate_adjacent_support_ids
                if support_id != fixture_id
            ]
            if len(eligible_support_ids) == 1:
                fixture_machine_state["adjacent_location_id"] = eligible_support_ids[0]
                continue
            adjacent_location_id = fixture_machine_state.get("adjacent_location_id")
            if isinstance(adjacent_location_id, str):
                continue
            synthesized_surface_id = f"{fixture_id}_adjacent_surface"
            if synthesized_surface_id not in fixture_ids:
                fixtures_by_id[synthesized_surface_id] = {"fixture_type": "counter"}
                fixture_ids.add(synthesized_surface_id)
                if isinstance(initial_state, dict):
                    initial_state.setdefault("fixtures", {})[synthesized_surface_id] = {
                        "fixture_type": "counter"
                    }
            fixture_machine_state["adjacent_location_id"] = synthesized_surface_id

            grounding = normalized_payload.get("grounding")
            if not isinstance(grounding, dict):
                continue
            grounding_symbols = grounding.setdefault("symbols", {})
            if not isinstance(grounding_symbols, dict):
                continue
            grounding_symbols.setdefault(
                synthesized_surface_id,
                {
                    "entity_type": "fixture",
                    "resolver": "nearest_placeable_surface_to_fixture",
                    "fixture_symbol": fixture_id,
                    "role": synthesized_surface_id,
                },
            )
    if isinstance(give_space_spec, dict):
        normalized_allowed_fixture_ids = [
            fixture_id
            for fixture_id in (give_space_spec.get("allowed_fixture_ids") or [])
            if isinstance(fixture_id, str) and fixture_id in fixture_ids
        ]
        for fixture_id in sorted(referenced_give_space_fixture_ids):
            if fixture_id not in normalized_allowed_fixture_ids:
                normalized_allowed_fixture_ids.append(fixture_id)
        if normalized_allowed_fixture_ids:
            give_space_spec["allowed_fixture_ids"] = normalized_allowed_fixture_ids

    grounding = normalized_payload.get("grounding")
    if isinstance(grounding, dict):
        grounding_symbols = grounding.get("symbols") or {}
        if isinstance(grounding_symbols, dict):
            for symbol_name, symbol_spec in grounding_symbols.items():
                if not isinstance(symbol_spec, dict):
                    continue
                if (
                    symbol_spec.get("resolver") == "nearest_placeable_surface_to_fixture"
                    and "fixture_symbol" not in symbol_spec
                    and isinstance(symbol_spec.get("reference_fixture_id"), str)
                ):
                    symbol_spec["fixture_symbol"] = symbol_spec.pop(
                        "reference_fixture_id"
                    )
                if symbol_spec.get("entity_type") != "fixture":
                    continue
                if symbol_spec.get("resolver") != "support_fixture_for_object":
                    continue
                object_symbol = symbol_spec.get("object_symbol")
                object_location = (
                    (initial_state.get("objects", {}) if isinstance(initial_state, dict) else {})
                    .get(object_symbol, {})
                    .get("location")
                )
                if isinstance(object_symbol, str) and object_location == symbol_name:
                    symbol_spec["resolver"] = "source_fixture_for_object"
        _normalize_grounding_symbols(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            grounding=grounding,
        )

    task_preconditions = [
        dict(condition)
        for condition in (normalized_payload.get("task_preconditions") or [])
        if isinstance(condition, dict)
    ]
    promoted_goal_conditions: list[dict[str, Any]] = []
    retained_preconditions: list[dict[str, Any]] = []
    for condition in task_preconditions:
        if condition.get("kind") in {"fixture_control_state", "fixture_part_state"}:
            promoted_goal_conditions.append(dict(condition))
            continue
        retained_preconditions.append(condition)
    task_preconditions = retained_preconditions
    task_preconditions = _drop_impossible_location_preconditions(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        task_preconditions=task_preconditions,
        trajectory=trajectory,
    )
    task_preconditions = _rewrite_container_pickup_preconditions(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        task_preconditions=task_preconditions,
        trajectory=trajectory,
    )
    task_preconditions = _synthesize_fixture_part_action_preconditions(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        trajectory=trajectory,
        task_preconditions=task_preconditions,
    )
    for condition in task_preconditions:
        if (
            isinstance(condition, dict)
            and condition.get("kind")
            in {
                "fixture_part_state_required_for_pickup",
                "fixture_part_state_required_for_action",
            }
        ):
            normalized_state = _normalize_part_state_value(
                condition.get("required_state")
            )
            if isinstance(normalized_state, str):
                condition["required_state"] = normalized_state
    normalized_payload["task_preconditions"] = task_preconditions

    _move_guarded_communicate_effects_to_following_action(
        task_effects=task_effects,
        trajectory=trajectory,
    )
    _normalize_guarded_communicate_effects(task_effects)
    _add_water_state_guards_for_completion_effects(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        task_goal=str(normalized_payload.get("task_goal") or ""),
        task_effects=task_effects,
        trajectory=trajectory,
    )

    final_object_locations = _infer_final_object_locations(initial_state, trajectory)

    goal_conditions = normalized_payload.get("goal_conditions")
    if isinstance(goal_conditions, list):
        goal_conditions = [*promoted_goal_conditions, *goal_conditions]
        goal_conditions = _synthesize_timed_completion_goal_and_effect(
            composite_task=str(normalized_payload.get("composite_task") or ""),
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            task_goal=str(normalized_payload.get("task_goal") or ""),
            goal_conditions=[
                dict(condition) for condition in goal_conditions if isinstance(condition, dict)
            ],
            task_effects=task_effects,
            trajectory=trajectory,
            final_object_locations=final_object_locations,
        )
        goal_conditions = _rewrite_shared_location_pool_goals(
            [dict(condition) for condition in goal_conditions if isinstance(condition, dict)],
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            final_object_locations=final_object_locations,
        )
        rewriteable_exclusive_groups: dict[tuple[str, ...], dict[str, str]] = {}
        exclusive_goal_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for condition in goal_conditions:
            if condition.get("kind") != "object_at_location_one_of":
                continue
            if not condition.get("exclusive", False):
                continue
            object_id = condition.get("object_id")
            locations = condition.get("locations")
            if not isinstance(object_id, str) or not isinstance(locations, list):
                continue
            normalized_locations = tuple(
                location for location in locations if isinstance(location, str)
            )
            if normalized_locations:
                exclusive_goal_groups.setdefault(normalized_locations, []).append(
                    condition
                )
        for locations, grouped_conditions in exclusive_goal_groups.items():
            if len(grouped_conditions) <= len(set(locations)):
                continue
            concrete_assignments: dict[str, str] = {}
            for condition in grouped_conditions:
                object_id = condition["object_id"]
                final_location = final_object_locations.get(object_id)
                if not isinstance(final_location, str) or final_location not in locations:
                    concrete_assignments = {}
                    break
                concrete_assignments[object_id] = final_location
            if concrete_assignments:
                rewriteable_exclusive_groups[locations] = concrete_assignments

        normalized_goal_conditions: list[Any] = []
        for condition in goal_conditions:
            if (
                condition.get("kind") == "object_at_location_one_of"
                and condition.get("exclusive", False)
            ):
                object_id = condition.get("object_id")
                locations = condition.get("locations")
                if isinstance(object_id, str) and isinstance(locations, list):
                    normalized_locations = tuple(
                        location for location in locations if isinstance(location, str)
                    )
                    final_location = rewriteable_exclusive_groups.get(
                        normalized_locations, {}
                    ).get(object_id)
                    if isinstance(final_location, str):
                        normalized_goal_conditions.append(
                            {
                                "kind": "object_at_location",
                                "object_id": object_id,
                                "location": final_location,
                            }
                        )
                        continue
            if (
                condition.get("kind") == "machine_flag_true"
                and "value" in condition
            ):
                rewritten_condition = dict(condition)
                rewritten_condition["kind"] = "machine_flag_equals"
                rewritten_condition["value"] = _normalize_part_state_value(
                    rewritten_condition.get("value")
                )
                normalized_goal_conditions.append(rewritten_condition)
                continue
            if (
                condition.get("kind") == "machine_flag_equals"
                and isinstance(condition.get("value"), str)
                and condition.get("value") in _PART_STATE_VALUES
            ):
                condition = dict(condition)
                condition["value"] = _normalize_part_state_value(condition.get("value"))
                machine_path = condition.get("machine_path")
                if isinstance(machine_path, list) and len(machine_path) == 2:
                    fixture_id, machine_key = machine_path
                    fixture_state = (
                        initial_state.get("fixtures", {}).get(fixture_id, {})
                        if isinstance(initial_state, dict)
                        else {}
                    )
                    fixture_parts = (
                        fixture_state.get("parts", {})
                        if isinstance(fixture_state, dict)
                        else {}
                    )
                    if (
                        isinstance(fixture_id, str)
                        and isinstance(machine_key, str)
                        and machine_key.endswith("_state")
                        and isinstance(fixture_parts, dict)
                        and len(fixture_parts) == 1
                    ):
                        part_id = next(iter(fixture_parts))
                        normalized_goal_conditions.append(
                            {
                                "kind": "fixture_part_state",
                                "fixture_id": fixture_id,
                                "part_id": part_id,
                                "state": condition["value"],
                            }
                        )
                        continue
            if condition.get("kind") == "fixture_part_state":
                rewritten_condition = dict(condition)
                rewritten_condition["state"] = _normalize_part_state_value(
                    rewritten_condition.get("state")
                )
                normalized_goal_conditions.append(rewritten_condition)
                continue
            normalized_goal_conditions.append(condition)
        normalized_goal_conditions = _rewrite_control_state_goals(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
        )
        normalized_goal_conditions = _add_location_goals_for_proxy_placement_effects(
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
            final_object_locations=final_object_locations,
        )
        normalized_goal_conditions = _rewrite_fixture_object_goals_to_support_sites(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            goal_conditions=normalized_goal_conditions,
            final_object_locations=final_object_locations,
        )
        normalized_goal_conditions = _add_zero_count_goals_for_either_or_rules(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            goal_conditions=normalized_goal_conditions,
            extra_execution_rules=normalized_payload.get("extra_execution_rules") or (),
        )
        normalized_goal_conditions = _add_persistent_control_state_goals_from_rules(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            goal_conditions=normalized_goal_conditions,
            extra_execution_rules=normalized_payload.get("extra_execution_rules") or (),
        )
        normalized_goal_conditions = _add_persistent_initial_on_control_goals(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            goal_conditions=normalized_goal_conditions,
            trajectory=trajectory,
        )
        normalized_goal_conditions = _rewrite_multi_state_control_goals(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
            trajectory=trajectory,
        )
        normalized_goal_conditions = _rewrite_partial_control_visit_goals(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
            trajectory=trajectory,
        )
        _guard_visited_effects_with_water_on(task_effects)
        normalized_goal_conditions = _add_activation_machine_goals_from_controls(
            composite_task=str(normalized_payload.get("composite_task") or ""),
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
            trajectory=trajectory,
            task_goal=str(normalized_payload.get("task_goal") or ""),
            extra_execution_rules=normalized_payload.get("extra_execution_rules") or (),
        )
        normalized_goal_conditions = _add_upright_machine_flags_from_rules(
            composite_task=str(normalized_payload.get("composite_task") or ""),
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
            trajectory=trajectory,
            extra_execution_rules=normalized_payload.get("extra_execution_rules") or (),
        )
        normalized_goal_conditions = _consolidate_upright_machine_flags(
            initial_state=initial_state if isinstance(initial_state, dict) else None,
            initial_public_state=(
                normalized_payload.get("initial_public_state")
                if isinstance(normalized_payload.get("initial_public_state"), dict)
                else None
            ),
            goal_conditions=normalized_goal_conditions,
            task_effects=task_effects,
        )
        normalized_goal_conditions = _drop_redundant_water_off_goals(
            normalized_goal_conditions
        )
        _prioritize_guarded_effects(task_effects)
        normalized_payload["goal_conditions"] = normalized_goal_conditions
    return normalized_payload


def _generate_one_spec(
    candidate: TaskAnalysis,
    *,
    client: BaseGenerationClient,
    model: str,
    temperature: float,
    examples: tuple[FewShotExample, ...],
    repair_context: SpecRepairContext | None = None,
) -> SpecGenerationResult:
    """Run one LLM generation call for a single candidate task."""

    try:
        source_code = _load_task_source(candidate)
    except OSError as exc:
        return SpecGenerationResult(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            spec_payload=None,
            error=f"failed to read source file: {exc}",
        )

    prompt = build_spec_generation_prompt(
        task_name=candidate.task_name,
        source_python=source_code,
        source_module=candidate.module_path,
        metadata=candidate.to_dict(),
        examples=examples,
        previous_spec_payload=(
            repair_context.previous_spec_payload if repair_context is not None else None
        ),
        repair_feedback_lines=(
            repair_context.feedback_lines if repair_context is not None else ()
        ),
    )

    try:
        # `thinking_level="LOW"` caps Gemini 2.5's internal reasoning budget
        # so it leaves enough of the 32k output budget for the TaskSpec
        # itself. At default (HIGH) thinking, specs were truncated mid-JSON
        # because reasoning consumed ~31k of the 32k budget.
        result = client.generate(
            model=model,
            prompt=prompt,
            response_schema=SPEC_GENERATION_RESPONSE_SCHEMA,
            temperature=temperature,
            thinking_level="LOW",
        )
    except Exception as exc:  # noqa: BLE001 — surface any client error per-task
        return SpecGenerationResult(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            spec_payload=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    payload = getattr(result, "payload", None)
    spec_payload, parse_error = _parse_spec_payload(payload)
    if spec_payload is not None:
        try:
            previous_payload_json: str | None = None
            for _ in range(2):
                spec_payload = _postprocess_spec_payload(spec_payload)
                current_payload_json = json.dumps(spec_payload, sort_keys=True)
                if current_payload_json == previous_payload_json:
                    break
                previous_payload_json = current_payload_json
        except ValueError as exc:
            return SpecGenerationResult(
                task_name=candidate.task_name,
                module_path=candidate.module_path,
                batch=candidate.batch,
                spec_payload=None,
                error=f"spec post-processing failed: {exc}",
            )

    return SpecGenerationResult(
        task_name=candidate.task_name,
        module_path=candidate.module_path,
        batch=candidate.batch,
        spec_payload=spec_payload,
        error=parse_error or None,
    )


def generate_specs(
    candidates: list[TaskAnalysis],
    *,
    client: BaseGenerationClient,
    model: str,
    examples: tuple[FewShotExample, ...],
    temperature: float = 0.2,
    workers: int = 4,
    progress_callback: Any = None,
    repair_feedback_by_task: dict[str, SpecRepairContext] | None = None,
) -> list[SpecGenerationResult]:
    """Generate specs for candidates in parallel, preserving input order."""

    results: list[SpecGenerationResult | None] = [None] * len(candidates)
    total = len(candidates)
    completed = 0
    lock = threading.Lock()

    if workers <= 1 or total <= 1:
        for index, candidate in enumerate(candidates):
            results[index] = _generate_one_spec(
                candidate,
                client=client,
                model=model,
                temperature=temperature,
                examples=examples,
                repair_context=(
                    repair_feedback_by_task.get(candidate.task_name)
                    if repair_feedback_by_task is not None
                    else None
                ),
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, candidate.task_name)
        return [r for r in results if r is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _generate_one_spec,
                candidate,
                client=client,
                model=model,
                temperature=temperature,
                examples=examples,
                repair_context=(
                    repair_feedback_by_task.get(candidate.task_name)
                    if repair_feedback_by_task is not None
                    else None
                ),
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            with lock:
                completed += 1
                current = completed
            if progress_callback is not None:
                progress_callback(current, total, candidates[index].task_name)

    return [r for r in results if r is not None]


def _spec_filename(task_name: str) -> str:
    """Mirror the `load_task_spec` naming convention used by the runtime."""

    normalized = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in task_name
    )
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return f"{collapsed}.json"


def _summarize_results(results: list[SpecGenerationResult]) -> dict[str, Any]:
    successes = sum(1 for r in results if r.spec_payload is not None)
    failures = len(results) - successes
    return {
        "total": len(results),
        "successes": successes,
        "failures": failures,
    }


def run_phase1(
    candidates: list[TaskAnalysis],
    *,
    output_dir: Path,
    model: str | None = None,
    workers: int = 4,
    temperature: float = 0.2,
    project: str | None = None,
    location: str = DEFAULT_LOCATION,
    sdk: str = DEFAULT_SDK,
    dry_run: bool = False,
    client: BaseGenerationClient | None = None,
    progress_callback: Any = None,
    generation_timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
    repair_feedback_by_task: dict[str, SpecRepairContext] | None = None,
) -> list[SpecGenerationResult]:
    """Execute Phase 1 and persist per-task spec JSONs under `phase1/`."""

    resolved_model = model or DEFAULT_MODEL

    if dry_run:
        return []

    examples = load_few_shot_examples()

    if client is None:
        client = build_generation_client(
            sdk=sdk,
            project=project,
            location=location,
            timeout_sec=generation_timeout_sec,
        )

    results = generate_specs(
        candidates,
        client=client,
        model=resolved_model,
        examples=examples,
        temperature=temperature,
        workers=workers,
        progress_callback=progress_callback,
        repair_feedback_by_task=repair_feedback_by_task,
    )

    phase_dir = output_dir / "phase1"
    specs_dir = phase_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    generation_log_by_task: dict[str, dict[str, Any]] = {}
    generation_log_path = phase_dir / "generation_log.json"
    if generation_log_path.exists():
        try:
            existing_generation_log = json.loads(
                generation_log_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            existing_generation_log = []
        if isinstance(existing_generation_log, list):
            for entry in existing_generation_log:
                if not isinstance(entry, dict):
                    continue
                task_name = entry.get("task_name")
                if isinstance(task_name, str) and task_name:
                    generation_log_by_task[task_name] = dict(entry)

    for result in results:
        spec_path = specs_dir / _spec_filename(result.task_name)
        entry = {
            "task_name": result.task_name,
            "module_path": result.module_path,
            "batch": result.batch,
            "error": result.error,
            "spec_path": None,
        }
        if result.spec_payload is not None:
            try:
                normalized_payload, sim_errors = normalize_spec_payload_against_simulation(
                    result.spec_payload,
                    task_name=result.task_name,
                )
            except Exception as exc:  # noqa: BLE001 — surface simulator normalization failures
                result.spec_payload = None
                result.error = (
                    f"simulation normalization failed: {type(exc).__name__}: {exc}"
                )
            else:
                if sim_errors:
                    result.spec_payload = None
                    error_text = "; ".join(sim_errors[:5])
                    if len(sim_errors) > 5:
                        error_text += f"; ... (+{len(sim_errors) - 5} more)"
                    result.error = f"simulation normalization failed: {error_text}"
                else:
                    result.spec_payload = normalized_payload
            entry["error"] = result.error

        if result.spec_payload is not None:
            with spec_path.open("w", encoding="utf-8") as f:
                json.dump(result.spec_payload, f, indent=2)
            entry["spec_path"] = str(spec_path.relative_to(output_dir))
        elif spec_path.exists():
            spec_path.unlink()
        generation_log_by_task[result.task_name] = entry

    generation_log = [
        generation_log_by_task[task_name]
        for task_name in sorted(generation_log_by_task)
    ]

    with generation_log_path.open("w", encoding="utf-8") as f:
        json.dump(generation_log, f, indent=2)

    summary = {
        "total": len(generation_log),
        "successes": sum(
            1 for entry in generation_log if isinstance(entry.get("spec_path"), str)
        ),
        "failures": sum(
            1 for entry in generation_log if not isinstance(entry.get("spec_path"), str)
        ),
    }
    summary["model"] = resolved_model
    with (phase_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return results
