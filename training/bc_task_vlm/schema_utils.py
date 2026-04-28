"""Validation and normalization helpers for one-step structured VLM outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from data_generation.task_level.tasks.shared.schema import build_task_response_schema


def build_single_step_response_schema(
    *,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds the task-level schema and narrows it to exactly one predicted step."""

    schema = build_task_response_schema(
        agent_ids=agent_ids,
        allowed_tool_specs=allowed_tool_specs,
        min_steps=1,
    )
    schema = deepcopy(schema)
    schema["properties"]["steps"]["maxItems"] = 1
    return schema


def compact_json_dumps(value: Any) -> str:
    """Serializes JSON in a stable compact format for targets and manifests."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def parse_first_json_object(text: str) -> Any:
    """Extracts the first balanced JSON object from a model response."""

    start = text.find("{")
    if start < 0:
        raise ValueError("Response did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("Response contained an unterminated JSON object.")


def canonicalize_for_comparison(value: Any) -> Any:
    """Normalizes JSON-like values for exact structured comparison."""

    if isinstance(value, dict):
        return {key: canonicalize_for_comparison(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize_for_comparison(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _normalize_required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _normalize_string_array(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    return [_normalize_required_string(item, f"{field_name}[]") for item in value]


def validate_single_step_payload(
    payload: Any,
    *,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validates and normalizes one structured model output payload."""

    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("Payload.steps must contain exactly one item.")

    step = steps[0]
    if not isinstance(step, dict):
        raise ValueError("Payload.steps[0] must be an object.")

    normalized_step_index = _normalize_integer(step.get("step"), "steps[0].step")
    normalized_agent = _normalize_required_string(step.get("agent"), "steps[0].agent")
    if normalized_agent not in agent_ids:
        raise ValueError(f"steps[0].agent must be one of {agent_ids}.")

    normalized_tool = _normalize_required_string(step.get("tool"), "steps[0].tool")
    if normalized_tool not in allowed_tool_specs:
        raise ValueError(f"steps[0].tool must be one of {tuple(allowed_tool_specs)}.")

    raw_args = step.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError("steps[0].args must be an object.")

    tool_spec = allowed_tool_specs[normalized_tool]
    expected_args = list(tool_spec.get("tool_args", ()))
    provided_arg_names = set(raw_args)
    missing_arg_names = [
        arg_name for arg_name in expected_args if arg_name not in raw_args
    ]
    if missing_arg_names:
        missing = ", ".join(missing_arg_names)
        raise ValueError(f"steps[0].args is missing required fields: {missing}.")

    unexpected_arg_names = provided_arg_names.difference(expected_args)
    if unexpected_arg_names:
        unexpected = ", ".join(sorted(unexpected_arg_names))
        raise ValueError(
            f"steps[0].args contains unsupported fields for {normalized_tool}: {unexpected}."
        )

    tool_arg_types = dict(tool_spec.get("tool_arg_types", {}))
    normalized_args: dict[str, Any] = {}
    for arg_name in expected_args:
        schema_type = tool_arg_types.get(arg_name, "STRING")
        raw_value = raw_args[arg_name]
        if schema_type == "STRING":
            normalized_value = _normalize_required_string(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        elif schema_type == "INTEGER":
            normalized_value = _normalize_integer(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        elif schema_type == "STRING_ARRAY":
            normalized_value = _normalize_string_array(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        else:
            raise ValueError(
                f"Unsupported schema type {schema_type!r} for argument {arg_name}."
            )

        if arg_name in {"agent", "agent_id", "to"}:
            allowed_values = set(agent_ids)
            if (
                isinstance(normalized_value, str)
                and normalized_value not in allowed_values
            ):
                allowed_text = ", ".join(sorted(allowed_values))
                raise ValueError(
                    f"steps[0].args.{arg_name} must be one of {allowed_text}."
                )

        allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
        if allowed_ids_key and allowed_ids_key in tool_spec:
            allowed_values = set(tool_spec[allowed_ids_key])
            if isinstance(normalized_value, str):
                if normalized_value not in allowed_values:
                    allowed_text = ", ".join(sorted(allowed_values))
                    raise ValueError(
                        f"steps[0].args.{arg_name} must be one of {allowed_text}."
                    )
            elif isinstance(normalized_value, list):
                invalid_values = [
                    value for value in normalized_value if value not in allowed_values
                ]
                if invalid_values:
                    invalid_text = ", ".join(sorted(invalid_values))
                    allowed_text = ", ".join(sorted(allowed_values))
                    raise ValueError(
                        f"steps[0].args.{arg_name} contains invalid values {invalid_text}; "
                        f"allowed values: {allowed_text}."
                    )

        normalized_args[arg_name] = normalized_value

    normalized_reasoning = _normalize_required_string(
        step.get("reasoning"),
        "steps[0].reasoning",
    )

    return {
        "steps": [
            {
                "step": normalized_step_index,
                "agent": normalized_agent,
                "tool": normalized_tool,
                "args": normalized_args,
                "reasoning": normalized_reasoning,
            }
        ]
    }
