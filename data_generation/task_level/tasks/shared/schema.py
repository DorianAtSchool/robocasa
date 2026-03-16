"""Build and validate shared task-level JSON response schemas."""

from __future__ import annotations

from typing import Any, Sequence

from .errors import TrajectoryStructureValidationError


def _normalize_text(value: Any, field_name: str) -> str:
    """Normalizes and validates a required string field."""

    if not isinstance(value, str):
        raise TrajectoryStructureValidationError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise TrajectoryStructureValidationError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Normalizes and validates a required object field."""

    if not isinstance(value, dict):
        raise TrajectoryStructureValidationError(f"{field_name} must be an object.")
    return dict(value)


def _append_unique_field_names(
    destination: list[str],
    field_names: Sequence[str],
) -> None:
    """Appends schema field names while preserving their first-seen order."""

    for field_name in field_names:
        if field_name not in destination:
            destination.append(field_name)


def _build_symbolic_field_schema(
    field_name: str,
    agent_ids: Sequence[str],
    schema_type: str = "STRING",
) -> dict[str, Any]:
    """Builds a schema property entry for one symbolic response field."""

    if field_name in {"agent", "agent_id", "to"}:
        return {
            "type": "STRING",
            "enum": list(agent_ids),
        }
    return {"type": schema_type}


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    """Maps a symbolic tool argument like fixture_id to its allowed_* override key."""

    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _resolve_tool_arg_schema_type(
    field_name: str,
    tool_spec: dict[str, Any],
) -> str:
    """Resolves the shared response-schema type for one tool argument."""

    tool_arg_types = tool_spec.get("tool_arg_types", {})
    if not isinstance(tool_arg_types, dict):
        raise ValueError("tool_arg_types must be a mapping when provided.")
    schema_type = tool_arg_types.get(field_name, "STRING")
    if schema_type not in {"STRING", "INTEGER"}:
        raise ValueError(f"Unsupported schema type {schema_type!r} for {field_name}.")
    return schema_type


def build_task_response_schema(
    *,
    agent_ids: Sequence[str],
    allowed_tool_specs: dict[str, dict[str, Any]],
    min_steps: int = 3,
) -> dict[str, Any]:
    """Builds the shared task-level response schema from agent and tool metadata."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if not allowed_tool_specs:
        raise ValueError("allowed_tool_specs must contain at least one tool.")
    if min_steps < 1:
        raise ValueError("min_steps must be at least 1.")

    tool_arg_names: list[str] = []
    tool_arg_schema_types: dict[str, str] = {}
    for tool_spec in allowed_tool_specs.values():
        # Preserve the tool registry order so schema rendering stays stable.
        for field_name in tool_spec.get("tool_args", ()):
            if field_name not in tool_arg_schema_types:
                tool_arg_schema_types[field_name] = _resolve_tool_arg_schema_type(
                    field_name,
                    tool_spec,
                )
            elif tool_arg_schema_types[field_name] != _resolve_tool_arg_schema_type(
                field_name,
                tool_spec,
            ):
                raise ValueError(
                    f"Conflicting schema types were configured for tool arg {field_name}."
                )
        _append_unique_field_names(tool_arg_names, tool_spec.get("tool_args", ()))

    return {
        "type": "OBJECT",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "ARRAY",
                "minItems": min_steps,
                "items": {
                    "type": "OBJECT",
                    "required": [
                        "step",
                        "agent",
                        "tool",
                        "args",
                        "reasoning",
                    ],
                    "properties": {
                        "step": {"type": "INTEGER"},
                        "agent": _build_symbolic_field_schema(
                            "agent",
                            agent_ids,
                        ),
                        "tool": {
                            "type": "STRING",
                            "enum": list(allowed_tool_specs),
                        },
                        "args": {
                            "type": "OBJECT",
                            "properties": {
                                field_name: _build_symbolic_field_schema(
                                    field_name,
                                    agent_ids,
                                    tool_arg_schema_types[field_name],
                                )
                                for field_name in tool_arg_names
                            },
                        },
                        "reasoning": {"type": "STRING"},
                    },
                },
            },
        },
    }
