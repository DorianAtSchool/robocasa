"""Centralize shared task-level subatomic tool metadata and task-specific spec builders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from data_generation.task_level.subatomic_tool_calls import discover_subatomic_tools

BASIC_TASK_TOOL_NAMES = ("give_space",)


def _build_subatomic_allowed_tool_specs() -> dict[str, dict[str, Any]]:
    """Builds the shared allowed-tool metadata for every subatomic tool."""

    subatomic_allowed_tool_specs: dict[str, dict[str, Any]] = {}
    for tool_spec in discover_subatomic_tools():
        argument_names = [argument.name for argument in tool_spec.constructor_args]
        tool_arg_types = {
            argument.name: argument.schema_type
            for argument in tool_spec.constructor_args
            if argument.schema_type != "STRING"
        }
        subatomic_allowed_tool_spec = {
            "description": tool_spec.description,
            "tool_args": list(argument_names),
        }
        if tool_arg_types:
            subatomic_allowed_tool_spec["tool_arg_types"] = tool_arg_types
        if tool_spec.name == "place_next_to":
            # `place_next_to` can anchor relative to either a movable object
            # or a fixture with an explicit adjacent symbolic surface.
            subatomic_allowed_tool_spec["tool_args"] = ["object_id"]
            subatomic_allowed_tool_spec["tool_arg_any_of"] = [
                ["reference_object_id", "reference_fixture_id"]
            ]
        subatomic_allowed_tool_specs[tool_spec.name] = subatomic_allowed_tool_spec
    return subatomic_allowed_tool_specs


# Keep the shared tool metadata centralized so individual task files only add
# task-specific symbolic constraints such as allowed fixture or object IDs.
SUBATOMIC_ALLOWED_TOOL_SPECS = _build_subatomic_allowed_tool_specs()


TASK_LEVEL_ALLOWED_TOOL_SPECS = {
    "communicate": {
        "description": (
            "Send a short coordination message to the other agent in the scene. "
            "args.to must be that agent's exact ID."
        ),
        "tool_args": ["to", "message"],
    },
    **SUBATOMIC_ALLOWED_TOOL_SPECS,
}


def build_allowed_tool_specs(
    tool_names: Sequence[str],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Builds a task-local allowed-tool map from the shared task-level registry.

    Args:
        tool_names: Ordered tool names to expose for a specific task, such as
            ``("communicate", "navigate_to_fixture", "pick_up_object")``.
            Shared basic coordination tools such as ``give_space`` are appended
            automatically when omitted.
        overrides: Optional per-tool metadata patches applied on top of the
            shared registry entry. Use this to add task-specific symbolic
            constraints without redefining the base tool spec. For example,
            ``{"pick_up_object": {"allowed_object_ids": ["mug"]}}`` keeps the
            shared description and ``tool_args`` fields, then
            adds the task-specific ``allowed_object_ids`` constraint.

    Returns:
        A new dictionary containing only the requested tools, with any override
        fields merged into the copied base specification for each tool.

    Raises:
        KeyError: If ``tool_names`` includes an unknown tool or if
            ``overrides`` contains a tool name that was not requested.
    """

    requested_tool_names = list(tool_names)
    for basic_tool_name in BASIC_TASK_TOOL_NAMES:
        if basic_tool_name not in requested_tool_names:
            requested_tool_names.append(basic_tool_name)

    selected_tool_specs: dict[str, dict[str, Any]] = {}
    tool_overrides = dict(overrides or {})

    for tool_name in requested_tool_names:
        if tool_name not in TASK_LEVEL_ALLOWED_TOOL_SPECS:
            raise KeyError(f"Unknown task-level tool name: {tool_name}")
        selected_tool_specs[tool_name] = deepcopy(
            TASK_LEVEL_ALLOWED_TOOL_SPECS[tool_name]
        )
        if tool_name in tool_overrides:
            # Merge task-specific symbolic constraints into the shared base spec.
            selected_tool_specs[tool_name].update(
                deepcopy(dict(tool_overrides[tool_name]))
            )

    # Keep give_space aligned with navigation fixture constraints when tasks do
    # not need to repeat the same allowed fixture list twice.
    give_space_spec = selected_tool_specs.get("give_space")
    navigate_tool_spec = selected_tool_specs.get("navigate_to_fixture")
    if (
        give_space_spec is not None
        and "allowed_fixture_ids" not in give_space_spec
        and navigate_tool_spec is not None
        and "allowed_fixture_ids" in navigate_tool_spec
    ):
        give_space_spec["allowed_fixture_ids"] = deepcopy(
            navigate_tool_spec["allowed_fixture_ids"]
        )

    unknown_override_names = set(tool_overrides) - set(requested_tool_names)
    if unknown_override_names:
        unknown_names = ", ".join(sorted(unknown_override_names))
        raise KeyError(
            f"Overrides were provided for unavailable tool names: {unknown_names}"
        )

    return selected_tool_specs
