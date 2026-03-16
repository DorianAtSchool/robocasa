"""Expose the public task-level generation helpers and shared tool metadata."""

from data_generation.task_level.subatomic_tool_calls import (
    SubatomicToolArg,
    SubatomicToolSpec,
    discover_subatomic_tools,
    render_subatomic_tool_catalog,
)
from data_generation.task_level.subatomic_tool_specs import (
    TASK_LEVEL_ALLOWED_TOOL_SPECS,
    build_allowed_tool_specs,
)

__all__ = [
    "SubatomicToolArg",
    "SubatomicToolSpec",
    "TASK_LEVEL_ALLOWED_TOOL_SPECS",
    "build_allowed_tool_specs",
    "discover_subatomic_tools",
    "render_subatomic_tool_catalog",
    "RuntimeConfig",
    "generate_trajectories",
]


def __getattr__(name):
    if name in {"RuntimeConfig", "generate_trajectories"}:
        from data_generation.task_level.raw_generation import (
            RuntimeConfig,
            generate_trajectories,
        )

        exports = {
            "RuntimeConfig": RuntimeConfig,
            "generate_trajectories": generate_trajectories,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
