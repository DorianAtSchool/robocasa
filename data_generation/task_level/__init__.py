from data_generation.task_level.tool_calls import (
    AtomicToolSpec,
    ConstructorArg,
    discover_atomic_tools,
    render_atomic_tool_catalog,
)

__all__ = [
    "AtomicToolSpec",
    "ConstructorArg",
    "discover_atomic_tools",
    "render_atomic_tool_catalog",
    "RuntimeConfig",
    "generate_trajectories",
]


def __getattr__(name):
    if name in {"RuntimeConfig", "generate_trajectories"}:
        from data_generation.task_level.trajectory_generation import (
            RuntimeConfig,
            generate_trajectories,
        )

        exports = {
            "RuntimeConfig": RuntimeConfig,
            "generate_trajectories": generate_trajectories,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
