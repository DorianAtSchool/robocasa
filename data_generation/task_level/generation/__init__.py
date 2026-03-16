"""Expose task-level generation helpers without eagerly importing the CLI."""

from __future__ import annotations

__all__ = [
    "RuntimeConfig",
    "generate_single_trajectory",
    "generate_trajectories",
    "main",
    "parse_args",
    "run_cli",
]


def __getattr__(name: str):
    """Loads public generation exports lazily to avoid preloading the CLI module."""

    if name == "RuntimeConfig":
        from data_generation.task_level.generation.config import RuntimeConfig

        return RuntimeConfig
    if name in {"generate_single_trajectory", "generate_trajectories"}:
        from data_generation.task_level.generation.orchestrator import (
            generate_single_trajectory,
            generate_trajectories,
        )

        exports = {
            "generate_single_trajectory": generate_single_trajectory,
            "generate_trajectories": generate_trajectories,
        }
        return exports[name]
    if name in {"main", "parse_args", "run_cli"}:
        from data_generation.task_level.generation.cli import main, parse_args, run_cli

        exports = {
            "main": main,
            "parse_args": parse_args,
            "run_cli": run_cli,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
