"""Dispatch generation requests to on-demand or batch runtimes."""

from __future__ import annotations

from typing import Any

from data_generation.task_level.generation.raw.config import (
    RuntimeConfig,
    _validate_runtime_config,
)
from data_generation.task_level.generation.raw.runtime_support import (
    _resolve_task_definition_or_raise,
)


def generate_single_trajectory(
    *,
    trajectory_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    client_factory: Any = None,
    overall_progress: Any | None = None,
    trajectory_progress: Any | None = None,
    seen_signatures: set[str] | None = None,
    seen_signatures_lock: threading.Lock | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    # Keep the public entrypoint stable while the implementation lives in the
    # on-demand runtime module.
    from data_generation.task_level.runtime.on_demand_generation import (
        generate_single_trajectory as generate_single_trajectory_on_demand,
    )

    return generate_single_trajectory_on_demand(
        trajectory_index=trajectory_index,
        runtime_config=runtime_config,
        task_definition=task_definition,
        client_factory=client_factory,
        overall_progress=overall_progress,
        trajectory_progress=trajectory_progress,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
    )


def generate_trajectories(
    runtime_config: RuntimeConfig,
    *,
    client_factory: Any = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    task_definition = _resolve_task_definition_or_raise(runtime_config.composite_task)
    # Centralize runtime selection here so the CLI and tests share one dispatch
    # point regardless of execution mode.
    _validate_runtime_config(runtime_config)
    if runtime_config.batch_processing:
        from data_generation.task_level.runtime.batch_generation import (
            generate_trajectories_batch,
        )

        return generate_trajectories_batch(
            runtime_config,
            task_definition=task_definition,
            show_progress=show_progress,
        )
    from data_generation.task_level.runtime.on_demand_generation import (
        generate_trajectories_on_demand,
    )

    return generate_trajectories_on_demand(
        runtime_config,
        task_definition=task_definition,
        client_factory=client_factory,
        show_progress=show_progress,
    )
