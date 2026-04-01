"""Build symbolic per-run task instances and canonical raw trajectory payloads."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Callable, Sequence

from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.utils import stable_json_sha256

from .types import TaskInstance


def _sample_agent_locations(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    fixture_ids: Sequence[str],
    run_index: int,
) -> dict[str, str]:
    """Samples deterministic starting fixtures for each agent independently."""

    seed_material = stable_json_sha256(
        {
            "composite_task": composite_task,
            "run_index": run_index,
            "fixture_ids": tuple(fixture_ids),
            "agent_ids": tuple(agent_ids),
        }
    )
    rng = random.Random(seed_material)
    # Sample each agent independently so different runs may start agents on the
    # same fixture or on different fixtures.
    return {agent_id: rng.choice(tuple(fixture_ids)) for agent_id in agent_ids}


def resolve_initial_position_fixture_ids(
    *,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Resolves the fixture IDs agents may use as randomized starting positions."""

    navigate_spec = allowed_tool_specs.get("navigate_to_fixture", {})
    allowed_fixture_ids = navigate_spec.get("allowed_fixture_ids")
    if isinstance(allowed_fixture_ids, list) and allowed_fixture_ids:
        if not all(isinstance(fixture_id, str) for fixture_id in allowed_fixture_ids):
            raise ValueError(
                "navigate_to_fixture.allowed_fixture_ids must be a list of strings."
            )
        return tuple(allowed_fixture_ids)

    fixture_state = initial_state.get("fixtures", {})
    if isinstance(fixture_state, dict) and fixture_state:
        fixture_ids = tuple(fixture_state)
        if not all(isinstance(fixture_id, str) for fixture_id in fixture_ids):
            raise ValueError("initial_state.fixtures keys must be strings.")
        return fixture_ids

    raise ValueError(
        "Tasks must define at least one fixture position for initial agent placement."
    )


def build_randomized_fixture_task_instance(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
    run_index: int,
    runtime_config: Any | None = None,
) -> TaskInstance:
    """Builds one per-run task instance with configurable agent start fixtures."""

    sampled_initial_state = deepcopy(initial_state)
    random_start_location = True
    if runtime_config is not None:
        random_start_location = bool(
            getattr(runtime_config, "random_start_location", True)
        )
    if not random_start_location:
        for agent_id in agent_ids:
            agent_state = sampled_initial_state.setdefault("agents", {}).setdefault(
                agent_id, {}
            )
            agent_state.setdefault("held_object", None)
        return TaskInstance(initial_state=sampled_initial_state)

    fixture_ids = resolve_initial_position_fixture_ids(
        initial_state=initial_state,
        allowed_tool_specs=allowed_tool_specs,
    )
    sampled_agent_locations = _sample_agent_locations(
        composite_task=composite_task,
        agent_ids=agent_ids,
        fixture_ids=fixture_ids,
        run_index=run_index,
    )

    for agent_id in agent_ids:
        agent_state = sampled_initial_state.setdefault("agents", {}).setdefault(
            agent_id, {}
        )
        agent_state["location"] = sampled_agent_locations[agent_id]
        agent_state.setdefault("held_object", None)

    return TaskInstance(initial_state=sampled_initial_state)


def build_symbolic_trajectory_record(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
    task_instance: TaskInstance,
) -> dict[str, Any]:
    """Builds the persisted raw trajectory payload for one symbolic task."""

    return {
        "trajectory_id": trajectory_id,
        "composite_task": composite_task,
        "agents": build_canonical_agents(agent_ids),
        "initial_state": task_instance.initial_state,
        "steps": candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
        "grounding_map": build_grounding_map_for_task(
            composite_task,
            task_instance.initial_state,
        ),
    }


def build_canonical_agents(agent_ids: Sequence[str]) -> list[dict[str, str]]:
    """Builds the shared persisted agent roster for trajectories."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    return [{"agent": agent_id} for agent_id in agent_ids]


def make_symbolic_trajectory_record_builder(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
) -> Callable[
    [dict[str, Any], dict[str, Any], str, dict[str, Any], TaskInstance], dict[str, Any]
]:
    """Builds a reusable symbolic raw-trajectory writer for one task."""

    def build_trajectory_record(
        candidate: dict[str, Any],
        validation: dict[str, Any],
        trajectory_id: str,
        generation_usage: dict[str, Any],
        task_instance: TaskInstance,
    ) -> dict[str, Any]:
        """Builds one symbolic raw trajectory record."""

        return build_symbolic_trajectory_record(
            composite_task=composite_task,
            agent_ids=agent_ids,
            candidate=candidate,
            validation=validation,
            trajectory_id=trajectory_id,
            generation_usage=generation_usage,
            task_instance=task_instance,
        )

    return build_trajectory_record
