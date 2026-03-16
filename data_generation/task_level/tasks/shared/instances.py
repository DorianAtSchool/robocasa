"""Build concrete per-run task instances and canonical agent payloads."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Sequence

from data_generation.utils import stable_json_sha256

from .types import TaskInstance


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
) -> TaskInstance:
    """Builds one deterministic per-run task instance with randomized start fixtures."""

    fixture_ids = resolve_initial_position_fixture_ids(
        initial_state=initial_state,
        allowed_tool_specs=allowed_tool_specs,
    )
    sampled_initial_state = deepcopy(initial_state)
    seed_material = stable_json_sha256(
        {
            "composite_task": composite_task,
            "run_index": run_index,
            "fixture_ids": fixture_ids,
            "agent_ids": tuple(agent_ids),
        }
    )
    rng = random.Random(seed_material)

    for agent_id in agent_ids:
        agent_state = sampled_initial_state.setdefault("agents", {}).setdefault(
            agent_id, {}
        )
        agent_state["location"] = rng.choice(fixture_ids)
        agent_state.setdefault("held_object", None)

    return TaskInstance(initial_state=sampled_initial_state)


def build_canonical_agents(agent_ids: Sequence[str]) -> list[dict[str, str]]:
    """Builds the shared persisted agent roster for trajectories."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    return [{"agent": agent_id} for agent_id in agent_ids]
