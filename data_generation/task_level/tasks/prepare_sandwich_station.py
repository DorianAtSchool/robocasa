"""Define the PrepareSandwichStation task prompt, schema, and semantic validator."""

from __future__ import annotations

from typing import Any

from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.shared.errors import (
    TaskPreconditionSemanticValidationError,
)
from data_generation.task_level.tasks.shared.fsm import FiniteStateTaskValidator
from data_generation.task_level.tasks.shared.instances import (
    build_canonical_agents,
    build_randomized_fixture_task_instance,
)
from data_generation.task_level.tasks.shared.prompting import make_task_prompt_builder
from data_generation.task_level.tasks.shared.schema import build_task_response_schema
from data_generation.task_level.tasks.shared.state import TaskRuntimeState
from data_generation.task_level.tasks.shared.types import (
    PreflightTokenEstimate,
    TaskDefinition,
    TaskInstance,
)

MAX_REASONING_CHARS = 200
AGENT_IDS = ("agent_0", "agent_1")

PREPARE_SANDWICH_STATION_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "ingredient_source_fixture",
            "held_object": None,
        },
        "agent_1": {
            "location": "ingredient_source_fixture",
            "held_object": None,
        },
    },
    "objects": {
        "ingredient_bowl": {
            "object_type": "bowl",
            "location": "ingredient_source_fixture",
        },
        "baguette": {
            "object_type": "baguette",
            "location": "ingredient_source_fixture",
        },
        "tomato_slice": {
            "object_type": "tomato_slice",
            "location": "ingredient_bowl",
        },
        "pickle_slice": {
            "object_type": "pickle_slice",
            "location": "ingredient_bowl",
        },
        "turkey_slice": {
            "object_type": "turkey_slice",
            "location": "ingredient_bowl",
        },
    },
    "fixtures": {
        "ingredient_source_fixture": {"fixture_type": "fridge"},
        "staging_surface": {"fixture_type": "counter"},
        "toaster_oven": {"fixture_type": "toaster_oven"},
    },
    "machine_state": {
        "toaster_oven": {
            "adjacent_location_id": "staging_surface",
        },
        "prepare_sandwich_station": {
            "ingredient_bowl_staged": False,
            "baguette_staged": False,
        },
    },
}

PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "navigate_to_fixture",
        "pick_up_object",
        "place_next_to",
    ),
    overrides={
        "navigate_to_fixture": {
            "allowed_fixture_ids": ["ingredient_source_fixture", "staging_surface"],
        },
        "pick_up_object": {
            "allowed_object_ids": ["ingredient_bowl", "baguette"],
            "allowed_source_ids": ["ingredient_source_fixture", "staging_surface"],
        },
        "place_next_to": {
            "allowed_object_ids": ["ingredient_bowl", "baguette"],
            "allowed_reference_object_ids": ["toaster_oven"],
        },
    },
)

PREPARE_SANDWICH_STATION_RESPONSE_SCHEMA = build_task_response_schema(
    agent_ids=AGENT_IDS,
    allowed_tool_specs=PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS,
)

PREPARE_SANDWICH_STATION_TASK_GOAL = (
    "retrieve ingredient_bowl and baguette from ingredient_source_fixture, then "
    "place both next to toaster_oven on staging_surface to stage them near the "
    "toaster oven."
)
PREPARE_SANDWICH_STATION_NON_COMMUNICATE_TOOL_NAMES = tuple(
    tool_name
    for tool_name in PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS
    if tool_name != "communicate"
)

# Tokens approximated from tasks with similar prompt and symbolic complexity.
PREPARE_SANDWICH_STATION_PREFLIGHT_TOKEN_ESTIMATE = PreflightTokenEstimate(
    prompt_tokens=3250,
    output_tokens=3900,
)


build_prepare_sandwich_station_prompt = make_task_prompt_builder(
    composite_task="PrepareSandwichStation",
    task_goal=PREPARE_SANDWICH_STATION_TASK_GOAL,
    initial_state=PREPARE_SANDWICH_STATION_INITIAL_STATE,
    allowed_tool_specs=PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS,
    non_communicate_tool_names=PREPARE_SANDWICH_STATION_NON_COMMUNICATE_TOOL_NAMES,
    extra_execution_rules=(
        "Pick up ingredient_bowl and baguette from ingredient_source_fixture before staging them on staging_surface.",
        "Use place_next_to with reference_object_id toaster_oven so both items end up on staging_surface near the toaster oven.",
        "Keep tomato_slice, pickle_slice, and turkey_slice inside ingredient_bowl throughout the trajectory.",
    ),
)


class PrepareSandwichStationValidator(FiniteStateTaskValidator):
    """Validates PrepareSandwichStation subatomic trajectories with the shared FSM."""

    def __init__(self, task_instance: TaskInstance | None = None) -> None:
        """Initializes the shared FSM with PrepareSandwichStation-specific state."""

        effective_initial_state = (
            task_instance.initial_state
            if task_instance is not None
            else PREPARE_SANDWICH_STATION_INITIAL_STATE
        )
        super().__init__(
            composite_task="PrepareSandwichStation",
            agent_ids=AGENT_IDS,
            initial_state=effective_initial_state,
            allowed_tool_specs=PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS,
            checks=(
                "initial_communication",
                "allowed_tools",
                "navigation_preconditions",
                "manipulation_preconditions",
                "effects",
                "final_success",
            ),
            max_reasoning_chars=MAX_REASONING_CHARS,
            initial_public_state={
                "ingredient_bowl_location": effective_initial_state["objects"][
                    "ingredient_bowl"
                ]["location"],
                "baguette_location": effective_initial_state["objects"]["baguette"][
                    "location"
                ],
                "ingredient_bowl_staged": effective_initial_state["machine_state"][
                    "prepare_sandwich_station"
                ]["ingredient_bowl_staged"],
                "baguette_staged": effective_initial_state["machine_state"][
                    "prepare_sandwich_station"
                ]["baguette_staged"],
            },
        )

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Checks that the sandwich ingredients remain in the bowl while staging."""

        for ingredient_id in ("tomato_slice", "pickle_slice", "turkey_slice"):
            ingredient_location = runtime_state.objects[ingredient_id]["location"]
            if ingredient_location != "ingredient_bowl":
                raise TaskPreconditionSemanticValidationError(
                    f"{ingredient_id} must remain in ingredient_bowl during PrepareSandwichStation.",
                    details={
                        "tool": step["tool"],
                        "ingredient_id": ingredient_id,
                        "required_location": "ingredient_bowl",
                        "actual_location": ingredient_location,
                    },
                )

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks success using the symbolic staging goal state."""

        ingredient_bowl_on_counter = (
            runtime_state.objects["ingredient_bowl"]["location"] == "staging_surface"
        )
        baguette_on_counter = (
            runtime_state.objects["baguette"]["location"] == "staging_surface"
        )
        ingredient_bowl_staged = runtime_state.machine_state[
            "prepare_sandwich_station"
        ]["ingredient_bowl_staged"]
        baguette_staged = runtime_state.machine_state["prepare_sandwich_station"][
            "baguette_staged"
        ]
        return (
            ingredient_bowl_on_counter
            and baguette_on_counter
            and ingredient_bowl_staged
            and baguette_staged
        )

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Updates task-local sandwich staging flags after each action."""

        if (
            step["tool"] == "place_next_to"
            and step["args"]["reference_object_id"] == "toaster_oven"
        ):
            if step["args"]["object_id"] == "ingredient_bowl":
                runtime_state.machine_state["prepare_sandwich_station"][
                    "ingredient_bowl_staged"
                ] = True
            if step["args"]["object_id"] == "baguette":
                runtime_state.machine_state["prepare_sandwich_station"][
                    "baguette_staged"
                ] = True

        runtime_state.public_state["ingredient_bowl_location"] = runtime_state.objects[
            "ingredient_bowl"
        ]["location"]
        runtime_state.public_state["baguette_location"] = runtime_state.objects[
            "baguette"
        ]["location"]
        runtime_state.public_state["ingredient_bowl_staged"] = (
            runtime_state.machine_state["prepare_sandwich_station"][
                "ingredient_bowl_staged"
            ]
        )
        runtime_state.public_state["baguette_staged"] = runtime_state.machine_state[
            "prepare_sandwich_station"
        ]["baguette_staged"]


def build_prepare_sandwich_station_trajectory_record(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
    task_instance: TaskInstance,
) -> dict[str, Any]:
    """Builds the persisted trajectory payload for PrepareSandwichStation."""

    return {
        "trajectory_id": trajectory_id,
        "composite_task": "PrepareSandwichStation",
        "agents": build_canonical_agents(AGENT_IDS),
        "initial_state": task_instance.initial_state,
        "grounding_map": build_grounding_map_for_task(
            "PrepareSandwichStation",
            task_instance.initial_state,
        ),
        "steps": candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
    }


PREPARE_SANDWICH_STATION_TASK = TaskDefinition(
    composite_task="PrepareSandwichStation",
    response_schema=PREPARE_SANDWICH_STATION_RESPONSE_SCHEMA,
    preflight_token_estimate=PREPARE_SANDWICH_STATION_PREFLIGHT_TOKEN_ESTIMATE,
    build_task_instance=lambda run_index: build_randomized_fixture_task_instance(
        composite_task="PrepareSandwichStation",
        agent_ids=AGENT_IDS,
        initial_state=PREPARE_SANDWICH_STATION_INITIAL_STATE,
        allowed_tool_specs=PREPARE_SANDWICH_STATION_ALLOWED_TOOL_SPECS,
        run_index=run_index,
    ),
    build_prompt=build_prepare_sandwich_station_prompt,
    build_trajectory_record=build_prepare_sandwich_station_trajectory_record,
    validator_factory=PrepareSandwichStationValidator,
)
