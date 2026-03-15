"""Define the HotDogSetup task prompt, schema, and semantic validator."""

from __future__ import annotations

from typing import Any

from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.base import (
    build_randomized_fixture_task_instance,
    build_canonical_agents,
    build_task_response_schema,
    FiniteStateTaskValidator,
    PreflightTokenEstimate,
    TaskPreconditionSemanticValidationError,
    TaskDefinition,
    TaskInstance,
    TaskRuntimeState,
    make_task_prompt_builder,
)


MAX_REASONING_CHARS = 200
AGENT_IDS = ("agent_0", "agent_1")

HOT_DOG_SETUP_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "staging_area",
            "held_object": None,
        },
        "agent_1": {
            "location": "staging_area",
            "held_object": None,
        },
    },
    "objects": {
        "hotdog_bun_1": {
            "object_type": "hotdog_bun",
            "location": "counter_1",
        },
        "sausage_1": {
            "object_type": "sausage",
            "location": "fridge_1",
        },
        "condiment_1": {
            "object_type": "condiment_bottle",
            "location": "cabinet_1",
        },
        "plate_1": {
            "object_type": "plate",
            "location": "dining_table_1",
        },
    },
    "fixtures": {
        "cabinet_1": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "open",
                }
            },
        },
        "counter_1": {"fixture_type": "counter"},
        "fridge_1": {"fixture_type": "fridge"},
        "dining_table_1": {"fixture_type": "dining_table"},
    },
    "machine_state": {
        "hot_dog_setup": {
            "condiment_placed_next_to_plate": False,
        }
    },
}

HOT_DOG_SETUP_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "navigate_to_fixture",
        "pick_up_object",
        "place_on_object",
        "place_next_to",
        "wait",
    ),
    overrides={
        "navigate_to_fixture": {
            "allowed_fixture_ids": ["cabinet_1", "counter_1", "fridge_1", "dining_table_1"],
        },
        "pick_up_object": {
            "allowed_object_ids": ["hotdog_bun_1", "sausage_1", "condiment_1"],
            "allowed_source_ids": ["cabinet_1", "counter_1", "fridge_1"],
        },
        "place_on_object": {
            "allowed_object_ids": ["hotdog_bun_1", "sausage_1"],
            "allowed_support_object_ids": ["plate_1"],
        },
        "place_next_to": {
            "allowed_object_ids": ["condiment_1"],
            "allowed_reference_object_ids": ["plate_1"],
        },
    },
)

HOT_DOG_SETUP_RESPONSE_SCHEMA = build_task_response_schema(
    agent_ids=AGENT_IDS,
    allowed_tool_specs=HOT_DOG_SETUP_ALLOWED_TOOL_SPECS,
)

HOT_DOG_SETUP_TASK_GOAL = (
    "move hotdog_bun_1 and sausage_1 onto plate_1 on dining_table_1, then place "
    "condiment_1 next to plate_1 on dining_table_1."
)
HOT_DOG_SETUP_NON_COMMUNICATE_TOOL_NAMES = tuple(
    tool_name
    for tool_name in HOT_DOG_SETUP_ALLOWED_TOOL_SPECS
    if tool_name != "communicate"
)

# Tokens approximated from a task prompt with similar symbolic complexity.
HOT_DOG_SETUP_PREFLIGHT_TOKEN_ESTIMATE = PreflightTokenEstimate(
    prompt_tokens=3250,
    output_tokens=3900,
)


build_hot_dog_setup_prompt = make_task_prompt_builder(
    composite_task="HotDogSetup",
    task_goal=HOT_DOG_SETUP_TASK_GOAL,
    initial_state=HOT_DOG_SETUP_INITIAL_STATE,
    allowed_tool_specs=HOT_DOG_SETUP_ALLOWED_TOOL_SPECS,
    non_communicate_tool_names=HOT_DOG_SETUP_NON_COMMUNICATE_TOOL_NAMES,
    extra_execution_rules=(
        "Move hotdog_bun_1 and sausage_1 onto plate_1, not directly onto dining_table_1.",
        "Only use place_next_to for condiment_1 with reference_object_id set to plate_1.",
        "Keep plate_1 on dining_table_1 throughout the trajectory.",
    ),
)


class HotDogSetupValidator(FiniteStateTaskValidator):
    """Validates HotDogSetup subatomic trajectories with the shared FSM."""

    def __init__(self, task_instance: TaskInstance | None = None) -> None:
        """Initializes the shared FSM with HotDogSetup-specific configuration."""

        effective_initial_state = (
            task_instance.initial_state
            if task_instance is not None
            else HOT_DOG_SETUP_INITIAL_STATE
        )
        super().__init__(
            composite_task="HotDogSetup",
            agent_ids=AGENT_IDS,
            initial_state=effective_initial_state,
            allowed_tool_specs=HOT_DOG_SETUP_ALLOWED_TOOL_SPECS,
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
                "bun_location": effective_initial_state["objects"]["hotdog_bun_1"]["location"],
                "sausage_location": effective_initial_state["objects"]["sausage_1"]["location"],
                "condiment_location": effective_initial_state["objects"]["condiment_1"]["location"],
                "plate_location": effective_initial_state["objects"]["plate_1"]["location"],
                "condiment_placed_next_to_plate": effective_initial_state["machine_state"]["hot_dog_setup"]["condiment_placed_next_to_plate"],
            },
        )

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Checks the HotDogSetup-specific stationary plate invariant."""

        if runtime_state.objects["plate_1"]["location"] != "dining_table_1":
            raise TaskPreconditionSemanticValidationError(
                "plate_1 must remain on dining_table_1 throughout HotDogSetup.",
                details={
                    "object_id": "plate_1",
                    "required_location": "dining_table_1",
                    "actual_location": runtime_state.objects["plate_1"]["location"],
                    "tool": step["tool"],
                },
            )

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks success using the symbolic hot-dog setup goal state."""

        bun_on_plate = runtime_state.objects["hotdog_bun_1"]["location"] == "plate_1"
        sausage_on_plate = runtime_state.objects["sausage_1"]["location"] == "plate_1"
        plate_on_table = runtime_state.objects["plate_1"]["location"] == "dining_table_1"
        condiment_on_table = runtime_state.objects["condiment_1"]["location"] == "dining_table_1"
        condiment_next_to_plate = runtime_state.machine_state["hot_dog_setup"]["condiment_placed_next_to_plate"]
        return (
            bun_on_plate
            and sausage_on_plate
            and plate_on_table
            and condiment_on_table
            and condiment_next_to_plate
        )

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Updates HotDogSetup-specific symbolic summary state after each action."""

        if (
            step["tool"] == "place_next_to"
            and step["args"]["object_id"] == "condiment_1"
            and step["args"]["reference_object_id"] == "plate_1"
        ):
            runtime_state.machine_state["hot_dog_setup"]["condiment_placed_next_to_plate"] = True

        runtime_state.public_state["bun_location"] = runtime_state.objects["hotdog_bun_1"]["location"]
        runtime_state.public_state["sausage_location"] = runtime_state.objects["sausage_1"]["location"]
        runtime_state.public_state["condiment_location"] = runtime_state.objects["condiment_1"]["location"]
        runtime_state.public_state["plate_location"] = runtime_state.objects["plate_1"]["location"]
        runtime_state.public_state["condiment_placed_next_to_plate"] = runtime_state.machine_state["hot_dog_setup"]["condiment_placed_next_to_plate"]


def build_hot_dog_setup_trajectory_record(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
    task_instance: TaskInstance,
) -> dict[str, Any]:
    """Builds the persisted trajectory payload for HotDogSetup."""

    return {
        "trajectory_id": trajectory_id,
        "composite_task": "HotDogSetup",
        "agents": build_canonical_agents(AGENT_IDS),
        "initial_state": task_instance.initial_state,
        "steps": candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
    }


HOT_DOG_SETUP_TASK = TaskDefinition(
    composite_task="HotDogSetup",
    response_schema=HOT_DOG_SETUP_RESPONSE_SCHEMA,
    preflight_token_estimate=HOT_DOG_SETUP_PREFLIGHT_TOKEN_ESTIMATE,
    build_task_instance=lambda run_index: build_randomized_fixture_task_instance(
        composite_task="HotDogSetup",
        agent_ids=AGENT_IDS,
        initial_state=HOT_DOG_SETUP_INITIAL_STATE,
        allowed_tool_specs=HOT_DOG_SETUP_ALLOWED_TOOL_SPECS,
        run_index=run_index,
    ),
    build_prompt=build_hot_dog_setup_prompt,
    build_trajectory_record=build_hot_dog_setup_trajectory_record,
    validator_factory=HotDogSetupValidator,
)
