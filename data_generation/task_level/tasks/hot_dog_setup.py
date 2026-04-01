"""Define the HotDogSetup task prompt, schema, and semantic validator."""

from __future__ import annotations

from typing import Any

from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.shared.errors import (
    TaskPreconditionSemanticValidationError,
)
from data_generation.task_level.tasks.shared.fsm import FiniteStateTaskValidator
from data_generation.task_level.tasks.shared.instances import (
    build_randomized_fixture_task_instance,
    make_symbolic_trajectory_record_builder,
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

HOT_DOG_SETUP_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "condiment_source_fixture",
            "held_object": None,
        },
        "agent_1": {
            "location": "condiment_source_fixture",
            "held_object": None,
        },
    },
    "objects": {
        "bun": {
            "object_type": "hotdog_bun",
            "location": "bun_source_fixture",
        },
        "sausage": {
            "object_type": "sausage",
            "location": "sausage_source_fixture",
        },
        "condiment": {
            "object_type": "condiment_bottle",
            "location": "condiment_source_fixture",
        },
        "serving_plate": {
            "object_type": "plate",
            "location": "serving_surface",
        },
    },
    "fixtures": {
        "condiment_source_fixture": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "open",
                }
            },
        },
        "bun_source_fixture": {"fixture_type": "counter"},
        "sausage_source_fixture": {"fixture_type": "fridge"},
        "serving_surface": {"fixture_type": "dining_table"},
    },
    "machine_state": {
        "hot_dog_setup": {
            "condiment_placed_next_to_serving_plate": False,
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
    ),
    overrides={
        "navigate_to_fixture": {
            "allowed_fixture_ids": [
                "condiment_source_fixture",
                "bun_source_fixture",
                "sausage_source_fixture",
                "serving_surface",
            ],
        },
        "pick_up_object": {
            "allowed_object_ids": ["bun", "sausage", "condiment"],
            "allowed_source_ids": [
                "condiment_source_fixture",
                "bun_source_fixture",
                "sausage_source_fixture",
            ],
        },
        "place_on_object": {
            "allowed_object_ids": ["bun", "sausage"],
            "allowed_support_object_ids": ["serving_plate"],
        },
        "place_next_to": {
            "allowed_object_ids": ["condiment"],
            "allowed_reference_object_ids": ["serving_plate"],
        },
    },
)

HOT_DOG_SETUP_RESPONSE_SCHEMA = build_task_response_schema(
    agent_ids=AGENT_IDS,
    allowed_tool_specs=HOT_DOG_SETUP_ALLOWED_TOOL_SPECS,
)

HOT_DOG_SETUP_TASK_GOAL = (
    "move bun and sausage onto serving_plate on serving_surface, then place "
    "condiment next to serving_plate on serving_surface."
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
        "Move bun and sausage onto serving_plate, not directly onto serving_surface.",
        "Only use place_next_to for condiment with reference_object_id set to serving_plate.",
        "Keep serving_plate on serving_surface throughout the trajectory.",
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
                "bun_location": effective_initial_state["objects"]["bun"]["location"],
                "sausage_location": effective_initial_state["objects"]["sausage"][
                    "location"
                ],
                "condiment_location": effective_initial_state["objects"]["condiment"][
                    "location"
                ],
                "serving_plate_location": effective_initial_state["objects"][
                    "serving_plate"
                ]["location"],
                "condiment_placed_next_to_serving_plate": effective_initial_state[
                    "machine_state"
                ]["hot_dog_setup"]["condiment_placed_next_to_serving_plate"],
            },
        )

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Checks the HotDogSetup-specific stationary plate invariant."""

        if runtime_state.objects["serving_plate"]["location"] != "serving_surface":
            raise TaskPreconditionSemanticValidationError(
                "serving_plate must remain on serving_surface throughout HotDogSetup.",
                details={
                    "object_id": "serving_plate",
                    "required_location": "serving_surface",
                    "actual_location": runtime_state.objects["serving_plate"][
                        "location"
                    ],
                    "tool": step["tool"],
                },
            )

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks success using the HotDogSetup goal state."""

        bun_on_plate = runtime_state.objects["bun"]["location"] == "serving_plate"
        sausage_on_plate = (
            runtime_state.objects["sausage"]["location"] == "serving_plate"
        )
        plate_on_table = (
            runtime_state.objects["serving_plate"]["location"] == "serving_surface"
        )
        condiment_on_table = (
            runtime_state.objects["condiment"]["location"] == "serving_surface"
        )
        condiment_next_to_plate = runtime_state.machine_state["hot_dog_setup"][
            "condiment_placed_next_to_serving_plate"
        ]
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
        """Updates HotDogSetup-specific summary state after each action."""

        if (
            step["tool"] == "place_next_to"
            and step["args"]["object_id"] == "condiment"
            and step["args"]["reference_object_id"] == "serving_plate"
        ):
            runtime_state.machine_state["hot_dog_setup"][
                "condiment_placed_next_to_serving_plate"
            ] = True

        runtime_state.public_state["bun_location"] = runtime_state.objects["bun"][
            "location"
        ]
        runtime_state.public_state["sausage_location"] = runtime_state.objects[
            "sausage"
        ]["location"]
        runtime_state.public_state["condiment_location"] = runtime_state.objects[
            "condiment"
        ]["location"]
        runtime_state.public_state["serving_plate_location"] = runtime_state.objects[
            "serving_plate"
        ]["location"]
        runtime_state.public_state["condiment_placed_next_to_serving_plate"] = (
            runtime_state.machine_state["hot_dog_setup"][
                "condiment_placed_next_to_serving_plate"
            ]
        )


build_hot_dog_setup_trajectory_record = make_symbolic_trajectory_record_builder(
    composite_task="HotDogSetup",
    agent_ids=AGENT_IDS,
)


HOT_DOG_SETUP_TASK = TaskDefinition(
    composite_task="HotDogSetup",
    response_schema=HOT_DOG_SETUP_RESPONSE_SCHEMA,
    preflight_token_estimate=HOT_DOG_SETUP_PREFLIGHT_TOKEN_ESTIMATE,
    build_task_instance=lambda run_index, runtime_config=None: build_randomized_fixture_task_instance(
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
