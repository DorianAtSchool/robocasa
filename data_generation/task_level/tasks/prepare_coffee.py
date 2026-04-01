"""Define the PrepareCoffee task prompt, schema, and semantic validator."""

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

PREPARE_COFFEE_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "mug_source_fixture",
            "held_object": None,
        },
        "agent_1": {
            "location": "mug_source_fixture",
            "held_object": None,
        },
    },
    "objects": {
        "mug": {
            "object_type": "mug",
            "location": "mug_source_fixture",
        }
    },
    "fixtures": {
        "mug_source_fixture": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "closed",
                }
            },
        },
        "staging_surface": {"fixture_type": "counter"},
        "coffee_machine": {
            "fixture_type": "coffee_machine",
            "controls": {
                "start_button": {
                    "control_type": "button",
                }
            },
        },
    },
    "machine_state": {
        "coffee_machine": {
            "started": False,
            "dispenser_id": "coffee_machine_dispenser",
        }
    },
}

PREPARE_COFFEE_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "navigate_to_fixture",
        "open_hinged_part",
        "pick_up_object",
        "place_on_surface",
        "place_under",
        "press_button",
    ),
    overrides={
        "navigate_to_fixture": {
            "allowed_fixture_ids": [
                "mug_source_fixture",
                "staging_surface",
                "coffee_machine",
            ],
        },
        "open_hinged_part": {
            "allowed_target_ids": ["mug_source_fixture"],
            "allowed_part_ids": ["door"],
        },
        "pick_up_object": {
            "allowed_object_ids": ["mug"],
            "allowed_source_ids": ["mug_source_fixture", "staging_surface"],
        },
        "place_on_surface": {
            "allowed_object_ids": ["mug"],
            "allowed_support_ids": ["staging_surface"],
        },
        "place_under": {
            "allowed_object_ids": ["mug"],
            "allowed_reference_fixture_ids": ["coffee_machine"],
        },
        "press_button": {
            "allowed_target_ids": ["coffee_machine"],
            "allowed_control_ids": ["start_button"],
        },
    },
)

PREPARE_COFFEE_RESPONSE_SCHEMA = build_task_response_schema(
    agent_ids=AGENT_IDS,
    allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
)

PREPARE_COFFEE_TASK_GOAL = (
    "retrieve mug from mug_source_fixture, place it on staging_surface, move it "
    "under coffee_machine, then press coffee_machine.start_button."
)
PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES = tuple(
    tool_name
    for tool_name in PREPARE_COFFEE_ALLOWED_TOOL_SPECS
    if tool_name != "communicate"
)

# Tokens approximated using some preliminary runs.
PREPARE_COFFEE_PREFLIGHT_TOKEN_ESTIMATE = PreflightTokenEstimate(
    prompt_tokens=3110,
    output_tokens=3600,
)


build_prepare_coffee_prompt = make_task_prompt_builder(
    composite_task="PrepareCoffee",
    task_goal=PREPARE_COFFEE_TASK_GOAL,
    initial_state=PREPARE_COFFEE_INITIAL_STATE,
    allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
    non_communicate_tool_names=PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES,
    extra_execution_rules=(
        "Open mug_source_fixture.door before using pick_up_object on mug from mug_source_fixture.",
        "Only press coffee_machine.start_button after mug is already at coffee_machine_dispenser.",
        "If an agent is blocked because the other agent still needs to open the cabinet, move the mug, or place the mug under coffee_machine, use communicate to explain what it is waiting on before the other agent proceeds.",
    ),
)


class PrepareCoffeeValidator(FiniteStateTaskValidator):
    """Validates PrepareCoffee subatomic trajectories with the shared FSM."""

    def __init__(self, task_instance: TaskInstance | None = None) -> None:
        """Initializes the shared FSM with PrepareCoffee-specific configuration."""

        effective_initial_state = (
            task_instance.initial_state
            if task_instance is not None
            else PREPARE_COFFEE_INITIAL_STATE
        )
        super().__init__(
            composite_task="PrepareCoffee",
            agent_ids=AGENT_IDS,
            initial_state=effective_initial_state,
            allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
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
                "mug_location": effective_initial_state["objects"]["mug"]["location"],
                "coffee_machine_started": effective_initial_state["machine_state"][
                    "coffee_machine"
                ]["started"],
            },
        )

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Checks the PrepareCoffee-specific preconditions not covered generically."""

        if (
            step["tool"] == "pick_up_object"
            and step["args"]["source_id"] == "mug_source_fixture"
            and runtime_state.fixtures["mug_source_fixture"]["parts"]["door"]["state"]
            != "open"
        ):
            raise TaskPreconditionSemanticValidationError(
                "pick_up_object from mug_source_fixture requires the cabinet door to be open.",
                details={
                    "tool": step["tool"],
                    "fixture_id": "mug_source_fixture",
                    "part_id": "door",
                    "required_state": "open",
                    "actual_state": runtime_state.fixtures["mug_source_fixture"][
                        "parts"
                    ]["door"]["state"],
                },
            )

        if (
            step["tool"] == "press_button"
            and runtime_state.objects["mug"]["location"] != "coffee_machine_dispenser"
        ):
            raise TaskPreconditionSemanticValidationError(
                "press_button on the coffee machine requires the mug under the coffee machine dispenser.",
                details={
                    "tool": step["tool"],
                    "object_id": "mug",
                    "required_location": "coffee_machine_dispenser",
                    "actual_location": runtime_state.objects["mug"]["location"],
                },
            )

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks success using the PrepareCoffee goal state."""

        mug_under_dispenser = (
            runtime_state.objects["mug"]["location"] == "coffee_machine_dispenser"
        )
        coffee_machine_started = runtime_state.machine_state["coffee_machine"][
            "started"
        ]
        return mug_under_dispenser and coffee_machine_started

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Updates the PrepareCoffee-specific state after each action."""

        if step["tool"] == "press_button":
            runtime_state.machine_state["coffee_machine"]["started"] = True

        runtime_state.public_state["mug_location"] = runtime_state.objects["mug"][
            "location"
        ]
        runtime_state.public_state["coffee_machine_started"] = (
            runtime_state.machine_state["coffee_machine"]["started"]
        )


build_prepare_coffee_trajectory_record = make_symbolic_trajectory_record_builder(
    composite_task="PrepareCoffee",
    agent_ids=AGENT_IDS,
)


PREPARE_COFFEE_TASK = TaskDefinition(
    composite_task="PrepareCoffee",
    response_schema=PREPARE_COFFEE_RESPONSE_SCHEMA,
    preflight_token_estimate=PREPARE_COFFEE_PREFLIGHT_TOKEN_ESTIMATE,
    build_task_instance=lambda run_index, runtime_config=None: build_randomized_fixture_task_instance(
        composite_task="PrepareCoffee",
        agent_ids=AGENT_IDS,
        initial_state=PREPARE_COFFEE_INITIAL_STATE,
        allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
        run_index=run_index,
        runtime_config=runtime_config,
    ),
    build_prompt=build_prepare_coffee_prompt,
    build_trajectory_record=build_prepare_coffee_trajectory_record,
    validator_factory=PrepareCoffeeValidator,
)
