"""Define the PrepareCoffee task prompt, schema, and semantic validator."""

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

PREPARE_COFFEE_INITIAL_STATE = {
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
        "mug_1": {
            "object_type": "mug",
            "location": "cabinet_1",
        }
    },
    "fixtures": {
        "cabinet_1": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "closed",
                }
            },
        },
        "counter_1": {"fixture_type": "counter"},
        "coffee_machine_1": {
            "fixture_type": "coffee_machine",
            "controls": {
                "start_button": {
                    "control_type": "button",
                }
            },
        },
    },
    "machine_state": {
        "coffee_machine_1": {
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
        "place_under_dispenser",
        "press_button",
        "wait",
    ),
    overrides={
        "navigate_to_fixture": {
            "allowed_fixture_ids": ["cabinet_1", "counter_1", "coffee_machine_1"],
        },
        "open_hinged_part": {
            "allowed_target_ids": ["cabinet_1"],
            "allowed_part_ids": ["door"],
        },
        "pick_up_object": {
            "allowed_object_ids": ["mug_1"],
            "allowed_source_ids": ["cabinet_1", "counter_1"],
        },
        "place_on_surface": {
            "allowed_object_ids": ["mug_1"],
            "allowed_support_ids": ["counter_1"],
        },
        "place_under_dispenser": {
            "allowed_object_ids": ["mug_1"],
            "allowed_dispenser_ids": ["coffee_machine_dispenser"],
        },
        "press_button": {
            "allowed_target_ids": ["coffee_machine_1"],
            "allowed_control_ids": ["start_button"],
        },
    },
)

PREPARE_COFFEE_RESPONSE_SCHEMA = build_task_response_schema(
    agent_ids=AGENT_IDS,
    allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
)

PREPARE_COFFEE_TASK_GOAL = (
    "retrieve mug_1 from cabinet_1, place it on counter_1, move it under the "
    "coffee machine dispenser, then press the coffee machine start button."
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
        "Open cabinet_1.door before using pick_up_object on mug_1 from cabinet_1.",
        "Only press coffee_machine_1.start_button after mug_1 is already at coffee_machine_dispenser.",
        "If an agent is blocked because the other agent still needs to open the cabinet, move the mug, or place the mug under the dispenser, use wait with a short positive duration.",
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
                "mug_location": effective_initial_state["objects"]["mug_1"]["location"],
                "coffee_machine_started": effective_initial_state["machine_state"]["coffee_machine_1"]["started"],
            },
        )

    def resolve_required_fixture(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Maps the coffee-machine dispenser token to its owning fixture."""

        if step["tool"] == "place_under_dispenser":
            return "coffee_machine_1"
        return super().resolve_required_fixture(step, runtime_state)

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Checks the PrepareCoffee-specific preconditions not covered generically."""

        if (
            step["tool"] == "pick_up_object"
            and step["args"]["source_id"] == "cabinet_1"
            and runtime_state.fixtures["cabinet_1"]["parts"]["door"]["state"] != "open"
        ):
            raise TaskPreconditionSemanticValidationError(
                "pick_up_object from cabinet_1 requires the cabinet door to be open.",
                details={
                    "tool": step["tool"],
                    "fixture_id": "cabinet_1",
                    "part_id": "door",
                    "required_state": "open",
                    "actual_state": runtime_state.fixtures["cabinet_1"]["parts"]["door"]["state"],
                },
            )

        if (
            step["tool"] == "press_button"
            and runtime_state.objects["mug_1"]["location"] != "coffee_machine_dispenser"
        ):
            raise TaskPreconditionSemanticValidationError(
                "press_button on the coffee machine requires mug_1 under the coffee machine dispenser.",
                details={
                    "tool": step["tool"],
                    "object_id": "mug_1",
                    "required_location": "coffee_machine_dispenser",
                    "actual_location": runtime_state.objects["mug_1"]["location"],
                },
            )

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks success usihng goal expression."""

        mug_under_dispenser = (
            runtime_state.objects["mug_1"]["location"] == "coffee_machine_dispenser"
        )
        coffee_machine_started = runtime_state.machine_state["coffee_machine_1"]["started"]
        return mug_under_dispenser and coffee_machine_started

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Updates the PrepareCoffee-specific symbolic state after each action."""

        if step["tool"] == "press_button":
            runtime_state.machine_state["coffee_machine_1"]["started"] = True

        runtime_state.public_state["mug_location"] = runtime_state.objects["mug_1"]["location"]
        runtime_state.public_state["coffee_machine_started"] = runtime_state.machine_state["coffee_machine_1"]["started"]


def build_prepare_coffee_trajectory_record(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
    task_instance: TaskInstance,
) -> dict[str, Any]:
    """Builds the persisted trajectory payload for PrepareCoffee."""

    return {
        "trajectory_id": trajectory_id,
        "composite_task": "PrepareCoffee",
        "agents": build_canonical_agents(AGENT_IDS),
        "initial_state": task_instance.initial_state,
        "steps": candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
    }


PREPARE_COFFEE_TASK = TaskDefinition(
    composite_task="PrepareCoffee",
    response_schema=PREPARE_COFFEE_RESPONSE_SCHEMA,
    preflight_token_estimate=PREPARE_COFFEE_PREFLIGHT_TOKEN_ESTIMATE,
    build_task_instance=lambda run_index: build_randomized_fixture_task_instance(
        composite_task="PrepareCoffee",
        agent_ids=AGENT_IDS,
        initial_state=PREPARE_COFFEE_INITIAL_STATE,
        allowed_tool_specs=PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
        run_index=run_index,
    ),
    build_prompt=build_prepare_coffee_prompt,
    build_trajectory_record=build_prepare_coffee_trajectory_record,
    validator_factory=PrepareCoffeeValidator,
)
