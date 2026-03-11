from __future__ import annotations

import json
from typing import Any

from data_generation.task_level.tasks.base import TaskDefinition, TrajectoryValidationError
from data_generation.task_level.tool_calls import (
    discover_atomic_tools,
    render_atomic_tool_catalog,
    render_synthetic_communication_tool,
)
from data_generation.utils import stable_json_sha256


MAX_REASONING_CHARS = 200
AGENT_IDS = ("agent_0", "agent_1")

PREPARE_COFFEE_INITIAL_STATE = {
    "objects": {
        "mug_1": {
            "object_type": "mug",
            "location": "cabinet_1",
        }
    },
    "fixtures": {
        "cabinet_1": {"fixture_type": "cabinet"},
        "counter_1": {"fixture_type": "counter"},
        "coffee_machine_1": {"fixture_type": "coffee_machine"},
    },
    "machine_state": {
        "coffee_machine_1": {
            "started": False,
            "dispenser_target_location": "coffee_machine_dispenser",
        }
    },
}

PREPARE_COFFEE_ALLOWED_TOOL_SPECS = {
    "communicate": {
        "description": "Send a short coordination message to the other agent.",
        "tool_args": ["to_agent_id", "message"],
        "entity_refs": ["from_agent_id", "to_agent_id"],
    },
    "PickPlaceCabinetToCounter": {
        "description": "Move mug_1 from cabinet_1 to counter_1.",
        "tool_args": ["object_id", "source_fixture_id", "target_fixture_id"],
        "entity_refs": ["object_id", "source_fixture_id", "target_fixture_id"],
    },
    "CoffeeSetupMug": {
        "description": "Move mug_1 from counter_1 to coffee_machine_1 dispenser position.",
        "tool_args": ["object_id", "source_fixture_id", "target_fixture_id"],
        "entity_refs": ["object_id", "source_fixture_id", "target_fixture_id"],
    },
    "StartCoffeeMachine": {
        "description": "Press the coffee machine start button with mug_1 under the dispenser.",
        "tool_args": ["fixture_id", "object_id"],
        "entity_refs": ["fixture_id", "object_id"],
    },
}

PREPARE_COFFEE_EXPECTED_ACTIONS = {
    "PickPlaceCabinetToCounter": {
        "tool_args": {
            "object_id": "mug_1",
            "source_fixture_id": "cabinet_1",
            "target_fixture_id": "counter_1",
        },
        "entity_refs": {
            "object_id": "mug_1",
            "source_fixture_id": "cabinet_1",
            "target_fixture_id": "counter_1",
        },
    },
    "CoffeeSetupMug": {
        "tool_args": {
            "object_id": "mug_1",
            "source_fixture_id": "counter_1",
            "target_fixture_id": "coffee_machine_1",
        },
        "entity_refs": {
            "object_id": "mug_1",
            "source_fixture_id": "counter_1",
            "target_fixture_id": "coffee_machine_1",
        },
    },
    "StartCoffeeMachine": {
        "tool_args": {
            "fixture_id": "coffee_machine_1",
            "object_id": "mug_1",
        },
        "entity_refs": {
            "fixture_id": "coffee_machine_1",
            "object_id": "mug_1",
        },
    },
}

PREPARE_COFFEE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["agents", "steps"],
    "properties": {
        "agents": {
            "type": "ARRAY",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "OBJECT",
                "required": ["agent_id"],
                "properties": {
                    "agent_id": {
                        "type": "STRING",
                        "enum": list(AGENT_IDS),
                    },
                },
            },
        },
        "steps": {
            "type": "ARRAY",
            "minItems": 3,
            "items": {
                "type": "OBJECT",
                "required": [
                    "step_index",
                    "agent_id",
                    "tool_name",
                    "tool_args",
                    "entity_refs",
                    "reasoning",
                ],
                "properties": {
                    "step_index": {"type": "INTEGER"},
                    "agent_id": {
                        "type": "STRING",
                        "enum": list(AGENT_IDS),
                    },
                    "tool_name": {
                        "type": "STRING",
                        "enum": list(PREPARE_COFFEE_ALLOWED_TOOL_SPECS),
                    },
                    "tool_args": {
                        "type": "OBJECT",
                        "properties": {
                            "to_agent_id": {
                                "type": "STRING",
                                "enum": list(AGENT_IDS),
                            },
                            "message": {"type": "STRING"},
                            "object_id": {"type": "STRING"},
                            "source_fixture_id": {"type": "STRING"},
                            "target_fixture_id": {"type": "STRING"},
                            "fixture_id": {"type": "STRING"},
                        },
                    },
                    "entity_refs": {
                        "type": "OBJECT",
                        "properties": {
                            "from_agent_id": {
                                "type": "STRING",
                                "enum": list(AGENT_IDS),
                            },
                            "to_agent_id": {
                                "type": "STRING",
                                "enum": list(AGENT_IDS),
                            },
                            "object_id": {"type": "STRING"},
                            "source_fixture_id": {"type": "STRING"},
                            "target_fixture_id": {"type": "STRING"},
                            "fixture_id": {"type": "STRING"},
                        },
                    },
                    "reasoning": {"type": "STRING"},
                },
            },
        },
    },
}

PREPARE_COFFEE_PREFLIGHT_REFERENCE_CANDIDATE = {
    "agents": [
        {"agent_id": "agent_0"},
        {"agent_id": "agent_1"},
    ],
    "steps": [
        {
            "step_index": 0,
            "agent_id": "agent_0",
            "tool_name": "communicate",
            "tool_args": {
                "to_agent_id": "agent_1",
                "message": "I will move the mug to the counter first.",
            },
            "entity_refs": {
                "from_agent_id": "agent_0",
                "to_agent_id": "agent_1",
            },
            "reasoning": "We need a shared plan before acting.",
        },
        {
            "step_index": 1,
            "agent_id": "agent_1",
            "tool_name": "communicate",
            "tool_args": {
                "to_agent_id": "agent_0",
                "message": "I will finish setup at the coffee machine.",
            },
            "entity_refs": {
                "from_agent_id": "agent_1",
                "to_agent_id": "agent_0",
            },
            "reasoning": "I should confirm the handoff sequence.",
        },
        {
            "step_index": 2,
            "agent_id": "agent_0",
            "tool_name": "PickPlaceCabinetToCounter",
            "tool_args": {
                "object_id": "mug_1",
                "source_fixture_id": "cabinet_1",
                "target_fixture_id": "counter_1",
            },
            "entity_refs": {
                "object_id": "mug_1",
                "source_fixture_id": "cabinet_1",
                "target_fixture_id": "counter_1",
            },
            "reasoning": "The mug starts in the cabinet.",
        },
        {
            "step_index": 3,
            "agent_id": "agent_1",
            "tool_name": "CoffeeSetupMug",
            "tool_args": {
                "object_id": "mug_1",
                "source_fixture_id": "counter_1",
                "target_fixture_id": "coffee_machine_1",
            },
            "entity_refs": {
                "object_id": "mug_1",
                "source_fixture_id": "counter_1",
                "target_fixture_id": "coffee_machine_1",
            },
            "reasoning": "The mug must reach the dispenser next.",
        },
        {
            "step_index": 4,
            "agent_id": "agent_1",
            "tool_name": "StartCoffeeMachine",
            "tool_args": {
                "fixture_id": "coffee_machine_1",
                "object_id": "mug_1",
            },
            "entity_refs": {
                "fixture_id": "coffee_machine_1",
                "object_id": "mug_1",
            },
            "reasoning": "The mug is ready for brewing.",
        },
    ],
}


def build_prepare_coffee_prompt(variation_key: str) -> str:
    atomic_catalog = render_atomic_tool_catalog(discover_atomic_tools())
    allowed_tools = json.dumps(
        PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
        indent=2,
        sort_keys=True,
    )
    initial_state = json.dumps(
        PREPARE_COFFEE_INITIAL_STATE,
        indent=2,
        sort_keys=True,
    )

    return f"""
You are simulating two cooperative robot agents in a RoboCasa kitchen.
Generate a single valid multi-agent task-level trajectory for the composite task PrepareCoffee.

Important rules:
- Simulate both agents: agent_0 and agent_1.
- The agents must coordinate through communication tool calls before any task action.
- Use only the allowed tools for this task, even though a larger atomic tool catalog is provided.
- Every step must be executable and symbolically valid.
- Keep reasoning short and explicit. Each step reasoning must be a single short sentence.
- Agents can pass each other freely in the kitchen, including around the island.
- Output JSON only, with no markdown.
- Make this trajectory distinct from previous attempts by following variation key: {variation_key}

Composite task:
- PrepareCoffee
- Goal: move mug_1 from cabinet_1 to counter_1, then under coffee_machine_1 dispenser, then start the coffee machine.

Initial symbolic state:
{initial_state}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools}

Communication tool:
{render_synthetic_communication_tool()}

Full RoboCasa atomic tool catalog for context:
{atomic_catalog}

Output requirements:
- Return an object with keys: agents, steps.
- agents must contain exactly two entries: agent_0 and agent_1.
- Each agent entry must contain only: agent_id.
- steps must be an interleaved timeline ordered by step_index starting at 0 with no gaps.
- Each step must contain: step_index, agent_id, tool_name, tool_args, entity_refs, reasoning.
- Do not include role or initial_plan anywhere in the output.
- coordination steps must use tool_name communicate and include both to_agent_id and message inside tool_args.
- Non-communicate steps must use one of:
  PickPlaceCabinetToCounter, CoffeeSetupMug, StartCoffeeMachine.
- Use the exact symbolic IDs from the initial state.
""".strip()


def _normalize_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TrajectoryValidationError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise TrajectoryValidationError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrajectoryValidationError(f"{field_name} must be an object.")
    return dict(value)


class PrepareCoffeeValidator:
    def __init__(self) -> None:
        self._all_checks = [
            "initial_communication",
            "allowed_tools",
            "entity_continuity",
            "preconditions",
            "effects",
            "final_success",
        ]

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        agents = self._normalize_agents(candidate.get("agents"))
        steps = self._normalize_steps(candidate.get("steps"))

        communicated_agents: set[str] = set()
        first_action_seen = False
        mug_location = "cabinet_1"
        coffee_machine_started = False

        seen_step_indexes: set[int] = set()
        for expected_index, step in enumerate(steps):
            if step["step_index"] != expected_index:
                raise TrajectoryValidationError(
                    f"step_index {step['step_index']} does not match expected index {expected_index}."
                )
            if step["step_index"] in seen_step_indexes:
                raise TrajectoryValidationError(
                    f"Duplicate step_index {step['step_index']}."
                )
            seen_step_indexes.add(step["step_index"])

            if step["tool_name"] == "communicate":
                self._validate_communicate_step(step)
                communicated_agents.add(step["agent_id"])
                continue

            if communicated_agents != set(AGENT_IDS):
                raise TrajectoryValidationError(
                    "Both agents must coordinate via communication before the first task action."
                )

            first_action_seen = True
            if coffee_machine_started:
                raise TrajectoryValidationError(
                    "No task actions are allowed after the coffee machine has been started."
                )

            if step["tool_name"] == "PickPlaceCabinetToCounter":
                self._validate_expected_action(step, "PickPlaceCabinetToCounter")
                if mug_location != "cabinet_1":
                    raise TrajectoryValidationError(
                        "PickPlaceCabinetToCounter requires mug_1 to start in cabinet_1."
                    )
                mug_location = "counter_1"
            elif step["tool_name"] == "CoffeeSetupMug":
                self._validate_expected_action(step, "CoffeeSetupMug")
                if mug_location != "counter_1":
                    raise TrajectoryValidationError(
                        "CoffeeSetupMug requires mug_1 to start on counter_1."
                    )
                mug_location = "coffee_machine_dispenser"
            elif step["tool_name"] == "StartCoffeeMachine":
                self._validate_expected_action(step, "StartCoffeeMachine")
                if mug_location != "coffee_machine_dispenser":
                    raise TrajectoryValidationError(
                        "StartCoffeeMachine requires mug_1 under the coffee machine dispenser."
                    )
                coffee_machine_started = True
            else:
                raise TrajectoryValidationError(
                    f"Tool {step['tool_name']} is not allowed for PrepareCoffee."
                )

        if not first_action_seen:
            raise TrajectoryValidationError(
                "Trajectory did not contain any task action steps."
            )
        if not coffee_machine_started:
            raise TrajectoryValidationError(
                "Trajectory never started the coffee machine."
            )

        signature = self.canonical_signature(
            {
                "agents": agents,
                "steps": steps,
            }
        )
        return {
            "is_valid": True,
            "checks": list(self._all_checks),
            "final_state": {
                "mug_location": mug_location,
                "coffee_machine_started": coffee_machine_started,
            },
            "signature": signature,
        }

    def canonical_signature(self, candidate: dict[str, Any]) -> str:
        normalized = {
            "agents": sorted(candidate["agents"], key=lambda agent: agent["agent_id"]),
            "steps": sorted(candidate["steps"], key=lambda step: step["step_index"]),
        }
        return stable_json_sha256(normalized)

    def _normalize_agents(self, agents_value: Any) -> list[dict[str, str]]:
        if not isinstance(agents_value, list) or len(agents_value) != 2:
            raise TrajectoryValidationError(
                "agents must be a list containing exactly two agents."
            )

        normalized_agents: list[dict[str, str]] = []
        seen_agent_ids: set[str] = set()
        for agent in agents_value:
            if not isinstance(agent, dict):
                raise TrajectoryValidationError("Each agent entry must be an object.")
            agent_id = _normalize_text(agent.get("agent_id"), "agent_id")
            if agent_id not in AGENT_IDS:
                raise TrajectoryValidationError(f"Unsupported agent_id {agent_id}.")
            if agent_id in seen_agent_ids:
                raise TrajectoryValidationError(f"Duplicate agent_id {agent_id}.")
            seen_agent_ids.add(agent_id)
            normalized_agents.append(
                {
                    "agent_id": agent_id,
                }
            )

        if seen_agent_ids != set(AGENT_IDS):
            raise TrajectoryValidationError(
                "agents must contain exactly agent_0 and agent_1."
            )
        return normalized_agents

    def _normalize_steps(self, steps_value: Any) -> list[dict[str, Any]]:
        if not isinstance(steps_value, list) or not steps_value:
            raise TrajectoryValidationError("steps must be a non-empty list.")

        normalized_steps: list[dict[str, Any]] = []
        for raw_step in steps_value:
            if not isinstance(raw_step, dict):
                raise TrajectoryValidationError("Each step must be an object.")
            if not isinstance(raw_step.get("step_index"), int):
                raise TrajectoryValidationError("step_index must be an integer.")

            reasoning = _normalize_text(
                raw_step.get("reasoning"),
                f"step[{raw_step.get('step_index')}].reasoning",
            )
            if len(reasoning) > MAX_REASONING_CHARS:
                raise TrajectoryValidationError(
                    f"Reasoning for step {raw_step['step_index']} exceeds {MAX_REASONING_CHARS} characters."
                )

            normalized_steps.append(
                {
                    "step_index": raw_step["step_index"],
                    "agent_id": _normalize_text(
                        raw_step.get("agent_id"),
                        f"step[{raw_step['step_index']}].agent_id",
                    ),
                    "tool_name": _normalize_text(
                        raw_step.get("tool_name"),
                        f"step[{raw_step['step_index']}].tool_name",
                    ),
                    "tool_args": _normalize_mapping(
                        raw_step.get("tool_args", {}),
                        f"step[{raw_step['step_index']}].tool_args",
                    ),
                    "entity_refs": _normalize_mapping(
                        raw_step.get("entity_refs", {}),
                        f"step[{raw_step['step_index']}].entity_refs",
                    ),
                    "reasoning": reasoning,
                }
            )
        return normalized_steps

    def _validate_communicate_step(self, step: dict[str, Any]) -> None:
        tool_args = step["tool_args"]
        entity_refs = step["entity_refs"]
        to_agent = tool_args.get("to_agent_id")
        message = tool_args.get("message")
        if to_agent not in AGENT_IDS or to_agent == step["agent_id"]:
            raise TrajectoryValidationError(
                "communicate requires to_agent_id to reference the other agent."
            )
        if not isinstance(message, str) or not " ".join(message.strip().split()):
            raise TrajectoryValidationError(
                "communicate requires a non-empty message in tool_args."
            )

        expected_refs = {
            "from_agent_id": step["agent_id"],
            "to_agent_id": to_agent,
        }
        if entity_refs != expected_refs:
            raise TrajectoryValidationError(
                f"communicate entity_refs must equal {expected_refs}."
            )
        if tool_args != {
            "to_agent_id": to_agent,
            "message": " ".join(message.strip().split()),
        }:
            raise TrajectoryValidationError(
                "communicate tool_args may only contain to_agent_id and message."
            )

    def _validate_expected_action(self, step: dict[str, Any], tool_name: str) -> None:
        expected = PREPARE_COFFEE_EXPECTED_ACTIONS[tool_name]
        if step["tool_args"] != expected["tool_args"]:
            raise TrajectoryValidationError(
                f"{tool_name} tool_args must equal {expected['tool_args']}."
            )
        if step["entity_refs"] != expected["entity_refs"]:
            raise TrajectoryValidationError(
                f"{tool_name} entity_refs must equal {expected['entity_refs']}."
            )


def build_prepare_coffee_trajectory_record(
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
) -> dict[str, Any]:
    normalized_candidate = candidate
    if validation.get("is_valid"):
        validator = PrepareCoffeeValidator()
        normalized_candidate = {
            "agents": validator._normalize_agents(candidate.get("agents")),
            "steps": validator._normalize_steps(candidate.get("steps")),
        }

    return {
        "trajectory_id": trajectory_id,
        "composite_task": "PrepareCoffee",
        "agents": normalized_candidate.get("agents"),
        "initial_state": PREPARE_COFFEE_INITIAL_STATE,
        "steps": normalized_candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
    }


PREPARE_COFFEE_TASK = TaskDefinition(
    composite_task="PrepareCoffee",
    response_schema=PREPARE_COFFEE_RESPONSE_SCHEMA,
    preflight_reference_candidate=PREPARE_COFFEE_PREFLIGHT_REFERENCE_CANDIDATE,
    build_prompt=build_prepare_coffee_prompt,
    build_trajectory_record=build_prepare_coffee_trajectory_record,
    validator_factory=PrepareCoffeeValidator,
)
