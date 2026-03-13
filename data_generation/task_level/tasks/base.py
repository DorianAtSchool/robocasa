"""Define shared task-level validation errors, FSM validators, and task definitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Callable, Protocol, Sequence

from data_generation.utils import stable_json_sha256


# Keep the shared FSM vocabulary small so task validators stay intuitive.
NAVIGATION_TOOL_NAMES = frozenset({"navigate_to_fixture"})
ACQUIRE_TOOL_NAMES = frozenset({"pick_up_object"})
RELEASE_TOOL_NAMES = frozenset(
    {
        "place_in_receptacle",
        "place_on_object",
        "place_on_surface",
        "place_under_dispenser",
    }
)
OPEN_PART_TOOL_NAMES = frozenset({"open_hinged_part", "open_sliding_part"})
CLOSE_PART_TOOL_NAMES = frozenset({"close_hinged_part", "close_sliding_part"})
PLACE_LOCATION_ARG_NAMES = (
    "support_id",
    "receptacle_id",
    "support_object_id",
    "dispenser_id",
)


class TrajectoryValidationError(ValueError):
    """Base class for task-level validation failures."""


class ResponseFormatValidationError(TrajectoryValidationError):
    """Raised when the model output does not match the JSON contract."""


class DuplicateTrajectoryValidationError(TrajectoryValidationError):
    """Raised when a candidate duplicates an existing saved trajectory."""


class TrajectoryStructureValidationError(TrajectoryValidationError):
    """Raised when a candidate fails structural schema-like checks."""


class TaskSemanticValidationError(TrajectoryValidationError):
    """Raised when a candidate violates task semantics or FSM transitions."""


class TaskValidator(Protocol):
    """Protocol implemented by task validators used by generation runtime."""

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Validates one candidate trajectory and returns normalized metadata."""


@dataclass(frozen=True)
class PreflightTokenEstimate:
    """Stores the manual token estimate used for preflight cost projection."""

    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0


@dataclass
class AgentRuntimeState:
    """Tracks the mutable symbolic state for one agent during FSM validation."""

    location: str | None
    held_object: str | None


@dataclass
class TaskRuntimeState:
    """Tracks shared mutable symbolic state while the FSM walks each trajectory."""

    agents: dict[str, AgentRuntimeState]  # Current symbolic state for each agent.
    objects: dict[str, dict[str, Any]]  # Mutable symbolic state for movable objects.
    fixtures: dict[str, dict[str, Any]]  # Mutable symbolic state for fixtures and parts.
    machine_state: dict[str, dict[str, Any]]  # Task-local machine or appliance flags.
    communicated_agents: set[str] = field(default_factory=set)  # Agents that have coordinated so far.
    public_state: dict[str, Any] = field(default_factory=dict)  # Extra task-specific summary fields for output.


def _format_agent_id_list(agent_ids: Sequence[str]) -> str:
    """Formats agent IDs into a short prompt-facing list."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if len(agent_ids) == 1:
        return agent_ids[0]
    if len(agent_ids) == 2:
        return f"{agent_ids[0]} and {agent_ids[1]}"
    return f"{', '.join(agent_ids[:-1])}, and {agent_ids[-1]}"


def _normalize_text(value: Any, field_name: str) -> str:
    """Normalizes and validates a required string field."""

    if not isinstance(value, str):
        raise TrajectoryStructureValidationError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise TrajectoryStructureValidationError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Normalizes and validates a required object field."""

    if not isinstance(value, dict):
        raise TrajectoryStructureValidationError(f"{field_name} must be an object.")
    return dict(value)


def _append_unique_field_names(
    destination: list[str],
    field_names: Sequence[str],
) -> None:
    """Appends schema field names while preserving their first-seen order."""

    for field_name in field_names:
        if field_name not in destination:
            destination.append(field_name)


def _build_symbolic_field_schema(
    field_name: str,
    agent_ids: Sequence[str],
) -> dict[str, Any]:
    """Builds a schema property entry for one symbolic response field."""

    if field_name in {"agent_id", "from_agent_id", "to_agent_id"}:
        return {
            "type": "STRING",
            "enum": list(agent_ids),
        }
    return {"type": "STRING"}


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    """Maps a symbolic tool argument like fixture_id to its allowed_* override key."""

    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def build_task_response_schema(
    *,
    agent_ids: Sequence[str],
    allowed_tool_specs: dict[str, dict[str, Any]],
    min_steps: int = 3,
) -> dict[str, Any]:
    """Builds the shared task-level response schema from agent and tool metadata."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if not allowed_tool_specs:
        raise ValueError("allowed_tool_specs must contain at least one tool.")
    if min_steps < 1:
        raise ValueError("min_steps must be at least 1.")

    tool_arg_names: list[str] = []
    entity_ref_names: list[str] = []
    for tool_spec in allowed_tool_specs.values():
        # Preserve the tool registry order so schema rendering stays stable.
        _append_unique_field_names(tool_arg_names, tool_spec.get("tool_args", ()))
        _append_unique_field_names(
            entity_ref_names,
            tool_spec.get("entity_refs", ()),
        )

    return {
        "type": "OBJECT",
        "required": ["agents", "steps"],
        "properties": {
            "agents": {
                "type": "ARRAY",
                "minItems": len(agent_ids),
                "maxItems": len(agent_ids),
                "items": {
                    "type": "OBJECT",
                    "required": ["agent_id"],
                    "properties": {
                        "agent_id": _build_symbolic_field_schema(
                            "agent_id",
                            agent_ids,
                        ),
                    },
                },
            },
            "steps": {
                "type": "ARRAY",
                "minItems": min_steps,
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
                        "agent_id": _build_symbolic_field_schema(
                            "agent_id",
                            agent_ids,
                        ),
                        "tool_name": {
                            "type": "STRING",
                            "enum": list(allowed_tool_specs),
                        },
                        "tool_args": {
                            "type": "OBJECT",
                            "properties": {
                                field_name: _build_symbolic_field_schema(
                                    field_name,
                                    agent_ids,
                                )
                                for field_name in tool_arg_names
                            },
                        },
                        "entity_refs": {
                            "type": "OBJECT",
                            "properties": {
                                field_name: _build_symbolic_field_schema(
                                    field_name,
                                    agent_ids,
                                )
                                for field_name in entity_ref_names
                            },
                        },
                        "reasoning": {"type": "STRING"},
                    },
                },
            },
        },
    }


class FiniteStateTaskValidator:
    """Applies a legality-and-goal FSM over task-level tool calls."""

    def __init__(
        self,
        *,
        composite_task: str,
        agent_ids: Sequence[str],
        initial_state: dict[str, Any],
        allowed_tool_specs: dict[str, Any],
        checks: Sequence[str] | None = None,
        max_reasoning_chars: int = 200,
        initial_public_state: dict[str, Any] | None = None,
    ) -> None:
        """Stores task-local configuration for the FSM validator.

        Args:
            composite_task: Human-readable task name; used in validation errors.
            agent_ids: Ordered agent IDs the validator expects in the trajectory.
            initial_state: Symbolic initial world state.
            allowed_tool_specs: Task-local tool registry used to reject unsupported tools.
            checks: Validation check labels returned on successful validation.
            max_reasoning_chars: Maximum allowed character count for each reasoning string.
            initial_public_state: Extra task-specific summary fields seeded into final_state.
        """

        self.composite_task = composite_task
        self.agent_ids = tuple(agent_ids)
        self._agent_id_set = set(self.agent_ids)
        self.initial_state = deepcopy(initial_state)
        self.allowed_tool_specs = deepcopy(allowed_tool_specs)
        self.max_reasoning_chars = max_reasoning_chars
        self._all_checks = list(
            checks
            or (
                "initial_communication",
                "allowed_tools",
                "navigation_preconditions",
                "manipulation_preconditions",
                "effects",
                "final_success",
            )
        )
        self._initial_public_state = deepcopy(initial_public_state or {})

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Validates a candidate trajectory by replaying legal steps to a goal."""

        agents = self._normalize_agents(candidate.get("agents"))
        steps = self._normalize_steps(candidate.get("steps"))
        runtime_state = self._build_runtime_state(agents)
        first_action_seen = False
        goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)

        for expected_index, step in enumerate(steps):
            if step["step_index"] != expected_index:
                raise TaskSemanticValidationError(
                    f"step_index {step['step_index']} does not match expected index {expected_index}."
                )

            # Once the task goal is satisfied, any later step is extra work.
            if goal_state_satisfied:
                raise TaskSemanticValidationError(
                    f"No steps are allowed after the {self.composite_task} goal state is satisfied."
                )

            if step["tool_name"] == "communicate":
                self._validate_communicate_step(step)
                runtime_state.communicated_agents.add(step["agent_id"])
                continue

            if runtime_state.communicated_agents != self._agent_id_set:
                raise TaskSemanticValidationError(
                    "Both agents must coordinate via communication before the first task action."
                )

            if step["tool_name"] not in self.allowed_tool_specs:
                raise TaskSemanticValidationError(
                    f"Tool {step['tool_name']} is not allowed for {self.composite_task}."
                )

            self._validate_task_local_symbolic_constraints(step)
            self._validate_generic_transition(step, runtime_state)
            self.validate_task_preconditions(step, runtime_state)
            self._apply_generic_effects(step, runtime_state)
            self.apply_task_effects(step, runtime_state)
            first_action_seen = True
            goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)

        if not first_action_seen:
            raise TaskSemanticValidationError(
                "Trajectory did not contain any task action steps."
            )
        if not goal_state_satisfied:
            raise TaskSemanticValidationError(
                f"Trajectory never satisfied the {self.composite_task} goal state."
            )

        signature = self.trajectory_signature({"agents": agents, "steps": steps})
        return {
            "is_valid": True,
            "checks": list(self._all_checks),
            "final_state": self._build_final_state(runtime_state),
            # Reuse the normalized trajectory when persisting successful outputs.
            "normalized_candidate": {
                "agents": agents,
                "steps": steps,
            },
            "signature": signature,
        }

    def trajectory_signature(self, candidate: dict[str, Any]) -> str:
        """Builds a stable signature for duplicate-trajectory rejection."""

        normalized = {
            "agents": sorted(candidate["agents"], key=lambda agent: agent["agent_id"]),
            "steps": sorted(candidate["steps"], key=lambda step: step["step_index"]),
        }
        return stable_json_sha256(normalized)

    def _build_runtime_state(
        self,
        agents: list[dict[str, str]],
    ) -> TaskRuntimeState:
        """Builds the mutable runtime state snapshot used by the FSM."""

        agent_states: dict[str, AgentRuntimeState] = {}
        for agent in agents:
            initial_agent_state = self.initial_state["agents"][agent["agent_id"]]
            agent_states[agent["agent_id"]] = AgentRuntimeState(
                location=initial_agent_state.get("location"),
                held_object=initial_agent_state.get("held_object"),
            )

        return TaskRuntimeState(
            agents=agent_states,
            objects=deepcopy(self.initial_state.get("objects", {})),
            fixtures=deepcopy(self.initial_state.get("fixtures", {})),
            machine_state=deepcopy(self.initial_state.get("machine_state", {})),
            public_state=deepcopy(self._initial_public_state),
        )

    def _build_final_state(self, runtime_state: TaskRuntimeState) -> dict[str, Any]:
        """Builds the shared validation payload from the final FSM state."""

        final_state = deepcopy(runtime_state.public_state)
        final_state["agents"] = {
            agent_id: {
                "location": agent_state.location,
                "held_object": agent_state.held_object,
            }
            for agent_id, agent_state in runtime_state.agents.items()
        }
        final_state["objects"] = deepcopy(runtime_state.objects)
        final_state["fixtures"] = deepcopy(runtime_state.fixtures)
        final_state["machine_state"] = deepcopy(runtime_state.machine_state)
        return final_state

    def _normalize_agents(self, agents_value: Any) -> list[dict[str, str]]:
        """Normalizes the agent roster required by the task-level schema."""

        if not isinstance(agents_value, list) or len(agents_value) != len(self.agent_ids):
            raise TrajectoryStructureValidationError(
                f"agents must be a list containing exactly {len(self.agent_ids)} agents."
            )

        normalized_agents: list[dict[str, str]] = []
        seen_agent_ids: set[str] = set()
        for agent in agents_value:
            if not isinstance(agent, dict):
                raise TrajectoryStructureValidationError(
                    "Each agent entry must be an object."
                )
            agent_id = _normalize_text(agent.get("agent_id"), "agent_id")
            if agent_id not in self._agent_id_set:
                raise TrajectoryStructureValidationError(
                    f"Unsupported agent_id {agent_id}."
                )
            if agent_id in seen_agent_ids:
                raise TrajectoryStructureValidationError(
                    f"Duplicate agent_id {agent_id}."
                )
            seen_agent_ids.add(agent_id)
            normalized_agents.append({"agent_id": agent_id})

        if seen_agent_ids != self._agent_id_set:
            expected_agents = _format_agent_id_list(self.agent_ids)
            raise TrajectoryStructureValidationError(
                f"agents must contain exactly {expected_agents}."
            )
        return normalized_agents

    def _normalize_steps(self, steps_value: Any) -> list[dict[str, Any]]:
        """Normalizes step objects before semantic validation starts."""

        if not isinstance(steps_value, list) or not steps_value:
            raise TrajectoryStructureValidationError("steps must be a non-empty list.")

        normalized_steps: list[dict[str, Any]] = []
        for raw_step in steps_value:
            if not isinstance(raw_step, dict):
                raise TrajectoryStructureValidationError("Each step must be an object.")
            if not isinstance(raw_step.get("step_index"), int):
                raise TrajectoryStructureValidationError(
                    "step_index must be an integer."
                )

            step_index = raw_step["step_index"]
            agent_id = _normalize_text(
                raw_step.get("agent_id"),
                f"step[{step_index}].agent_id",
            )
            if agent_id not in self._agent_id_set:
                raise TrajectoryStructureValidationError(
                    f"Unsupported step[{step_index}].agent_id {agent_id}."
                )

            reasoning = _normalize_text(
                raw_step.get("reasoning"),
                f"step[{step_index}].reasoning",
            )
            if len(reasoning) > self.max_reasoning_chars:
                raise TrajectoryStructureValidationError(
                    f"Reasoning for step {step_index} exceeds {self.max_reasoning_chars} characters."
                )

            normalized_steps.append(
                {
                    "step_index": step_index,
                    "agent_id": agent_id,
                    "tool_name": _normalize_text(
                        raw_step.get("tool_name"),
                        f"step[{step_index}].tool_name",
                    ),
                    "tool_args": _normalize_mapping(
                        raw_step.get("tool_args", {}),
                        f"step[{step_index}].tool_args",
                    ),
                    "entity_refs": _normalize_mapping(
                        raw_step.get("entity_refs", {}),
                        f"step[{step_index}].entity_refs",
                    ),
                    "reasoning": reasoning,
                }
            )
        return normalized_steps

    def _validate_communicate_step(self, step: dict[str, Any]) -> None:
        """Validates the shared synthetic communication tool."""

        tool_args = step["tool_args"]
        entity_refs = step["entity_refs"]
        to_agent = tool_args.get("to_agent_id")
        message = tool_args.get("message")
        if to_agent not in self._agent_id_set or to_agent == step["agent_id"]:
            raise TaskSemanticValidationError(
                "communicate requires to_agent_id to reference the other agent."
            )
        if not isinstance(message, str) or not " ".join(message.strip().split()):
            raise TaskSemanticValidationError(
                "communicate requires a non-empty message in tool_args."
            )

        normalized_message = " ".join(message.strip().split())
        expected_refs = {
            "from_agent_id": step["agent_id"],
            "to_agent_id": to_agent,
        }
        if entity_refs != expected_refs:
            raise TaskSemanticValidationError(
                f"communicate entity_refs must equal {expected_refs}."
            )
        if tool_args != {
            "to_agent_id": to_agent,
            "message": normalized_message,
        }:
            raise TaskSemanticValidationError(
                "communicate tool_args may only contain to_agent_id and message."
            )

    def _validate_generic_transition(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Enforces the shared manipulation and navigation rules for every task."""

        agent_state = runtime_state.agents[step["agent_id"]]
        tool_name = step["tool_name"]
        tool_args = step["tool_args"]
        required_fixture = self.resolve_required_fixture(step, runtime_state)

        if tool_name in ACQUIRE_TOOL_NAMES:
            if agent_state.held_object is not None:
                raise TaskSemanticValidationError(
                    f"{step['agent_id']} cannot pick up a second object while already holding {agent_state.held_object}."
                )
            if required_fixture is not None:
                self._require_agent_location(
                    step["agent_id"],
                    agent_state.location,
                    required_fixture,
                )
            object_id = tool_args["object_id"]
            source_id = tool_args["source_id"]
            object_location = runtime_state.objects.get(object_id, {}).get("location")
            if object_location != source_id:
                raise TaskSemanticValidationError(
                    f"pick_up_object requires {object_id} to start at {source_id}."
                )
            return

        # Once an agent is holding something, the only shared safe actions are
        # moving to the next fixture or placing that same object down.
        if (
            agent_state.held_object is not None
            and tool_name not in NAVIGATION_TOOL_NAMES
            and tool_name not in RELEASE_TOOL_NAMES
        ):
            raise TaskSemanticValidationError(
                f"{step['agent_id']} must place {agent_state.held_object} before using {tool_name}."
            )

        if tool_name in RELEASE_TOOL_NAMES:
            if required_fixture is not None:
                self._require_agent_location(
                    step["agent_id"],
                    agent_state.location,
                    required_fixture,
                )
            self._require_held_object(
                step["agent_id"],
                agent_state,
                tool_args["object_id"],
            )
            return

        if tool_name not in NAVIGATION_TOOL_NAMES and required_fixture is not None:
            self._require_agent_location(
                step["agent_id"],
                agent_state.location,
                required_fixture,
            )

    def _validate_task_local_symbolic_constraints(self, step: dict[str, Any]) -> None:
        """Rejects symbolic IDs that violate the task-local allowed_* tool overrides."""

        tool_spec = self.allowed_tool_specs[step["tool_name"]]
        tool_args = step["tool_args"]

        for arg_name in tool_spec.get("tool_args", ()):
            allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
            if allowed_ids_key is None or allowed_ids_key not in tool_spec:
                continue

            allowed_ids = tool_spec[allowed_ids_key]
            if not isinstance(allowed_ids, list) or not all(
                isinstance(allowed_id, str) for allowed_id in allowed_ids
            ):
                raise ValueError(
                    f"{self.composite_task} configured {step['tool_name']}.{allowed_ids_key} "
                    "with a non-string list."
                )

            arg_value = tool_args.get(arg_name)
            if arg_value not in allowed_ids:
                raise TaskSemanticValidationError(
                    f"{step['tool_name']} requires {arg_name} to be one of "
                    f"{allowed_ids}, got {arg_value!r}."
                )

    def _apply_generic_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Applies the shared symbolic state changes for the current step."""

        agent_state = runtime_state.agents[step["agent_id"]]
        tool_name = step["tool_name"]
        tool_args = step["tool_args"]

        if tool_name in NAVIGATION_TOOL_NAMES:
            agent_state.location = tool_args["fixture_id"]
            return

        if tool_name in OPEN_PART_TOOL_NAMES:
            self._set_part_state(
                runtime_state=runtime_state,
                target_id=tool_args["target_id"],
                part_id=tool_args["part_id"],
                part_state="open",
            )
            return

        if tool_name in CLOSE_PART_TOOL_NAMES:
            self._set_part_state(
                runtime_state=runtime_state,
                target_id=tool_args["target_id"],
                part_id=tool_args["part_id"],
                part_state="closed",
            )
            return

        if tool_name in ACQUIRE_TOOL_NAMES:
            object_id = tool_args["object_id"]
            agent_state.held_object = object_id
            runtime_state.objects.setdefault(object_id, {})["location"] = (
                f"held_by_{step['agent_id']}"
            )
            return

        if tool_name in RELEASE_TOOL_NAMES:
            object_id = tool_args["object_id"]
            agent_state.held_object = None
            runtime_state.objects.setdefault(object_id, {})["location"] = (
                self._resolve_release_location(tool_args)
            )

    def _resolve_release_location(self, tool_args: dict[str, Any]) -> str:
        """Resolves where a released object should live after placement."""

        for arg_name in PLACE_LOCATION_ARG_NAMES:
            location_id = tool_args.get(arg_name)
            if isinstance(location_id, str):
                return location_id
        raise TaskSemanticValidationError(
            "Placement tools must include a symbolic destination."
        )

    def _set_part_state(
        self,
        *,
        runtime_state: TaskRuntimeState,
        target_id: str,
        part_id: str,
        part_state: str,
    ) -> None:
        """Updates the symbolic open or closed state for a fixture part."""

        target_state = runtime_state.fixtures.setdefault(target_id, {})
        parts_state = target_state.setdefault("parts", {})
        part_entry = parts_state.setdefault(part_id, {})
        part_entry["state"] = part_state

    def _require_agent_location(
        self,
        agent_id: str,
        current_location: str | None,
        expected_location: str,
    ) -> None:
        """Checks that an agent navigated to the fixture before interacting there."""

        if current_location != expected_location:
            raise TaskSemanticValidationError(
                f"{agent_id} must navigate to {expected_location} before interacting there."
            )

    def _require_held_object(
        self,
        agent_id: str,
        agent_state: AgentRuntimeState,
        object_id: str,
    ) -> None:
        """Checks that the acting agent is holding the required object."""

        if agent_state.held_object != object_id:
            raise TaskSemanticValidationError(
                f"{agent_id} must be holding {object_id} before placing it."
            )

    def resolve_required_fixture(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Resolves which fixture an agent must already be at for a step."""

        tool_args = step["tool_args"]
        for arg_name in ("target_id", "source_id", "support_id", "receptacle_id"):
            fixture_id = tool_args.get(arg_name)
            if isinstance(fixture_id, str):
                return fixture_id

        support_object_id = tool_args.get("support_object_id")
        if isinstance(support_object_id, str):
            support_location = runtime_state.objects.get(support_object_id, {}).get(
                "location"
            )
            if isinstance(support_location, str):
                return support_location
        return None

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Allows task validators to add small task-local preconditions."""

        _ = (step, runtime_state)

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Allows task validators to update task-local symbolic state."""

        _ = (step, runtime_state)

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks whether the replayed symbolic state satisfies the task goal."""

        raise NotImplementedError(
            "Task validators must implement is_goal_state_satisfied()."
        )


def make_task_prompt_builder(
    *,
    composite_task: str,
    task_goal: str,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, Any],
    non_communicate_tool_names: Sequence[str],
    full_tool_catalog: str,
    environment_description: str = "RoboCasa kitchen",
    agent_ids: Sequence[str] = ("agent_0", "agent_1"),
) -> Callable[[str], str]:
    """Builds a reusable task prompt function from the shared prompt template.

    Args:
        composite_task: Task name shown to the model and stored in generated data.
        task_goal: One-sentence goal description for the task.
        initial_state: Symbolic initial state presented to the model.
        allowed_tool_specs: Task-specific allowed tools and symbolic constraints.
        non_communicate_tool_names: Non-communication task tools allowed in steps.
        full_tool_catalog: Full catalog text shown for broader subatomic context.
        environment_description: Short environment phrase inserted into the prompt.
        agent_ids: Ordered agent IDs the task expects the model to simulate.

    Returns:
        A callable that renders the task prompt for a specific variation key.
    """

    agent_id_list_text = _format_agent_id_list(agent_ids)
    agent_count = len(agent_ids)
    allowed_tools_text = json.dumps(
        allowed_tool_specs,
        indent=2,
        sort_keys=True,
    )
    initial_state_text = json.dumps(
        initial_state,
        indent=2,
        sort_keys=True,
    )
    non_communicate_tool_text = ", ".join(non_communicate_tool_names)

    def build_prompt(variation_key: str) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        return f"""
You are simulating {agent_count} cooperative robot agents in a {environment_description}.
Generate a single valid multi-agent task-level trajectory for the composite task {composite_task}.

Important rules:
- Simulate both agents: {agent_id_list_text}.
- The agents must coordinate through communication tool calls before any task action.
- Use only the allowed tools for this task, even though a larger shared subatomic catalog is provided.
- Every step must be executable and symbolically valid.
- Keep reasoning short and explicit. Each step reasoning must be a single short sentence.
- Agents can pass each other freely in the kitchen, including around the island.
- Output JSON only, with no markdown.
- Make this trajectory distinct from previous attempts by following variation key: {variation_key}

Composite task:
- {composite_task}
- Goal: {task_goal}

Initial symbolic state:
{initial_state_text}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools_text}

Communication tool:
- communicate: Send a short coordination message to the other agent. Inputs: to_agent_id, message.

Full subatomic tool catalog for context:
{full_tool_catalog}

Output requirements:
- Return an object with keys: agents, steps.
- agents must contain exactly {agent_count} entries: {agent_id_list_text}.
- Each agent entry must contain only: agent_id.
- steps must be an interleaved timeline ordered by step_index starting at 0 with no gaps.
- Each step must contain: step_index, agent_id, tool_name, tool_args, entity_refs, reasoning.
- Do not include role or initial_plan anywhere in the output.
- coordination steps must use tool_name communicate and include both to_agent_id and message inside tool_args.
- Non-communicate steps must use only:
  {non_communicate_tool_text}.
- Use the exact symbolic IDs from the initial state and allowed tools.
""".strip()

    return build_prompt


@dataclass(frozen=True)
class TaskDefinition:
    """Stores the prompt, schema, and validator factory for one composite task."""

    composite_task: str
    response_schema: dict[str, Any]
    preflight_token_estimate: PreflightTokenEstimate
    build_prompt: Callable[[str], str]
    build_trajectory_record: Callable[
        [dict[str, Any], dict[str, Any], str, dict[str, Any]],
        dict[str, Any],
    ]
    validator_factory: Callable[[], TaskValidator]
