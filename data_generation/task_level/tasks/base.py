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
        "place_next_to",
        "place_on_object",
        "place_on_surface",
        "place_under",
        "place_under_dispenser",
    }
)
OBSERVATION_TOOL_NAMES = frozenset({"get_image"})
WAIT_TOOL_NAMES = frozenset({"wait"})
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

    def __init__(
        self,
        message: str,
        *,
        step: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Stores optional step and machine-readable details for one failure."""

        super().__init__(message)
        self.step = step
        self.details = dict(details or {})

    @property
    def error_type(self) -> str:
        """Returns the concrete exception class name used for reporting."""

        return type(self).__name__

    @property
    def error_base_type(self) -> str:
        """Returns the top-level validation family for aggregation."""

        for cls in type(self).mro():
            if TrajectoryValidationError in cls.__bases__:
                return cls.__name__
        return self.error_type

    def with_step(self, step: int | None) -> TrajectoryValidationError:
        """Clones the exception with a step number when one is missing."""

        if self.step is not None or step is None:
            return self
        return type(self)(str(self), step=step, details=self.details)


class ResponseFormatValidationError(TrajectoryValidationError):
    """Raised when the model output does not match the JSON contract."""


class DuplicateTrajectoryValidationError(TrajectoryValidationError):
    """Raised when a candidate duplicates an existing saved trajectory."""


class TrajectoryStructureValidationError(TrajectoryValidationError):
    """Raised when a candidate fails structural schema-like checks."""


class TaskSemanticValidationError(TrajectoryValidationError):
    """Raised when a candidate violates task semantics or FSM transitions."""


class UnexpectedStepIndexSemanticValidationError(TaskSemanticValidationError):
    """Raised when step numbering diverges from the replay order."""


class PostGoalActionSemanticValidationError(TaskSemanticValidationError):
    """Raised when a non-observation action appears after the goal is reached."""


class MissingInitialCommunicationSemanticValidationError(TaskSemanticValidationError):
    """Raised when agents skip required coordination before acting."""


class UnsupportedToolSemanticValidationError(TaskSemanticValidationError):
    """Raised when a trajectory uses a tool outside the task tool registry."""


class MissingTaskActionSemanticValidationError(TaskSemanticValidationError):
    """Raised when a trajectory never performs a real task action."""


class UnsatisfiedGoalSemanticValidationError(TaskSemanticValidationError):
    """Raised when replay ends before the task goal is satisfied."""


class CommunicationStepSemanticValidationError(TaskSemanticValidationError):
    """Raised when a communicate step uses invalid symbolic arguments."""


class ObservationSequenceSemanticValidationError(TaskSemanticValidationError):
    """Raised when required observation bracketing is missing."""


class ObjectStateSemanticValidationError(TaskSemanticValidationError):
    """Raised when an object's symbolic state conflicts with the action."""


class HeldObjectSemanticValidationError(TaskSemanticValidationError):
    """Raised when held-object constraints are violated."""


class NavigationSemanticValidationError(TaskSemanticValidationError):
    """Raised when an agent acts at the wrong symbolic location."""


class ToolArgumentSemanticValidationError(TaskSemanticValidationError):
    """Raised when a tool argument is missing, malformed, or disallowed."""


class WaitDurationSemanticValidationError(ToolArgumentSemanticValidationError):
    """Raised when wait.seconds is not a positive integer."""


class PlacementDestinationSemanticValidationError(TaskSemanticValidationError):
    """Raised when a placement tool omits its symbolic destination."""


class TaskPreconditionSemanticValidationError(TaskSemanticValidationError):
    """Raised when task-local symbolic preconditions are not met."""


class TaskValidator(Protocol):
    """Protocol implemented by task validators used by generation runtime."""

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Validates one candidate trajectory and returns normalized metadata."""


class TaskPromptBuilder(Protocol):
    """Protocol implemented by task prompt builders used by generation runtime."""

    def __call__(
        self,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds one task prompt, optionally including retry feedback."""


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
    schema_type: str = "STRING",
) -> dict[str, Any]:
    """Builds a schema property entry for one symbolic response field."""

    if field_name in {"agent", "agent_id", "to"}:
        return {
            "type": "STRING",
            "enum": list(agent_ids),
        }
    return {"type": schema_type}


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    """Maps a symbolic tool argument like fixture_id to its allowed_* override key."""

    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _resolve_tool_arg_schema_type(
    field_name: str,
    tool_spec: dict[str, Any],
) -> str:
    """Resolves the shared response-schema type for one tool argument."""

    tool_arg_types = tool_spec.get("tool_arg_types", {})
    if not isinstance(tool_arg_types, dict):
        raise ValueError("tool_arg_types must be a mapping when provided.")
    schema_type = tool_arg_types.get(field_name, "STRING")
    if schema_type not in {"STRING", "INTEGER"}:
        raise ValueError(f"Unsupported schema type {schema_type!r} for {field_name}.")
    return schema_type


def _build_fsm_prompt_rules(
    allowed_tool_specs: dict[str, dict[str, Any]],
    *,
    extra_rules: Sequence[str] | None = None,
) -> list[str]:
    """Builds concise prompt rules that mirror the FSM validator."""

    allowed_tool_names = set(allowed_tool_specs)
    prompt_rules: list[str] = []

    if "communicate" in allowed_tool_names:
        prompt_rules.append(
            "Before the first task action, both agents must communicate at least once."
        )
    if "navigate_to_fixture" in allowed_tool_names:
        prompt_rules.append(
            "Before interacting with a fixture, surface, receptacle, or dispenser, first navigate to that place."
        )
    if allowed_tool_names & (OPEN_PART_TOOL_NAMES | CLOSE_PART_TOOL_NAMES):
        prompt_rules.append(
            "Only open or close a fixture part after navigating to that same fixture."
        )
    if allowed_tool_names & ACQUIRE_TOOL_NAMES:
        prompt_rules.append(
            "Only use pick_up_object when the object is still at the listed source_id, and never pick up a second object while already holding one."
        )
        prompt_rules.append(
            "After picking up an object, that agent should only navigate or place that same object until it is no longer holding anything."
        )
    if allowed_tool_names & RELEASE_TOOL_NAMES:
        prompt_rules.append(
            "Only use a placement tool for the exact object the acting agent is currently holding."
        )
    if allowed_tool_names & WAIT_TOOL_NAMES:
        prompt_rules.append(
            "Use wait only to pause in place when a delay is necessary, and provide a positive integer number of seconds."
        )
    # Keep later symbolic references aligned with prior FSM effects.
    prompt_rules.append(
        "Keep object locations consistent across steps. After an object moves, later source_id and destination references must match its new symbolic location."
    )
    prompt_rules.append(
        "Stop as soon as the goal state is satisfied. Do not add extra task actions afterward."
    )

    for rule in extra_rules or ():
        normalized_rule = " ".join(rule.strip().split())
        if normalized_rule:
            prompt_rules.append(normalized_rule)
    return prompt_rules


def build_canonical_agents(agent_ids: Sequence[str]) -> list[dict[str, str]]:
    """Builds the shared persisted agent roster for trajectories."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    return [{"agent": agent_id} for agent_id in agent_ids]


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
    tool_arg_schema_types: dict[str, str] = {}
    for tool_spec in allowed_tool_specs.values():
        # Preserve the tool registry order so schema rendering stays stable.
        for field_name in tool_spec.get("tool_args", ()):
            if field_name not in tool_arg_schema_types:
                tool_arg_schema_types[field_name] = _resolve_tool_arg_schema_type(
                    field_name,
                    tool_spec,
                )
            elif tool_arg_schema_types[field_name] != _resolve_tool_arg_schema_type(
                field_name,
                tool_spec,
            ):
                raise ValueError(
                    f"Conflicting schema types were configured for tool arg {field_name}."
                )
        _append_unique_field_names(tool_arg_names, tool_spec.get("tool_args", ()))

    return {
        "type": "OBJECT",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "ARRAY",
                "minItems": min_steps,
                "items": {
                    "type": "OBJECT",
                    "required": [
                        "step",
                        "agent",
                        "tool",
                        "args",
                        "reasoning",
                    ],
                    "properties": {
                        "step": {"type": "INTEGER"},
                        "agent": _build_symbolic_field_schema(
                            "agent",
                            agent_ids,
                        ),
                        "tool": {
                            "type": "STRING",
                            "enum": list(allowed_tool_specs),
                        },
                        "args": {
                            "type": "OBJECT",
                            "properties": {
                                field_name: _build_symbolic_field_schema(
                                    field_name,
                                    agent_ids,
                                    tool_arg_schema_types[field_name],
                                )
                                for field_name in tool_arg_names
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
        # Some tasks ask the model to emit observation steps directly, while
        # others synthesize them later during post-processing.
        self._requires_observation_steps = any(
            tool_name in OBSERVATION_TOOL_NAMES
            for tool_name in self.allowed_tool_specs
        )
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
            try:
                if step["step"] != expected_index:
                    raise UnexpectedStepIndexSemanticValidationError(
                        f"step {step['step']} does not match expected index {expected_index}.",
                        details={
                            "actual_step": step["step"],
                            "expected_step": expected_index,
                        },
                    )

                # Preserve a final post-condition snapshot after the goal is reached,
                # but reject any later non-observation work.
                if goal_state_satisfied and step["tool"] not in OBSERVATION_TOOL_NAMES:
                    raise PostGoalActionSemanticValidationError(
                        f"No steps are allowed after the {self.composite_task} goal state is satisfied.",
                        details={
                            "tool": step["tool"],
                            "composite_task": self.composite_task,
                        },
                    )

                if step["tool"] == "communicate":
                    self._validate_communicate_step(step)
                    runtime_state.communicated_agents.add(step["agent"])
                    continue

                if runtime_state.communicated_agents != self._agent_id_set:
                    raise MissingInitialCommunicationSemanticValidationError(
                        "Both agents must coordinate via communication before the first task action.",
                        details={
                            "agent": step["agent"],
                            "tool": step["tool"],
                            "communicated_agents": sorted(runtime_state.communicated_agents),
                            "required_agents": sorted(self._agent_id_set),
                        },
                    )

                if step["tool"] not in self.allowed_tool_specs:
                    raise UnsupportedToolSemanticValidationError(
                        f"Tool {step['tool']} is not allowed for {self.composite_task}.",
                        details={
                            "tool": step["tool"],
                            "composite_task": self.composite_task,
                        },
                    )

                if self._requires_observation_steps:
                    self._validate_required_observation_sequence(steps, expected_index)
                self._validate_task_local_symbolic_constraints(step)
                self._validate_generic_transition(step, runtime_state)
                self.validate_task_preconditions(step, runtime_state)
                self._apply_generic_effects(step, runtime_state)
                self.apply_task_effects(step, runtime_state)
                first_action_seen = True
                goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)
            except TrajectoryValidationError as exc:
                raise self._validation_error_with_step(exc, step["step"]) from exc

        if not first_action_seen:
            raise MissingTaskActionSemanticValidationError(
                "Trajectory did not contain any task action steps."
            )
        if not goal_state_satisfied:
            raise UnsatisfiedGoalSemanticValidationError(
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
            "agents": sorted(candidate["agents"], key=lambda agent: agent["agent"]),
            "steps": sorted(candidate["steps"], key=lambda step: step["step"]),
        }
        return stable_json_sha256(normalized)

    def _build_runtime_state(
        self,
        agents: list[dict[str, str]],
    ) -> TaskRuntimeState:
        """Builds the mutable runtime state snapshot used by the FSM."""

        agent_states: dict[str, AgentRuntimeState] = {}
        for agent in agents:
            initial_agent_state = self.initial_state["agents"][agent["agent"]]
            agent_states[agent["agent"]] = AgentRuntimeState(
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

        if agents_value is None:
            return build_canonical_agents(self.agent_ids)
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
            agent_id = _normalize_text(agent.get("agent"), "agent")
            if agent_id not in self._agent_id_set:
                raise TrajectoryStructureValidationError(
                    f"Unsupported agent {agent_id}."
                )
            if agent_id in seen_agent_ids:
                raise TrajectoryStructureValidationError(
                    f"Duplicate agent {agent_id}."
                )
            seen_agent_ids.add(agent_id)
            normalized_agents.append({"agent": agent_id})

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
            if not isinstance(raw_step.get("step"), int):
                raise TrajectoryStructureValidationError(
                    "step must be an integer."
                )

            step_index = raw_step["step"]
            try:
                agent_id = _normalize_text(
                    raw_step.get("agent"),
                    f"step[{step_index}].agent",
                )
                if agent_id not in self._agent_id_set:
                    raise TrajectoryStructureValidationError(
                        f"Unsupported step[{step_index}].agent {agent_id}."
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
                        "step": step_index,
                        "agent": agent_id,
                        "tool": _normalize_text(
                            raw_step.get("tool"),
                            f"step[{step_index}].tool",
                        ),
                        "args": _normalize_mapping(
                            raw_step.get("args", {}),
                            f"step[{step_index}].args",
                        ),
                        "reasoning": reasoning,
                    }
                )
            except TrajectoryValidationError as exc:
                raise self._validation_error_with_step(exc, step_index) from exc
        return normalized_steps

    def _validation_error_with_step(
        self,
        exc: TrajectoryValidationError,
        step: int | None,
    ) -> TrajectoryValidationError:
        """Attaches the first failing step number to a validation exception."""

        if exc.step is not None:
            return exc
        return exc.with_step(step)

    def _validate_communicate_step(self, step: dict[str, Any]) -> None:
        """Validates the shared synthetic communication tool."""

        tool_args = step["args"]
        to_agent = tool_args.get("to")
        message = tool_args.get("message")
        if to_agent not in self._agent_id_set or to_agent == step["agent"]:
            raise CommunicationStepSemanticValidationError(
                "communicate requires to to reference the other agent.",
                details={"agent": step["agent"], "to": to_agent},
            )
        if not isinstance(message, str) or not " ".join(message.strip().split()):
            raise CommunicationStepSemanticValidationError(
                "communicate requires a non-empty message in args.",
                details={"agent": step["agent"], "to": to_agent},
            )

        if set(tool_args) != {"to", "message"}:
            raise CommunicationStepSemanticValidationError(
                "communicate args may only contain to and message.",
                details={"arg_names": sorted(tool_args)},
            )
        normalized_message = " ".join(message.strip().split())
        step["args"] = {
            "to": to_agent,
            "message": normalized_message,
        }

    def _validate_required_observation_sequence(
        self,
        steps: Sequence[dict[str, Any]],
        step_index: int,
    ) -> None:
        """Requires each non-communication action to be bracketed by get_image."""

        step = steps[step_index]
        if step["tool"] == "communicate" or step["tool"] in OBSERVATION_TOOL_NAMES:
            return

        if step_index == 0 or steps[step_index - 1]["tool"] not in OBSERVATION_TOOL_NAMES:
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately preceded by get_image.",
                details={"tool": step["tool"], "step": step["step"], "position": "before"},
            )
        if step_index + 1 >= len(steps) or steps[step_index + 1]["tool"] not in OBSERVATION_TOOL_NAMES:
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately followed by get_image.",
                details={"tool": step["tool"], "step": step["step"], "position": "after"},
            )

    def _validate_generic_transition(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Enforces the shared manipulation and navigation rules for every task."""

        agent_state = runtime_state.agents[step["agent"]]
        tool_name = step["tool"]
        tool_args = step["args"]
        required_fixture = self.resolve_required_fixture(step, runtime_state)

        if tool_name in ACQUIRE_TOOL_NAMES:
            if agent_state.held_object is not None:
                raise HeldObjectSemanticValidationError(
                    f"{step['agent']} cannot pick up a second object while already holding {agent_state.held_object}.",
                    details={
                        "agent": step["agent"],
                        "held_object": agent_state.held_object,
                        "tool": tool_name,
                    },
                )
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
            object_id = tool_args["object_id"]
            source_id = tool_args["source_id"]
            object_location = runtime_state.objects.get(object_id, {}).get("location")
            if object_location != source_id:
                raise ObjectStateSemanticValidationError(
                    f"pick_up_object requires {object_id} to start at {source_id}.",
                    details={
                        "object_id": object_id,
                        "expected_location": source_id,
                        "actual_location": object_location,
                    },
                )
            return

        # Once an agent is holding something, the only shared safe actions are
        # moving to the next fixture or placing that same object down.
        if (
            agent_state.held_object is not None
            and tool_name not in NAVIGATION_TOOL_NAMES
            and tool_name not in RELEASE_TOOL_NAMES
            and tool_name not in OBSERVATION_TOOL_NAMES
            and tool_name not in WAIT_TOOL_NAMES
        ):
            raise HeldObjectSemanticValidationError(
                f"{step['agent']} must place {agent_state.held_object} before using {tool_name}.",
                details={
                    "agent": step["agent"],
                    "held_object": agent_state.held_object,
                    "tool": tool_name,
                },
            )

        if tool_name in OBSERVATION_TOOL_NAMES or tool_name in WAIT_TOOL_NAMES:
            return

        if tool_name in RELEASE_TOOL_NAMES:
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
            self._require_held_object(
                step["agent"],
                agent_state,
                tool_args["object_id"],
            )
            return

        if tool_name not in NAVIGATION_TOOL_NAMES and required_fixture is not None:
            self._require_agent_location(
                step=step,
                current_location=agent_state.location,
                expected_location=required_fixture,
            )

    def _validate_task_local_symbolic_constraints(self, step: dict[str, Any]) -> None:
        """Rejects symbolic IDs that violate the task-local allowed_* tool overrides."""

        tool_spec = self.allowed_tool_specs[step["tool"]]
        tool_args = step["args"]

        for arg_name in tool_spec.get("tool_args", ()):
            arg_value = tool_args.get(arg_name)
            arg_schema_type = _resolve_tool_arg_schema_type(arg_name, tool_spec)
            if arg_schema_type == "STRING":
                if not isinstance(arg_value, str) or not " ".join(arg_value.strip().split()):
                    raise ToolArgumentSemanticValidationError(
                        f"{step['tool']} requires {arg_name} to be a non-empty string.",
                        details={
                            "tool": step["tool"],
                            "arg_name": arg_name,
                            "arg_value": arg_value,
                        },
                    )
                tool_args[arg_name] = " ".join(arg_value.strip().split())
            elif arg_schema_type == "INTEGER":
                if not isinstance(arg_value, int) or isinstance(arg_value, bool):
                    raise ToolArgumentSemanticValidationError(
                        f"{step['tool']} requires {arg_name} to be an integer.",
                        details={
                            "tool": step["tool"],
                            "arg_name": arg_name,
                            "arg_value": arg_value,
                        },
                    )
                if step["tool"] == "wait" and arg_name == "seconds" and arg_value < 1:
                    raise WaitDurationSemanticValidationError(
                        "wait requires seconds to be a positive integer.",
                        details={"tool": step["tool"], "arg_name": arg_name, "arg_value": arg_value},
                    )
            allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
            if allowed_ids_key is None or allowed_ids_key not in tool_spec:
                continue

            allowed_ids = tool_spec[allowed_ids_key]
            if not isinstance(allowed_ids, list) or not all(
                isinstance(allowed_id, str) for allowed_id in allowed_ids
            ):
                raise ValueError(
                    f"{self.composite_task} configured {step['tool']}.{allowed_ids_key} "
                    "with a non-string list."
                )

            if tool_args[arg_name] not in allowed_ids:
                raise ToolArgumentSemanticValidationError(
                    f"{step['tool']} requires {arg_name} to be one of "
                    f"{allowed_ids}, got {tool_args[arg_name]!r}.",
                    details={
                        "tool": step["tool"],
                        "arg_name": arg_name,
                        "arg_value": tool_args[arg_name],
                        "allowed_values": list(allowed_ids),
                    },
                )

    def _apply_generic_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Applies the shared symbolic state changes for the current step."""

        agent_state = runtime_state.agents[step["agent"]]
        tool_name = step["tool"]
        tool_args = step["args"]

        if tool_name in NAVIGATION_TOOL_NAMES:
            agent_state.location = tool_args["fixture_id"]
            return

        if tool_name in OBSERVATION_TOOL_NAMES:
            return

        if tool_name in WAIT_TOOL_NAMES:
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
                f"held_by_{step['agent']}"
            )
            return

        if tool_name in RELEASE_TOOL_NAMES:
            object_id = tool_args["object_id"]
            agent_state.held_object = None
            runtime_state.objects.setdefault(object_id, {})["location"] = (
                self._resolve_release_location(step, runtime_state)
            )

    def _resolve_release_location(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str:
        """Resolves where a released object should live after placement."""

        tool_args = step["args"]
        for arg_name in PLACE_LOCATION_ARG_NAMES:
            location_id = tool_args.get(arg_name)
            if isinstance(location_id, str):
                return location_id

        reference_object_id = tool_args.get("reference_object_id")
        if isinstance(reference_object_id, str):
            reference_location = runtime_state.objects.get(reference_object_id, {}).get(
                "location"
            )
            if isinstance(reference_location, str):
                return reference_location
            raise PlacementDestinationSemanticValidationError(
                f"{step['tool']} requires {reference_object_id} to have a known symbolic location.",
                details={
                    "tool": step["tool"],
                    "reference_object_id": reference_object_id,
                    "reference_location": reference_location,
                },
            )

        reference_fixture_id = tool_args.get("reference_fixture_id")
        if isinstance(reference_fixture_id, str):
            # Prefer a fixture's symbolic dispenser output when the task state exposes one.
            fixture_machine_state = runtime_state.machine_state.get(reference_fixture_id, {})
            if isinstance(fixture_machine_state, dict):
                dispenser_id = fixture_machine_state.get("dispenser_id")
                if isinstance(dispenser_id, str):
                    return dispenser_id
            return reference_fixture_id

        raise PlacementDestinationSemanticValidationError(
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
        *,
        step: dict[str, Any],
        current_location: str | None,
        expected_location: str,
    ) -> None:
        """Checks that an agent navigated to the fixture before interacting there."""

        if current_location != expected_location:
            step_number = step.get("step")
            step_prefix = (
                f"Step {step_number} ({step['tool']}): "
                if isinstance(step_number, int)
                else ""
            )
            current_location_label = current_location or "unknown location"
            raise NavigationSemanticValidationError(
                f"{step_prefix}{step['agent']} is at {current_location_label} and must "
                f"use navigate_to_fixture to reach {expected_location} before using "
                f"{step['tool']}.",
                step=step_number if isinstance(step_number, int) else None,
                details={
                    "agent": step["agent"],
                    "tool": step["tool"],
                    "current_location": current_location,
                    "expected_location": expected_location,
                },
            )

    def _require_held_object(
        self,
        agent_id: str,
        agent_state: AgentRuntimeState,
        object_id: str,
    ) -> None:
        """Checks that the acting agent is holding the required object."""

        if agent_state.held_object != object_id:
            raise HeldObjectSemanticValidationError(
                f"{agent_id} must be holding {object_id} before placing it.",
                details={
                    "agent": agent_id,
                    "expected_object": object_id,
                    "held_object": agent_state.held_object,
                },
            )

    def resolve_required_fixture(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Resolves which fixture an agent must already be at for a step."""

        tool_args = step["args"]
        for arg_name in (
            "target_id",
            "source_id",
            "support_id",
            "receptacle_id",
            "reference_fixture_id",
        ):
            fixture_id = tool_args.get(arg_name)
            if isinstance(fixture_id, str):
                return fixture_id

        for arg_name in ("support_object_id", "reference_object_id"):
            reference_object_id = tool_args.get(arg_name)
            if not isinstance(reference_object_id, str):
                continue
            support_location = runtime_state.objects.get(reference_object_id, {}).get(
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
    extra_execution_rules: Sequence[str] | None = None,
    agent_ids: Sequence[str] = ("agent_0", "agent_1"),
) -> TaskPromptBuilder:
    """Builds a reusable task prompt function from the shared prompt template.

    Args:
        composite_task: Task name shown to the model and stored in generated data.
        task_goal: One-sentence goal description for the task.
        initial_state: Symbolic initial state presented to the model.
        allowed_tool_specs: Task-specific allowed tools and symbolic constraints.
        non_communicate_tool_names: Non-communication task tools allowed in steps.
        extra_execution_rules: Optional task-specific sequencing rules enforced by validation.
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
    execution_rules_text = "\n".join(
        f"- {rule}"
        for rule in _build_fsm_prompt_rules(
            allowed_tool_specs,
            extra_rules=extra_execution_rules,
        )
    )

    def build_prompt(
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        prompt = f"""
You are simulating {agent_count} cooperative robot agents in a physical kitchen environment.
Generate a single valid multi-agent task-level trajectory for the composite task {composite_task}.
Think about the physical constraints and limitations when building the trajectory.

Important rules:
- Simulate both agents: {agent_id_list_text}.
- In the initial steps, the agents must coordinate through communication tool calls before any task action. Both agents must communicate during this time.
- Both agents must cooperatively complete the task, a single agent should not do all subtasks.
- Use only the allowed tools for this task.
- Every step must be executable and symbolically valid.
- Track each agent’s current fixture after every navigation and verify that each non-navigation action matches that current fixture.
- Number steps consecutively starting at 0 with no gaps.
- The reasoning text should explain why the agent is using the tool call, referencing what happened before or what the agent plans on doing. Keep reasoning text short and explicit. Each reasoning text must be a single short sentence.
- In reasoning text and communicate.message text, refer to agents using exact IDs like agent_0 and agent_1, not Agent 0 or Agent 1.
- Agents can pass each other freely in the kitchen, including around the island.
- If an agent has no immediate legal task action because it is waiting on the other agent, emit wait(seconds) instead of skipping that agent.
- Make this trajectory distinct from previous attempts by following variation key: {variation_key}
- Output JSON only, with no markdown.

Simple execution rules:
{execution_rules_text}

Composite task:
- {composite_task}
- Goal: {task_goal}

Initial symbolic state:
{initial_state_text}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools_text}

""".strip()
        if retry_feedback is None:
            return prompt
        return f"{prompt}\n\n{retry_feedback.strip()}"

    return build_prompt


@dataclass(frozen=True)
class TaskDefinition:
    """Stores the prompt, schema, and validator factory for one composite task."""

    composite_task: str
    response_schema: dict[str, Any]
    preflight_token_estimate: PreflightTokenEstimate
    build_prompt: TaskPromptBuilder
    build_trajectory_record: Callable[
        [dict[str, Any], dict[str, Any], str, dict[str, Any]],
        dict[str, Any],
    ]
    validator_factory: Callable[[], TaskValidator]
