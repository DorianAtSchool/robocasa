"""Apply the shared symbolic finite-state validator to task trajectories."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from data_generation.utils import stable_json_sha256

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    PLACE_LOCATION_ARG_NAMES,
    RELEASE_TOOL_NAMES,
    WAIT_TOOL_NAMES,
)
from .errors import (
    CommunicationStepSemanticValidationError,
    HeldObjectSemanticValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    NavigationSemanticValidationError,
    ObjectStateSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    PlacementDestinationSemanticValidationError,
    PostGoalActionSemanticValidationError,
    TaskPreconditionSemanticValidationError,
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
    WaitDurationSemanticValidationError,
)
from .instances import build_canonical_agents
from .prompting import _format_agent_id_list
from .schema import (
    _allowed_ids_key_for_arg_name,
    _normalize_mapping,
    _normalize_text,
    _resolve_tool_arg_schema_type,
)
from .state import AgentRuntimeState, TaskRuntimeState


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
            tool_name in OBSERVATION_TOOL_NAMES for tool_name in self.allowed_tool_specs
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
                            "communicated_agents": sorted(
                                runtime_state.communicated_agents
                            ),
                            "required_agents": sorted(self._agent_id_set),
                        },
                    )

                if step[
                    "tool"
                ] not in self.allowed_tool_specs and not self._is_allowed_observation_tool(
                    step["tool"]
                ):
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
            "initial_state": self.initial_state,
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
        if not isinstance(agents_value, list) or len(agents_value) != len(
            self.agent_ids
        ):
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
                raise TrajectoryStructureValidationError(f"Duplicate agent {agent_id}.")
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
                raise TrajectoryStructureValidationError("step must be an integer.")

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
        """Requires each non-communication action to be bracketed by observation steps."""

        step = steps[step_index]
        if step["tool"] == "communicate" or step["tool"] in OBSERVATION_TOOL_NAMES:
            return

        if (
            step_index == 0
            or steps[step_index - 1]["tool"] not in OBSERVATION_TOOL_NAMES
        ):
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately preceded by an observation step.",
                details={
                    "tool": step["tool"],
                    "step": step["step"],
                    "position": "before",
                },
            )
        if (
            step_index + 1 >= len(steps)
            or steps[step_index + 1]["tool"] not in OBSERVATION_TOOL_NAMES
        ):
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately followed by an observation step.",
                details={
                    "tool": step["tool"],
                    "step": step["step"],
                    "position": "after",
                },
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
            and tool_name not in GIVE_SPACE_TOOL_NAMES
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

        if tool_name in GIVE_SPACE_TOOL_NAMES:
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
                self._require_other_agent_at_fixture(
                    step=step,
                    runtime_state=runtime_state,
                    fixture_id=required_fixture,
                )
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

    def _is_allowed_observation_tool(self, tool_name: str) -> bool:
        """Accepts post-processed observation tools when a task supports observation steps."""

        return self._requires_observation_steps and tool_name in OBSERVATION_TOOL_NAMES

    def _validate_task_local_symbolic_constraints(self, step: dict[str, Any]) -> None:
        """Rejects symbolic IDs that violate the task-local allowed_* tool overrides."""

        if self._is_allowed_observation_tool(step["tool"]) and (
            step["tool"] not in self.allowed_tool_specs
        ):
            return

        tool_spec = self.allowed_tool_specs[step["tool"]]
        tool_args = step["args"]

        for arg_name in tool_spec.get("tool_args", ()):
            arg_value = tool_args.get(arg_name)
            arg_schema_type = _resolve_tool_arg_schema_type(arg_name, tool_spec)
            if arg_schema_type == "STRING":
                if not isinstance(arg_value, str) or not " ".join(
                    arg_value.strip().split()
                ):
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
                        details={
                            "tool": step["tool"],
                            "arg_name": arg_name,
                            "arg_value": arg_value,
                        },
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

        if tool_name in GIVE_SPACE_TOOL_NAMES:
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
            runtime_state.objects.setdefault(object_id, {})[
                "location"
            ] = f"held_by_{step['agent']}"
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
            fixture_machine_state = runtime_state.machine_state.get(
                reference_fixture_id, {}
            )
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

    def _require_other_agent_at_fixture(
        self,
        *,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
        fixture_id: str,
    ) -> None:
        """Checks that give_space is only used for an occupied or coordinated shared fixture."""

        other_agents_at_fixture = sorted(
            agent_id
            for agent_id, agent_state in runtime_state.agents.items()
            if agent_id != step["agent"] and agent_state.location == fixture_id
        )
        if other_agents_at_fixture:
            return

        # Allow the yielding agent to clear a fixture before the incoming agent arrives
        # once both agents have already coordinated through communicate.
        if runtime_state.communicated_agents == self._agent_id_set:
            return

        step_number = step.get("step")
        step_prefix = (
            f"Step {step_number} ({step['tool']}): "
            if isinstance(step_number, int)
            else ""
        )
        raise TaskPreconditionSemanticValidationError(
            f"{step_prefix}{step['agent']} can use give_space at {fixture_id} only "
            "when the agents have already coordinated and the fixture needs to be cleared.",
            step=step_number if isinstance(step_number, int) else None,
            details={
                "agent": step["agent"],
                "fixture_id": fixture_id,
                "other_agent_locations": {
                    agent_id: agent_state.location
                    for agent_id, agent_state in runtime_state.agents.items()
                    if agent_id != step["agent"]
                },
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
            "fixture_id",
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
