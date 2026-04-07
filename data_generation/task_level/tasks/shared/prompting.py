"""Render shared task prompts from task metadata and task state."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    INTERACTION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
)
from .types import TaskInstance, TaskPromptBuilder


def _format_agent_id_list(agent_ids: Sequence[str]) -> str:
    """Formats agent IDs into a short prompt-facing list."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if len(agent_ids) == 1:
        return agent_ids[0]
    if len(agent_ids) == 2:
        return f"{agent_ids[0]} and {agent_ids[1]}"
    return f"{', '.join(agent_ids[:-1])}, and {agent_ids[-1]}"


def _format_initial_agent_positions(
    initial_state: dict[str, Any],
    agent_ids: Sequence[str],
) -> str:
    """Formats the prompt-facing list of starting fixture positions."""

    position_lines: list[str] = []
    for agent_id in agent_ids:
        agent_state = initial_state.get("agents", {}).get(agent_id, {})
        location = agent_state.get("location")
        location_label = (
            location if isinstance(location, str) and location else "unknown"
        )
        position_lines.append(f"- {agent_id}: {location_label}")
    return "\n".join(position_lines)


def _build_fsm_prompt_rules(
    allowed_tool_specs: dict[str, dict[str, Any]],
    *,
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    extra_rules: Sequence[str] | None = None,
) -> list[str]:
    """Builds concise prompt rules that mirror the FSM validator."""

    # Keep the prompt rules derived from the same tool set the validator uses so
    # the model sees the key legality constraints up front.
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
    if allowed_tool_names & GIVE_SPACE_TOOL_NAMES:
        prompt_rules.append(
            "Use give_space only at the fixture where that agent is already positioned, after the agents communicate that another agent is about to navigate there, so the yielding agent clears the space before the other agent arrives."
        )
        prompt_rules.append(
            "After an agent executes give_space, that agent is no longer at the fixture. "
            "Before that agent can interact there again (pick up, place, open, close, or press_button), "
            "it must navigate_to_fixture first. Similarly, the arriving agent must give_space in turn "
            "before the original agent can navigate back."
        )
    if allowed_tool_names & INTERACTION_TOOL_NAMES:
        prompt_rules.append(
            "Interaction tools (press_button) require the agent to be at the target fixture. Navigate to the fixture first."
        )
    # Keep later references aligned with prior FSM effects.
    prompt_rules.append(
        "Keep object locations consistent across steps. After an object moves, later source_id and destination references must match its new location."
    )
    prompt_rules.append(
        "Stop as soon as the goal state is satisfied. Do not add extra task actions afterward."
    )

    for condition in task_preconditions or ():
        condition_kind = condition.get("kind")
        if condition_kind == "fixture_part_state_required_for_pickup":
            prompt_rules.append(
                "Open "
                f"{condition['fixture_id']}.{condition['part_id']} before using "
                f"{condition['tool']} from {condition['source_id']}."
            )
        elif condition_kind == "object_location_required_for_action":
            prompt_rules.append(
                f"Only use {condition['tool']} after {condition['object_id']} is "
                f"already at {condition['required_location']}."
            )

    for rule in extra_rules or ():
        normalized_rule = " ".join(rule.strip().split())
        if normalized_rule:
            prompt_rules.append(normalized_rule)

    deduped_rules: list[str] = []
    seen_rules: set[str] = set()
    for rule in prompt_rules:
        normalized_rule = " ".join(rule.strip().split())
        if not normalized_rule or normalized_rule in seen_rules:
            continue
        seen_rules.add(normalized_rule)
        deduped_rules.append(normalized_rule)
    return deduped_rules


def make_task_prompt_builder(
    *,
    composite_task: str,
    task_goal: str,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, Any],
    non_communicate_tool_names: Sequence[str],
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    extra_execution_rules: Sequence[str] | None = None,
    agent_ids: Sequence[str] = ("agent_0", "agent_1"),
) -> TaskPromptBuilder:
    """Builds a reusable task prompt function from the shared prompt template.

    Args:
        composite_task: Task name shown to the model and stored in generated data.
        task_goal: One-sentence goal description for the task.
        initial_state: Initial state presented to the model.
        allowed_tool_specs: Task-specific allowed tools and id constraints.
        non_communicate_tool_names: Non-communication task tools allowed in steps.
        extra_execution_rules: Optional task-specific sequencing rules enforced by validation.
        agent_ids: Ordered agent IDs the task expects the model to simulate.

    Returns:
        A callable that renders the task prompt for a specific variation key.
    """

    agent_id_list_text = _format_agent_id_list(agent_ids)
    agent_count = len(agent_ids)
    _ = non_communicate_tool_names

    def build_prompt(
        variation_key: str,
        task_instance: TaskInstance | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        # Copy the initial state per run so prompt rendering can reflect sampled
        # task instances without mutating the task definition defaults.
        prompt_initial_state = (
            deepcopy(task_instance.initial_state)
            if task_instance is not None
            else deepcopy(initial_state)
        )
        prompt_allowed_tool_specs = (
            deepcopy(task_instance.allowed_tool_specs)
            if task_instance is not None
            and task_instance.allowed_tool_specs is not None
            else deepcopy(allowed_tool_specs)
        )
        prompt_task_goal = (
            task_instance.task_goal
            if task_instance is not None and isinstance(task_instance.task_goal, str)
            else task_goal
        )
        prompt_extra_execution_rules = (
            task_instance.extra_execution_rules
            if task_instance is not None and task_instance.extra_execution_rules
            else tuple(extra_execution_rules or ())
        )
        allowed_tools_text = json.dumps(
            prompt_allowed_tool_specs,
            indent=2,
            sort_keys=True,
        )
        execution_rules_text = "\n".join(
            f"- {rule}"
            for rule in _build_fsm_prompt_rules(
                prompt_allowed_tool_specs,
                task_preconditions=task_preconditions,
                extra_rules=prompt_extra_execution_rules,
            )
        )
        initial_state_text = json.dumps(
            prompt_initial_state,
            indent=2,
            sort_keys=True,
        )
        initial_position_text = _format_initial_agent_positions(
            prompt_initial_state,
            agent_ids,
        )
        prompt = f"""
You are simulating {agent_count} cooperative robot agents in a physical kitchen environment.
Generate a single valid multi-agent task-level trajectory for the composite task {composite_task}.

Important rules:
- Simulate both agents: {agent_id_list_text}.
- Keep track of what object each agent is holding and where the agent's location is at all times.
- Keep track of all agent's locations which can only be at fixture locations. Be sure that the agent is not "teleporting" across the environment to complete tasks; the agent should navigate first via a tool call.
- If agent_A plans to navigate to a fixture where agent_B is already positioned, have the agents communicate first about that upcoming navigation, then have agent_B execute give_space(fixture_id) at that fixture before agent_A arrives so they avoid a location conflict.
- In the initial steps, the agents must coordinate through communication tool calls before any task action. Both agents must communicate during this time.
- Throughout the trajectory, both agents should actively communicate with each other to communicate intentions, plans, and needs, not just in the initial steps.
- Each communicate step sends a message to the other agent in the scene, so args.to must be the exact ID of that other agent.
- For each step, args must contain exactly the argument names required by that tool. Do not omit required args and do not invent extra arg keys.
- In args, use the exact IDs shown in the allowed tools block for this task.
- Keep args as a flat object that contains only that step's tool inputs.
- If an agent is not performing an action, be sure the agent communicates what the agent is waiting for so that no agent is doing nothing.
- Both agents must cooperatively complete the task, a single agent should not do all subtasks.
- Use only the allowed tools for this task.
- Every step must be executable and valid for the current task state.
- Track each agent’s current fixture after every navigation and verify that each non-navigation action matches that current fixture.
- Number steps consecutively starting at 0 with no gaps.
- The reasoning text should explain why the agent is using the tool call from a first-person point-of-view. Each reasoning text must be a single short sentence.
- In reasoning text and communicate.message text, refer to agents using exact IDs like agent_0 and agent_1, not Agent 0 or Agent 1.
- Agents can pass each other freely in the kitchen, including around the island.
- If an agent has no immediate legal task action because it is waiting on the other agent, use communicate to explain the dependency before the other agent proceeds.
- Make this trajectory distinct from previous attempts by following variation key: {variation_key}
- Output JSON only, with no markdown.

Simple execution rules:
{execution_rules_text}

Composite task:
- {composite_task}
- Goal: {prompt_task_goal}

Initial agent positions:
{initial_position_text}

Initial task state:
{initial_state_text}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools_text}

""".strip()
        if retry_feedback is None:
            return prompt
        return f"{prompt}\n\n{retry_feedback.strip()}"

    return build_prompt
