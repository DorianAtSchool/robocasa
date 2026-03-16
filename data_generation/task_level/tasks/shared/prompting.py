"""Render shared task prompts from task metadata and symbolic state."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
    WAIT_TOOL_NAMES,
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
    if allowed_tool_names & GIVE_SPACE_TOOL_NAMES:
        prompt_rules.append(
            "Use give_space only at the fixture where that agent is already positioned, after the agents communicate that another agent is about to navigate there, so the yielding agent clears the space before the other agent arrives."
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
    execution_rules_text = "\n".join(
        f"- {rule}"
        for rule in _build_fsm_prompt_rules(
            allowed_tool_specs,
            extra_rules=extra_execution_rules,
        )
    )
    _ = non_communicate_tool_names

    def build_prompt(
        variation_key: str,
        task_instance: TaskInstance | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        prompt_initial_state = (
            deepcopy(task_instance.initial_state)
            if task_instance is not None
            else deepcopy(initial_state)
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
- If an agent is not performing an action, be sure the agent communicates what the agent is waiting for so that no agent is doing nothing.
- Both agents must cooperatively complete the task, a single agent should not do all subtasks.
- Use only the allowed tools for this task.
- Every step must be executable and symbolically valid.
- Track each agent’s current fixture after every navigation and verify that each non-navigation action matches that current fixture.
- Number steps consecutively starting at 0 with no gaps.
- The reasoning text should explain why the agent is using the tool call, referencing what happened before or what the agent plans on doing. Each reasoning text must be a single short sentence.
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

Initial agent positions:
{initial_position_text}

Initial symbolic state:
{initial_state_text}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools_text}

""".strip()
        if retry_feedback is None:
            return prompt
        return f"{prompt}\n\n{retry_feedback.strip()}"

    return build_prompt
