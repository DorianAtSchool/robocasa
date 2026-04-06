"""Prompt construction helpers for task-level VLM fine-tuning."""

from __future__ import annotations

from typing import Any, Iterable

from training.bc_task_vlm.schema_utils import compact_json_dumps

SYSTEM_PROMPT = (
    "You are a robot task planner. Predict exactly one next symbolic action step "
    "as JSON. Use the images only as scene context. Do not output image paths, "
    "do not output get_image, and do not describe the images."
)


def _format_allowed_values(tool_spec: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in sorted(tool_spec):
        if not key.startswith("allowed_"):
            continue
        value = tool_spec[key]
        if not isinstance(value, list) or not value:
            continue
        allowed_values = ", ".join(str(item) for item in value)
        chunks.append(f"{key}=[{allowed_values}]")
    return "; ".join(chunks)


def format_allowed_tool_block(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> str:
    """Renders task-local tool constraints for the prompt."""

    lines: list[str] = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        tool_args = ", ".join(tool_spec.get("tool_args", ())) or "no args"
        description = tool_spec.get("description", "").strip()
        line = f"- {tool_name}({tool_args}): {description}"
        allowed_values = _format_allowed_values(tool_spec)
        if allowed_values:
            line = f"{line} Constraints: {allowed_values}."
        lines.append(line)
    return "\n".join(lines)


def format_history_steps(history_steps: Iterable[dict[str, Any]]) -> str:
    """Renders prior symbolic action history compactly."""

    rendered_steps = []
    for step in history_steps:
        rendered_steps.append(
            f"- step={step['step']} agent={step['agent']} tool={step['tool']} "
            f"args={compact_json_dumps(step['args'])}"
        )
    if not rendered_steps:
        return "- none"
    return "\n".join(rendered_steps)


def build_user_prompt(
    *,
    composite_task: str,
    task_instruction: str,
    agent_id: str,
    next_step_index: int,
    observation_views: list[str],
    history_steps: list[dict[str, Any]],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> str:
    """Builds the text block that accompanies the current image observation."""

    observations_text = ", ".join(observation_views) if observation_views else "unknown"
    return (
        f"Task family: {composite_task}\n"
        f"Task instruction: {task_instruction}\n"
        f"Current acting agent: {agent_id}\n"
        f"Next global step index: {next_step_index}\n"
        f"Observation views attached in order: {observations_text}\n\n"
        "Previous executed symbolic action history:\n"
        f"{format_history_steps(history_steps)}\n\n"
        "Available tools for this task:\n"
        f"{format_allowed_tool_block(allowed_tool_specs)}\n\n"
        "Return JSON only using this exact shape:\n"
        '{"steps":[{"step":<int>,"agent":"<agent_id>","tool":"<tool_name>",'
        '"args":{...},"reasoning":"<short text>"}]}\n\n'
        "Rules:\n"
        "- Predict exactly one next step.\n"
        "- Use symbolic IDs only, never concrete simulator IDs.\n"
        "- The agent field must match the current acting agent.\n"
        "- The tool must be one of the allowed tools listed above.\n"
        "- The args object must contain exactly the arguments required by that tool.\n"
        "- Keep reasoning short and action-focused.\n"
        "- Do not emit markdown, prose, or any text outside the JSON object."
    )


def build_messages(
    *,
    user_prompt: str,
    num_images: int,
    target_text: str | None = None,
) -> list[dict[str, Any]]:
    """Builds one chat conversation for training or generation."""

    user_content = [{"type": "image"} for _ in range(num_images)]
    user_content.append({"type": "text", "text": user_prompt})

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
    return messages
