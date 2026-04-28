"""Prompt template for Phase 0b transferability rating.

Rates a single-agent composite task for its suitability as a 2-agent
collaborative task. The rating is not a hard filter — low-rated tasks are
still recorded so decisions are auditable.
"""

from __future__ import annotations

from typing import Any

RATING_VALUES: tuple[str, ...] = ("HIGH", "MEDIUM", "LOW")

# JSON response schema passed to the Gemini generation config so the model
# is forced to emit a structured rating instead of free-form text.
TRANSFERABILITY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "string",
            "enum": list(RATING_VALUES),
        },
        "rationale": {
            "type": "string",
        },
    },
    "required": ["rating", "rationale"],
}


_RUBRIC = """\
You are rating a RoboCasa kitchen task for how suitable it is as a 2-agent
collaborative task. Both agents share the same kitchen and can each operate
their own gripper. A good 2-agent task is one where the workload genuinely
splits between the two agents — either by running sub-actions in parallel
OR by coordinating sequentially (one agent enables an action that the other
performs). Sequencing is NOT a downgrade: dependency-driven coordination
(hold-and-act, handoffs, relays, knob-after-placement) is exactly the kind
of behavior we want to capture.

Use exactly one of these ratings:

- HIGH: The workload splits cleanly between two agents and the resulting
  trajectory is meaningfully shorter or richer than a single agent could
  produce. This includes BOTH:
    * Parallel patterns — multiple objects at different fixtures fetched
      and placed concurrently (e.g. "ingredients from pantry + fridge to
      counter", "organize items into bins").
    * Strong sequential-coordination patterns — one agent opens/holds a
      door or drawer while the other reaches in; one agent positions an
      object and the other immediately operates a dependent control (e.g.
      pan-then-knob, bread-then-toaster-lever); a handoff where one agent
      places the precondition for the next agent's action.

- MEDIUM: Two agents help, but the split is small or the bottleneck is
  noticeable. Typical shape: only two micro-actions to divide; the agents
  share a single tight workspace; or a tag-team relay where the second
  agent only acts after the first finishes a single placement (e.g.
  ordered stacking).

- LOW: The task realistically requires only one gripper from start to
  finish. Typical shape: a single tool the second agent cannot share
  (one sponge wiping a counter), continuous single-handed manipulation
  (holding an object under running water), or a single trivial action
  after a wait (turn one knob off). The second agent would simply stand
  idle. Multi-step or sequential is NOT enough on its own to be LOW —
  there must be no meaningful sub-action a second agent can take.

Be conservative on the LOW boundary in particular: if the source code
contains TWO or more discrete sub-actions (open + reach, place + flip,
pick + press, stack + stack), the task is at least MEDIUM, even if the
sub-actions must run in strict order. Reserve LOW for the truly
single-gripper cases above. Return a 1-2 sentence rationale grounded in
what the task actually does (objects, fixtures, sequencing), not generic
statements.
"""


def _format_metadata(metadata: dict[str, Any]) -> str:
    """Render the Phase 0a metadata snippet into a compact text block."""

    lines: list[str] = []
    lines.append(f"- activity: {metadata.get('activity', 'unknown')}")
    lines.append(f"- batch: {metadata.get('batch', 'unknown')}")
    lines.append(f"- has_hinged_parts: {bool(metadata.get('has_hinged_parts'))}")

    obj_configs = metadata.get("obj_configs") or []
    if obj_configs:
        lines.append("- objects:")
        for cfg in obj_configs:
            name = cfg.get("name", "?")
            groups = cfg.get("obj_groups")
            fixture = cfg.get("placement_fixture_attr")
            distr = " (distractor)" if cfg.get("is_distractor") else ""
            container = cfg.get("try_to_place_in")
            extras: list[str] = []
            if fixture:
                extras.append(f"fixture={fixture}")
            if container:
                extras.append(f"container={container}")
            extras_str = f" [{', '.join(extras)}]" if extras else ""
            lines.append(f"    * {name}: groups={groups!r}{extras_str}{distr}")

    fixture_refs = metadata.get("fixture_refs") or []
    if fixture_refs:
        lines.append("- fixture_refs:")
        for ref in fixture_refs:
            name = ref.get("name", "?")
            ftype = ref.get("fixture_type")
            hinged = " (hinged)" if ref.get("has_hinged_parts") else ""
            lines.append(f"    * {name}: type={ftype}{hinged}")

    return "\n".join(lines)


def build_transferability_prompt(
    *,
    task_name: str,
    source_code: str,
    metadata: dict[str, Any],
) -> str:
    """Builds the full prompt sent to the LLM for one task.

    The prompt includes the rubric, Phase 0a metadata (objects, fixtures,
    batch), and the full task source so the model can reason about the
    actual task behavior rather than just its signature.
    """

    metadata_block = _format_metadata(metadata)
    return "\n".join(
        [
            _RUBRIC,
            "",
            f"Task: {task_name}",
            "",
            "Extracted metadata:",
            metadata_block,
            "",
            "Task source code:",
            "```python",
            source_code.rstrip(),
            "```",
            "",
            "Return a JSON object with fields `rating` and `rationale`.",
        ]
    )
