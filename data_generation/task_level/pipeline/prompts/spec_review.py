"""Prompt template for Phase 2.5 semantic TaskSpec review."""

from __future__ import annotations

import json
from typing import Any


SPEC_REVIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {
            "type": "boolean",
        },
        "review_summary": {
            "type": "string",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggested_fixes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["approved", "review_summary", "issues", "suggested_fixes"],
}


_REVIEW_RUBRIC = """\
You are reviewing a generated RoboCasa TaskSpec against its original source task.

Your job is semantic review, not stylistic editing. The spec has already passed
static schema/FSM checks, so only reject it for real source-to-spec mismatches.

Review it against the CURRENT TaskSpec framework, not an idealized superset.
This framework intentionally uses one concrete symbolic instantiation of a task,
and it supports only a small set of goal/precondition/effect kinds.

Supported goal kinds:
- `object_at_location`
- `object_count_at_location`
- `object_at_location_one_of`
- `machine_flag_true`
- `machine_flag_equals`
- `fixture_part_state`
- `fixture_control_state`

Supported precondition kinds:
- `object_must_remain_at_location`
- `fixture_part_state_required_for_pickup`
- `fixture_part_state_required_for_action`
- `object_location_required_for_action`

Supported effect kinds:
- `set_machine_flag_on_action`
  Optional effect guards are allowed via `required_object_locations` and
  `required_machine_values`.

Important framework semantics:
- Shared `object_at_location_one_of` goals with `exclusive: true` are
  enforced one-to-one across the shared location pool at runtime. Do not
  reject those as failing to distribute objects unless the pool itself is
  semantically wrong.
- A temporary setup condition from the source (for example "water must be
  on while rinsing counts") may be modeled as an effect guard instead of
  a final goal.
- When the schema cannot directly express something like orientation or
  continuous timing, a guarded machine-flag approximation is acceptable
  if it preserves the task's core semantics.
- Internal support-site IDs may appear only in
  `initial_state.fixtures[*].support_sites` and trajectory/tool args; the
  grounding block does not need standalone support-site symbols.
- If the source allows "any counter except X" or another family of
  equivalent fixtures, grounding one concrete valid fixture symbol from
  that family is acceptable.
- `object_by_type` plus `preferred_fixture_types` is acceptable grounding
  when it is consistent with the chosen concrete fixture instantiation.

Treat the spec as acceptable when it picks ONE consistent concrete symbolic
instantiation of source randomness, such as:
- choosing a fixed number of objects from a variable-size source task,
- assigning specific interchangeable objects to specific targets,
- choosing one valid arrangement among several equivalent ones.

Do not reject just because the source code allows more permutations than the
spec represents, as long as the spec still describes a valid instance of the
same task family and the example trajectory witnesses that instance.

Be conservative about rejecting details that the current schema does not model
directly. For example, gripper-distance checks, object tilt, continuous timing,
exact burner identity, and other low-level simulator predicates may be omitted
or approximated if the main task semantics are still captured faithfully.

Approve the spec if all of the following are true:
- The task_goal matches the source task's intended success condition.
- goal_conditions capture every independent success criterion in `_check_success`.
- task_preconditions and task_effects are semantically appropriate.
- The spec includes the task-relevant objects/fixtures and excludes distractors.
- The example_trajectory is a plausible symbolic witness for the intended task.
- grounding symbols and entity roles are materially correct.

Do NOT reject for:
- Different but equivalent wording.
- Minor naming choices when the symbolic meaning is still correct.
- Omitted distractors.
- Preflight token estimate values.
- Choosing one concrete valid assignment among interchangeable objects.
- Not modeling source-side variability beyond the chosen symbolic instantiation.
- Omitting simulator predicates that are outside the supported TaskSpec schema,
  unless doing so changes the task's core identity.
- A concrete goal choice that is narrower than the source's full set of
  valid permutations, as long as it remains a valid instance of the same
  task family.

When you reject:
- List only concrete, source-grounded issues.
- Keep issues short and specific.
- Suggested fixes must be actionable within the supported TaskSpec schema and tool vocabulary above.
- Do not invent unsupported goal/precondition/effect kinds or new tools.
- Prefer rejecting only when the issue would materially change the generated
  task data, not when the spec is merely a coarser but still faithful abstraction.
"""


def build_spec_review_prompt(
    *,
    task_name: str,
    source_python: str,
    spec_payload: dict[str, Any],
) -> str:
    """Build the review prompt for one source/spec pair."""

    spec_json = json.dumps(spec_payload, indent=2, sort_keys=True)
    return "\n".join(
        [
            _REVIEW_RUBRIC,
            "",
            f"Task: {task_name}",
            "",
            "Original source code:",
            "```python",
            source_python.rstrip(),
            "```",
            "",
            "Generated TaskSpec JSON:",
            "```json",
            spec_json,
            "```",
            "",
            "Return a JSON object with:",
            "- `approved`: true only if the spec is semantically faithful.",
            "- `review_summary`: 1-2 sentences.",
            "- `issues`: concrete source-to-spec mismatches only.",
            "- `suggested_fixes`: concrete repair suggestions aligned with the issues.",
        ]
    )
