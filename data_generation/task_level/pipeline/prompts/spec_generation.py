"""Prompt template for Phase 1 TaskSpec generation.

Given a RoboCasa composite task source file, ask the LLM to emit a complete
TaskSpec JSON matching the schema in ``data_generation.task_level.tasks.specs``.
The prompt carries few-shot examples (source + spec pairs), a canonical tool
catalog, and an explicit field-by-field schema description. Phase 1 later
rebuilds the shared tool metadata programmatically, so the prompt only asks
the model for task-local tool constraints and the rest of the TaskSpec.
"""

from __future__ import annotations

import json
from typing import Any

from data_generation.task_level.subatomic_tool_specs import TASK_LEVEL_ALLOWED_TOOL_SPECS

from ..few_shot import FewShotExample

# Response schema passed to the Gemini generation config.
#
# Gemini's structured-output mode requires every `object` node to enumerate
# its sub-properties up front — it cannot express a dictionary with dynamic
# string keys. TaskSpec uses dynamic-keyed objects heavily
# (`initial_state.objects`, `allowed_tool_specs`, `grounding.symbols`,
# etc.) so there is no schema we can pass that would both (a) leave those
# maps free and (b) force Gemini to emit content for them.
#
# We therefore disable structured output for Phase 1 by passing `None`.
# The client still sets `response_mime_type=application/json` so the
# model returns raw JSON text, and Phase 2 validates the shape.
SPEC_GENERATION_RESPONSE_SCHEMA: dict[str, Any] | None = None


_SHARED_TOOL_METADATA_KEYS = frozenset(
    {"description", "tool_args", "tool_arg_types", "tool_arg_any_of"}
)


def _render_tool_arg_signature(tool_spec: dict[str, Any]) -> str:
    """Render canonical arg requirements, including alternative arg groups."""

    parts: list[str] = [
        f"`{arg_name}`" for arg_name in tool_spec.get("tool_args", ()) if isinstance(arg_name, str)
    ]
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        normalized_group = [
            f"`{arg_name}`" for arg_name in arg_group if isinstance(arg_name, str)
        ]
        if normalized_group:
            parts.append(f"exactly one of ({', '.join(normalized_group)})")
    return ", ".join(parts) or "(none)"


def _build_canonical_tool_catalog() -> str:
    """Render the exact runtime tool names and argument signatures."""

    lines: list[str] = []
    for tool_name, tool_spec in TASK_LEVEL_ALLOWED_TOOL_SPECS.items():
        arg_text = _render_tool_arg_signature(tool_spec)
        lines.append(
            f"- `{tool_name}`: {tool_spec['description']} Exact args: {arg_text}."
        )
    return "\n".join(lines)


_CANONICAL_TOOL_CATALOG = _build_canonical_tool_catalog()


_SCHEMA_DESCRIPTION = """\
Every TaskSpec JSON must contain these top-level fields:

- `spec_version` (int): always 1.
- `composite_task` (string): the Python class name of the task.
- `source_python_module` (string): dotted Python module path to the source
  file that defines the task class.
- `agent_ids` (list[string]): always ["agent_0", "agent_1"].
- `max_reasoning_chars` (int): always 200.
- `validator_checks` (list[string]): always the full list
  ["initial_communication", "allowed_tools", "navigation_preconditions",
  "manipulation_preconditions", "effects", "final_success"].
- `preflight_token_estimate` (object): {prompt_tokens: 3000, output_tokens:
  3500, reasoning_tokens: 0}. Tune these modestly (2500–4500) based on the
  task's complexity, but these defaults are fine for most tasks.
- `initial_state` (object): the symbolic starting state. Contains:
    * `agents`: {agent_0: {location, held_object: null}, agent_1: {...}}.
      Both agents usually start at the same fixture (the primary workspace).
    * `objects`: {id: {object_type, location}}. The `location` is another
      object id (stacked inside) or a fixture id (sitting on it).
    * `fixtures`: {id: {fixture_type, parts?, controls?, support_sites?}}.
      Only include fixtures that appear in the trajectory. `parts` entries
      describe hinged_part or sliding_part with a canonical `state` of only
      "open" or "closed". `controls` entries describe buttons / levers /
      rotary controls with their current symbolic `state`. `support_sites`
      names sub-locations on a fixture that matter for placement or
      navigation semantics, such as stove burners or toaster slots.
    * `machine_state` (optional): {machine_id: {flag_name: bool, ...}}. Use
      this when the task has a spatial-proximity or state check that
      cannot be expressed by object_at_location alone.
- `allowed_tool_specs` (object): map from canonical `tool_name` to a
  task-local override object. The pipeline injects shared `description`,
  `tool_args`, and `tool_arg_types` programmatically from the canonical
  tool catalog below, so each value should contain ONLY task-specific
  constraint fields such as `allowed_object_ids`, `allowed_source_ids`,
  `allowed_receptacle_ids`, `allowed_support_object_ids`,
  `allowed_support_ids`, `allowed_target_ids`, `allowed_part_ids`,
  `allowed_control_ids`, `allowed_fixture_ids`,
  `allowed_reference_object_ids`, or `allowed_reference_fixture_ids`.
  Do NOT repeat shared metadata fields. Only list the tools the
  trajectory actually needs.
  Always include `communicate` and `navigate_to_fixture`. Include
  `give_space` when two agents share a fixture. Include
  `open_hinged_part`/`close_hinged_part` when the task opens/closes a
  hinged door. Include `open_sliding_part`/`close_sliding_part` when the
  task opens/closes a drawer. For placements, choose ONE of
  `place_in_receptacle` (inside a container, e.g. tray/bowl),
  `place_on_object` (on top of a movable support like a plate/cutting
  board), `place_on_surface` (onto an exposed surface; note the arg name
  is `support_id`, not `fixture_id`), `place_next_to` (adjacent on the
  same support; uses exactly one of `reference_object_id` or
  `reference_fixture_id`), or `place_under` (under a dispenser spout or
  fixture).
  Restrict the `allowed_*_ids` lists to the exact symbolic IDs the task
  should permit — do NOT include distractors.
- `task_goal` (string): a single-sentence natural-language description of
  what success looks like, grounded in the symbolic IDs used in the spec.
- `extra_execution_rules` (list[string]): short imperative rules that
  clarify placement tool choice, fixture constraints, and what agents must
  NOT do (e.g. "Do not pick up the cupcake_container.").
- `initial_public_state` (object): flat map with `{id}_location` keys for
  every movable task-relevant object, plus any machine flag the goal
  references. Values mirror `initial_state` at t=0.
- `task_preconditions` (list): see the kinds section below.
- `goal_conditions` (list): see the kinds section below. MUST include at
  least one condition. MUST cover every independent success criterion in
  `_check_success`.
- `task_effects` (list): see the kinds section below. Only used for
  machine-flag tasks.
- `grounding` (object): with `legacy_symbol_aliases` (map of
  `<legacy>_N -> id`) and `symbols` (map of id -> {entity_type, resolver,
  role, preferred_fixture_types, ...}). Resolvers in use:
    * `object_by_type` for every object.
    * `source_fixture_for_object` for the fixture an object starts at.
    * `support_fixture_for_object` for the fixture where placements happen.
  Grounding pins each symbolic id to real sim entities when the spec is
  instantiated into a randomized scene. Additional supported fixture
  resolvers include `unique_fixture_type` for fixtures uniquely identified
  by type and `nearest_placeable_surface_to_fixture` for a nearby support
  surface next to another fixture symbol.
- `example_trajectory` (object): {agents: [{agent: agent_0}, {agent:
  agent_1}], steps: [...]}. Each step: {step, agent, tool, args, reasoning}.
  MUST start with two communicate steps (one per agent) establishing the
  plan. MUST end at the step that first satisfies `goal_conditions` — no
  follow-up steps after that, not even re-checks.
- `notes` (optional list[string]): short maintenance notes.
"""


_GOAL_KINDS = """\
Goal condition `kind` values (use exactly these, nothing else):

- `object_at_location`: {object_id, location}. True when the object's
  location exactly matches.
- `object_count_at_location`: {object_ids: [id, ...], location, count}.
  True when exactly `count` of the listed objects are currently at
  `location`. Use this for exact allocation/count requirements such as
  "exactly one chocolate in the glass" or "two yogurts on each plate".
- `object_at_location_one_of`: {object_id, locations: [id, ...],
  exclusive?: bool}. True when the object is at any one of the listed
  locations. Only set `exclusive: true` when the number of distinct
  target locations is at least equal to the number of goals sharing the
  same `locations` list (e.g. two bowls that each must land on a
  different plate). If more objects than locations share the list (e.g.
  four ice cubes split between two cups), leave `exclusive` unset or
  false — otherwise the goal is unsatisfiable. When the source requires
  exact counts per receptacle, prefer `object_count_at_location` over a
  broad shared `one_of` pool.
- `machine_flag_true`: {machine_path: [machine_id, flag_name]}. True when
  that nested value in `machine_state` is truthy. Used for
  spatial-proximity checks that `_check_success` computes in Python code.
- `machine_flag_equals`: {machine_path: [machine_id, flag_name], value}.
  True when the nested value exactly equals `value`. Use this when
  success requires a machine flag to be false or match a specific state.
- `fixture_part_state`: {fixture_id, part_id, state}. True when a fixture
  part in `initial_state.fixtures[*].parts[*]` currently has that exact
  state. Use this for goals like "drawer closed" or "door open" instead
  of inventing machine-state mirrors for fixture-part state.
- `fixture_control_state`: {fixture_id, control_id, state}. True when a
  fixture control in `initial_state.fixtures[*].controls[*]` currently
  has that exact state. Use this for knobs, handles, buttons, or levers
  instead of inventing machine-state mirrors for control state.
"""

_PRECONDITION_KINDS = """\
Task precondition `kind` values (use exactly these, nothing else):

- `object_must_remain_at_location`: {object_id, location, message}. Fails
  if the object ever leaves that location. Use for containers/plates that
  should stay put throughout the trajectory.
- `fixture_part_state_required_for_pickup`: {tool, source_id, fixture_id,
  part_id, required_state, message}. Gates a pickup on a fixture part
  state (e.g. "cabinet door must be open before pick_up_object with
  source_id=cabinet").
- `fixture_part_state_required_for_action`: {tool, fixture_id, part_id,
  required_state, message, arg_name?, arg_value?}. Gates a non-pickup
  action on a fixture part state, optionally only when the step uses a
  specific symbolic arg value (for example placing into a specific open
  drawer or rack).
- `object_location_required_for_action`: {tool, object_id,
  required_location, message, arg_name?, arg_value?}. Gates an action on
  an object's current location. Use `arg_name`/`arg_value` when the
  acted-on symbol is different from the object whose location matters
  (for example, "only pick up bowl after cup is already in bowl").
"""

_EFFECT_KINDS = """\
Task effect `kind` values (use exactly these, nothing else):

- `set_machine_flag_on_action`: {tool, args, machine_path, value}. When
  the trajectory executes `tool` with args matching every key/value in
  `args`, set `machine_path` in machine_state to `value`. When completion
  also depends on symbolic state at that moment, add optional
  `required_object_locations: [{object_id, location}, ...]` and/or
  `required_machine_values: [{machine_path, value}, ...]` guards. Pair
  each machine-flag goal with an effect that sets it under the right
  conditions.
"""

_GUIDANCE = """\
Follow these rules when writing the spec:

1. IDs should be short, snake_case, and unique. Plates/trays usually get
   semantic names (`skewer_plate`, `cupcake_container`). Multiple instances
   get numeric suffixes (`bowl1`, `bowl2`). Distractor objects defined in
   the source but unused by the trajectory MUST be excluded from the spec.

2. When the source uses `try_to_place_in=<container>` for one of the
   object cfgs, the container is a sim-created object (not a fixture).
   Add it to `initial_state.objects` with the container's `object_type`
   (e.g. "plate"), location set to the actual fixture id, and reference
   it from `preferred_fixture_types` on the contained object's grounding.

3. Pick the placement tool that matches the semantic relationship from
   the source's `_check_success`:
   - `OU.check_obj_in_receptacle(obj, container)` -> `place_in_receptacle`
     when the container is interior (tray/oven_tray/bowl/basket) or
     `place_on_object` when it is a flat movable support (plate,
     cutting_board).
   - XY-distance proximity check -> `place_next_to` AND a machine flag.
     Use `reference_object_id` for movable anchors. Use
     `reference_fixture_id` only when `machine_state[fixture_id]`
     includes `adjacent_location_id`, and make that adjacent support
     fixture explicit in `initial_state.fixtures`. Objects placed next
     to the fixture land at that `adjacent_location_id`, not at the
     fixture id itself.
   - Direct counter/fixture contact without a movable support ->
     `place_on_surface`.

4. When a goal depends on a spatial proximity check (XY distance) rather
   than a simple "in" check, introduce a machine flag via `task_effects`
   and reference it with `machine_flag_true`. See the GarnishCupcake
   example. Do not collapse a flexible adjacency task into exact count
   goals unless the source truly requires exact counts.

5. When success requires a temporal completion event after an earlier
   setup step (for example "rinse, then turn water off"), set the
   completion machine flag on the final action that actually completes
   the subtask, not on an earlier placement or setup step. If that final
   action should count only when an object is in place or a machine is
   already on, encode those requirements with effect guards
   (`required_object_locations` / `required_machine_values`) so later
   generated trajectories cannot set the flag incorrectly. If completion
   is an abstract milestone like "waiting finished" or "agent observes the
   state change", you may attach the guarded effect to an explicit
   `communicate` step that marks that milestone. Do not make exact
   `communicate.message` text the only way to satisfy a task when a
   physical action can carry the completion effect instead.

6. The example_trajectory must start with two `communicate` steps (one
   per agent). Before any non-communication action, resolve which
   fixture that step operates at from its args (`fixture_id`,
   `target_id`, `source_id`, support/receptacle ids, or a referenced
   object's current support). If that fixture is not the agent's current
   location, insert `navigate_to_fixture` immediately before the action.
   Physical proximity does not waive navigation. Include at least one
   `give_space` call when both agents need to use the same fixture area.
   After an agent uses `give_space`, that agent is no longer positioned
   at the fixture, so it must `navigate_to_fixture` before its next
   non-communication action.

7. Do NOT add extra steps after the step that first satisfies all
   goal_conditions. This is enforced by the validator: the trajectory
   must end exactly at goal satisfaction.

8. Only include fixture-part-state preconditions for fixtures that
   actually have hinged_part or sliding_part entries in
   `initial_state.fixtures`. Use
   `fixture_part_state_required_for_pickup` for pickup steps gated by an
   open part, and `fixture_part_state_required_for_action` for other
   actions gated by an open part (for example placing into a drawer or
   rack). Also add the corresponding
   `open_hinged_part`/`open_sliding_part` tool entry to
   `allowed_tool_specs`, and make the example_trajectory execute that
   open step before the first gated action.

9. `grounding.legacy_symbol_aliases` must map any legacy IDs the source
   task uses (e.g. `skewer_plate_1` -> `skewer_plate`, `cabinet_1` ->
   `cabinet`). When in doubt, add one alias per id that appends `_1`.

10. Every object id referenced in `goal_conditions`, `task_preconditions`,
   `task_effects`, and `example_trajectory` MUST appear in
   `initial_state.objects` or `initial_state.fixtures`. The static
   validator in Phase 2 will reject the spec otherwise.

11. Use only the canonical tool names listed below. Never invent wrapper
    tools like `turn_on_machine`, `turn_on_sink`, or `turn_off_knob`.
    Represent controls with the canonical tools instead:
    `press_button`, `press_lever`, or `set_rotary_control`.

12. The canonical tool arg names matter. Examples:
    `place_on_surface` uses `support_id`, not `fixture_id`;
    `press_button`/`press_lever`/`set_rotary_control` use `target_id`
    plus `control_id`; `set_rotary_control` also needs `goal`. Do not
    invent alternate arg names like `machine_id`. `place_next_to` uses
    exactly one of `reference_object_id` or `reference_fixture_id`, not
    both. `set_rotary_control.args.goal` must be a string state like
    `"on"` or `"off"`, not a numeric knob value like `1.0`.

13. Every required step arg must be a concrete, non-null symbolic value.
    Never emit `null` for `support_id`, `receptacle_id`, `control_id`,
    `part_id`, or any other required tool argument.

14. `object_id` must always refer to a movable object in
    `initial_state.objects`. Fixture ids belong in fields like
    `fixture_id`, `target_id`, `reference_fixture_id`, or `support_id`,
    never in `object_id`.

15. When the source requires exact counts across multiple receptacles,
    do not use a broad shared `object_at_location_one_of` pool unless it
    is genuinely flexible in the source. Prefer `object_count_at_location`
    for interchangeable objects or per-type subgroups. Example: if
    plate1 and plate2 must each end with one bun and one sausage, write
    bun-count and sausage-count goals per plate, not four shared `one_of`
    goals over `[plate1, plate2]`. Use concrete `object_at_location`
    only when the source distinguishes individual objects.

16. Do not invent new `kind` values. Goals may use only
    `object_at_location`, `object_count_at_location`,
    `object_at_location_one_of`, `machine_flag_true`,
    `machine_flag_equals`, `fixture_part_state`, or
    `fixture_control_state`. Preconditions may use only
    `object_must_remain_at_location`,
    `fixture_part_state_required_for_pickup`, or
    `fixture_part_state_required_for_action`, or
    `object_location_required_for_action`. Effects may use only
    `set_machine_flag_on_action`.

17. Ground primary fixtures directly when possible. If a fixture is
    uniquely identified by type in the scene, use `unique_fixture_type`
    instead of grounding it through an object that already starts on that
    same fixture. Use `support_fixture_for_object` only for the actual
    placement/support fixture of an object, not to recover the object's
    current source fixture. If the referenced object already starts on the
    fixture, use `source_fixture_for_object`, not `support_fixture_for_object`.

18. If the source distinguishes fixture sub-locations such as stove
    burners, toaster slots, sink basins, or dishwasher racks, model them
    as concrete symbolic IDs. Put attachment-style placement targets in
    `fixtures[*].support_sites` and use `place_on_surface.support_id`
    with those IDs. The agent still navigates to the parent fixture.

19. When an agent is holding any object, its next physical steps should
    only be `navigate_to_fixture` or a placement of that same object.
    Do not open or close fixture parts, and do not operate controls while
    still holding something.

20. If the source checks an appliance state like `turned_on` or
    `started`, model that with a machine flag plus a `set_machine_flag_on_action`
    effect on the activating control step, not only with a final
    `fixture_control_state` on a button or lever. Likewise, if a
    temporary state only matters while some "visited" or completion flag
    is being earned, encode that temporary requirement as an effect guard
    (`required_machine_values`) rather than as a final goal.

21. Do not create separate grounding symbols for internal support sites
    or controls. Put those symbolic IDs under
    `initial_state.fixtures[*].support_sites` / `controls` and ground the
    parent fixture itself.
"""


def _render_example_spec_json(spec_json: str) -> str:
    """Render few-shot specs with only task-local tool overrides exposed."""

    try:
        payload = json.loads(spec_json)
    except json.JSONDecodeError:
        return spec_json.rstrip()

    allowed_tool_specs = payload.get("allowed_tool_specs")
    if isinstance(allowed_tool_specs, dict):
        payload["allowed_tool_specs"] = {
            str(tool_name): (
                {
                    key: value
                    for key, value in dict(tool_spec).items()
                    if key not in _SHARED_TOOL_METADATA_KEYS
                }
                if isinstance(tool_spec, dict)
                else tool_spec
            )
            for tool_name, tool_spec in allowed_tool_specs.items()
        }
    return json.dumps(payload, indent=2)


def _format_example(index: int, example: FewShotExample) -> str:
    return "\n".join(
        [
            f"### Example {index}: {example.task_name}",
            "",
            "Source Python:",
            "```python",
            example.source_python.rstrip(),
            "```",
            "",
            "Generated TaskSpec JSON:",
            "```json",
            _render_example_spec_json(example.spec_json),
            "```",
        ]
    )


def build_spec_generation_prompt(
    *,
    task_name: str,
    source_python: str,
    source_module: str,
    metadata: dict[str, Any],
    examples: tuple[FewShotExample, ...],
    previous_spec_payload: dict[str, Any] | None = None,
    repair_feedback_lines: tuple[str, ...] = (),
) -> str:
    """Build the full Phase 1 prompt for one task."""

    example_blocks = [
        _format_example(index + 1, example) for index, example in enumerate(examples)
    ]

    metadata_lines: list[str] = []
    for cfg in metadata.get("obj_configs") or []:
        name = cfg.get("name", "?")
        groups = cfg.get("obj_groups")
        fixture = cfg.get("placement_fixture_attr")
        container = cfg.get("try_to_place_in")
        distr = " (distractor — exclude from spec)" if cfg.get("is_distractor") else ""
        extras: list[str] = []
        if fixture:
            extras.append(f"fixture={fixture}")
        if container:
            extras.append(f"try_to_place_in={container}")
        metadata_lines.append(
            f"  - {name}: groups={groups!r}"
            f"{' [' + ', '.join(extras) + ']' if extras else ''}{distr}"
        )
    objects_block = (
        "Object configs (from AST analysis):\n" + "\n".join(metadata_lines)
        if metadata_lines
        else "Object configs: (none extracted)"
    )

    fixture_lines: list[str] = []
    for ref in metadata.get("fixture_refs") or []:
        name = ref.get("name", "?")
        ftype = ref.get("fixture_type")
        hinged = " (hinged)" if ref.get("has_hinged_parts") else ""
        fixture_lines.append(f"  - {name}: type={ftype}{hinged}")
    fixtures_block = (
        "Fixture refs (from AST analysis):\n" + "\n".join(fixture_lines)
        if fixture_lines
        else "Fixture refs: (none extracted)"
    )

    repair_block: list[str] = []
    if previous_spec_payload is not None or repair_feedback_lines:
        repair_block.extend(
            [
                "## Repair context",
                "You are revising a previously generated TaskSpec.",
                "Keep valid parts unchanged, but fix every issue below and any",
                "dependent inconsistency those fixes introduce.",
            ]
        )
        if repair_feedback_lines:
            repair_block.extend(
                [
                    "",
                    "Feedback to address:",
                    *[f"- {line}" for line in repair_feedback_lines],
                ]
            )
        if previous_spec_payload is not None:
            repair_block.extend(
                [
                    "",
                    "Previous TaskSpec JSON:",
                    "```json",
                    json.dumps(previous_spec_payload, indent=2),
                    "```",
                ]
            )

    return "\n".join(
        [
            "You generate TaskSpec JSONs for 2-agent RoboCasa kitchen tasks.",
            "",
            "## Task schema",
            _SCHEMA_DESCRIPTION,
            "",
            "## Canonical tool catalog",
            _CANONICAL_TOOL_CATALOG,
            "",
            "## Supported goal condition kinds",
            _GOAL_KINDS,
            "",
            "## Supported precondition kinds",
            _PRECONDITION_KINDS,
            "",
            "## Supported effect kinds",
            _EFFECT_KINDS,
            "",
            "## Authoring guidance",
            _GUIDANCE,
            "",
            "## Few-shot examples",
            *example_blocks,
            "",
            "## Your task",
            f"Generate a TaskSpec JSON for composite task `{task_name}`.",
            f"Source module: `{source_module}`",
            "",
            objects_block,
            "",
            fixtures_block,
            *([""] + repair_block if repair_block else []),
            "",
            "Task source code:",
            "```python",
            source_python.rstrip(),
            "```",
            "",
            "Return ONLY the complete TaskSpec JSON object as raw JSON.",
            "Do not wrap it in any envelope object or in markdown fences,",
            "and do not include any commentary. Every top-level TaskSpec",
            "field documented above must be present.",
        ]
    )
