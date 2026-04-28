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
    {
        "description",
        "tool_args",
        "optional_tool_args",
        "tool_arg_types",
        "tool_arg_any_of",
    }
)
# Few-shot examples are disabled by default. They were useful for early shape
# bootstrapping, but the curated set is small and can over-anchor generation to
# stale patterns. Phase 2 is now strong enough to enforce the contract.
_FEW_SHOT_EXAMPLE_COUNT = 0
_EXAMPLE_OMIT_TOP_LEVEL_FIELDS = frozenset(
    {
        "spec_version",
        "source_python_module",
        "agent_ids",
        "max_reasoning_chars",
        "validator_checks",
        "preflight_token_estimate",
        "notes",
    }
)


def _prompt_canonical_any_of_groups(
    tool_name: str,
    tool_spec: dict[str, Any],
) -> list[tuple[str, ...]]:
    """Return generation-facing arg alternatives without legacy aliases."""

    canonical_overrides: dict[str, list[tuple[str, ...]]] = {
        "place_on_surface": [("support_id",)],
        "place_in_receptacle": [("receptacle_id",)],
        "place_on_object": [("support_object_id",)],
        "place_under": [("reference_fixture_id",)],
        "place_next_to": [("reference_object_id", "reference_fixture_id")],
    }
    if tool_name in canonical_overrides:
        return canonical_overrides[tool_name]

    groups: list[tuple[str, ...]] = []
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        normalized = tuple(arg_name for arg_name in arg_group if isinstance(arg_name, str))
        if normalized:
            groups.append(normalized)
    return groups


def _prompt_canonical_optional_args(
    tool_name: str,
    tool_spec: dict[str, Any],
) -> list[str]:
    """Return generation-facing optional args, hiding legacy aliases."""

    legacy_aliases = {"target_id", "reference_id"}
    return [
        arg_name
        for arg_name in tool_spec.get("optional_tool_args", ())
        if isinstance(arg_name, str) and arg_name not in legacy_aliases
    ]


def _render_tool_arg_signature(tool_spec: dict[str, Any]) -> str:
    """Render canonical arg requirements, including alternative arg groups."""

    parts: list[str] = [
        f"`{arg_name}`" for arg_name in tool_spec.get("tool_args", ()) if isinstance(arg_name, str)
    ]
    tool_name = str(tool_spec.get("_tool_name_for_prompt", ""))
    for arg_group in _prompt_canonical_any_of_groups(tool_name, tool_spec):
        normalized_group = [
            f"`{arg_name}`" for arg_name in arg_group if isinstance(arg_name, str)
        ]
        if normalized_group:
            parts.append(f"exactly one of ({', '.join(normalized_group)})")
    optional_args = [
        f"`{arg_name}`"
        for arg_name in _prompt_canonical_optional_args(tool_name, tool_spec)
    ]
    if optional_args:
        parts.append(f"optional: {', '.join(optional_args)}")
    return ", ".join(parts) or "(none)"


def _build_canonical_tool_catalog() -> str:
    """Render the exact runtime tool names and argument signatures."""

    lines: list[str] = []
    for tool_name, tool_spec in TASK_LEVEL_ALLOWED_TOOL_SPECS.items():
        prompt_tool_spec = dict(tool_spec)
        prompt_tool_spec["_tool_name_for_prompt"] = tool_name
        arg_text = _render_tool_arg_signature(prompt_tool_spec)
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
    * `objects`: {id: {object_type, location, preserve_pose?}}. The
      `location` is another object id (stacked inside) or a fixture id
      (sitting on it). Set `preserve_pose: true` on duplicate,
      interchangeable ingredients (e.g. two sugar cubes that start
      together, ice cubes already inside a bucket) so the simulator
      leaves their scene-generated starting poses untouched instead of
      re-sampling new positions. Only use it when the exact starting
      pose does not matter for the task semantics and the object is
      already in the correct fixture/container.
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
  board), `place_on_surface` (onto an exposed surface; use `support_id`,
  plus optional `target_site_id` /
  `relative_position` when the exact burner, shelf, basin, or side
  matters), `place_next_to` (adjacent on the same support; uses exactly
  one of `reference_object_id` or `reference_fixture_id`, with optional `target_site_id` /
  `relative_position`), or `place_under` (under a dispenser spout or
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

3. Different symbolic objects must correspond to different physical
   instances unless the source truly describes one shared object. If the
   task needs two bowls, two decorations, or two condiments, define two
   distinct symbolic object ids in `initial_state.objects`. Do not assume
   runtime grounding will duplicate one scene object to satisfy repeated
   symbols.

4. Pick the placement tool that matches the semantic relationship from
   the source's `_check_success`:
   - `OU.check_obj_in_receptacle(obj, container)` -> `place_in_receptacle`
     when the container is interior (tray/oven_tray/bowl/basket) or
     `place_on_object` when it is a flat movable support (plate,
     cutting_board).
   - XY-distance proximity check -> `place_next_to` AND a machine flag.
     Use `reference_object_id` for movable anchors. Use
     `reference_fixture_id` for fixture anchors, and only when `machine_state[fixture_id]`
     includes `adjacent_location_id`, and make that adjacent support
     fixture explicit in `initial_state.fixtures`. Objects placed next
     to the fixture land at that `adjacent_location_id`, not at the
     fixture id itself.
   - Direct counter/fixture contact without a movable support ->
     `place_on_surface`.

5. When a goal depends on a spatial proximity check (XY distance) rather
   than a simple "in" check, introduce a machine flag via `task_effects`
   and reference it with `machine_flag_true`. See the GarnishCupcake
   example. Do not collapse a flexible adjacency task into exact count
   goals unless the source truly requires exact counts.

6. When success requires a temporal completion event after an earlier
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

7. The example_trajectory must start with two `communicate` steps (one
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

8. Do NOT add extra steps after the step that first satisfies all
   goal_conditions. This is enforced by the validator: the trajectory
   must end exactly at goal satisfaction.

9. Only include fixture-part-state preconditions for fixtures that
   actually have hinged_part or sliding_part entries in
   `initial_state.fixtures`. Use
   `fixture_part_state_required_for_pickup` for pickup steps gated by an
   open part, and `fixture_part_state_required_for_action` for other
   actions gated by an open part (for example placing into a drawer or
   rack). Also add the corresponding
   `open_hinged_part`/`open_sliding_part` tool entry to
   `allowed_tool_specs`, and make the example_trajectory execute that
   open step before the first gated action.

10. `grounding.legacy_symbol_aliases` must map any legacy IDs the source
   task uses (e.g. `skewer_plate_1` -> `skewer_plate`, `cabinet_1` ->
   `cabinet`). When in doubt, add one alias per id that appends `_1`.

11. Every object id referenced in `goal_conditions`, `task_preconditions`,
   `task_effects`, and `example_trajectory` MUST appear in
   `initial_state.objects` or `initial_state.fixtures`. The static
   validator in Phase 2 will reject the spec otherwise.

12. Use only the canonical tool names listed below. Never invent wrapper
    tools like `turn_on_machine`, `turn_on_sink`, or `turn_off_knob`.
    Represent controls with the canonical tools instead:
    `press_button`, `press_lever`, or `set_rotary_control`.

13. The canonical tool arg names matter. Examples:
    `place_on_surface` uses `support_id`;
    `press_button`/`press_lever`/`set_rotary_control` use `target_id`
    plus `control_id`; `set_rotary_control` also needs `goal`. Do not
    invent alternate arg names like `machine_id`. `place_next_to` uses
    exactly one of `reference_object_id` or `reference_fixture_id`, not
    multiple. `set_rotary_control.args.goal`
    must be a string state like `"on"` or `"off"`, not a numeric knob
    value like `1.0`.

14. Every required step arg must be a concrete, non-null symbolic value.
    Never emit `null` for `support_id`, `receptacle_id`, `control_id`,
    `part_id`, or any other required tool argument.

15. `object_id` must always refer to a movable object in
    `initial_state.objects`. Fixture ids belong in fields like
    `fixture_id`, `target_id`, `reference_fixture_id`, or `support_id`,
    never in `object_id`.

16. When the source requires exact counts across multiple receptacles,
    do not use a broad shared `object_at_location_one_of` pool unless it
    is genuinely flexible in the source. Prefer `object_count_at_location`
    for interchangeable objects or per-type subgroups. Example: if
    plate1 and plate2 must each end with one bun and one sausage, write
    bun-count and sausage-count goals per plate, not four shared `one_of`
    goals over `[plate1, plate2]`. Use concrete `object_at_location`
    only when the source distinguishes individual objects.

17. Do not invent new `kind` values. Goals may use only
    `object_at_location`, `object_count_at_location`,
    `object_at_location_one_of`, `machine_flag_true`,
    `machine_flag_equals`, `fixture_part_state`, or
    `fixture_control_state`. Preconditions may use only
    `object_must_remain_at_location`,
    `fixture_part_state_required_for_pickup`, or
    `fixture_part_state_required_for_action`, or
    `object_location_required_for_action`. Effects may use only
    `set_machine_flag_on_action`.

18. Ground primary fixtures directly when possible. If a fixture is
    uniquely identified by type in the scene, use `unique_fixture_type`
    instead of grounding it through an object that already starts on that
    same fixture. Use `support_fixture_for_object` only for the actual
    placement/support fixture of an object, not to recover the object's
    current source fixture. If the referenced object already starts on the
    fixture, use `source_fixture_for_object`, not `support_fixture_for_object`.

19. If the source distinguishes fixture sub-locations such as stove
    burners, cabinet shelves, drawer slots, fridge shelves, sink basins,
    blender bowls, stand-mixer bowls, toaster slots, or dishwasher racks,
    model them as concrete symbolic IDs. Put those IDs in
    `fixtures[*].support_sites`. For partitioned fixtures, use the
    concrete support-site id in `initial_state.objects[*].location` when
    an object starts at a specific sub-location, and use
    `target_site_id` / `source_site_id` on tool steps whenever the exact
    sub-location matters. Do not rely on fixture-level placement for
    partitioned fixtures, and do not assume runtime fallback to another
    site when a site is specified.

20. Keep fixture ids and support-site ids in separate fields. Use
    `support_id` / `receptacle_id` / `reference_fixture_id`
    for the parent fixture or receptacle anchor, and use
    `target_site_id` / `source_site_id` for the exact sub-location. Do
    not combine them into one string such as `toaster_oven_rack`, and do
    not put a fixture symbol into `reference_object_id`.

21. When an agent is holding any object, its next physical steps should
    only be `navigate_to_fixture` or a placement of that same object.
    Do not open or close fixture parts, and do not operate controls while
    still holding something.

22. If the source checks an appliance state like `turned_on` or
    `started`, model that with a machine flag plus a `set_machine_flag_on_action`
    effect on the activating control step, not only with a final
    `fixture_control_state` on a button or lever. Likewise, if a
    temporary state only matters while some "visited" or completion flag
    is being earned, encode that temporary requirement as an effect guard
    (`required_machine_values`) rather than as a final goal.

23. Do not create separate grounding symbols for internal support sites
    or controls. Put those symbolic IDs under
    `initial_state.fixtures[*].support_sites` / `controls` and ground the
    parent fixture itself.

24. For hinged fixtures, prefer the generic part id `hinged` unless the
    source truly distinguishes multiple independently meaningful parts.
    Do not invent `left_door`, `right_door`, or bare `door`.

25. For sliding fixtures, prefer the generic part id `sliding` unless the
    source truly distinguishes multiple independently meaningful slide
    joints. Do not invent ad hoc drawer-part names.

26. Few-shot examples illustrate the JSON shape and overall style, but
    these site-specific and distinct-instance rules override any older
    example that is silent about partitioned fixtures.
"""

_FORBIDDEN_NAME_GUIDANCE = """\
Forbidden names and common hallucinations:
- Do not invent part ids like `left_door`, `right_door`, `door_left`, `door_right`, or bare `door`; prefer the generic hinged alias `hinged` unless the task truly needs a different exact part id.
- Do not invent site ids like `stove_burner`; use the exact burner/support-site id such as `front_left`, `front_right`, `rear_left`, or `rear_right`.
- `place_under` never takes `control_id`; use `object_id`, `reference_fixture_id`, and optional `target_site_id`.
- Do not emit legacy alias args (`target_id` as a placement alias, `reference_id`) in newly generated specs. Use canonical anchor args (`support_id`, `receptacle_id`, `support_object_id`, `reference_object_id`, `reference_fixture_id`).
"""

_COMPACT_SCHEMA_DESCRIPTION = """\
Emit one complete TaskSpec JSON object with all required top-level fields:
`spec_version`, `composite_task`, `source_python_module`, `agent_ids`,
`max_reasoning_chars`, `validator_checks`, `preflight_token_estimate`,
`initial_state`, `allowed_tool_specs`, `task_goal`, `extra_execution_rules`,
`initial_public_state`, `task_preconditions`, `goal_conditions`,
`task_effects`, `grounding`, `example_trajectory`, and optional `notes`.

Use these defaults unless the task requires otherwise:
- `spec_version`: 1.
- `agent_ids`: ["agent_0", "agent_1"].
- `max_reasoning_chars`: 200.
- `validator_checks`: ["initial_communication", "allowed_tools",
  "navigation_preconditions", "manipulation_preconditions", "effects",
  "final_success"].
- `preflight_token_estimate`: {"prompt_tokens": 3000, "output_tokens": 3500,
  "reasoning_tokens": 0}.

`initial_state`:
- `agents`: starting `{location, held_object: null}` for each agent.
- `objects`: every movable task object with `{object_type, location,
  preserve_pose?}`. Locations are fixture ids, support-site ids, or object ids.
- `fixtures`: every used fixture with `{fixture_type, parts?, controls?,
  support_sites?}`.
- `machine_state`: optional flags used by `machine_flag_*` goals/effects.

`allowed_tool_specs` should contain only task-local allowlists such as
`allowed_object_ids`, `allowed_source_ids`, `allowed_receptacle_ids`,
`allowed_target_ids`, `allowed_control_ids`, `allowed_part_ids`,
`allowed_fixture_ids`, `allowed_reference_object_ids`,
`allowed_reference_fixture_ids`, and `allowed_target_site_ids`. Do not repeat
shared tool metadata such as descriptions or arg schemas.

`example_trajectory.steps` must start with two `communicate` steps and must end
immediately when `goal_conditions` first become true.
"""

_CANONICAL_JSON_SHAPE_EXAMPLE = """\
Canonical JSON shape example (copy this object-vs-array structure):
{
  "spec_version": 1,
  "composite_task": "ExampleTask",
  "source_python_module": "package.module",
  "agent_ids": ["agent_0", "agent_1"],
  "max_reasoning_chars": 200,
  "validator_checks": ["initial_communication", "allowed_tools", "navigation_preconditions", "manipulation_preconditions", "effects", "final_success"],
  "preflight_token_estimate": {"prompt_tokens": 3000, "output_tokens": 3500, "reasoning_tokens": 0},
  "initial_state": {
    "agents": {
      "agent_0": {"location": "cabinet", "held_object": null},
      "agent_1": {"location": "counter", "held_object": null}
    },
    "objects": {
      "bowl1": {"object_type": "bowl", "location": "cabinet"},
      "bowl2": {"object_type": "bowl", "location": "cabinet"}
    },
    "fixtures": {
      "cabinet": {
        "fixture_type": "cabinet",
        "parts": {"hinged": {"part_type": "hinged_part", "state": "closed"}}
      },
      "counter": {"fixture_type": "counter"}
    },
    "machine_state": {}
  },
  "allowed_tool_specs": {
    "communicate": {},
    "navigate_to_fixture": {"allowed_fixture_ids": ["cabinet", "counter"]},
    "open_hinged_part": {"allowed_target_ids": ["cabinet"], "allowed_part_ids": ["hinged"]},
    "pick_up_object": {"allowed_object_ids": ["bowl1", "bowl2"], "allowed_source_ids": ["cabinet"]},
    "place_on_surface": {"allowed_object_ids": ["bowl1", "bowl2"], "allowed_target_ids": ["counter"]}
  },
  "task_goal": "Move both bowls from the cabinet to the counter.",
  "extra_execution_rules": ["Open cabinet.hinged before picking up bowls."],
  "initial_public_state": {"bowl1_location": "cabinet", "bowl2_location": "cabinet"},
  "task_preconditions": [
    {"kind": "fixture_part_state_required_for_pickup", "tool": "pick_up_object", "source_id": "cabinet", "fixture_id": "cabinet", "part_id": "hinged", "required_state": "open"}
  ],
  "goal_conditions": [
    {"kind": "object_at_location", "object_id": "bowl1", "location": "counter"},
    {"kind": "object_at_location", "object_id": "bowl2", "location": "counter"}
  ],
  "task_effects": [],
  "grounding": {
    "legacy_symbol_aliases": {},
    "symbols": {
      "bowl1": {"entity_type": "object", "resolver": "object_by_type", "object_type": "bowl"},
      "bowl2": {"entity_type": "object", "resolver": "object_by_type", "object_type": "bowl"},
      "cabinet": {"entity_type": "fixture", "resolver": "source_fixture_for_object", "object_id": "bowl1"},
      "counter": {"entity_type": "fixture", "resolver": "unique_fixture_type", "fixture_type": "counter"}
    }
  },
  "example_trajectory": {
    "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
    "steps": [
      {"step": 0, "agent": "agent_0", "tool": "communicate", "args": {"to": "agent_1", "message": "I will open the cabinet."}, "reasoning": "I am coordinating first."},
      {"step": 1, "agent": "agent_1", "tool": "communicate", "args": {"to": "agent_0", "message": "I will wait at the counter."}, "reasoning": "I am acknowledging the plan."}
    ]
  },
  "notes": []
}
"""

_COMPACT_GUIDANCE = """\
Write the spec from the source `_check_success` logic, not from guesses.

Core rules:
1. Different symbolic objects must correspond to different physical instances
   unless the source truly uses one shared object. Exclude distractors.
2. If a trajectory step uses `target_site_id` / `source_site_id`, then final
   `goal_conditions.object_at_location.location` must use that exact concrete
   support-site id when the object is expected to finish there. Examples:
   blender -> `int`, sink -> `basin_left`/`basin_right`, toaster oven -> `rack_0`.
   Do not write the parent fixture as the exact goal location in that case.
3. Keep fixture ids and support-site ids separate. Parent fixture/receptacle
   fields use `support_id`, `receptacle_id`,
   `reference_fixture_id`; exact sub-locations use `target_site_id` /
   `source_site_id`. Do not combine them into strings like `toaster_oven_rack`.
4. If a partitioned fixture has a concrete support-site id, use that concrete
   support-site id and do not rely on fallback to another site.
5. Use `place_in_receptacle` for interior containers/sites, `place_on_object`
   for movable supports like plates/cutting boards, `place_on_surface` for
   fixture surfaces, `place_next_to` for adjacency, and `place_under` for
   dispenser/basin-under placement.
6. Spatial-proximity success should use a machine flag set by a
   `set_machine_flag_on_action` effect on the placement step. Do not add
   an `object_at_location` goal that keeps the object at its source unless
   the example trajectory explicitly places the object back there at the end.
7. If an appliance action only counts after objects are in place, put those
   objects in `required_object_locations` on the effect. Parent fixture
   requirements may be used there; exact goal locations should still be exact.
8. Insert `navigate_to_fixture` before any physical step if the acting agent is
   not already at that step's fixture. Do not navigate to a fixture only to call
   `give_space`; `give_space` means the agent is already there and is clearing
   that area for another agent. After `give_space`, navigate again before the
   next physical action at that fixture.
9. When two agents use the same fixture or movable support on that fixture in
   sequence, the agent already at that fixture should call `give_space` before
   the other agent navigates there or performs a placement there.
10. When holding an object, the next physical action should be navigation or
   placement of that same object. Do not operate parts/controls while holding.
11. Use only supported kinds:
   goals: `object_at_location`, `object_count_at_location`,
   `object_at_location_one_of`, `machine_flag_true`, `machine_flag_equals`,
   `fixture_part_state`, `fixture_control_state`.
   preconditions: `object_must_remain_at_location`,
   `fixture_part_state_required_for_pickup`,
   `fixture_part_state_required_for_action`,
   `object_location_required_for_action`.
   effects: `set_machine_flag_on_action`.
11. Prefer generic part ids `hinged` and `sliding` unless the source truly
   distinguishes multiple meaningful parts.
12. Ground movable objects with `object_by_type`, source fixtures with
   `source_fixture_for_object`, unique fixtures with `unique_fixture_type`, and
   adjacent supports with `nearest_placeable_surface_to_fixture`.
"""


def _canonicalize_generic_example_part_id(part_id: Any) -> Any:
    if not isinstance(part_id, str):
        return part_id
    normalized = "_".join(part for part in part_id.strip().lower().split("_") if part)
    if not normalized:
        return part_id
    if "hinge" in normalized or "joint" in normalized:
        return part_id
    if (
        normalized in {"door", "left_door", "right_door", "door_left", "door_right", "hinged"}
        or normalized.endswith("_door")
        or normalized.startswith("door_")
    ):
        return "hinged"
    if (
        normalized in {"drawer", "slide", "sliding"}
        or normalized.endswith("_drawer")
        or normalized.endswith("_slide")
        or normalized.startswith("drawer_")
    ):
        return "sliding"
    return part_id


def _compact_example_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact_payload = {
        key: value
        for key, value in payload.items()
        if key not in _EXAMPLE_OMIT_TOP_LEVEL_FIELDS
    }

    def _rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            rewritten: dict[str, Any] = {}
            for key, nested_value in value.items():
                if key == "reasoning" and isinstance(nested_value, str):
                    rewritten[key] = "..."
                elif key == "part_id":
                    rewritten[key] = _canonicalize_generic_example_part_id(nested_value)
                elif key == "allowed_part_ids" and isinstance(nested_value, list):
                    canonical_part_ids: list[str] = []
                    for part_id in nested_value:
                        canonical_part_id = _canonicalize_generic_example_part_id(part_id)
                        if (
                            isinstance(canonical_part_id, str)
                            and canonical_part_id not in canonical_part_ids
                        ):
                            canonical_part_ids.append(canonical_part_id)
                    rewritten[key] = canonical_part_ids
                elif key == "parts" and isinstance(nested_value, dict):
                    rewritten_parts: dict[str, Any] = {}
                    for part_id, part_state in nested_value.items():
                        canonical_part_id = _canonicalize_generic_example_part_id(part_id)
                        if not isinstance(canonical_part_id, str):
                            continue
                        rewritten_parts[canonical_part_id] = _rewrite(part_state)
                    rewritten[key] = rewritten_parts
                else:
                    rewritten[key] = _rewrite(nested_value)
            return rewritten
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        return value

    return _rewrite(compact_payload)


def _render_example_spec_json(spec_json: str) -> str:
    """Render few-shot specs with only task-local tool overrides exposed."""

    try:
        payload = json.loads(spec_json)
    except json.JSONDecodeError:
        return spec_json.rstrip()

    payload = _compact_example_payload(payload)

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


def _build_detected_pattern_guidance(
    *,
    task_name: str,
    source_python: str,
    metadata: dict[str, Any],
) -> list[str]:
    """Return compact task-pattern hints derived from static source signals."""

    normalized_source = source_python.lower()
    object_names = [
        str(cfg.get("name"))
        for cfg in metadata.get("obj_configs") or []
        if isinstance(cfg, dict) and isinstance(cfg.get("name"), str)
    ]
    fixture_names = [
        str(ref.get("name"))
        for ref in metadata.get("fixture_refs") or []
        if isinstance(ref, dict) and isinstance(ref.get("name"), str)
    ]
    normalized_task_name = task_name.strip().lower()

    has_unique_assignment_xy_pattern = (
        "assigned_" in normalized_source
        and (
            "transform_global_to_local" in normalized_source
            or ("x_dist" in normalized_source and "y_dist" in normalized_source)
        )
        and (
            "best_candidate" in normalized_source
            or "closest_" in normalized_source
            or "best_score" in normalized_source
        )
    )
    has_counter_or_table_contact = (
        "check_obj_any_counter_contact" in normalized_source
        or "check_obj_fixture_contact" in normalized_source
    )
    has_simple_proximity_pattern = (
        "np.linalg.norm" in normalized_source
        and any(token in normalized_source for token in ("_near_", "near_", "_close"))
        and has_counter_or_table_contact
    )

    stool_refs = [name for name in fixture_names if name.startswith("stool")]
    bowl_objects = [name for name in object_names if name.startswith("bowl")]
    has_setup_bowls_pattern = (
        normalized_task_name == "setupbowls"
        or (
            "assigned_stools" in normalized_source
            and "stool_positions" in normalized_source
            and "check_obj_any_counter_contact" in normalized_source
        )
    )
    if has_setup_bowls_pattern and not bowl_objects:
        bowl_objects = ["bowl1", "bowl2"]
    if has_setup_bowls_pattern:
        if not stool_refs:
            stool_refs = ["stool1", "stool2"]
        elif stool_refs == ["stool"]:
            stool_refs = ["stool1", "stool2"]

    hints: list[str] = []
    if has_unique_assignment_xy_pattern:
        hints.extend(
            [
                "Detected task pattern: unique XY proximity assignment.",
                "The source `_check_success` assigns each movable object to a distinct nearby anchor using XY thresholds or local-coordinate distances.",
                "Spec recipe: express the physical placement with `place_next_to` against the corresponding anchor (`reference_object_id` for movable anchors such as plates/trays, `reference_fixture_id` for passive fixtures such as stools). Use one `machine_flag_true` per required assignment, set by a `set_machine_flag_on_action` effect on the matching placement step.",
                "For this pattern, do not set proximity machine flags from `place_on_surface`; that loses the anchor relation and will fail validation. The effect tool should be `place_next_to` with the same anchor args used by the trajectory step.",
                "If the anchor is a passive fixture like a stool, navigate to the support/work surface such as the dining counter before `place_next_to`; do not navigate to the passive anchor just to place near it.",
                "Do not add final `object_at_location` goals that keep a moved object at its source. If exact location goals are needed, they must describe the final support/contact location, not the starting source.",
                "If the source also checks counter/table contact, keep that as the placement support context in the trajectory/effect semantics rather than as a source-location goal.",
            ]
        )
        if object_names:
            hints.append(f"Candidate movable object symbols from metadata: {', '.join(object_names)}.")
        if fixture_names:
            hints.append(f"Candidate fixture/anchor symbols from metadata: {', '.join(fixture_names)}.")

    if has_simple_proximity_pattern and not has_unique_assignment_xy_pattern:
        hints.extend(
            [
                "Detected task pattern: object proximity plus support/contact constraint.",
                "The source `_check_success` computes an XY distance between objects/fixtures and also checks counter/table/support contact.",
                "Spec recipe: use `place_next_to` for the proximity relation and a `machine_flag_true` set by that `place_next_to` action. Do not represent proximity by leaving the object at its source location or by setting a proximity flag from `place_on_surface`.",
            ]
        )
        if object_names:
            hints.append(f"Candidate object symbols from metadata: {', '.join(object_names)}.")
        if fixture_names:
            hints.append(f"Candidate fixture symbols from metadata: {', '.join(fixture_names)}.")

    if has_setup_bowls_pattern and bowl_objects and stool_refs:
        if hints:
            hints.append("")
        hints.extend(
            [
                "Concrete inferred pattern for SetupBowls:",
                f"Objects to place: {', '.join(bowl_objects)}.",
                f"Passive anchors: {', '.join(stool_refs)}.",
                "The source `_check_success` assigns each bowl to a distinct stool by XY thresholds and also requires counter contact.",
                "Spec recipe: use `place_next_to` for each bowl with `reference_fixture_id` set to a distinct stool symbol, and set one `machine_flag_true` per bowl/stool placement via `set_machine_flag_on_action`.",
                "The example trajectory should navigate to `dining_counter` before each `place_next_to`; it should not navigate to `stool1` or `stool2` for the placement step.",
                "Do not add final `object_at_location` goals that keep bowls at the cabinet source. If you add exact location goals at all, they must describe the final counter-side placement, not the source.",
                "Allowed tools should include `place_next_to` with `allowed_reference_fixture_ids` for the stool symbols and `allowed_object_ids` for the bowls.",
            ]
        )

    return hints


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

    if _FEW_SHOT_EXAMPLE_COUNT <= 0:
        selected_examples: tuple[FewShotExample, ...] = ()
    elif len(examples) > _FEW_SHOT_EXAMPLE_COUNT:
        selected_examples = (examples[0], examples[-1])
    else:
        selected_examples = examples
    example_blocks = [
        _format_example(index + 1, example)
        for index, example in enumerate(selected_examples)
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
    detected_pattern_guidance = _build_detected_pattern_guidance(
        task_name=task_name,
        source_python=source_python,
        metadata=metadata,
    )
    detected_pattern_block = (
        ["## Detected task pattern", *detected_pattern_guidance]
        if detected_pattern_guidance
        else []
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
            "## TaskSpec contract",
            _COMPACT_SCHEMA_DESCRIPTION,
            "",
            "## Canonical JSON Shape",
            _CANONICAL_JSON_SHAPE_EXAMPLE,
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
            "## Authoring rules",
            _COMPACT_GUIDANCE,
            "",
            "## Forbidden names",
            _FORBIDDEN_NAME_GUIDANCE,
            "",
            *(
                ["## Few-shot examples", *example_blocks, ""]
                if example_blocks
                else []
            ),
            "## Your task",
            f"Generate a TaskSpec JSON for composite task `{task_name}`.",
            f"Source module: `{source_module}`",
            "",
            objects_block,
            "",
            fixtures_block,
            *([""] + detected_pattern_block if detected_pattern_block else []),
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
