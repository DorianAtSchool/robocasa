# TaskSpec Reference

This page is the user-facing reference for the JSON `TaskSpec` format used by
the task-level data-generation pipeline.

If the `TaskSpec` contract changes, update this page in the same PR as the code
change. In practice that usually means updating this doc together with:

- `data_generation/task_level/pipeline/prompts/spec_generation.py`
- `data_generation/task_level/pipeline/phase2.py`
- `data_generation/task_level/tasks/specs/__init__.py`
- any affected tests under `tests/`

## What A TaskSpec Is

A `TaskSpec` is the symbolic contract between:

- Phase 1 spec generation
- Phase 2 validation
- Phase 3 trajectory generation
- Phase 4 trajectory execution / sweep

It describes:

- the symbolic initial scene state
- the allowed tool vocabulary and task-local constraints
- the goal / precondition / effect logic for the symbolic validator
- a small example trajectory that should satisfy the spec

The runtime loader is [tasks/specs/__init__.py](/home/dorian/Projects/robocasa/data_generation/task_level/tasks/specs/__init__.py:31).

## Pipeline Behavior

### Phase 1

Phase 1 generates a `TaskSpec` with the LLM.

- `--phase1-sim-normalization` is optional and off by default.
- When enabled, Phase 1 normalizes simulator-facing ids like fixture parts,
  controls, support sites, and some locations against live simulator metadata.

### Phase 2

Phase 2 always validates the spec before trajectory generation.

- Schema loading via `TaskSpec.from_dict(...)`
- supported goal / precondition / effect kinds
- referential integrity across `initial_state`, goals, effects, and
  `example_trajectory`
- simulator-backed fixture / site reference validation
- simulator-backed alignment checks via
  `collect_simulation_alignment_errors(...)`
- symbolic dry-run of `example_trajectory`

There is currently no CLI flag that disables the Phase 2 simulator-backed
validation path. This is separate from Phase 1 sim normalization.

### Phase 3

Phase 3 trajectory generation sees the full `initial_state`,
`allowed_tool_specs`, goal logic, and execution rules. The shared prompt builder
serializes the current `initial_state` directly into the model prompt.

Raw trajectory generation (`python -m data_generation.task_level.generation.raw.cli`)
now also runs a default-on static referential validation pass before FSM replay.
This check does not initialize the simulator. It validates simulator-facing
symbolic ids in candidate steps against TaskSpec-declared ids (for example
`part_id`, `control_id`, `source_site_id`, `target_site_id`) and rejects
unknown tool args.

- Default: enabled
- Disable flag: `--disable-static-referential-validation`

This complements Phase 2 validation by enforcing the same class of id
consistency checks even when trajectories are generated outside the pipeline.

### Phase 4

Phase 4 adapts the generated trajectory to a concrete scene and executes it in
sim. Fields like `preserve_pose` are consumed here by
`SimToolExecutor.load_initial_state(...)`.

## Top-Level Shape

Current top-level fields:

```jsonc
{
  "spec_version": 1,
  "composite_task": "PrepareCoffee",
  "source_python_module": "robocasa.environments.kitchen.multi_stage.brewing.prepare_coffee",
  "agent_ids": ["agent_0", "agent_1"],
  "max_reasoning_chars": 200,
  "validator_checks": [
    "initial_communication",
    "allowed_tools",
    "navigation_preconditions",
    "manipulation_preconditions",
    "effects",
    "final_success"
  ],
  "preflight_token_estimate": {
    "prompt_tokens": 3000,
    "output_tokens": 3500,
    "reasoning_tokens": 0
  },
  "initial_state": {},
  "allowed_tool_specs": {},
  "task_goal": "Natural-language success condition.",
  "extra_execution_rules": [],
  "initial_public_state": {},
  "task_preconditions": [],
  "goal_conditions": [],
  "task_effects": [],
  "grounding": {},
  "example_trajectory": {},
  "notes": []
}
```

`TaskSpec.from_dict(...)` currently requires the fields above except `notes`,
which is optional.

## `initial_state`

`initial_state` is the symbolic scene state at time `t=0`.

```jsonc
{
  "agents": {
    "agent_0": {"location": "coffee_machine", "held_object": null},
    "agent_1": {"location": "coffee_machine", "held_object": null}
  },
  "objects": {
    "mug": {
      "object_type": "mug",
      "location": "staging_surface"
    },
    "kettle": {
      "object_type": "kettle",
      "location": "stove",
      "target_site_id": "front_left"
    },
    "sugar_cube_1": {
      "object_type": "sugar_cube",
      "location": "dining_counter",
      "preserve_pose": true
    }
  },
  "fixtures": {
    "coffee_machine": {
      "fixture_type": "coffee_machine"
    },
    "cabinet": {
      "fixture_type": "cabinet_double_door",
      "parts": {
        "door": {"state": "closed"}
      }
    },
    "stove": {
      "fixture_type": "stove",
      "support_sites": {
        "front_left": {},
        "front_right": {}
      }
    }
  },
  "machine_state": {
    "coffee_machine": {
      "power_on": false
    }
  }
}
```

### `initial_state.agents`

Each agent usually includes:

- `location`: symbolic fixture id where the agent starts
- `held_object`: usually `null` at task start

### `initial_state.objects`

Each object entry usually includes:

- `object_type`: RoboCasa object category or task-relevant subtype
- `location`: one of
  - a fixture id
  - another object id
  - a concrete support-site id when the exact sub-location matters
- `target_site_id` optional:
  - use when the object starts on a specific support site of a fixture
- `preserve_pose` optional boolean:
  - when `true`, the executor keeps the scene-generated starting pose instead
    of re-placing the object
  - use this for duplicate or interchangeable objects that already start in the
    correct fixture or container and whose exact pose is not semantically
    important

`preserve_pose` is a runtime executor feature. It is currently documented in
the Phase 1 spec-generation prompt and honored in
[sim_tool_executor.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_executor.py:1170),
but it is not yet enforced by a dedicated typed `TaskSpec` schema.

### `initial_state.fixtures`

Each fixture entry includes:

- `fixture_type`
- optional `parts`
- optional `controls`
- optional `support_sites`

Use exact symbolic ids for fixture internals. Do not invent aliases like
`left_door`, `door_left`, or `stove_burner` unless those exact ids are really
declared for that fixture.

### `initial_state.machine_state`

Optional symbolic machine flags for cases where simple object locations are not
enough. This is used by goal conditions and task effects that depend on
internal machine state.

## `allowed_tool_specs`

`allowed_tool_specs` is the per-task tool contract.

Each key is a canonical tool name such as:

- `communicate`
- `navigate_to_fixture`
- `pick_up_object`
- `place_on_surface`
- `place_in_receptacle`
- `place_on_object`
- `place_next_to`
- `place_under`
- `open_hinged_part`
- `close_hinged_part`
- `press_button`

Each value contains the canonical tool metadata plus task-local constraints.
Stored specs typically include fields like:

- `description`
- `tool_args`
- `optional_tool_args`
- `tool_arg_any_of`
- `tool_arg_types`
- task-local `allowed_*` constraint lists

Common task-local constraint fields include:

- `allowed_object_ids`
- `allowed_source_ids`
- `allowed_support_ids`
- `allowed_target_ids`
- `allowed_receptacle_ids`
- `allowed_support_object_ids`
- `allowed_reference_object_ids`
- `allowed_reference_fixture_ids`
- `allowed_fixture_ids`
- `allowed_part_ids`
- `allowed_control_ids`

Important rules:

- ids in these allowlists must come from the symbolic ids declared elsewhere in
  the spec
- fixture-part and support-site ids must match real ids for the referenced
  fixture
- `place_under` currently allows `object_id`, one of `reference_fixture_id` or
  `target_id` depending on schema context, and optional `target_site_id`
- `place_under` does not take `control_id`

### Why Optional Tool Args Exist

Optional args exist so TaskSpecs and trajectories can stay semantically clear
without forcing one rigid spelling for every placement / reference operation.
They preserve high-level intent while still allowing the executor to resolve
scene-specific details.

- `source_site_id` (`pick_up_object`)
  Use when pickup must come from a specific sub-location inside a fixture
  (for example a specific rack / basin / shelf). Needed to disambiguate
  multiple valid pickup regions within the same `source_id`.

- `target_id` (alias on several tools)
  Alias for the primary target field used by that tool family:
  `support_id`, `receptacle_id`, `support_object_id`, or
  `reference_fixture_id` depending on tool. Needed so specs generated from
  varied prompts / templates can remain valid without brittle key-name coupling.

- `support_id` (`place_on_surface`)
  Explicit naming for surface placement target. Needed for semantic clarity
  when the task is explicitly "on a surface", while still interoperating with
  alias form (`target_id`).

- `receptacle_id` (`place_in_receptacle`)
  Explicit naming for container / interior target. Needed to keep "in" semantics
  separate from surface semantics in specs and validator logic.

- `target_site_id` (placement tools and `place_under`)
  Selects a specific support site within a fixture (burner, rack, shelf, basin,
  slot, dispenser-adjacent site). Needed when fixture-level target is not
  specific enough and wrong site choice causes semantic or physical failure.

- `reference_id` (`place_next_to`)
  Generic anchor id when planner output does not pre-classify the reference as
  object vs fixture. Needed as an interoperability field for upstream planners.

- `reference_object_id` / `reference_fixture_id` (`place_next_to`)
  Explicit anchor typing. Needed to avoid ambiguity and let validation enforce
  correct id domain.

- `relative_position` (placement tools)
  Soft directional preference (`left`, `right`, `front`, etc.) relative to the
  target / reference. Needed for tasks where arrangement intent matters, but
  exact metric coordinates should still be solved by executor geometry checks.

## Behavioral Fields

### `task_goal`

Short natural-language statement of the intended success condition.

### `extra_execution_rules`

Task-specific guidance that augments the shared prompt rules.

### `initial_public_state`

Symbolic public facts exposed to the planner. These should mirror the initial
task state at `t=0`.

### `task_preconditions`

Current supported kinds:

- `object_must_remain_at_location`
- `fixture_part_state_required_for_pickup`
- `fixture_part_state_required_for_action`
- `object_location_required_for_action`

### `goal_conditions`

Current supported kinds:

- `object_at_location`
- `object_count_at_location`
- `object_at_location_one_of`
- `machine_flag_true`
- `machine_flag_equals`
- `fixture_part_state`
- `fixture_control_state`

### `task_effects`

Current supported kinds:

- `set_machine_flag_on_action`

### `grounding`

Maps symbolic ids in the spec to source-task roles and grounding hints. This is
used to preserve semantic identity through the spec and trajectory pipeline.

### `example_trajectory`

A small symbolic trajectory that should be valid under the spec. Phase 2 dry
runs this trajectory through the symbolic FSM validator. If the example
trajectory disagrees with the declared initial state, goals, or tool
constraints, Phase 2 should fail.

### `notes`

Optional human notes. Not part of execution semantics.

## Common Failure Modes

- invented part ids such as `left_door` or `door_left`
- invented support-site ids such as `stove_burner`
- fixture ids placed in `reference_object_id`
- `place_under(..., control_id=...)`
- symbolic object ids whose requested `object_type` has no plausible concrete
  scene candidate
- missing `target_site_id` when the task depends on an exact burner / rack /
  basin / slot

## Authoring Guidance

- Keep symbolic ids stable across `initial_state`, goals, effects, and
  `example_trajectory`.
- Prefer exact support-site ids when the task depends on a specific burner,
  rack, basin, tray, shelf, or slot.
- Use `preserve_pose: true` sparingly, only when the scene generator already
  provides a good starting pose and task semantics do not depend on exact
  re-placement.
- Treat duplicate objects as distinct symbolic ids even if they share the same
  `object_type`.
- Keep `example_trajectory` minimal but valid; it is a validator target, not a
  free-form demonstration blob.

## Source Of Truth

The authoritative implementation lives in code, especially:

- [phase2.py](/home/dorian/Projects/robocasa/data_generation/task_level/pipeline/phase2.py:824)
- [spec_generation.py](/home/dorian/Projects/robocasa/data_generation/task_level/pipeline/prompts/spec_generation.py:85)
- [tasks/specs/__init__.py](/home/dorian/Projects/robocasa/data_generation/task_level/tasks/specs/__init__.py:31)
- [sim_tool_executor.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_executor.py:1170)

This page should stay synchronized with those files.
