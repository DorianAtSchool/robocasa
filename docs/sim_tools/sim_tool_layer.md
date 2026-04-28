# Sim Tool Layer

This repository exposes a symbolic simulator tool layer on top of RoboCasa.

The goal is not to replay dataset actions exactly. The goal is to give planners
and external LLMs a stable, validated action interface that can be grounded and
executed in a live scene.

The concrete executor for these tools is documented in
[sim_tool_executor.md](./sim_tool_executor.md).

## Why This Layer Exists

The raw environment is too low-level for planning:

- object and fixture ids are scene-specific
- part ids and control ids depend on the concrete fixture instance
- many task instructions are easier to express semantically than as direct
  MuJoCo state edits

The tool layer provides:

- a planner-facing API with explicit semantic intent
- validation against the live scene
- a separation between planner semantics and pragmatic executor logic

## Available Tools

The current tool surface is defined in
[sim_tool_specs.py](../../robocasa/utils/sim_tool_specs.py).

### Fixture Articulation

- `open_hinged_part(target_id, part_id)`
- `close_hinged_part(target_id, part_id)`
- `open_sliding_part(target_id, part_id)`
- `close_sliding_part(target_id, part_id)`

### Navigation

- `navigate_to_fixture(fixture_id)`
- `give_space(fixture_id)`

### Object Manipulation

- `pick_up_object(object_id, source_id)`
- `place_in_receptacle(object_id, receptacle_id | target_id, target_site_id?, relative_position?)`
- `place_on_object(object_id, support_object_id, anchor_fixture_id)`
- `place_on_surface(object_id, support_id | target_id, target_site_id?, relative_position?)`
- `place_next_to(object_id, reference_object_id | reference_fixture_id | reference_id, target_site_id?, relative_position?)`
- `place_under(object_id, reference_fixture_id | target_id, target_site_id?)`

### Controls

- `press_button(target_id, control_id)`
- `press_lever(target_id, control_id)`
- `set_rotary_control(target_id, control_id, goal)`

### Coordination

- `communicate(to, message)`
- `wait()`

## Parameter Types

### Scene-Bound Symbols

These should come from the live scene description:

- `object_id`
- `fixture_id`
- `source_id`
- `support_id`
- `receptacle_id`
- `support_object_id`
- `anchor_fixture_id`
- `reference_object_id`
- `reference_fixture_id`
- `target_id`
- `part_id`
- `control_id`

### Planner Literals

These are planner-provided values rather than scene ids:

- `goal`
- `to`
- `message`

Recommended generation flow:

1. build a live scene
2. extract the scene symbols
3. give the planner only those valid ids
4. validate each tool call before execution

## Spatial Semantics

Already represented directly in the tool surface:

- `on` via `place_on_surface(...)`
- `on_object` via `place_on_object(...)`
- `in` via `place_in_receptacle(...)`
- `next_to` via `place_next_to(...)`
- `under` via `place_under(...)`

Still planner-level:

- `near`
- richer `inside` semantics beyond `place_in_receptacle(...)`
- fixture-selection phrases such as "counter near toaster oven"

## Planner vs Executor

The planner-facing API and the executor backend are intentionally different.

- The planner API should stay semantically clean.
- The executor can use teleportation, direct state edits, and geometry
  heuristics to make those semantics executable.

This separation matters because multiple planner tools may share the same
runner-side implementation without being the same action semantically.

## Compact LLM Context

The generator for compact planner-facing scene/tool descriptions lives in
[generate_llm_task_descriptions.py](../../robocasa/utils/generate_llm_task_descriptions.py).

Useful context usually includes:

- task instruction
- robots
- tool specs
- objects with type and current location
- fixtures with type, nearby fixtures, parts, and controls
- rules forbidding invented ids

## Current Limits

This symbolic layer works well for:

- fixture articulation
- teleport-based fixture navigation
- fixture and support-object placement
- appliance actuation
- multi-robot coordination

It is weaker for:

- long-horizon appliance processes with rich temporal state
- liquid, washing, and pouring semantics
- contact-rich insertion or fine manipulation
- physically realistic grasping and support stability
- ambiguous spatial relations that require richer scene reasoning
- dataset-faithful replay
