# Sim Tool Layer

This repository now exposes a symbolic simulator tool layer on top of RoboCasa.
The goal is not to replay dataset actions exactly. The goal is to give planners
and external LLMs a stable, validated action interface that can be executed in a
live scene.

## Why This Layer Exists

The raw environment is too low-level for planning:

- object and fixture ids are scene-specific
- part ids and control ids depend on the concrete fixture instance
- many tasks are easiest to describe semantically, not by direct MuJoCo edits

The tool layer provides:

- a planner-facing API with explicit semantic intent
- validation against the live scene
- a place to separate planning semantics from executor pragmatics

## Available Tools

The current tool surface is defined in [robocasa/utils/sim_tool_specs.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_specs.py).

### Fixture Articulation

- `open_hinged_part(target_id, part_id)`
- `close_hinged_part(target_id, part_id)`
- `open_sliding_part(target_id, part_id)`
- `close_sliding_part(target_id, part_id)`

Use these for cabinets, drawers, fridges, lids, and similar articulated parts.

### Navigation

- `navigate_to_fixture(fixture_id)`

Moves a robot base to the working pose associated with a fixture.

### Object Manipulation

- `pick_up_object(object_id, source_id)`
- `place_in_receptacle(object_id, receptacle_id)`
- `place_on_object(object_id, support_object_id, anchor_fixture_id)`
- `place_on_surface(object_id, support_id)`
- `place_under_dispenser(object_id, dispenser_id)`

These are distinct planner-level actions even when some of them share backend
execution patterns.

### Control / Activation

- `press_button(target_id, control_id)`
- `press_lever(target_id, control_id)`
- `set_rotary_control(target_id, control_id, goal)`

Use these for microwaves, coffee machines, toasters, kettles, knobs, and other
fixture controls.

### Multi-Agent Coordination

- `communicate(to, message)`
- `wait()`

These are semantic coordination tools. They do not primarily exist to change
physics state; they exist to structure multi-robot plans.

## Symbol Types

These ids must come from the current live scene, not from free-form language:

- `object_id`
- `fixture_id`
- `source_id`
- `support_id`
- `receptacle_id`
- `support_object_id`
- `anchor_fixture_id`
- `dispenser_id`
- `target_id`
- `part_id`
- `control_id`

The safest generation flow is:

1. build a live scene
2. extract its symbol table
3. give the planner only those valid symbols
4. validate planned actions before execution

## Spatial Qualifiers

Primitive tools are not enough to express every task. A more flexible planning
layer should include spatial qualifiers such as:

- `on`
- `in`
- `under`
- `near`
- `next_to`
- `inside`
- `at_fixture`

Examples:

- `place condiment next_to plate`
- `place mug under coffee_machine dispenser`
- `put sausage on plate`

These qualifiers should be interpreted by a semantic planner and then lowered to
primitive sim tools.

## Task-State Predicates

Useful planner-level predicates include:

- `holding(robot, object)`
- `robot_at(robot, fixture)`
- `object_at(object, target)`
- `object_on(object, support_object)`
- `object_in(object, receptacle)`
- `fixture_open(fixture)`
- `fixture_closed(fixture)`
- `control_set(fixture, control, goal)`
- `area_clear(fixture)`
- `subtask_done(name)`

These predicates let a planner reason about progress, preconditions, and
multi-agent coordination without directly touching MuJoCo state.

## Planner vs Executor

The planner-facing API and the executor backend are intentionally different:

- The planner API should stay semantically clean.
- The executor can be pragmatic and use a mix of RoboCasa helpers and direct sim
  state edits.

This separation matters because multiple tools may share the same low-level
backend while still representing different planning semantics.

## Compact LLM Context

Passing the full scene JSON to an external LLM is unnecessary. The compact
planner context should contain:

- task instruction
- robots
- tool specs
- objects with type and current location
- fixtures with type, nearby fixtures, parts, and controls
- hard rules that forbid invented ids

The generator for this lives in
[robocasa/scripts/generate_llm_task_descriptions.py](/home/dorian/Projects/robocasa/robocasa/scripts/generate_llm_task_descriptions.py).

## Current Limits

This approach scales well for symbolic kitchen workflows such as:

- open / close fixture parts
- navigate to fixture
- pick from fixture
- place on fixture or support object
- actuate appliances
- coordinate between robots

It is weaker for:

- precise relative placement such as `next_to` without a lowering rule
- liquid and washing processes
- long temporal appliance state changes
- contact-rich insertion or fine manipulation
- dataset-faithful replay

Those cases need richer semantics, better state predicates, or a more physical
controller backend.
