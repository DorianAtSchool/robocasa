# Sim Tool Layer

This repository exposes a symbolic simulator tool layer on top of RoboCasa.

The goal is not to replay dataset actions exactly. The goal is to give planners
and external LLMs a stable, validated action interface that can be grounded and
executed in a live scene.

The concrete executor for these tools is documented in
[sim_tool_executor.md](/Users/dorian/Documents/robocasa/docs/sim_tool_executor.md).

## Why This Layer Exists

The raw environment is too low-level for planning:

- object and fixture ids are scene-specific
- part ids and control ids depend on the concrete fixture instance
- many task instructions are easiest to express semantically, not as direct
  MuJoCo state edits

The tool layer provides:

- a planner-facing API with explicit semantic intent
- validation against the live scene
- a separation between clean planning semantics and pragmatic executor logic

## Available Tools

The current tool surface is defined in
[sim_tool_specs.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_specs.py).

### Fixture Articulation

- `open_hinged_part(target_id, part_id)`
- `close_hinged_part(target_id, part_id)`
- `open_sliding_part(target_id, part_id)`
- `close_sliding_part(target_id, part_id)`

Use these for fridges, cabinets, drawers, lids, racks, and similar articulated
fixture parts.

### Navigation

- `navigate_to_fixture(fixture_id)`
- `give_space(fixture_id)`

`navigate_to_fixture(...)` moves a robot to the working pose associated with a
fixture. `give_space(...)` is the explicit coordination tool for backing away
from a fixture so another robot can use it.

### Object Manipulation

- `pick_up_object(object_id, source_id)`
- `place_in_receptacle(object_id, receptacle_id)`
- `place_on_object(object_id, support_object_id, anchor_fixture_id)`
- `place_on_surface(object_id, support_id)`
- `place_next_to(object_id, reference_object_id)`
- `place_under(object_id, reference_fixture_id)`

These are distinct planner-level actions even when some of them share backend
execution patterns.

Important semantic distinctions:

- `place_on_surface(...)` means "put the object somewhere valid on this support surface"
- `place_on_object(...)` means "stack on this movable support object"
- `place_in_receptacle(...)` means "place into a container-like or interior target"
- `place_next_to(...)` means "place adjacent to another object on the same support surface"
- `place_under(...)` means "place directly beneath a reference fixture" and is also used for dispenser-like fixtures such as sinks and coffee machines

### Controls / Activation

- `press_button(target_id, control_id)`
- `press_lever(target_id, control_id)`
- `set_rotary_control(target_id, control_id, goal)`

Use these for microwaves, coffee machines, toasters, kettles, knobs, and other
fixture controls.

### Multi-Agent Coordination

- `communicate(to, message)`
- `wait()`

These are semantic coordination tools. They primarily structure multi-robot
plans rather than changing physics state.

## Parameter Types

Some tool parameters must come from the current live scene. Others are small
planner-provided literals.

### Scene-Bound Symbols

These should come from the current scene description, not from free-form text:

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

These are planner-provided values, not scene ids:

- `goal`
- `to`
- `message`

The safest generation flow is:

1. build a live scene
2. extract its symbol table
3. give the planner only those valid ids
4. validate planned actions before execution

## Spatial Semantics

Some spatial relations are now first-class tools instead of implicit lowering rules.

Already represented directly in the tool surface:

- `on` via `place_on_surface(...)`
- `on_object` via `place_on_object(...)`
- `in` via `place_in_receptacle(...)`
- `next_to` via `place_next_to(...)`
- `under` via `place_under(...)`

Still planner-level / higher-level semantics:

- `near`
- `inside` when it is richer than `place_in_receptacle(...)`
- fixture-selection semantics such as "on the counter near the toaster oven"

Example:

- a planner may still need to resolve "counter near toaster oven" into a
  concrete `support_id`, but once that support is chosen the executor already
  has the tool needed for `place_next_to(...)` or `place_on_surface(...)`

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

The planner-facing API and the executor backend are intentionally different.

- The planner API should stay semantically clean.
- The executor can be pragmatic and use teleportation, direct state edits,
  and geometry heuristics.

This separation matters because multiple tools may share the same low-level
backend while still representing different planning semantics.

## Compact LLM Context

Passing the full scene JSON to an external planner is unnecessary. A compact
planner context should contain:

- task instruction
- robots
- tool specs
- objects with type and current location
- fixtures with type, nearby fixtures, parts, and controls
- hard rules that forbid invented ids

The generator for this lives in
[generate_llm_task_descriptions.py](/Users/dorian/Documents/robocasa/robocasa/scripts/generate_llm_task_descriptions.py).

## Current Limits

This symbolic layer scales well for:

- opening / closing fixture parts
- navigating to fixture working poses
- picking from fixtures and surfaces
- placing on fixtures, on support objects, next to objects, or under fixtures
- actuating appliances
- coordinating between robots

It is weaker for:

- long-horizon appliance processes with rich temporal state
- liquid, washing, and pouring semantics
- contact-rich insertion or fine manipulation
- physically realistic grasping / support stability
- ambiguous spatial relations that require richer scene reasoning than the
  current semantic refs and lowering helpers provide
- dataset-faithful replay

Those cases need richer predicates, richer symbolic lowering, or a more
physical controller backend.
