# Sim Tool Executor

This document explains the current implementation of
[sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py).

It is narrower than
[sim_tool_layer.md](./sim_tool_layer.md).
That document describes the planner-facing tool surface. This document focuses
on the concrete executor that runs those tools inside a live RoboCasa scene.

For implementation history, see:

- [sim_tool_executor_phase_2.md](./sim_tool_executor_phase_2.md)
- [sim_tool_executor_phase_3.md](./sim_tool_executor_phase_3.md)
- [continuous_vs_grid_placement.md](./continuous_vs_grid_placement.md)

## Purpose

`SimToolExecutor` is a pragmatic symbolic-execution layer over a live RoboCasa /
MuJoCo environment.

It bridges:

- grounded symbolic tool calls such as `pick_up_object(...)` and `place_on_surface(...)`
- a concrete sampled RoboCasa task instance

It is not:

- a low-level motion controller
- dataset-faithful demo replay
- a learned manipulation policy

Instead, it executes symbolic plans by combining:

- robot base teleportation to working poses
- direct object pose updates
- fixture / control state updates
- semantic grounding from task-level references to scene ids
- rendering of stepwise frames and videos

## Execution Model

The executor is explicitly teleport-based.

There are three main state-update paths under the hood.

### 1. Robot Base Positioning

Robot motion is delegated to
[trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py),
primarily through `runner._move_robot_near_fixture(...)`.

That path no longer relies on a single fixture-relative helper. It now uses the
Phase 2 placement subsystem:

- shared face / front-alignment helpers in [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)
- a grid backend in [occupancy_grid.py](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py)
- a continuous backend in [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)

So for robot placement:

- the symbolic tool identifies a target fixture or anchor fixture
- the runner computes a layout-aware working pose
- the robot base pose is teleported into the simulator

For enclosing fixtures such as fridges and cabinets, the executor also uses
front-readiness checks so interaction tools can decide whether a robot is
already in a usable front working pose.

### 2. Object Positioning

Object motion is a mix of runner-side fixture placement and direct pose writes.

For fixture placement:

- `place_on_surface(...)`
- fixture-target `place_in_receptacle(...)`
- `place_next_to(...)`
- generic `place_under(...)`

all route through the runner's collision-aware fixture-placement logic from
Phase 3.

That path:

- samples candidate poses from fixture reset regions
- filters candidates against existing objects and nearby fixture geometry
- prefers poses near a semantic target XY when one exists
- falls back conservatively if a geometry helper cannot evaluate one case

For direct placement paths such as object-on-object stacking and dispenser-site
placement, the executor writes object qpos directly with MuJoCo state updates.

### 3. Fixture / Control State Updates

Fixture articulation and controls use a mix of:

- RoboCasa fixture helpers such as `open_door()` / `close_door()`
- `fixture.set_joint_state(...)`
- fixture-specific helpers for appliances such as microwaves, coffee machines,
  kettles, and toasters

## High-Level Flow

The executor lifecycle is:

1. construct a live environment through `TrajectoryRunner`
2. inspect and cache the current scene description
3. optionally ground a semantic plan template into concrete ids
4. execute each tool step
5. save frames, videos, and metadata

At construction time, `SimToolExecutor(...)` creates a
[TrajectoryRunner](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py),
stores it as `self.runner`, exposes `self.env`, and initializes:

- `self._held_objects: dict[int, str]`

That held-object map is the executor's symbolic grasp state.

## Core Responsibilities

`SimToolExecutor` has five main jobs.

### 1. Scene Access

It exposes scene and rendering helpers:

- `get_scene_description()`
- `render()`
- `save_scene_frames(...)`
- `save_placement_map(...)`

These are thin wrappers over the underlying runner.

### 2. Symbol Validation

Before executing actions, the executor validates scene ids:

- `_require_fixture(fixture_id)`
- `_require_object(object_id)`

This keeps plans tied to the current scene instead of invented strings.

### 3. Semantic Grounding

The executor supports semantic plan templates through:

1. `build_demo_plan_template(...)`
2. `ground_plan_template(...)`
3. `run_tool_plan(...)`

Templates can contain semantic references such as:

```json
{
  "$ref": "source_fixture",
  "object_id": "sausage",
  "preferred_fixture_types": ["fridge"]
}
```

or:

```json
{
  "$ref": "object_anchor_fixture",
  "object_id": "plate",
  "preferred_fixture_types": ["dining_counter", "island", "counter_non_dining"],
  "require_placeable": true
}
```

Those refs are resolved against the current scene at runtime.

### 4. Tool Dispatch

Each tool is implemented as a method on `SimToolExecutor`.

Examples:

- `navigate_to_fixture(...)`
- `pick_up_object(...)`
- `place_on_surface(...)`
- `place_in_receptacle(...)`
- `place_on_object(...)`
- `place_next_to(...)`
- `place_under(...)`
- `open_hinged_part(...)`
- `press_button(...)`
- `communicate(...)`

The generic entry point is:

- `execute(tool_name, robot_idx=0, **kwargs)`

### 5. Output Saving

`run_tool_plan(...)` executes a plan while saving:

- before / after PNGs
- per-camera MP4 videos
- `plan.json`
- `metadata.json`

The saved `plan.json` is the grounded plan that was actually executed.

## How State Is Represented

The executor mixes symbolic bookkeeping with direct simulator state edits.

### Held Objects

When `pick_up_object(...)` runs:

- the robot is moved near the source fixture if needed
- `self._held_objects[robot_idx]` is updated
- `_sync_held_object(...)` snaps the object near the robot end effector

This is symbolic grasp state, not contact-rich grasp simulation.

### Object Placement

There are three main placement styles:

- fixture placement through runner sampling
- object-on-object placement through bbox geometry
- explicit site placement for dispenser-like cases

Runner-side fixture placement is now collision-aware. Direct placement paths
still exist, but they update support grounding consistently and participate in
contained-object transport.

### Receptacle Carry Semantics

If a receptacle-like object is moved, its contents now move with it.

This is implemented through:

- runner-side contained-object transport in `move_object(...)`
- executor-side contained-object transport in `_set_object_pose(...)`
- held-object syncing through `_sync_held_object(...)`

So a bowl / plate / receptacle can carry its contents across both held-object
motion and placement motion.

### Fixture / Control State

Articulation and control tools use:

- fixture helper methods
- fixture-specific appliance helpers
- direct joint value updates

depending on the target fixture type.

## Tool Behavior Overview

### Navigation

`navigate_to_fixture(...)`

- validates the fixture id
- computes whether the fixture should use front-only semantics
- teleports the robot to a working pose
- re-syncs any held object

### Picking

`pick_up_object(...)`

- validates object and source ids
- moves the robot near the source fixture if it is not already in a usable pose
- **returns failure if the robot cannot navigate to the source fixture**
- marks the object as held
- snaps it near the end effector

For enclosing fixtures, the readiness check is stricter than simple proximity:
the robot must be on the correct front face and within a reasonable working
standoff.

Note: the executor does not automatically open enclosing fixtures (fridges,
cabinets) before picking. Trajectories must include explicit `open_hinged_part`
steps. This is still to be determined as a future improvement.

### Placement on Fixtures

`place_on_surface(...)`

- computes a collision-aware fixture target
- moves the robot near that target region
- places the object through the runner
- clears held-object state

### Placement on Objects

`place_on_object(...)`

- resolves or uses an explicit anchor fixture for robot approach
- moves the robot near the support object's anchor area
- computes the placed pose from support-object top and placed-object bottom bbox geometry
- clears held-object state

### Receptacles

`place_in_receptacle(...)`

- uses fixture placement if the target is a fixture
- uses object-on-object placement if the target is an object

### Spatial Tools

`place_next_to(...)`

- finds the support fixture of the reference object
- computes one or more preferred adjacent target XYs
- asks the runner for the nearest collision-free fixture pose

`place_under(...)`

- uses explicit dispenser sites for coffee machines / sinks
- otherwise finds the nearest support surface below the reference fixture and
  places there through the runner

### Controls

`press_button(...)`, `press_lever(...)`, `set_rotary_control(...)`

- move the robot near the target fixture
- update fixture state using helper methods or direct joint writes

### Coordination

`communicate(...)` and `wait()`

These are semantic coordination actions. They do not materially change physics
state, but they are recorded in the execution trace.

## Built-In Demo Plans

The executor supports named built-in plans through:

- `build_demo_plan_template(...)`
- `build_demo_plan(...)`

The current built-in demos are:

- `cooperative_hotdog_setup`
- `sandwich_station`

Important clarification:

- `cooperative_hotdog_setup` stages bun + sausage on the plate and then places
  the condiment on the same support surface
- `sandwich_station` currently places both the ingredient bowl and baguette on
  the counter near the toaster oven; it does not place the baguette inside the bowl

## Relation to `TrajectoryRunner`

The split is:

- `TrajectoryRunner`
  - builds the environment
  - manages cameras and rendering
  - computes robot working poses
  - computes collision-aware fixture placement poses
  - moves objects onto fixtures

- `SimToolExecutor`
  - validates tool calls
  - grounds semantic references
  - tracks held objects
  - dispatches symbolic tool methods
  - records execution outputs

So the executor is a symbolic dispatch layer over the runner.

## Camera Outputs

Rendered outputs come from `TrajectoryRunner`.

Important camera behavior:

- robot cameras are discovered from the live environment
- wrist / `eye_in_hand` views are included when available
- `room_view` is a derived free camera
- `top_view` is another derived free camera based on the scene footprint

The room view is scene-aware rather than purely layout-static.

## Limitations

The executor is intentionally pragmatic and therefore limited.

- no low-level physical grasp controller
- no dataset-faithful demo replay
- still teleport-based for navigation and manipulation
- direct object-on-object stacking is geometry-based, not contact-stabilized
- semantic spatial relations are still lowered by heuristics rather than a full symbolic geometry solver
- support grounding and relative placement are improved, but ambiguous scenes can still require conservative fallback behavior

That makes the executor useful for symbolic planning, debugging, and visual
rollout, but it should not be mistaken for a physically realistic motion policy.
