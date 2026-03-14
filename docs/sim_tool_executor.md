# Sim Tool Executor

This document explains how
[sim_tool_executor.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_executor.py)
works.

It is narrower than
[sim_tool_layer.md](/home/dorian/Projects/robocasa/docs/sim_tool_layer.md).
That document describes the planner-facing tool API. This document describes
the concrete executor that runs those tools inside a live RoboCasa simulation.

## Purpose

`SimToolExecutor` is a pragmatic bridge between:

- symbolic tool plans such as `pick_up_object(...)` or `place_on_surface(...)`
- a real RoboCasa / MuJoCo environment

It is not a policy controller and it is not dataset-faithful action replay.
Instead, it executes symbolic plans by combining:

- robot base teleportation near fixtures or object anchors
- direct object pose updates in sim state
- fixture joint / control state updates
- rendering of per-step frames and videos

## Teleportation Model

The executor is explicitly teleport-based.

It does **not** replay continuous robot actions from RoboCasa demos, and it does
not use a learned low-level manipulation policy. Instead, it performs symbolic
state changes in a live environment.

There are three different teleportation-like mechanisms under the hood:

### 1. Robot Base Repositioning

Robot motion is handled by teleporting the mobile base near a target fixture or
object anchor.

This path uses RoboCasa placement utilities rather than hard-coded world
coordinates:

- `EnvUtils.compute_robot_base_placement_pose(...)`
- `EnvUtils.set_robot_to_position(...)`

Those are called through
[trajectory_runner.py](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py)
when `runner._move_robot_near_fixture(...)` is used.

So for robot placement:

- the target fixture comes from the symbolic plan or semantic grounding
- the actual placement pose is computed using RoboCasa fixture geometry and
  layout-aware placement logic
- then the robot base pose is written into the simulator by RoboCasa helper code

### 2. Object Repositioning

Object motion is more direct.

For held objects and object-on-object placement, the executor writes object joint
poses directly with MuJoCo sim state updates:

- `env.sim.data.set_joint_qpos(...)`

This happens in:

- `_set_object_pose(...)`
- `_sync_held_object(...)`
- `_place_on_object_center(...)`

For fixture placement, the executor uses the runner's `move_object(...)`, which
computes a fixture-relative target point and then also writes the object's joint
qpos directly.

So yes: object teleportation ultimately overrides MuJoCo sim data directly.

### 3. Fixture / Control State Updates

For doors, drawers, buttons, levers, and knobs, the executor uses a mix of:

- RoboCasa fixture helper methods such as `open_door()` / `close_door()`
- `fixture.set_joint_state(...)`
- a few fixture-specific helper methods for appliances

So this part is partly RoboCasa-native and partly direct state mutation.

## High-Level Flow

The executor lifecycle is:

1. create a live environment through `TrajectoryRunner`
2. inspect the current scene
3. optionally ground a semantic plan template into concrete scene ids
4. execute each tool step
5. save frames, videos, and metadata

At construction time, `SimToolExecutor(...)` creates a
[TrajectoryRunner](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py),
stores `self.runner`, exposes `self.env`, and initializes a small held-object
cache:

- `self._held_objects: dict[int, str]`

That held-object cache is how the executor represents grasp state.

## Core Responsibilities

`SimToolExecutor` does five main jobs:

### 1. Scene Access

It exposes scene and rendering helpers:

- `get_scene_description()`
- `render()`
- `save_scene_frames(...)`

These are thin wrappers over the underlying `TrajectoryRunner`.

### 2. Symbol Validation

Before executing actions, the executor validates ids:

- `_require_fixture(fixture_id)`
- `_require_object(object_id)`

This keeps plans tied to the actual live scene rather than invented strings.

### 3. Semantic Grounding

The executor now supports semantic plan templates.

The template path is:

1. `build_demo_plan_template(...)`
2. `ground_plan_template(...)`
3. `run_tool_plan(...)`

Templates can contain semantic references like:

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

This is what makes the hotdog demo transferable across layouts.

### 4. Tool Execution

Each tool is implemented as a method on `SimToolExecutor`.

Examples:

- `navigate_to_fixture(...)`
- `pick_up_object(...)`
- `place_on_surface(...)`
- `place_on_object(...)`
- `open_hinged_part(...)`
- `press_button(...)`
- `communicate(...)`

The generic entry point is:

- `execute(tool_name, robot_idx=0, **kwargs)`

### 5. Rendering and Saving Outputs

`run_tool_plan(...)` executes a full plan while saving:

- before / after PNGs for every step
- per-camera MP4 videos
- `plan.json`
- `metadata.json`

The saved `plan.json` is the grounded plan that was actually executed, not the
unresolved semantic template.

The default saved cameras now include:

- per-robot agent views
- per-robot wrist / eye-in-hand views
- `room_view`
- `top_view`

## How State Is Represented

The executor mixes symbolic bookkeeping with direct sim state:

### Held Objects

When `pick_up_object(...)` runs:

- the robot is moved near the source fixture
- the object id is recorded in `self._held_objects[robot_idx]`
- `_sync_held_object(...)` snaps the object near the robot end effector

This is a symbolic grasp representation rather than a physical closed-loop
gripper controller.

In other words, the robot is not physically grasping through contact dynamics.
The executor records that a robot is "holding" an object and then keeps the
object snapped near the end effector by updating its pose directly.

### Object Placement

There are two main placement paths:

- fixture placement
- object-on-object placement

For fixture placement, the executor uses `TrajectoryRunner.move_object(...)`.

For object-on-object placement, the executor computes the support object's bbox
top and the placed object's bbox bottom, then writes the object pose directly.

### Fixture / Control State

Articulation and control tools use a mix of:

- fixture helpers such as `open_door()` / `close_door()`
- fixture-specific helpers for microwaves, toasters, kettles, coffee machines
- direct joint value updates through `set_joint_state(...)`

## Grounding Helpers

The grounding logic relies on a few internal helpers:

- `_get_scene_object_location(object_id)`
- `_find_nearest_fixture_for_object(...)`
- `_resolve_object_anchor_fixture(...)`
- `_infer_source_fixture(...)`

These helpers convert object-centric semantic references into concrete fixture
ids for the current scene.

Two common cases are:

- source fixture for an object
- placeable anchor fixture near a support object such as a plate

This means the task definition influences teleportation only indirectly.

The task provides:

- which objects exist
- which fixture roles exist
- the natural-language goal
- the sampled scene instance

But the executor itself decides:

- which concrete fixture id to navigate to
- which anchor fixture to use for support objects
- where to place the robot base relative to that fixture / object
- where to write object poses in sim

So teleportation is not coming from prerecorded task actions. It is coming from:

1. the current sampled task scene
2. semantic grounding logic in the executor
3. RoboCasa placement helpers
4. direct MuJoCo state edits

## Robot Positioning

Robot placement itself is delegated to
[trajectory_runner.py](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py).

The executor uses:

- `runner._move_robot_near_fixture(...)`

For support-object-centric actions such as `place_on_object(...)`, the executor
now resolves the support object's anchor fixture and moves the robot relative to
that object, not just the fixture center.

This is important for cases like a plate near the edge of a dining surface.

## How Individual Tools Work

### Navigation

`navigate_to_fixture(...)`

- validates the fixture id
- teleports the robot near the fixture
- re-syncs any held object

The teleport itself is computed with RoboCasa placement utilities and then
applied by setting the robot base pose in sim.

### Picking

`pick_up_object(...)`

- validates object and source ids
- moves the robot near the source fixture
- marks the object as held
- snaps it near the end effector

So "pick" is not a physical closing-gripper sequence. It is:

1. robot base teleport
2. symbolic held-object assignment
3. direct object pose snapping

### Placement on Fixtures

`place_on_surface(...)`

- moves the robot near the support fixture
- uses the runner to place the object on that fixture
- clears held-object state

The runner computes a fixture-relative target position, validates it against the
fixture region, and then writes the object qpos directly.

### Placement on Objects

`place_on_object(...)`

- resolves a placeable anchor near the support object
- uses the explicit `anchor_fixture_id` from the grounded trajectory when available
- moves the robot near that anchor with the support object as reference
- places the object on the support object's top surface
- clears held-object state

This path is directly geometry-based. The support object's bbox top and the
placed object's bbox bottom are used to compute the final pose, which is then
written into MuJoCo state.

### Receptacles

`place_in_receptacle(...)`

- if the target is a fixture id, it uses fixture placement
- if the target is an object id, it falls back to object-on-object placement

### Controls

`press_button(...)`, `press_lever(...)`, `set_rotary_control(...)`

- move the robot near the target fixture
- update fixture state using helper methods or joint writes

### Coordination

`communicate(...)` and `wait()`

These are semantic actions. They do not substantially alter sim state, but they
are recorded in metadata and useful for multi-agent plans.

## Built-In Demo Plans

The executor supports named built-in demo plans through:

- `build_demo_plan_template(...)`
- `build_demo_plan(...)`

At the moment the main built-in example is:

- `cooperative_hotdog_setup`

This demo is authored as a semantic template and grounded at runtime.

## CLI Usage

The file is runnable as a module:

```bash
python -m robocasa.utils.sim_tool_executor --help
```

It supports:

- saving current-scene frames only
- executing a JSON plan via `--plan`
- executing a built-in demo plan via `--demo-plan`

Example:

```bash
MUJOCO_GL=osmesa python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --demo-plan cooperative_hotdog_setup \
  --output-dir /tmp/hotdog_demo
```

## Relation to `TrajectoryRunner`

The split is:

- `TrajectoryRunner`
  - builds the env
  - renders cameras
  - teleports robots near fixtures
  - places objects on fixtures
  - manages room / top views

- `SimToolExecutor`
  - validates symbolic tool calls
  - grounds semantic refs to scene ids
  - tracks held objects
  - dispatches tool methods
  - saves plan execution outputs

So the executor is a symbolic dispatch layer over the runner.

## Camera Outputs

The rendered outputs come from `TrajectoryRunner`.

The important camera behavior is:

- robot cameras are discovered from the live env camera list
- each robot now includes wrist / `eye_in_hand` views when available
- `room_view` is a free camera, not a fixed MuJoCo named camera
- `top_view` is another free camera derived from the scene footprint

The room view is scene-aware rather than purely layout-static:

- it starts from a layout preset azimuth / elevation
- it re-centers on the current scene footprint
- it sets distance from the current scene extent and camera FOV

So different layouts, styles, and sampled scenes can change the final room-view
framing.

## Is It Using RoboCasa Tasks Or Overriding MuJoCo State?

Both, but in different ways.

### It uses RoboCasa tasks for:

- scene construction
- object sampling
- fixture sampling
- task language
- fixture references such as fridge / cabinet / dining table roles

### It uses RoboCasa helper logic for:

- robot base placement near fixtures
- some fixture interactions such as opening doors
- camera setup and scene rendering

### It overrides MuJoCo sim state for:

- object pose teleportation via `set_joint_qpos(...)`
- held-object snapping
- object-on-object placement
- some fixture joint / control state changes

So the executor should be understood as:

- RoboCasa task semantics and scene generation on the front end
- pragmatic MuJoCo state editing plus RoboCasa placement helpers on the backend

## Limitations

The executor is intentionally pragmatic and therefore limited:

- no low-level physical grasp controller
- no dataset-faithful action replay
- object locations in the scene description are still heuristic in some cases
- support-fixture grounding can still fail on ambiguous scenes
- precise spatial relations such as `next_to` still need richer lowering logic

This makes the executor useful for symbolic planning, debugging, and visual
plan rollout, but it should not be mistaken for a physically realistic motion
policy.
