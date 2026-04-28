# Sim Tool Executor

This document explains the current executor implementation centered on
[sim_tool_executor.py](../../robocasa/utils/sim_tool_executor.py).

It is narrower than [sim_tool_layer.md](./sim_tool_layer.md). That document
describes the planner-facing tool surface. This page focuses on the concrete
runtime that executes those tools inside a live RoboCasa scene.

Related notes:

- [sim_tool_executor_phase_2.md](./sim_tool_executor_phase_2.md)
- [sim_tool_executor_phase_3.md](./sim_tool_executor_phase_3.md)
- [continuous_vs_grid_placement.md](./continuous_vs_grid_placement.md)

## Purpose

`SimToolExecutor` is a symbolic execution layer over a live RoboCasa / MuJoCo
environment.

It bridges:

- validated tool calls such as `pick_up_object(...)` and `place_on_surface(...)`
- a concrete sampled kitchen scene

It is not:

- a low-level motion controller
- dataset-faithful replay
- a learned manipulation policy

The implementation remains pragmatic: robot bases are teleported to working
poses, objects are often moved with direct state updates, and fixture controls
are toggled through helper methods or direct joint edits.

## Current Module Layout

`SimToolExecutor` is still the stable import surface, but its implementation is
now split across smaller modules:

- [sim_tool_executor.py](../../robocasa/utils/sim_tool_executor.py): class definition, tool dispatch, most tool methods
- [sim_tool_executor_planning.py](../../robocasa/utils/sim_tool_executor_planning.py): demo-plan templates and semantic grounding
- [sim_tool_executor_execution.py](../../robocasa/utils/sim_tool_executor_execution.py): `run_tool_plan(...)`
- [sim_tool_executor_inspection.py](../../robocasa/utils/sim_tool_executor_inspection.py): rendering, map saving, and scene-inspection helpers
- [sim_tool_executor_state_loading.py](../../robocasa/utils/sim_tool_executor_state_loading.py): normalized initial-state loading
- [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py): environment construction, camera management, robot approach grounding, and fixture/object placement helpers

## Execution Model

There are four main state-update paths.

### 1. Robot Base Positioning

Robot motion is delegated to
[trajectory_runner.py](../../robocasa/utils/trajectory_runner.py), primarily
through `runner._move_robot_near_fixture(...)`.

The current runner uses:

- shared placement geometry helpers in [placement.py](../../robocasa/utils/placement.py)
- a grid backend in [occupancy_grid.py](../../robocasa/utils/occupancy_grid.py)

As of April 27, 2026, the older continuous robot-base placement backend has
been removed. Robot approach grounding is grid-only.

For enclosing fixtures such as fridges and cabinets, the runner and executor
also use front-readiness checks so a robot is only treated as "already there"
when it is actually in a usable front working pose.

### 2. Object Positioning

Object motion is a mix of:

- runner-side fixture placement
- direct object pose writes
- contained-object transport when receptacles move

Fixture placement for `place_on_surface(...)`, fixture-target
`place_in_receptacle(...)`, `place_next_to(...)`, and generic `place_under(...)`
routes through the runner's collision-aware sampler.

Direct placement still exists for object-on-object stacking and a few
fixture-specific placement paths such as dispenser sites.

### 3. Fixture And Control State

Fixture articulation and appliance controls use a mix of:

- fixture helper methods such as `open_door()` / `close_door()`
- direct joint writes
- fixture-specific helpers for appliances such as microwaves, coffee machines,
  kettles, and toasters

### 4. Initial-State Loading

When a trajectory carries a normalized `initial_state`, the executor applies it
through [sim_tool_executor_state_loading.py](../../robocasa/utils/sim_tool_executor_state_loading.py).

That pass can:

- set fixture door/drawer/joint state
- set machine start state
- place or preserve objects
- assign held objects to agents
- rebuild executor-side support and held-object bookkeeping

## High-Level Flow

The executor lifecycle is:

1. construct a live environment through `TrajectoryRunner`
2. inspect and cache the scene description
3. optionally ground a semantic plan template into concrete ids
4. execute each tool step
5. save images and metadata

At construction time, `SimToolExecutor(...)` creates a
[TrajectoryRunner](../../robocasa/utils/trajectory_runner.py), stores it as
`self.runner`, exposes `self.env`, and initializes symbolic state such as
`self._held_objects`.

## Core Responsibilities

### Scene Access

Scene and rendering helpers include:

- `get_scene_description()`
- `render()`
- `save_scene_frames(...)`
- `save_placement_map(...)`

### Symbol Validation

The executor validates fixture and object ids before acting on them.

### Semantic Grounding

Semantic template support lives in the planning mixin through:

- `build_demo_plan_template(...)`
- `ground_plan_template(...)`
- `build_demo_plan(...)`

### Tool Dispatch

Each tool remains a method on `SimToolExecutor`. The main dispatch entry point
is `execute(tool_name, robot_idx=0, **kwargs)`.

### Output Saving

`run_tool_plan(...)` executes a plan and writes:

- `plan.json`
- `metadata.json`
- any images requested by `get_image`

## Tool Behavior Overview

Main tool groups:

- navigation: `navigate_to_fixture(...)`, `give_space(...)`
- picking: `pick_up_object(...)`
- placement: `place_on_surface(...)`, `place_in_receptacle(...)`, `place_on_object(...)`, `place_next_to(...)`, `place_under(...)`
- articulation and controls: `open_hinged_part(...)`, `open_sliding_part(...)`, `press_button(...)`, `set_rotary_control(...)`
- coordination: `communicate(...)`, `wait()`

The executor keeps these tool names stable even when several of them share the
same lower-level runner utilities.

## Camera Outputs

Rendered outputs come from `TrajectoryRunner`.

Important camera behavior:

- robot cameras are discovered from the live environment
- `room_view` is a derived free camera
- `top_view` is a second derived free camera
- map images are occupancy-grid visualizations

Images are rendered only when a plan or trajectory contains `get_image` steps.

## Limitations

The executor is intentionally pragmatic:

- no low-level physical grasp controller
- still teleport-based for navigation and manipulation
- object-on-object stacking is geometry-based, not contact-stabilized
- some spatial relations are still lowered heuristically
- ambiguous scenes can still require conservative fallback behavior

That makes it useful for symbolic planning, debugging, and rendered rollout,
but not a realistic motion policy.
