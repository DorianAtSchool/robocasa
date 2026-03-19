# SimToolExecutor Phase 2: What Was Actually Implemented

This document records what Phase 2 turned into in practice.

The original Phase 2 plan in [sim_tool_executor_phases.md](./sim_tool_executor_phases.md) described a 2D occupancy grid for robot approach grounding. The final implementation was broader: it became a full robot-base placement subsystem with two interchangeable backends, shared front-approach semantics for enclosing fixtures, and sweep-based validation tooling.

## Summary

Phase 2 delivered a robust teleport-based robot approach layer for `SimToolExecutor`.

Instead of only adding a grid planner, the implementation introduced:

- a shared placement abstraction for robot base grounding
- a grid backend for coarse but layout-aware placement
- a continuous backend for exact face sampling without grid quantization
- front-approach semantics for enclosing / interactive fixtures such as fridges, cabinets, drawers, microwaves, ovens, and dishwashers
- executor-side readiness checks and retry logic so interaction tools use the same placement semantics
- placement-map and sweep tooling to debug layout/style failures across tasks

This phase improved placement robustness, but it did **not** implement true mobile navigation. Robots still teleport to working poses.

## What Was Built

### 1. Shared Placement Module

New shared placement logic lives in [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py).

It provides:

- fixture AABB extraction and obstacle filtering
- face ordering and face sampling utilities
- a continuous placement backend (`ContinuousPlacement`)
- shared front-alignment helpers for enclosing fixtures
- front working-band filtering so robots do not approach interactive fixtures from edge-hugging side/corner poses

The continuous backend samples candidate poses at configurable standoffs around fixture faces and rejects poses that:

- collide with fixture AABBs
- fall outside room bounds
- land in enclosed pockets
- are too close to another robot

### 2. Occupancy Grid Backend

New grid-based placement logic lives in [occupancy_grid.py](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py).

It builds a 2D occupancy grid from fixture footprints and uses it to ground robot base poses.

Implemented behaviors:

- rasterize ground-level fixtures into occupied cells
- save a pre-flood-fill fixture-only grid
- flood-fill room interior to seal unreachable pockets
- sample face-adjacent candidate cells around fixtures
- support front-only placement for interactive fixtures
- allow front poses to use the fixture-only grid so narrow but physically valid front gaps are not lost to reachability sealing

The final grid cell size used in practice was coarser than the original 5 cm sketch in the plan. The implementation settled on a configurable grid, commonly used at 10 cm, with continuous placement available as a higher-fidelity alternative.

### 3. TrajectoryRunner Placement Integration

[trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py) now owns the robot-base placement flow.

Implemented behaviors:

- initialize both placement backends at scene startup
- support selectable placement mode: `grid` or `continuous`
- use shared `_move_robot_near_fixture(...)` logic across tools
- fall back between backends when the primary placement mode fails
- validate placed poses before committing them
- resolve front working targets from fixture handle / door geometry when available
- correct robot-robot overlap by re-placing the moved robot
- expose `give_space(...)` to move blocking robots away from an active fixture

This is the main point where Phase 2 stopped being “just an occupancy grid” and became a general approach-grounding layer.

### 4. Enclosing Fixture Front Semantics

One of the biggest Phase 2 additions was the semantic distinction between:

- open support surfaces such as counters and islands
- enclosing / interactive fixtures such as fridges, cabinets, drawers, microwaves, ovens, and dishwashers

For enclosing fixtures, the final behavior is:

- use front-required placement
- ignore contained-object XY for lateral approach bias
- infer the active front from handle / door target geometry when possible
- project the working target onto the correct front face
- reject side-face and edge-hugging corner approaches with a front working band

This was necessary to make “pick up object from fridge/cabinet” mean “approach the fixture correctly” rather than “stand near the item’s XY projection.”

### 5. SimToolExecutor Integration

[sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py) was updated so the tool layer actually uses the new grounding semantics.

Implemented behaviors:

- classify enclosing fixtures as front-approach fixtures
- use front-readiness checks instead of a coarse “near fixture center” heuristic
- retry front placement after clearing nearby teammate blockers
- recenter off-axis robots before front interactions when needed
- route object-anchor movement through the same enclosing-fixture approach semantics

This made Phase 2 a real executor-facing behavior change, not just a runner-side utility.

### 6. Placement Visualization and Sweep Infrastructure

Phase 2 also added the tooling used to debug and validate placement across tasks:

- [placement_map.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement_map.py) for grid / continuous placement visualization
- [test_placement_sweep.py](/Users/dorian/Documents/robocasa/tests/test_placement_sweep.py) for task-layout-style-seed sweeps
- unit coverage in [test_placement.py](/Users/dorian/Documents/robocasa/tests/test_placement.py), [test_occupancy_grid.py](/Users/dorian/Documents/robocasa/tests/test_occupancy_grid.py), and [test_sim_tool_executor.py](/Users/dorian/Documents/robocasa/tests/test_sim_tool_executor.py)

These became the primary acceptance mechanism for Phase 2, especially for catching layout-specific regressions in fridges, cabinets, islands, and corners.

## Deviations From The Original Plan

Phase 2 did not land exactly as originally described.

What changed:

- it did not replace teleportation with navigation
- it did not become the final “foundation for the Phase 5 navigation controller” yet, though it can still serve that role later
- it introduced a continuous placement backend in addition to the grid backend
- it pulled front-approach semantics and placement retries into the executor, because runner-only grounding was not enough

In practice, Phase 2 became:

> a robust robot approach-grounding layer for teleport-based execution

rather than:

> only a 2D occupancy grid implementation

## Files Added Or Materially Changed

Core implementation:

- [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)
- [occupancy_grid.py](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py)
- [trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py)
- [sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py)
- [placement_map.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement_map.py)

Validation:

- [test_placement.py](/Users/dorian/Documents/robocasa/tests/test_placement.py)
- [test_occupancy_grid.py](/Users/dorian/Documents/robocasa/tests/test_occupancy_grid.py)
- [test_sim_tool_executor.py](/Users/dorian/Documents/robocasa/tests/test_sim_tool_executor.py)
- [test_placement_sweep.py](/Users/dorian/Documents/robocasa/tests/test_placement_sweep.py)

## End State

At the end of Phase 2, robot base grounding was no longer a brittle fixture-relative fallback. It became a tested subsystem with:

- layout-aware placement
- front-specific semantics for enclosing fixtures
- interchangeable grid and continuous backends
- multi-robot clearance handling
- map and sweep tooling for regression detection

That is the implementation baseline the next phases should assume.
