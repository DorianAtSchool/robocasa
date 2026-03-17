# SimToolExecutor: Phased Improvement Plan

This document describes the phased plan to make the `SimToolExecutor` scalable across arbitrary tasks, not just `HotDogSetup`.

## Problem Summary

The current executor works well for `HotDogSetup` across layout / style / seed variation, but breaks on tasks like `PrepareSandwichStation` because:

- Placement semantics only express "on this fixture", not spatial relations like "near the toaster oven"
- Robot base placement fails on corner fixtures and tight geometries
- Object placement uses random jitter inside fixture bbox with no collision awareness
- Moving a receptacle does not carry its contents

## Phase 1: General Spatial Relation Tools

**Goal:** Replace task-specific placement logic with general-purpose spatial relation tools that a VLM can naturally output.

New tools added to the tool surface:

| Tool | Description | Args |
|------|-------------|------|
| `place_next_to` | Place object adjacent to another object on the same surface | `object_id`, `reference_object_id` |
| `place_under` | Place object directly beneath a fixture (generalized from `place_under_dispenser`) | `object_id`, `reference_fixture_id` |

These are separate tool calls (not args on `place_on_object`) because they express spatial relations that require runtime resolution — the executor computes the actual position from reference object/fixture poses, which the VLM planner cannot know.

Note: `place_near` was considered but removed — it is equivalent to `place_next_to` from the executor's perspective. Fixture-level placement ("on the counter near the toaster") is the VLM planner's responsibility to resolve into the correct `place_on_surface(object, counter_id)` call, since the anchor pattern already exists.

Existing tools (`place_on_object`, `place_on_surface`, `place_in_receptacle`) remain unchanged.

**Files modified:**
- `robocasa/utils/sim_tool_specs.py` — new tool specs
- `robocasa/utils/sim_tool_executor.py` — new executor methods + spatial resolution helpers

## Phase 2: 2D Occupancy Grid for Robot Approach Grounding

**Goal:** Make robot base placement robust across all layouts, including corners, islands, and tight kitchens.

Build a 2D occupancy grid (~5cm cells) from fixture bounding boxes at scene init:
- Query grid for free cells within reach of target fixture
- Rank candidate cells by reachability and approach angle
- Handle corner fixtures naturally — fewer free cells means the robot approaches from the open side
- Handle island fixtures — robot can approach from either side

This grid replaces the current fixture-relative fallback in `_move_robot_near_fixture()` and becomes the foundation for the Phase 5 navigation controller.

**Files modified:**
- New: `robocasa/utils/occupancy_grid.py`
- `robocasa/utils/trajectory_runner.py` — use grid for base placement

## Phase 3: Collision-Aware Object Placement

**Goal:** Place objects on surfaces without colliding with other objects or appliances.

Given a target surface region (from Phase 1 spatial resolution or direct fixture placement):
1. Query positions of all objects currently on that surface
2. Query positions of appliances attached to or near that surface
3. Sample candidate placement poses
4. Reject candidates that overlap existing object/appliance bounding boxes
5. Pick the best valid candidate (closest to the spatial constraint target)

Replaces the current random jitter + "inside fixture bbox" validation.

Note: `place_under` for non-dispenser fixtures (the generic surface-projection path) currently places with no awareness of existing objects on the target surface. This phase should cover that case — after projecting XY under the reference fixture, the collision-aware sampler should verify the candidate doesn't overlap objects already on the surface.

**Files modified:**
- `robocasa/utils/trajectory_runner.py` — `_compute_object_target_pos` and `move_object`
- `robocasa/utils/sim_tool_executor.py` — pass collision context to placement

## Phase 4: Receptacle Carry Semantics

**Goal:** When a receptacle (bowl, basket) is moved, its contents move with it.

When `pick_up_object` or any placement tool moves a receptacle:
1. Detect contained objects via vertical overlap + bounding box intersection
2. Compute relative offsets of contents to the receptacle
3. Move all contained objects rigidly with the receptacle
4. Update containment state on placement

**Files modified:**
- `robocasa/utils/sim_tool_executor.py` — containment detection + rigid transport in `_set_object_pose`, `pick_up_object`, and placement methods

## Phase 5: Demo-Informed Navigation and Manipulation

**Goal:** Use existing demo data to design real controllers that replace teleportation.

Two sub-efforts:

### 5a: Navigation controller
- Use the Phase 2 occupancy grid for path planning (A* or similar)
- Extract navigation waypoints from demo trajectories as priors
- Replace `_move_robot_near_fixture` teleportation with planned navigation

### 5b: Manipulation state machines
- Analyze demo trajectories to extract manipulation subskill structure (approach, grasp, lift, transport, place)
- Design parameterized state machines for each tool (pick, place, open, close, etc.)
- Replace teleport manipulation with state machine execution

**Files modified:**
- New: `robocasa/utils/navigation_controller.py`
- New: `robocasa/utils/manipulation_state_machines.py`
- `robocasa/utils/sim_tool_executor.py` — swap teleport backends for controller backends

## Dependency Graph

```
Phase 1 (spatial tools) ──────────────────────┐
                                               ├── Phase 3 (collision-aware placement)
Phase 2 (occupancy grid) ─────────────────────┤
                                               └── Phase 5a (navigation controller)
Phase 4 (receptacle carry) ── independent

Phase 5b (manipulation state machines) ── independent, uses demo data
```

Phases 1 and 2 can proceed in parallel. Phase 3 builds on both. Phase 4 is independent. Phase 5 builds on Phase 2 and demo analysis.
