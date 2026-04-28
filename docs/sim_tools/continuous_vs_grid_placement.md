# Continuous Placement vs Grid Placement

This page is now a historical note.

As of April 27, 2026, `TrajectoryRunner` uses only the occupancy-grid backend in
[occupancy_grid.py](../../robocasa/utils/occupancy_grid.py) for robot-base
placement.

## Current State

The old split was:

- grid placement via `OccupancyGrid`
- continuous placement via `ContinuousPlacement`

The current codebase keeps only the grid path.

## What Was Removed

The refactor removed the continuous robot-base placement path and its related
debug surface:

- `ContinuousPlacement` from [placement.py](../../robocasa/utils/placement.py)
- `draw_continuous_map(...)` from [placement_map.py](../../robocasa/utils/placement_map.py)
- the continuous-only runner fallback code
- the obsolete unit test [tests/test_placement.py](../../tests/test_placement.py)

CLI and test entry points now only accept:

```text
--placement grid
```

## What Still Exists

The removal did not change the higher-level semantics that were built during the
original Phase 2 work:

- face-based candidate sampling around fixtures
- front-only approach semantics for enclosing fixtures
- multi-robot clearance checks
- occupancy-grid placement-map rendering

Those behaviors now all route through:

- [placement.py](../../robocasa/utils/placement.py) for shared geometry helpers
- [occupancy_grid.py](../../robocasa/utils/occupancy_grid.py) for robot-base grounding
- [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py) for integration into execution

## Why This File Still Exists

Older Phase 2 notes and earlier PRs referred to "grid vs continuous" as an
active choice. That is no longer true, but this file remains so those references
have a clear explanation instead of silently going stale.

If you still see `placement="continuous"` in old notes, scripts, or outputs,
read it as historical context rather than current behavior.

## Practical Takeaway

Today there is one active robot-base placement backend:

- grid placement, with front-approach semantics and occupancy-grid diagnostics

If robot placement regresses, the files to inspect are:

- [placement.py](../../robocasa/utils/placement.py)
- [occupancy_grid.py](../../robocasa/utils/occupancy_grid.py)
- [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py)
- [placement_map.py](../../robocasa/utils/placement_map.py)
