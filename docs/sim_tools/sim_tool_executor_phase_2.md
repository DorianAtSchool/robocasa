# SimToolExecutor Phase 2

This page records the lasting outcome of Phase 2.

The original Phase 2 work introduced robot-base grounding for teleport-based
execution. During development, that briefly included both a grid backend and an
experimental continuous backend. As of April 27, 2026, the continuous backend
has been removed and the surviving Phase 2 behavior is grid-only.

## Lasting Deliverables

Phase 2 still defines the current approach-grounding layer:

- shared face/front-alignment helpers in [placement.py](../../robocasa/utils/placement.py)
- occupancy-grid robot placement in [occupancy_grid.py](../../robocasa/utils/occupancy_grid.py)
- runner-side integration in [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py)
- executor-side front-readiness checks and retries in [sim_tool_executor.py](../../robocasa/utils/sim_tool_executor.py)
- placement visualization in [placement_map.py](../../robocasa/utils/placement_map.py)

## Semantics That Survived

The important Phase 2 semantics are unchanged:

- robots still teleport to working poses
- enclosing fixtures use front-only approach semantics
- front working bands prevent edge-hugging side/corner placements
- multi-robot clearance is enforced during placement
- sweep/debug tooling is used to catch layout-specific regressions

## What Changed Since The Original Phase

The original Phase 2 notes are no longer literally current because:

- `ContinuousPlacement` was removed from `placement.py`
- the runner no longer switches between grid and continuous backends
- `draw_continuous_map(...)` was removed from `placement_map.py`
- the old `tests/test_placement.py` coverage was deleted as obsolete

The final current interpretation of Phase 2 is:

> a grid-based robot approach-grounding subsystem with front-specific semantics
> for teleport-based execution

## Validation

Relevant remaining coverage lives primarily in:

- [tests/test_occupancy_grid.py](../../tests/test_occupancy_grid.py)
- [tests/test_sim_tool_executor.py](../../tests/test_sim_tool_executor.py)
- [tests/test_placement_sweep.py](../../tests/test_placement_sweep.py)
