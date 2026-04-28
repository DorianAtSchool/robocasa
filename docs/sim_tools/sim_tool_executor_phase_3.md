# SimToolExecutor Phase 3

This page records the lasting outcome of Phase 3.

Phase 3 replaced the old "sample one reset-region center and retry with jitter"
fixture-placement path with a deterministic, collision-aware sampler used by the
executor's surface-style placement tools.

## Lasting Deliverables

The Phase 3 behavior that still defines the current system is:

- collision-aware fixture placement in [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py)
- executor routing so surface-like tools use that shared sampler
- support-location tracking updates after direct placement calls
- graceful fallback when geometry helpers cannot evaluate a case cleanly

The executor implementation is now split across smaller modules, but the public
entry point remains [sim_tool_executor.py](../../robocasa/utils/sim_tool_executor.py).

## Current Behavior

For fixture placement, the runner now:

- samples candidate poses from reset regions
- shrinks those regions by the placed object's footprint
- scores candidates against preferred semantic targets when provided
- rejects collisions against objects and nearby fixture geometry
- falls back conservatively if geometry evaluation hits an edge case

Executor-side tools that depend on this path include:

- `place_on_surface(...)`
- fixture-target `place_in_receptacle(...)`
- `place_next_to(...)`
- generic `place_under(...)`

## End State

The enduring result of Phase 3 is:

> a shared collision-aware fixture placement subsystem used by the executor's
> placement tools

Relevant files:

- [trajectory_runner.py](../../robocasa/utils/trajectory_runner.py)
- [sim_tool_executor.py](../../robocasa/utils/sim_tool_executor.py)
- [tests/test_sim_tool_executor.py](../../tests/test_sim_tool_executor.py)
