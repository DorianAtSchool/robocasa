# SimToolExecutor Phase 3: What Was Actually Implemented

This document records what Phase 3 turned into in practice.

The original Phase 3 plan in [sim_tool_executor_phases.md](/Users/dorian/Documents/robocasa/docs/sim_tool_executor_phases.md) described collision-aware object placement on surfaces. The final implementation kept that goal, but it landed as a shared fixture-placement sampler in the runner plus executor-side routing changes so the high-level placement tools actually use it.

## Summary

Phase 3 replaced the old "sample one reset region center, add random jitter if needed" placement path with a deterministic, collision-aware sampler for fixture placement.

The final behavior is:

- sample candidate poses from fixture reset regions
- shrink each region by the placed object's footprint so candidates stay on the usable surface
- score candidates against a preferred XY target when the tool expresses a spatial relation
- reject candidates that intersect existing scene objects
- reject candidates that intersect nearby fixture / appliance geometry
- fall back to legacy fixture-region sampling if collision-aware target computation hits a geometry-helper edge case
- keep object-to-fixture grounding up to date after direct placement calls

This phase applies to fixture placement, not true physics-based packing or stacking.

## What Was Built

### 1. Collision-Aware Runner Placement

[trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py) now owns the core object-placement search.

Implemented pieces:

- fixture-local/world coordinate conversion helpers
- bbox-based object footprint metadata
- reset-region enumeration with minimum-size filtering
- lattice sampling inside each valid region
- preferred-target projection so tools like `place_next_to` and `place_under` can bias toward a semantic target without requiring the exact point to be free
- collision rejection against both scene objects and nearby fixture geometry
- cached object-location updates when objects move between fixtures

The runner now chooses a best valid candidate instead of retrying random local jitter.

### 2. Executor Routing Through The Shared Sampler

[sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py) was updated so surface-like tools use the runner's collision-aware target instead of bypassing it.

Changed behaviors:

- `place_on_surface` now precomputes a collision-aware landing pose and uses it consistently for both robot approach and object placement
- `place_in_receptacle` does the same when the receptacle target is a fixture
- `place_next_to` now resolves one or two preferred adjacent targets, asks the runner for the nearest collision-free pose on the support fixture, and places there
- generic `place_under` now projects toward the reference fixture but still runs through the support-surface sampler instead of forcing an exact XY pose
- executor-side placement target resolution now has a safe fallback path so a geometry exception does not turn a valid surface placement into a tool failure

This is what made Phase 3 a real executor-facing behavior change instead of only a runner utility.

### 3. Object Location Tracking Fixes

Before this phase, several placement methods updated simulator state directly but still resolved support fixtures from the original scene description.

That mismatch was fixed by:

- introducing a runner-side object-location cache update helper
- teaching [sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py) to consult the live runner cache before falling back to the original scene description
- updating placement methods to refresh object grounding after direct object-on-object or dispenser-style placements

This matters because spatial-relation tools such as `place_next_to` depend on knowing the object's current support surface.

### 4. Phase 4 Carry Logic Was Confirmed And Tightened

While implementing Phase 3, it became clear that Phase 4 was mostly present already but not fully normalized across all direct-placement paths.

As part of this pass:

- `_set_object_pose(...)` now carries contained objects rigidly with the receptacle/object being moved
- placement methods that still use direct pose updates now also refresh fixture grounding for moved contents
- runner-side `move_object(...)` continues to carry contained objects and now updates cached support locations for both the receptacle and its contents

So Phase 4 should now be treated as implemented, not merely partially implied by earlier code.

### 5. Post-Implementation Hardening

The first Phase 3 / 4 pass exposed two follow-up issues that were fixed immediately afterward.

Those fixes are part of the shipped behavior and should be considered part of the final Phase 3 state:

- front-readiness for enclosing fixtures was tightened so a robot is only treated as "already at the fridge/cabinet" if it is both front-aligned and at a reasonable working standoff, not merely somewhere on the correct side
- collision checks now fail soft when a geometry helper cannot evaluate one obstacle pair, and the executor falls back to legacy fixture-region sampling if collision-aware target computation itself raises

This hardening was necessary to prevent regressions such as skipped fridge teleports and `place_on_surface` failures on otherwise valid counters / islands.

## Deviations From The Original Plan

Phase 3 stayed close to the original goal, but there were still a few practical differences.

What changed:

- the implementation is reset-region based rather than arbitrary free-space search over an entire fixture bbox
- collision checking includes nearby fixture geometry, not just objects already on the support surface
- the final shipped behavior includes a conservative fallback path, not collision-aware placement as a hard requirement in every geometry edge case
- support-fixture grounding fixes became part of the phase because spatial-relation placement was unreliable without them
- some dispenser/object-on-object placement paths still use explicit target poses, but they now participate in containment transport and support-location updates

What did not change:

- task-level demo semantics were not changed by this phase
- in the current `PrepareSandwichStation` demo template, the ingredient bowl and baguette are both placed on the counter near the toaster oven; the baguette is not intentionally placed inside the bowl

In practice, Phase 3 became:

> a shared collision-aware fixture placement subsystem used by the executor's surface placement tools

rather than:

> only a replacement for random jitter inside `move_object`

## Files Added Or Materially Changed

Core implementation:

- [trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py)
- [sim_tool_executor.py](/Users/dorian/Documents/robocasa/robocasa/utils/sim_tool_executor.py)

Validation:

- [test_sim_tool_executor.py](/Users/dorian/Documents/robocasa/tests/test_sim_tool_executor.py)

Documentation:

- [sim_tool_executor_phase_3.md](/Users/dorian/Documents/robocasa/docs/sim_tool_executor_phase_3.md)

## End State

At the end of Phase 3, fixture placement is no longer a best-effort random jitter process.

The executor now has a shared placement path that:

- respects fixture reset regions
- avoids occupied surface space
- honors semantic spatial targets when possible
- degrades gracefully when collision geometry helpers are incomplete or brittle
- keeps object/support grounding in sync after placement

That is the baseline the remaining phases should assume.
