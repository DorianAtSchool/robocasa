# Continuous Placement vs Grid Placement

This document explains the two robot-base placement backends used by
[trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py):

- [OccupancyGrid](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py)
- [ContinuousPlacement](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)

This comparison is specifically about **robot base grounding** for teleport
placement near fixtures. It is not the same as Phase 3 object placement, which
uses a different collision-aware sampler for placing movable objects onto
fixtures.

## Summary

Both backends answer the same question:

> Where should the robot stand to work at this fixture?

They share the same high-level semantics:

- sample candidate poses around fixture faces
- support front-only behavior for enclosing fixtures such as fridges and cabinets
- honor multi-robot separation
- bias surface placement toward a reference object or reference XY when one exists

But they model free space differently:

- **Grid placement** reasons over rasterized cells and flood-filled reachability
- **Continuous placement** reasons over exact XY samples and circle-vs-AABB collision checks

Because of that, they can disagree even in the same layout.

## Shared Semantics

The two backends are not independent designs. They deliberately share a common
semantic layer from [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py).

Shared ideas include:

- fixture AABB extraction
- face ordering
- front-face inference from handle / door target geometry when available
- front working-band filtering
- fixed-standoff candidate sampling along fixture faces
- minimum robot-robot separation

That shared layer is why both backends generally agree on:

- which side of a fridge is the front
- how wide the valid front working band should be
- how reference targets should bias surface placement

## Grid Placement

Grid placement lives in
[occupancy_grid.py](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py).

### How It Represents Space

It rasterizes the kitchen into a 2D occupancy map.

At initialization it:

- gathers fixture footprints
- marks ground-level fixtures and walls as occupied cells
- stores a pre-flood-fill fixture-only grid
- flood-fills from the room interior
- marks unreached free cells as occupied

That flood-fill step is important. It seals off pockets that are not reachable
through connected free cells even if they are not directly inside a fixture.

### Candidate Generation

For a target fixture, grid placement:

- samples poses along each fixture face at a fixed standoff
- maps each sampled world pose to a grid cell
- rejects poses whose cells are occupied
- rejects poses whose cells are already used by another robot
- rejects poses that violate physical robot-robot separation

For front-required interactions, it:

- uses only the inferred front face
- checks candidates against the fixture-only grid rather than the flood-filled grid
- keeps only candidates inside the front working band
- gradually relaxes lateral tolerance along that same front face

That fixture-only exception exists because the robot teleports. A narrow front
gap can be physically usable for teleport placement even if grid reachability
would have sealed it as an enclosed pocket.

### Strengths

- fast and stable
- naturally encodes room reachability
- works well for tight layouts with many walls and counters
- robust for debugging because occupied vs free regions are easy to visualize

### Weaknesses

- quantization error: a small fixture overlap can block an entire cell
- coarse discretization can reject poses that are physically valid
- flood-fill can be too conservative in narrow teleport-valid gaps if the
  fixture-only exception is not used
- cell occupancy is only an approximation of physical base collision

## Continuous Placement

Continuous placement lives in
[placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)
inside `ContinuousPlacement`.

### How It Represents Space

It does not rasterize the room.

Instead it keeps:

- obstacle AABBs from ground-level fixtures / walls
- room bounds
- a robot radius approximation for circle-vs-AABB collision

Each candidate is evaluated directly in world coordinates.

### Candidate Generation

For a target fixture, continuous placement:

- samples exact XY poses along fixture faces at the configured standoff
- rejects poses that collide with obstacle AABBs
- rejects poses outside room bounds
- rejects poses that appear enclosed by nearby obstacles
- rejects poses too close to other robots

For front-required interactions, it uses the same front-face and working-band
semantics as grid placement:

- only the inferred front face is considered
- the exact projected front target is inserted as an explicit candidate
- candidates are filtered to stay within the front working band
- lateral tolerance is expanded gradually only along that front face

### Strengths

- no grid quantization
- exact face sampling
- better fidelity for narrow valid poses
- easier to keep aligned with geometric front targets

### Weaknesses

- depends on conservative AABBs and radius approximations
- "enclosed space" detection is heuristic, not true navigation reachability
- can accept poses that are geometrically valid but operationally awkward if
  higher-level readiness checks are too weak
- more sensitive to geometry-helper brittleness than the grid backend

## How The Runner Uses Both

[trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py)
initializes both backends regardless of the selected placement mode.

The selected mode only chooses the **primary** backend:

- `placement="grid"` means try `OccupancyGrid` first
- `placement="continuous"` means try `ContinuousPlacement` first

The runner then applies additional logic:

### 1. Cross-Validation

Continuous results are validated against both:

- continuous standability / obstacle checks
- grid reachability or fixture-only occupancy checks

This matters because the continuous backend does not model global reachability
the way the grid flood-fill does.

### 2. Fallback To The Other Backend

If the primary backend fails or produces a rejected pose, the runner tries the
other backend.

So "grid mode" does not mean the continuous code is unused, and "continuous
mode" does not mean grid reasoning is irrelevant.

### 3. Last-Resort Heuristic Search

If both structured backends fail, the runner can still try a last-resort set of
simple directional standoff poses.

That fallback is more permissive for surfaces than for front-required enclosing
fixtures.

### 4. Overlap Correction

After placement, if the two robots are still too close, the runner can re-place
the moved robot beside the other one using the same backend family and the same
front/surface semantics.

## Why They Can Disagree

Even with shared semantics, grid and continuous can produce different answers.

The main reasons are:

### 1. Quantization vs Exact Geometry

Grid placement collapses the world to cells. Continuous placement does not.

A pose that is free in continuous geometry may still land in an occupied cell in
the grid backend.

### 2. Reachability vs Local Collision

Grid placement encodes reachability through flood-fill.

Continuous placement encodes only local collision and enclosed-space heuristics.

That means continuous can propose a pose that looks locally valid while the grid
backend rejects it as effectively sealed off.

### 3. Obstacle Conservatism

Continuous placement depends heavily on conservative obstacle AABBs and a robot
radius approximation.

Grid placement depends on how those same fixtures rasterize into cells.

These are different approximations, so the same physical gap may look:

- too narrow in continuous because of radius-vs-AABB conservatism
- too narrow in grid because of cell blocking
- valid in one backend but not the other

### 4. Task-Specific Reference Targets

The layout alone does not determine the final pose.

The same fixture in the same layout can get different target bias depending on:

- `require_front`
- `ref_object_id`
- `ref_pos_override`
- inferred handle / door target
- multi-robot occupancy at that moment in the plan

So the same kitchen can yield different placements across tasks or even across
different steps of the same task.

### 5. Multi-Robot State

Both backends avoid other robots, but they do so differently:

- grid excludes robot cells and also checks physical distance
- continuous checks only physical distance against exact candidate poses

This can shift which candidate survives in cluttered two-robot interactions.

## Enclosing Fixtures

For fridges, cabinets, drawers, microwaves, ovens, and similar fixtures, the
important rule is:

> placement is about the enclosing fixture's front working line, not the
> contained object's raw XY

Both backends now support this by:

- inferring the front face from handle / door target geometry when available
- using front-only sampling for `require_front=True`
- enforcing a front working band
- expanding lateral tolerance only along that front face

The executor adds another layer on top of this:

- a robot is only considered "already near the fixture" if it is not just on
  the correct side, but also within a reasonable front working standoff

That readiness check is what prevents skipped fridge teleports in cases where a
robot is merely nearby, not actually in a usable front pose.

## Which One To Trust

There is no universal winner.

Use **grid-first** when:

- layout robustness matters most
- you want reachability-aware placement
- you are debugging walls, corners, and blocked pockets

Use **continuous-first** when:

- exact geometric alignment matters most
- cell quantization is causing false rejections
- you want smoother face-centered placement around large fixtures

In practice, the codebase treats them as complementary:

- the selected mode provides the primary answer
- the other backend remains a fallback / cross-check

## Debugging Guidance

When grid and continuous disagree, inspect:

1. the placement map
2. the inferred front face / front target
3. whether `require_front` is active
4. the current positions of the other robot(s)
5. whether the disagreement is due to grid reachability, cell quantization, or
   continuous collision conservatism

Useful files:

- [placement.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement.py)
- [occupancy_grid.py](/Users/dorian/Documents/robocasa/robocasa/utils/occupancy_grid.py)
- [trajectory_runner.py](/Users/dorian/Documents/robocasa/robocasa/utils/trajectory_runner.py)
- [placement_map.py](/Users/dorian/Documents/robocasa/robocasa/utils/placement_map.py)
- [test_placement_sweep.py](/Users/dorian/Documents/robocasa/tests/test_placement_sweep.py)

## Practical Takeaway

Grid and continuous placement are not competing product choices. They are two
different approximations of the same working-pose problem:

- grid is better at reasoning about reachable free space
- continuous is better at reasoning about exact geometry

The current system intentionally uses both because neither approximation is
reliably sufficient by itself across all kitchens, fixtures, and multi-robot
states.
