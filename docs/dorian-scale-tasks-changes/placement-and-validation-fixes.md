# Placement & Validation Fixes (A-E)

Patches for 5 root causes identified during trajectory sweep validation on the
`dorian-scale-tasks` branch.  All changes target the trajectory execution
pipeline (sim tool executor, trajectory runner) and the trajectory generation
FSM/prompt layer.

---

## A — `sim.forward()` after fixture state changes

**Root cause:** `fixture.open_door()` / `close_door()` call `set_joint_state()`
which sets `qpos` but never calls `sim.forward()`.  Objects placed inside a
fixture (e.g. fridge) after opening still see the old collision geometry — the
door appears closed to the physics engine until the next `forward()`.

**Fix:**  Added `self.env.sim.forward()` immediately after `open_door()` and
`close_door()` in `open_hinged_part` and `close_hinged_part`.

Note: `_set_named_joint()` (used by sliding parts) already calls
`sim.forward()`, so sliding parts were not affected.

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `open_hinged_part()`, `close_hinged_part()`

---

## B — Z-offset uses object `bottom_offset` instead of flat 2cm

**Root cause:** `trajectory_runner._iter_object_target_candidates()` applied a
flat `world[2] += 0.02` after converting a fixture-local offset to world
coordinates.  This does not account for object geometry — tall objects sink
into surfaces, small objects float above them.

Native RoboCasa placement (`placement_samplers.py:426`) correctly uses
`object_z -= obj.bottom_offset[-1]` to place the object's geometric bottom
on the surface.

**Fix:** Replaced the flat `+0.02` with `-obj.bottom_offset[-1]`, computed per
object.  Applied to both the main candidate loop and the fallback path.

**Files changed:**
- `robocasa/utils/trajectory_runner.py` — `_iter_object_target_candidates()`

---

## C — Same-fixture overlap detection in `load_initial_state`

**Root cause:** `load_initial_state()` skips objects already at their target
fixture (line 393) to avoid resampling positions.  When multiple objects share
the same fixture (e.g. sausage + cheese in the fridge), they may have been
placed at overlapping positions by the sim's initial sampler, and no pairwise
check catches this.

**Fix:** After the skip-or-move loop, objects are grouped by target fixture.
For each group with 2+ objects, pairwise XY distances are checked against the
sum of their `horizontal_radius` values.  Overlapping pairs are resolved by
nudging the smaller object horizontally along the separation vector until
clearance is achieved (with a 5mm margin).

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `load_initial_state()`

---

## D — Dispenser placement Z (scalable)

**Root cause:** `place_under()` for CoffeeMachine/Sink placed the object
*center* at the dispenser spout's world Z.  This means a mug's top half pokes
above the spout and its bottom floats above the drip tray.

**Fix:** Extracted a shared `_get_dispenser_site_name()` registry method that
maps fixtures to their dispenser site names.  Currently handles
`CoffeeMachine` (→ `receptacle_place_site`) and `Sink` (→ `water_site`).
Adding future dispensers (e.g. ElectricKettle, Blender) requires only a new
`elif` branch in this method.

`place_under()` now uses a single code path for all dispenser fixtures:
1. Read the dispenser site's world position via `sim.data.site_xpos`
2. Adjust Z so the object's **bottom** (not center) rests at spout height,
   using `obj.bottom_offset[-1]`
3. This positions the object inside the fixture where it can settle onto the
   drip tray / basin naturally

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `_get_dispenser_site_name()` (new),
  `place_under()` (refactored)

---

## E — `give_space` re-navigation rule (prompts + FSM)

**Root cause:** When an agent executes `give_space`, it physically steps back
from the fixture.  However:
1. The FSM did not update `agent_state.location` — it still thought the agent
   was at the fixture.
2. `press_button` was not in any explicit tool category, so it fell through to
   the generic catch-all which passed because the FSM had stale location info.
3. The LLM trajectory generator had no instruction that `give_space` requires
   re-navigation before further interaction.

This meant trajectories could generate `give_space → press_button` without an
intervening `navigate_to_fixture`, producing physically implausible robot
positioning.

**Fix (three parts):**

### E1 — `constants.py`
Added `INTERACTION_TOOL_NAMES = frozenset({"press_button"})`.  Future
interaction tools (`pull_lever`, `turn_knob`, etc.) should be added here.

### E2 — `fsm.py`
- `_apply_generic_effects()`: `give_space` now sets `agent_state.location = None`
  so the FSM correctly reflects that the agent has left the fixture.
- `_validate_generic_transition()`: Added explicit validation branch for
  `INTERACTION_TOOL_NAMES` that requires agent to be at the target fixture
  (same pattern as OPEN/CLOSE parts).  This catches invalid sequences where
  `give_space` precedes `press_button` without re-navigation.

### E3 — `prompting.py`
Added two prompt rules to the FSM-derived execution rules:
1. After `give_space`, the agent is no longer at the fixture and must
   `navigate_to_fixture` before any further interaction (pick up, place, open,
   close, press_button).  The arriving agent must also `give_space` before the
   original agent can navigate back.
2. Interaction tools (`press_button`) require the agent to be at the target
   fixture — navigate first.

**Files changed:**
- `data_generation/task_level/tasks/shared/constants.py`
- `data_generation/task_level/tasks/shared/fsm.py`
- `data_generation/task_level/tasks/shared/prompting.py`

---

## F — Apply fixture part states from trajectory initial_state

**Root cause:** `load_initial_state()` intentionally skipped fixture part
states (open/close) from the trajectory, assuming `_setup_scene` had set
them correctly.  But `_setup_scene` opens cabinets during object placement
(so objects can be placed inside), and the code never closed them back.
Result: cabinets that should start closed were visually open.

**Fix:** Replaced the skip logic with an explicit loop that applies each
fixture's `parts.{part_id}.state` from the trajectory's initial_state.
Supports both `"closed"` and `"open"` states, with `close_door()`/`open_door()`
for hinged parts and `_set_named_joint()` for named joints.  A single
`sim.forward()` follows to refresh all fixtures at once.

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `load_initial_state()`

---

## G — Conservative overlap repair for coincident placements

**Root cause:** The sim sometimes places two objects at the exact same
position on a fixture (e.g. sausage + cheese at the same fridge shelf
center).  The original repair (Patch C) was too aggressive — it used
`rad_a + rad_b` as the threshold, which triggered on objects that were
merely near each other on a counter (normal).  The `move_object()` re-place
strategy caused regressions: baguettes landing inside bowls, objects
teleporting to cutting boards.

**Fix:** Tightened to a 3cm coincident threshold — only fires when objects
are at near-identical XY positions (sim duplicate placement), not when
they're just close together on a surface.  Removed the `move_object()`
re-place strategy entirely (too unpredictable).  Instead, nudges the smaller
object laterally along the fixture's local X axis by `rad_a + rad_b + 5mm`
to ensure clearance while staying on the same shelf.

Objects whose `location` is another object (e.g. slices in a bowl) are
excluded from the overlap check by the `location not in objects` filter.

The lateral nudge now validates the new position is still on the fixture
(via `_validate_object_on_fixture`).  If the positive direction goes
off-fixture (e.g. off the counter edge onto a stove), it tries the negative
direction before falling back.

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `load_initial_state()` overlap repair

---

## I — Explicit object-in-object placement

**Root cause:** Objects whose trajectory location is another object (e.g.
`tomato_slice` at `ingredient_bowl`) were skipped in `load_initial_state()`
with the assumption that the sim's `_setup_scene` placed them correctly.
But the sim does not handle object-in-object placement — it places all
objects independently on fixture surfaces.  Slices ended up on random
fridge shelves instead of inside the bowl.

**Fix:** Added an object-in-object placement pass that teleports items to
their container's body center position (`_get_object_pose` → `_set_object_pose`).

**Ordering is critical:** This pass runs AFTER the overlap repair so that:
1. Containers (bowls) are at their final post-nudge positions
2. Items aren't present during the overlap repair, preventing them from being
   accidentally dragged by sibling objects (e.g. baguette nudging would pick up
   slices via `_find_contained_objects` if they were at the same position)

Execution order in `load_initial_state()`:
1. Fixture placement pass
2. Overlap repair (nudge coincident objects)
3. Object-in-object placement (slices → bowl's final position)
4. Robot placement

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `load_initial_state()`

---

## H — Spread multiple objects on same support (cutting board)

**Root cause:** `_place_on_object_center()` always placed at the support
object's center XY.  When multiple objects were placed on the same support
(e.g. sausage and cheese on a cutting board), they stacked at the same
position.

**Fix:** Before placing, `_place_on_object_center()` now calls
`_find_objects_on_support()` to detect existing objects on the support.
When others are present, the new object is offset along the support's
longer in-plane axis (computed from the support's bbox).  Objects are
distributed evenly across 70% of the support extent.

**Concave container handling:** Spreading is skipped for bowls and similar
concave containers (detected by `extent_z / max_xy >= 0.3`).  Items in
bowls are placed at the bowl's body center Z (inside the bowl) rather than
at `support_top_z` (the rim).  This allows natural piling without items
sitting on the rim edges.

New helper `_find_objects_on_support()` finds smaller objects within
`horizontal_radius * 1.2` XY and slightly above the support Z.

**Files changed:**
- `robocasa/utils/sim_tool_executor.py` — `_place_on_object_center()` (modified),
  `_find_objects_on_support()` (new)
