# Sim Tool Executor Handoff

This document captures the current state of the `SimToolExecutor` work so another model instance can continue the conversation with minimal loss of context.

## Current State

The current teleport-based `SimToolExecutor` approach is working reasonably well for `HotDogSetup` across multiple layouts / styles / seeds, but it is not yet robust across tasks.

What currently works well:

- Symbolic tool execution through `robocasa/utils/sim_tool_executor.py`
- Semantic plan templates plus runtime grounding
- Explicit `anchor_fixture_id` support for `place_on_object`
- Wrist cameras per robot
- Dynamic room view that is less zoomed out than before
- Hotdog semantic template runs across multiple layouts / styles / seeds

Important recent changes already in the codebase:

- `place_on_object` now uses:
  - `place_on_object(object_id, support_object_id, anchor_fixture_id)`
- Room-view fitting was tightened in:
  - [trajectory_runner.py](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py)
- Wrist cameras are enabled via:
  - `robotN_eye_in_hand`
- Hotdog support-object anchoring was improved so the dining-table anchor is explicit in grounded plans

Relevant files:

- [sim_tool_executor.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_executor.py)
- [sim_tool_specs.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_specs.py)
- [trajectory_runner.py](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py)
- [prepare_sandwich_station.py](/home/dorian/Projects/robocasa/robocasa/environments/kitchen/composite/preparing_sandwiches/prepare_sandwich_station.py)
- [sim_tool_executor.md](/home/dorian/Projects/robocasa/docs/sim_tool_executor.md)
- [sim_tool_layer.md](/home/dorian/Projects/robocasa/docs/sim_tool_layer.md)
- [hotdog_semantic_template_runs.md](/home/dorian/Projects/robocasa/docs/hotdog_semantic_template_runs.md)

## Main Open Problem

The current approach appears scalable across layout / style / seed variation for `HotDogSetup`, but not across arbitrary tasks.

The biggest current failure case discussed was `PrepareSandwichStation`, where several different issues may be happening:

- wrong fixture or support fixture grounding
- bad robot base teleport placement even if the chosen fixture is correct
- bad object placement region on the support fixture
- missing semantics for "near toaster oven"
- missing support for moving receptacles together with their contents

## Key Diagnosis

`HotDogSetup` fits the current abstraction well because:

- source objects are individually manipulable
- the target is a support object (`plate`)
- the semantic anchor is relatively easy to recover

`PrepareSandwichStation` is harder because:

- the task success condition is "on the counter near the toaster oven"
- the current plan / executor only expresses "on this fixture"
- placement currently validates fixture bbox containment, not relation-to-appliance or collision-free support-region placement
- the task includes a bowl containing ingredient objects, so receptacle carry semantics matter

## What a 2D Grid Would Help With

A 2D grid / occupancy map could help with:

- robot base spawnability
- not teleporting bases into counters or blocked regions
- finding approach poses with collision avoidance
- later replacing teleport navigation with planned navigation

A 2D grid would not by itself solve:

- choosing the correct support fixture
- choosing the right placement subregion on a support fixture
- avoiding toaster / appliance collisions for object placement
- carrying contained objects when moving a bowl / basket

## Recommended Problem Decomposition

The discussion converged on separating the problem into four layers:

1. Semantic grounding
- choose the right fixture / support object / support region

2. Robot approach grounding
- choose a reachable base pose for the selected anchor

3. Object placement grounding
- choose a collision-free pose on the support surface in the right semantic subregion

4. Carry semantics
- decide what moves together when a receptacle is moved

## Longer-Term Direction

The likely long-term architecture is:

- keep the symbolic tool layer
- improve semantic grounding and support-region reasoning
- add occupancy-based or motion-planned base navigation
- add tool-specific manipulation controllers or state machines later

The earlier discussion concluded that raw RoboCasa demo replay is not a general solution for new scenes, because demo replay depends on matching the recorded scene/model/state. Demo data may still be useful as priors for manipulation subskills or state-machine design, but not as a direct executor for arbitrary new layouts.

## Entire Last Message

Below is the full last assistant message, included verbatim as requested:

> Yes, we can run `SimToolExecutor` on a new task like `PrepareSandwichStation`, but that is exactly where the current abstraction starts to break.
>
> `HotDogSetup` works because it fits the executor well:
> - source objects are individually movable
> - destination is a clear support object (`plate`)
> - the semantic anchor is easy to recover
>
> `PrepareSandwichStation` is harder for three separate reasons.
>
> First, the task semantics are more specific than the current tool plan. The actual task success is “bowl and baguette on a counter near the toaster oven” in [prepare_sandwich_station.py](/home/dorian/Projects/robocasa/robocasa/environments/kitchen/composite/preparing_sandwiches/prepare_sandwich_station.py#L160), but the current plan only says “put them on `counter_X`”. That loses the “near toaster” part. Then [move_object()](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py#L1031) samples a generic point on the fixture and only validates “inside fixture bbox”, not “near toaster” or “not colliding with toaster”. So the toaster collision is mainly a placement-semantic problem, not a base-teleport problem.
>
> Second, receptacle transport is incomplete. If the bowl or basket contains other objects, teleporting the receptacle does not automatically carry its contents. `place_on_object` and `move_object` are moving the named object’s MuJoCo pose, not maintaining a rigid attachment set for contained items. So the “basket moved but contents did not move intuitively” issue is a real executor limitation, separate from navigation.
>
> Third, robot-base teleportation is still only fixture-relative. `_move_robot_near_fixture()` in [trajectory_runner.py](/home/dorian/Projects/robocasa/robocasa/utils/trajectory_runner.py#L652) uses `compute_robot_base_placement_pose(...)` when possible, then a coarse fallback in front of the fixture if that fails. If the chosen fixture is wrong, or if the correct fixture has awkward geometry like a corner/counter combination, the fallback can absolutely place the robot on a bad side. So your suspicion is right: some failures are wrong anchor selection, and some are bad placement even for the chosen anchor.
>
> A 2D spawn grid would help, but only for one slice of the problem. It would help with:
> - not teleporting robot bases into counters / blocked regions
> - choosing reachable approach poses with collision avoidance
> - later, navigation planning for continuous control
>
> It would not fix:
> - choosing the right support fixture or support region
> - placing “near toaster” instead of “somewhere on the counter”
> - carrying objects inside a moved receptacle
> - object-object / object-appliance collision at placement time
>
> So I would separate the problem into four layers:
>
> 1. Semantic grounding
> The trajectory must specify the right fixture, support object, and support region.
> Examples:
> - not just `counter_1_right_group_1`
> - but “the counter workspace near toaster_oven”
>
> 2. Robot approach grounding
> Given the right anchor, choose a reachable base pose.
> This is where a grid / occupancy map helps a lot.
>
> 3. Object placement grounding
> Given the right support fixture, choose a collision-free pose on that support, ideally in a task-relevant subregion.
> This needs more than the current random jitter inside fixture bbox.
>
> 4. Carry semantics
> If a receptacle is moved, decide what moves with it.
> For `ingredient_bowl`, contents should probably remain rigidly attached during teleport execution.
>
> That is why “arbitrary task” does not work yet. The executor is mostly generic, but the grounding and placement semantics are still too weak.
>
> On the longer-term controller question: I would not jump straight from this teleport system to full scripted continuous policies for everything. I’d do it in stages.
>
> Best short-term path:
> - keep the symbolic tool layer
> - improve grounding and placement semantics
> - add a robot-base occupancy / reachability map
> - add receptacle-carry support
> - add task-specific support regions like “near toaster”, “under dispenser”, “next to plate”
>
> Best medium-term path for VLA data:
> - use the same tool layer as the planner interface
> - replace teleport navigation with a real navigation controller / motion planner
> - replace teleport manipulation with tool-specific state machines
>
> I do not think raw RoboCasa demo replay is the general answer. Full replay works only when the scene matches the recorded model/state. That is exactly why earlier demo playback looked wrong across layouts. You could still use demos as priors for manipulation subskills, waypoint extraction, or state-machine design, but not as a drop-in executor for arbitrary new scenes.
>
> So my view is:
> - a grid is worth doing, but only for robot-base placement / navigation
> - the sandwich failures are not just “teleportation is bad”
> - the bigger missing pieces are support-region semantics, collision-aware placement, and receptacle-carry behavior
> - for VLA, the right architecture is still “semantic tools -> controller backend”, not “teleport forever” and not “replay demos verbatim”
>
> If you want, the next useful step is to write down a concrete failure taxonomy for `PrepareSandwichStation`:
> - wrong anchor fixture
> - bad base pose on correct fixture
> - bad support-region choice
> - bad placement collision
> - missing receptacle-carry propagation
>
> That would make the next implementation pass much cleaner.
