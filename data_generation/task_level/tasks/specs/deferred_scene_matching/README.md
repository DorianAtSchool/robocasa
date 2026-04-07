# Deferred Scene-Matching Specs

This folder holds task specs that are structurally valid under the current
`TaskSpec` schema, but are not safe to keep in the active spec registry yet.

## Why they are deferred

These RoboCasa tasks use a mismatch between:

- **semantic meaning for the planner**
  - names the trajectory should ideally reason about, such as `banana`,
    `apple`, `ketchup`, or `syrup_bottle`
- **sim object-slot identifiers**
  - names the environment instantiates directly, such as `fruit1`, `fruit2`,
    `obj1`, `obj2`, `spice`, or `bottle`

For these tasks, the RoboCasa source task often samples a concrete category into
an abstract slot. For example, one scene may instantiate:

- `fruit1 -> apple`
- `fruit2 -> banana`

while another may instantiate:

- `fruit1 -> orange`
- `fruit2 -> pear`

The current active task-level pipeline does **not** yet build a per-scene alias
map like:

- `banana -> fruit2`
- `apple -> fruit1`

before simulator execution.

## What the current active pipeline expects

The active spec-native path assumes that:

- the symbolic `initial_state` is authoritative for prompt generation
- object identifiers used by the trajectory, validator, and grounding map are
  stable
- `object_type` is concrete enough to resolve the intended scene object

That works well for tasks such as:

- `HotDogSetup`
- `PrepareCoffee`
- `PrepareSandwichStation`
- `PrepareSausageCheese`
- `PrepareCheeseStation`

because their semantic object names and sim expectations are close enough to one
another.

## Why listing every possible object in one JSON is not enough

Adding every candidate object to the spec would only change the **symbolic**
task state. It would not force RoboCasa to instantiate those exact concrete
objects in the scene, because scene creation is still controlled by the source
task's `_get_obj_cfgs()` logic.

So the missing piece is not a larger JSON. The missing piece is a per-instance
scene-matching phase that can map semantic names used by trajectories onto the
actual object-slot IDs created by the environment for that scene.

## Deferred tasks in this folder

- `GatherMarinadeIngredients`
- `SetUpSpiceStation`
- `OrganizeCondiments`
- `SetupFruitBowl`
- `OrganizeCoffeeCondiments`

These specs are kept here so the work is not lost, but they are intentionally
excluded from the active `TaskSpec` registry until scene-specific semantic
matching is added.
