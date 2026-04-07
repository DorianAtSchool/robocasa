# Next Task Candidates

This file tracks RoboCasa composite tasks that are good fits for the current
task-level symbolic generation stack.

## Active Safe Tasks

These are already implemented as active spec-native tasks because their object
semantics line up well enough with the current validator and grounding model:

- `HotDogSetup`
- `PrepareCoffee`
- `PrepareSandwichStation`
- `PrepareSausageCheese`
- `PrepareCheeseStation`

## Recommended Next Safe Tasks

These look like the best next additions if we want to preserve the same
semantic-to-sim guarantees as the current active set.

### 1. AddLemonToFish
Source: `robocasa/environments/kitchen/composite/garnishing_dishes/add_lemon_to_fish.py`

Why it is a good fit:
- Uses concrete semantic object names: `lemon_wedge`, `fish`.
- The goal is simple: move the lemon wedge from the fridge onto the fish plate.
- Good fridge-to-table transfer task without sampled category slots.

### 2. GarnishCupcake
Source: `robocasa/environments/kitchen/composite/garnishing_dishes/garnish_cupcake.py`

Why it is a good fit:
- Uses concrete semantic names: `cinnamon`, `chocolate`, `cupcake`.
- Two-agent split is natural: one agent can stage cinnamon while the other places chocolate on the plate.
- Very close to the existing `place_next_to` + `place_on_object` task pattern.

### 3. PrepareBroilingStation
Source: `robocasa/environments/kitchen/composite/broiling_fish/prepare_broiling_station.py`

Why it is a good fit:
- Uses fixed semantic names: `tongs`, `plate`, `oven`.
- Goal is simple staging near a known anchor fixture.
- Good oven-side placement task without sampled object categories.

### 4. PrepareDrinkStation
Source: `robocasa/environments/kitchen/composite/serving_beverages/prepare_drink_station.py`

Why it is a good fit:
- Uses fixed semantic object names: `tray`, `cup`, `mug`, `pitcher`.
- Strong two-agent coordination: tray loading plus pitcher staging.
- Fits the current symbolic patterns if modeled as `place_on_object` for tray items and `place_next_to` for pitcher.

### 5. SetBowlsForSoup
Source: `robocasa/environments/kitchen/composite/setting_the_table/set_bowls_for_soup.py`

Why it is a good fit:
- Uses fixed semantic names: `bowl1`, `bowl2`, `plate1`, `plate2`.
- Strong coordination signal with one bowl per agent.
- Harder than the current active tasks, but the semantics are still cleaner than sampled-category tasks.

### 6. SeasoningSteak
Source: `robocasa/environments/kitchen/composite/seasoning_food/seasoning_steak.py`

Why it is a good fit:
- Uses fixed semantic names: `shaker`, `steak`.
- Cabinet-to-table staging task with straightforward door semantics.
- The placement is somewhat geometric, but still much more semantically stable than sampled-slot tasks.

## Deferred Due To Scene-Matching Mismatch

These tasks were drafted as specs, but moved out of the active registry because
their semantic object meanings do not currently line up cleanly with the sim's
slot-style object IDs.

See:
- `data_generation/task_level/tasks/specs/deferred_scene_matching/README.md`

Deferred tasks:
- `GatherMarinadeIngredients`
- `SetUpSpiceStation`
- `OrganizeCondiments`
- `SetupFruitBowl`
- `OrganizeCoffeeCondiments`

## Coordination-Heavy But Harder

### SetupBowls
Source: `robocasa/environments/kitchen/composite/setting_the_table/setup_bowls.py`

Why it is compelling:
- Excellent two-agent coordination and communication task.
- Clean work split: one bowl per agent.

Why it is harder:
- Needs a symbolic notion like `front_of(stool)` rather than only
  `on_surface`, `in_receptacle`, or `next_to`.

### SetupWineGlasses
Source: `robocasa/environments/kitchen/composite/setting_the_table/setup_wine_glasses.py`

Why it is compelling:
- Also naturally parallel and communication-heavy.

Why it is harder:
- Needs stable symbolic reasoning about left/right or side-of-plate geometry.

### AlignSilverware
Source: `robocasa/environments/kitchen/composite/setting_the_table/align_silverware.py`

Why it is compelling:
- Strong communication and ordering signal.

Why it is harder:
- Requires symbolic `left_of` / `right_of` relations relative to a plate.
