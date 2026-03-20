# Trajectory Guessing And Recommended Generation Workflow

This note explains two things:

1. How the current trajectory adapter "guesses" scene ids when a trajectory uses abstract ids like `cabinet_1` or `counter_1`
2. What workflow is recommended if the goal is to generate full trajectories with images reliably

The current implementation lives primarily in:

- `robocasa/utils/trajectory_adapter.py`
- `robocasa/utils/sim_tool_executor.py`


## Why This Matters

The current symbolic executor can run a full trajectory and save images, but it is only reliable when the trajectory already uses the real scene ids.

When the trajectory uses invented or abstract ids such as:

- `cabinet_1`
- `counter_1`
- `coffee_machine_1`
- `mug_1`

the adapter has to map them to whatever ids actually exist in the current RoboCasa scene. That mapping is currently heuristic, not semantic.


## Current Guessing Algorithm

When the adapter sees a requested id that does not exist in the live scene, it resolves it in roughly this order.

### 1. Exact Match

If the requested id already exists in the scene, it is used directly.

Examples:

- `fridge_main_group`
- `coffee_machine_left_group`
- `counter_1_left_group`

This is the safe case.


### 2. Type / Type-Family Filtering

If exact match fails, the adapter looks at the type hint from `initial_state`.

Examples:

- `fixture_type = cabinet`
- `fixture_type = counter`
- `object_type = mug`

It then filters the live scene to candidates of the same type or type family.

Current fixture-family handling includes:

- `cabinet` -> `cabinet`, `cabinet_single_door`, `cabinet_double_door`, `cabinet_with_door`
- `counter` -> `counter`, `counter_non_dining`, `counter_non_corner`, `dining_counter`, `island`
- `drawer` -> `drawer`, `top_drawer`

This step narrows the search, but it does not decide which candidate is semantically correct.


### 3. Token Match

The adapter removes purely numeric underscore-separated chunks and compares the remaining token strings.

Examples:

- `cabinet_1` -> `cabinet`
- `coffee_machine_1` -> `coffee_machine`
- `cab_1_main_group` -> `cab_main_group`
- `coffee_machine_left_group` -> `coffee_machine_left_group`

This helps only when the scene id differs mostly by numbering. In practice it rarely helps with RoboCasa fixture ids because real ids often contain:

- abbreviations like `cab`
- spatial suffixes like `left_group`, `main_group`, `right_group`
- corner or group tags

So token match is usually bypassed.


### 4. Ordinal Guess

If the requested id ends with `_N`, the adapter picks candidate number `N` from the sorted candidate list.

This is what happened in the current coffee run.

Examples from `tmp/trajectory_run/adapted_trajectory.json`:

- `cabinet_1 -> cab_1_main_group`
  - method: `ordinal_guess`
  - reason: `Selected candidate #1 among 2 candidates`

- `counter_1 -> counter_1_left_group`
  - method: `ordinal_guess`
  - reason: `Selected candidate #1 among 6 candidates`

This is the main source of bad grounding right now.


### 5. First Candidate Fallback

If there is still no better signal and approximate matching is allowed, the adapter picks the first sorted candidate.

This is extremely weak. It is effectively "alphabetical fallback."


## What The Guesser Is Not Using

The current guesser does not reason over:

- which cabinet actually contains the mug
- which counter is near the coffee machine
- which fixture is closest to another selected fixture
- object adjacency
- task semantics like `dining_table` vs `dining_counter`
- step-to-step consistency across the whole trajectory

So the guesser is not really solving a planning problem. It is solving a string-and-type matching problem.


## Naming Patterns In RoboCasa Scene Ids

There is a pattern, but it is layout-oriented, not human-semantic.

Typical fixture ids look like:

- `coffee_machine_left_group`
- `counter_1_left_group`
- `counter_1_front_group`
- `counter_corner_main_group`
- `cab_1_right_group`
- `cab_main_main_group`
- `fridge_left_group`
- `sink_right_group`
- `stack_2_right_group_4`
- `window_group_1_room`

Typical object ids look like:

- `obj`
- `plate`
- `sausage`
- `condiment`
- `hotdog_bun`
- `hotdog_bun_container`

The naming often encodes:

- fixture family: `counter`, `cab`, `fridge`, `sink`, `stack`
- ordinal or local index: `1`, `2`, `3`
- spatial tag: `left`, `right`, `front`, `main`, `corner`, `room`, `island`
- grouping suffix: usually `group`
- sometimes a sub-index: drawer / stack ids like `stack_2_right_group_4`

This means the scene ids are often good for layout debugging, but not especially good as canonical planner-facing names.


## Concrete Live Examples

### PrepareCoffee Scene Examples

Observed live fixture ids:

- `cab_1_left_group => cabinet_single_door`
- `cab_1_main_group => cabinet_single_door`
- `cab_1_right_group => cabinet_double_door`
- `cab_2_left_group => cabinet_single_door`
- `cab_main_main_group => cabinet_double_door`
- `coffee_machine_left_group => coffee_machine`
- `counter_1_front_group => dining_counter`
- `counter_1_left_group => counter_non_dining`
- `counter_1_right_group => counter_non_dining`
- `counter_main_main_group => counter_non_dining`
- `counter_corner_main_group => counter_non_dining`
- `shelves_right_group => cabinet`

Observed live object ids:

- `obj => mug @ coffee_machine_left_group`
- `distr_cab => hotdog_bun @ cab_1_left_group`

Important implications:

- `mug_1` has no lexical similarity to `obj`
- `cabinet_1` only weakly resembles `cab_*`
- `counter_1` matches many counters, not one


### HotDogSetup Scene Examples

Observed live fixture ids:

- `cab_1_front_group => cabinet_double_door`
- `cab_1_right_group => cabinet_double_door`
- `cab_2_right_group => cabinet_double_door`
- `cab_3_main_group => cabinet`
- `fridge_main_group => fridge`
- `coffee_machine_right_group => coffee_machine`
- `counter_1_front_group => counter_non_dining`
- `counter_1_main_group => counter_non_dining`
- `counter_1_right_group => counter_non_dining`
- `counter_2_main_group => counter_non_dining`
- `island_island_group => island`

Observed live object ids:

- `condiment => condiment_bottle`
- `hotdog_bun => hotdog_bun`
- `hotdog_bun_container => plate`
- `plate => plate`
- `sausage => sausage`

Important implications:

- `fridge_1` is relatively safe because only one fridge exists
- `sausage_1` is relatively safe because only one sausage exists
- `plate_1` is dangerous because there are two `plate`-typed objects
- `cabinet_1` is dangerous because several cabinets exist
- `counter_1` is dangerous because several counters exist
- `dining_table_1` currently fails because the scene uses `dining_counter`, not `dining_table`


## Concrete Resolution Examples

### Safe-ish Resolutions

These are safe mainly because the type is unique in the scene.

- `coffee_machine_1 -> coffee_machine_left_group`
  - unique coffee machine candidate

- `mug_1 -> obj`
  - unique mug candidate

- `fridge_1 -> fridge_main_group`
  - unique fridge candidate in hotdog scene

- `sausage_1 -> sausage`
  - unique sausage object

- `condiment_1 -> condiment`
  - unique condiment bottle


### Risky Resolutions

These are risky because several same-family candidates exist.

- `cabinet_1 -> cab_1_main_group`
  - current method: ordinal guess
  - problem: does not know which cabinet contains the relevant object

- `counter_1 -> counter_1_left_group`
  - current method: ordinal guess
  - problem: does not know which counter is near the coffee machine

- `plate_1 -> ?`
  - hotdog scene has both `hotdog_bun_container` and `plate` with object type `plate`
  - current logic could guess the wrong one depending on sort order


### Failing Resolutions

These fail because the current family logic is incomplete or the state key is not actually a fixture id.

- `dining_table_1`
  - scene may use `dining_counter`
  - current matcher does not interpret that automatically

- `machine_state.hot_dog_setup`
  - current loader assumes machine-state keys are fixture ids
  - this is task-level state, not fixture-level state


## Why The Current Guessing Produces Bad Trajectories

The most common bad pattern is:

1. filter to all counters or all cabinets
2. sort alphabetically
3. pick candidate number `N`

That can produce ids that look plausible but are semantically wrong.

For example, in a coffee trajectory, `counter_1` might resolve to a counter near the fridge instead of the counter near the coffee machine. The trajectory still becomes executable, but the behavior looks wrong because the symbolic grounding was wrong before any motion happened.


## Recommended Workflow For Generating Trajectories With Images

The recommended workflow is to avoid guessing as the primary path.

### Recommended Pipeline

1. Build the live scene first

Freeze:

- task
- layout
- style
- seed

Then export:

- canonical fixture ids
- canonical object ids
- fixture types
- parts / controls
- nearby-fixture relations
- supported camera views


2. Prompt the trajectory generator with canonical ids

Do not ask the model to invent ids like:

- `cabinet_1`
- `counter_1`
- `mug_1`

if the scene already exposes:

- `cab_1_main_group`
- `counter_1_left_group`
- `obj`

The model should generate the full trajectory JSON, but using the real scene ids.


3. Include image steps directly in the trajectory

The planner can still emit:

- `get_image`
- `views`
- `image_paths`

That part is fine.


4. Run a static validation pass before execution

Check:

- every fixture id exists
- every object id exists
- every part / control exists
- every requested camera view exists
- initial state is self-consistent
- no low-confidence guessed id is being relied on


5. Execute strictly

If a required navigation or grounding step fails:

- stop
- do not let later manipulation steps silently continue

This matters because otherwise the trajectory may "complete" while being physically or semantically wrong.


6. Save full audit artifacts

For each run, save:

- the original trajectory JSON
- the adapted / normalized trajectory
- the resolved scene snapshot
- execution metadata
- final state
- saved images

This makes debugging possible.


## What Guessing Should Be Used For

Guessing is still useful, but only as fallback.

Good uses:

- legacy trajectories
- quick experiments
- partial recovery when only one id is abstract

Bad uses:

- final dataset generation
- multi-counter / multi-cabinet tasks
- trajectories where correctness matters more than throughput


## Bottom-Line Recommendation

For reliable trajectory generation with images:

- generate against the real scene description
- use canonical scene ids in the planner output
- keep adapter guessing only as a fallback
- reject low-confidence or failed grounding
- save scene + execution metadata with each run

The current guessing approach is acceptable for quick bootstrapping, but it is not a strong enough basis for producing clean large-scale trajectory data.
