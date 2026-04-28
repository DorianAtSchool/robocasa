# Trajectory ID Resolution And Recommended Generation Workflow

This note explains two things:

1. How the trajectory adapter resolves symbolic ids to concrete scene ids
2. What workflow is recommended if the goal is to generate full trajectories with images reliably

The current implementation lives primarily in:

- `robocasa/utils/trajectory_adapter.py`
- `robocasa/utils/sim_tool_executor.py`


## Why This Matters

The symbolic executor can run a full trajectory and save images, but it needs to map symbolic ids from the trajectory (like `bun`, `serving_surface`, `condiment_source_fixture`) to concrete scene ids (like `hotdog_bun`, `counter_2_main_group`, `cab_3_main_group`).


## Current Resolution: Sim Ground Truth

The adapter now uses **sim ground truth** as the primary resolution strategy, replacing the old heuristic guessing algorithm. This was necessary because heuristic resolution (type-family filtering, ordinal guessing, alphabetical fallback) produced semantically wrong results in multi-cabinet and multi-counter scenes.

### How It Works

The adapter's `_apply_sim_ground_truth()` method resolves ids using data the simulator already knows:

**Objects** are resolved in this order:

1. **Exact symbolic id match**: if the trajectory symbol already exists in `env.objects`, use it directly
2. **Exact object-type-as-id match**: if the trajectory's `object_type` is itself a concrete `env.objects` key, prefer that
3. **Generic object-type match**: otherwise match by `object_type`

The keys come from `_get_obj_cfgs()` name fields and are the task's semantic role names (e.g. `hotdog_bun`, `sausage`, `condiment`, `plate`). The object-type-as-id step matters in scenes like HotDogSetup, where both `plate` and `hotdog_bun_container` have `object_type == "plate"`: a symbolic role like `serving_plate` now prefers the concrete object id `plate` when that id exists.

**Fixtures** are resolved via two strategies:

1. **Object placement tracing**: If a fixture symbol is the location of a resolved object, look up `object_placements[resolved_obj_key]` to get the concrete fixture id. This uses `env.object_cfgs` which records which fixture each object was placed on.

2. **Fixture refs by type**: If no object traces to this fixture, match against `env.fixture_refs` (registered via `register_fixture_ref()` in each task's `_setup_kitchen_references()`).

All sim ground truth resolutions have confidence 1.0 and method `"sim_ground_truth"` in the resolution log.

### Scene Description Ground Truth

The scene description (`get_scene_description()`) now exposes:

- `fixture_refs`: maps task role names to concrete fixture ids (from `env.fixture_refs`)
- `object_placements`: maps object names to the fixture they were placed on (from `env.object_cfgs`)
- `init_robot_base_ref`: the concrete fixture id where the task says robots should start (from `env.init_robot_base_ref`)

These are built using reverse-lookup from `env.fixture_refs` and `env.object_cfgs`, not from the heuristic `_find_object_fixture()` 2D-distance method.

### Robot Spawn

Robot initial positions depend on the executor's `robot_spawn` mode.

- `trajectory` is the current default and places each robot at the fixture named
  in the trajectory's `initial_state`
- `sim` keeps the task's `init_robot_base_ref` placement from the simulator

The executor still stages robots at the sim ground-truth start early in setup so
debug maps and pre-initial-state renders can reflect the scene before
trajectory-driven repositioning.

### Heuristic Fallback

The old heuristic resolution methods (type filtering, token match, ordinal guess, first-candidate fallback) still exist as fallback for any symbols not resolved by ground truth. These are the same methods documented in the "Legacy Guessing Algorithm" section below and produce lower-confidence results.


## Legacy Guessing Algorithm

When ground truth cannot resolve a symbol (e.g., legacy trajectories without `initial_state` type hints, or tasks without `fixture_refs`), the adapter falls back to heuristic resolution.

### 1. Exact Match

If the requested id already exists in the scene, it is used directly.

### 2. Type / Type-Family Filtering

The adapter looks at the type hint from `initial_state` and filters to candidates of the same type family.

Current fixture-family handling includes:

- `cabinet` -> `cabinet`, `cabinet_single_door`, `cabinet_double_door`, `cabinet_with_door`
- `counter` -> `counter`, `counter_non_dining`, `counter_non_corner`, `dining_counter`, `island`
- `drawer` -> `drawer`, `top_drawer`

### 3. Token Match

Removes numeric chunks and compares token strings. Rarely helps with RoboCasa ids due to abbreviations and spatial suffixes.

### 4. Ordinal Guess

If the requested id ends with `_N`, picks candidate number `N` from sorted candidates. This was the main source of bad grounding in earlier versions.

### 5. First Candidate Fallback

Picks the first sorted candidate. Effectively alphabetical fallback.


## What The Legacy Guesser Does Not Use

The heuristic guesser does not reason over:

- which cabinet actually contains the relevant object
- which counter is near another selected fixture
- object adjacency or spatial relations
- task semantics like `dining_table` vs `dining_counter`
- step-to-step consistency across the trajectory


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
- a symbolic role with `object_type == "plate"` now prefers the concrete object id `plate`
- `plate_1` is still dangerous because there are two `plate`-typed objects
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
  - exact `object_type == "plate"` now prefers the concrete object id `plate`
  - abstract aliases like `plate_1` can still be ambiguous because neither the symbol nor the type uniquely identifies one candidate


### Failing Resolutions

These fail because the current family logic is incomplete or the state key is not actually a fixture id.

- `dining_table_1`
  - scene may use `dining_counter`
  - current matcher does not interpret that automatically

- `machine_state.hot_dog_setup`
  - current loader assumes machine-state keys are fixture ids
  - this is task-level state, not fixture-level state


## Why Heuristic Guessing Produces Bad Trajectories

When ground truth is unavailable and the adapter falls back to heuristics, the most common bad pattern is:

1. filter to all counters or all cabinets
2. sort alphabetically
3. pick candidate number `N`

That can produce ids that look plausible but are semantically wrong.

For example, in a coffee trajectory, `counter_1` might resolve to a counter near the fridge instead of the counter near the coffee machine. The trajectory still becomes executable, but the behavior looks wrong because the symbolic grounding was wrong before any motion happened.

The sim ground truth approach eliminates this class of errors for trajectories that include `initial_state` type hints, because it traces through the task's actual object placement configs rather than guessing by name.


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


## What Heuristic Guessing Should Be Used For

Heuristic guessing is still useful, but only as fallback for legacy trajectories without `initial_state` type hints.

Good uses:

- legacy trajectories without type hints
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
- ensure trajectories include `initial_state` with object/fixture type hints so sim ground truth can resolve them
- keep heuristic guessing only as a fallback for legacy data
- reject low-confidence or failed grounding
- save scene + execution metadata with each run

Sim ground truth resolution is now the primary path and handles most cases with full confidence. Heuristic fallback remains for edge cases.


## Known Limitations

### Fixture articulation (opening fridges, cabinets)

Opening enclosing fixtures before picking up objects inside them is not yet handled automatically by the executor. Trajectories that require opening a fridge or cabinet door must include explicit `open_hinged_part` steps. If the trajectory omits these steps (e.g., the LLM planner forgets), the robot will navigate to the fixture but the door will remain closed. This is a trajectory generation issue, not an executor bug — the executor faithfully executes whatever steps are provided but does not infer missing open/close actions. A reliable solution (e.g., auto-opening fixtures when pick_up_object targets an enclosing fixture) is still to be determined.
