# Commands

Run all commands from the repo root:

```bash
cd /Users/dorian/Documents/robocasa_mason
```

## Sim Tool Executor

Render the current scene only:

```bash
python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/executor_scene_grid
```

Run the built-in hotdog demo plan with grid placement:

```bash
python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --demo-plan cooperative_hotdog_setup \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/executor_hotdog_grid
```

Run the built-in sandwich demo plan with grid placement:

```bash
python -m robocasa.utils.sim_tool_executor \
  --task PrepareSandwichStation \
  --demo-plan sandwich_station \
  --robots 1 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/executor_sandwich_grid
```

Run the hotdog demo plan with continuous placement:

```bash
python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --demo-plan cooperative_hotdog_setup \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement continuous \
  --output-dir tmp/executor_hotdog_continuous
```

Run an external tool plan JSON:

```bash
python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --plan path/to/tool_plan.json \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/executor_tool_plan_grid
```

Run a full external trajectory JSON through the trajectory adapter:

```bash
python -m robocasa.utils.sim_tool_executor \
  --trajectory path/to/trajectory.json \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/executor_trajectory_grid
```

Run a trajectory with a custom cell size (default is 0.05 m):

```bash
python -m robocasa.utils.sim_tool_executor \
  --task HotDogSetup \
  --trajectory data_generation/task_level/data/raw/20260324T031125Z/hot_dog_setup/trajectories/traj_000000.json \
  --robots 2 \
  --layout 11 \
  --style 42 \
  --seed 42 \
  --placement grid \
  --cell-size 0.05 \
  --output-dir tmp/test_ground_truth
```

Example `get_image` tool call inside a plan or trajectory step:

```json
{
  "tool": "get_image",
  "args": {
    "views": ["top_view", "room_view", "map"],
    "image_paths": [
      "images/top.png",
      "images/room.png",
      "images/map.png"
    ]
  }
}
```

## Trajectory Sweep

Sweep all trajectories in a dataset directory:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output
```

Sweep a single task:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --task hot_dog_setup
```

Sweep specific trajectory indices:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --task hot_dog_setup --indices 0 1 2
```

Sweep across multiple layouts, styles, and seeds:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 56 \
  --styles 34 42 \
  --seeds 42 99
```

Dry run (preview without executing):

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 56 --styles 34 42 --seeds 42 99 \
  --dry-run
```

## Trajectory-Level Runs

Hotdog trajectory test, grid placement:

```bash
python tests/test_occupancy_grid_trajectories.py \
  --placement grid \
  --test hotdog \
  --task HotDogSetup \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --output tmp/traj_hotdog_grid
```

Sandwich trajectory test, grid placement:

```bash
python tests/test_occupancy_grid_trajectories.py \
  --placement grid \
  --test sandwich \
  --task PrepareSandwichStation \
  --robots 1 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --output tmp/traj_sandwich_grid
```

Generic trajectory stress test on another task:

```bash
python tests/test_occupancy_grid_trajectories.py \
  --placement grid \
  --test generic \
  --task MicrowaveThawing \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --output tmp/traj_generic_grid
```

Run multiple trajectory suites together:

```bash
python tests/test_occupancy_grid_trajectories.py \
  --placement continuous \
  --test hotdog,sandwich,collision \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --output tmp/traj_multi_continuous
```

## Sweep Tests

Quick grid smoke test:

```bash
python tests/test_placement_sweep.py \
  --placement grid \
  --tasks HotDogSetup \
  --layouts 11 \
  --styles 34 \
  --seeds 42 \
  --output tmp/quick_grid
```

Full default grid sweep:

```bash
python tests/test_placement_sweep.py \
  --placement grid \
  --output tmp/sweep_grid
```

Full default continuous sweep:

```bash
python tests/test_placement_sweep.py \
  --placement continuous \
  --output tmp/sweep_continuous
```

Targeted sweep over both demo-backed tasks:

```bash
python tests/test_placement_sweep.py \
  --placement grid \
  --tasks HotDogSetup,PrepareSandwichStation \
  --layouts 11,56 \
  --styles 34,42 \
  --seeds 42 \
  --output tmp/sweep_targeted_grid
```

Faster sweep without saving videos:

```bash
python tests/test_placement_sweep.py \
  --placement grid \
  --tasks HotDogSetup \
  --layouts 11 \
  --styles 34 \
  --seeds 42 \
  --output tmp/quick_grid_no_video \
  --no-video \
  --verbose
```

## Notes

- `cooperative_hotdog_setup` maps to `HotDogSetup`.
- `sandwich_station` maps to `PrepareSandwichStation`.
- `grid` is the default placement mode for the executor, trajectory tests, and sweep tests.
- Default grid cell size is 0.05 m (5 cm). Override with `--cell-size`.
- `get_image` accepts an array of views in one call. Supported environment views include `top_view`, `room_view`, and `map`.
- A trajectory JSON is not self-contained today unless it carries `scene_parameters` (or top-level `layout` / `style` / `seed`).
  Older symbolic trajectory JSONs still need those flags so the simulator can recreate a concrete kitchen instance.
- Trajectory adapter uses sim ground truth (`env.fixture_refs`, `env.object_cfgs`) as the primary resolution strategy. Heuristic fallback is used only for symbols not resolved by ground truth.
- Robot initial spawn uses the sim's placement system (`init_robot_base_ref`), not the trajectory's agent locations.
- `pick_up_object` returns failure if the robot cannot navigate to the source fixture.
- Opening enclosing fixtures (fridges, cabinets) before picking is **not** automatic — trajectories must include explicit `open_hinged_part` steps. This is still to be determined.
- Executor outputs include adapted trajectory, execution metadata, and images rendered by `get_image` steps.
- Sweep outputs write one folder per combo with a `result.json`.
