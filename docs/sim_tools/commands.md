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
- Executor outputs include placement maps, frame PNGs, videos, and metadata JSON.
- Sweep outputs write one folder per combo with a `result.json`.
