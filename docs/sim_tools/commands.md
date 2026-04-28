# Commands

Run commands from the repo root with the RoboCasa conda env active:

```bash
conda activate robocasa
cd robocasa
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

Run the built-in hotdog demo plan:

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

Run the built-in sandwich demo plan:

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

Override the occupancy-grid cell size:

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

Dry run:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 56 --styles 34 42 --seeds 42 99 \
  --dry-run
```

## Test And Smoke Commands

Hotdog trajectory smoke test:

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

Sandwich trajectory smoke test:

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

Quick placement sweep:

```bash
python tests/test_placement_sweep.py \
  --placement grid \
  --tasks HotDogSetup \
  --layouts 11 \
  --styles 34 \
  --seeds 42 \
  --output tmp/quick_grid
```

## Notes

- `grid` is the only supported placement mode.
- `--placement` is still present for compatibility, but it currently only accepts `grid`.
- Images are rendered only by explicit `get_image` tool calls.
- Sweep mode skips videos by default.
- Default grid cell size is `0.05` meters.
