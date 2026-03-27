# Running Trajectories

Commands for executing raw trajectory JSONs through the simulator to produce grounded trajectories with rendered images.

All commands from the repo root:

```bash
cd robocasa
```

## Single Trajectory

Run one trajectory through the executor:

```bash
python -m robocasa.utils.sim_tool_executor \
  --trajectory data_generation/task_level/data/image/20260324T031125Z/hot_dog_setup/trajectories/traj_000000.json \
  --robots 2 \
  --layout 11 \
  --style 34 \
  --seed 42 \
  --placement grid \
  --output-dir tmp/test_single
```

The `--task` flag is optional — it is read from the trajectory's `composite_task` field. Scene parameters (`--layout`, `--style`, `--seed`) default to 11, 34, 42 if not specified in the trajectory or on the command line.

### With videos

By default, per-camera MP4 videos are generated alongside the images. To skip them:

```bash
python -m robocasa.utils.sim_tool_executor \
  --trajectory path/to/trajectory.json \
  --robots 2 \
  --placement grid \
  --skip-videos \
  --output-dir tmp/test_no_video
```

Videos are one MP4 per camera (`room_view.mp4`, `top_view.mp4`, `robot0_agentview_center.mp4`, etc.). They show the trajectory as a slideshow of rendered frames at `--fps` (default 2). Useful for quick visual review but large — skip them for batch runs.

## Sweep (Batch)

`scripts/sweep_trajectories.py` discovers all `traj_*.json` files under a dataset directory and executes each one. Videos are always skipped in sweep mode.

### All trajectories

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output
```

### Single task

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --task hot_dog_setup
```

### Specific trajectory indices

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --task hot_dog_setup --indices 0 1 2
```

### Single trajectory via sweep

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --task hot_dog_setup --indices 0
```

### Sweep across layouts, styles, and seeds

Each trajectory is executed once per (layout, style, seed) combination. Outputs get an extra directory level `L<layout>_S<style>_sd<seed>`:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 56 \
  --styles 34 42 \
  --seeds 42 99
```

This runs every trajectory 2×2×2 = 8 times (one per combo).

### Dry run

Preview what would run without executing anything:

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 56 --styles 34 42 --seeds 42 99 \
  --dry-run
```

## Output Structure

### Single trajectory output

```
tmp/test_single/
  adapted_trajectory.json          # grounded trajectory with concrete IDs
  trajectory_execution_metadata.json  # per-step results + image paths
  plan.json                        # tool call list (standalone)
  metadata.json                    # execution log
  images/traj_000000/              # rendered by get_image steps
    0_top_view_agent_0.png
    0_room_view_agent_0.png
    0_map_agent_0.png
    3_wrist_agent_0.png
    3_agentview_center_agent_0.png
    ...
  room_view.mp4                    # per-camera videos (unless --skip-videos)
  top_view.mp4
  robot0_agentview_center.mp4
  ...
```

### Sweep output (single combo)

When using one layout/style/seed (the default), the structure is flat:

```
tmp/sweep_output/
  sweep_summary.json               # overall results for all trajectories
  hot_dog_setup/
    traj_000000/
      adapted_trajectory.json
      trajectory_execution_metadata.json
      plan.json
      metadata.json
      images/traj_000000/
        0_top_view_agent_0.png
        ...
    traj_000001/
      ...
  prepare_coffee/
    traj_000000/
      ...
```

### Sweep output (multiple combos)

When sweeping across layouts/styles/seeds, each combo gets its own subdirectory:

```
tmp/sweep_output/
  sweep_summary.json
  hot_dog_setup/
    traj_000000/
      L11_S34_sd42/
        adapted_trajectory.json
        trajectory_execution_metadata.json
        plan.json
        metadata.json
        images/traj_000000/
          ...
      L11_S42_sd42/
        ...
      L56_S34_sd42/
        ...
    traj_000001/
      L11_S34_sd42/
        ...
```

No videos are generated in sweep mode. Images are only those rendered by `get_image` tool calls in the trajectory — there is no bulk before/after frame dump.

## How Images Are Generated

Images are **not** rendered for every tool step. They are rendered only when the trajectory contains `get_image` tool calls, which the LLM planner interleaves around action steps:

```
get_image (before)  →  renders wrist + agentview
pick_up_object      →  executes action
get_image (after)   →  renders wrist + agentview
```

Each `get_image` step specifies:
- `views`: which cameras to render (e.g. `["wrist", "agentview_center"]`, `["top_view", "room_view", "map"]`)
- `image_paths`: where to save each rendered image

View names are agent-relative (`wrist`, `agentview_center`, `agentview_left`, `agentview_right`) and map to concrete camera names like `robot0_eye_in_hand`, `robot1_agentview_center` based on which agent requested the image. Global views (`top_view`, `room_view`, `map`) are shared.

## Sweep Summary

After a sweep, `sweep_summary.json` contains:

```json
{
  "input_dir": "...",
  "layouts": [11, 56],
  "styles": [34, 42],
  "seeds": [42, 99],
  "scene_combos": 8,
  "trajectories": 24,
  "total": 192,
  "succeeded": 188,
  "failed": 4,
  "results": [
    {
      "status": "ok",
      "task": "HotDogSetup",
      "steps_succeeded": 17,
      "steps_total": 17,
      "images_rendered": 22,
      "elapsed_s": 45.3,
      "task_dir": "hot_dog_setup",
      "traj_idx": 0,
      "layout": 11,
      "style": 34,
      "seed": 42
    },
    ...
  ]
}
```

## Sweep Script Options

```
--input-dir       Dataset root directory (required)
--output-dir      Output root directory (required)
--task            Filter to a single task directory name
--indices         Filter to specific trajectory indices (e.g. 0 1 2)
--layouts         Kitchen layout ids to sweep (default: 11)
--styles          Kitchen style ids to sweep (default: 34)
--seeds           Environment seeds to sweep (default: 42)
--robots          Number of robots (default: 2)
--placement       Robot placement strategy: grid or continuous (default: grid)
--cell-size       Grid cell size in meters (default: 0.05)
--dry-run         Print what would run without executing
```
