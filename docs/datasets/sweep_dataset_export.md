# Sweep Dataset Export

This document describes the current sweep-output compression policy, the Hugging Face dataset structure produced by `scripts/sweep_trajectories.py`, storage requirements, and example commands for local generation and `--push-to-hub`.

## Compression Policy

The current sweep/export path is optimized for storage efficiency:

- camera renders are saved as JPEG with quality `85`
- top-down and room renders use JPEG as well
- map images remain PNG
- large episode JSON is not duplicated into every parquet row
- per-camera `initial.jpg` and `pre_initial_state.jpg` debug frames are disabled by default
- MP4 videos are skipped by default unless `--videos` is set

### What Changed

Compared with the original eager-PNG / duplicated-JSON workflow:

- camera images are much smaller because they are stored as JPEG instead of PNG
- maps stay lossless because PNG compresses those flat-color images better than JPEG
- trajectory JSON is uploaded as sidecar files instead of being repeated in every dataset row

## Dataset Structure

The exported Hub dataset is a single flat step-level table.

Each row is one tool step, not one whole trajectory.

### Core Columns

- `episode_id`: unique run id, for example `hot_dog_setup/traj_000000/L11_S34_sd42`
- `task`
- `task_dir`
- `layout`
- `style`
- `seed`
- `num_steps`: total number of steps in the episode
- `step_index`
- `tool_name`
- `tool_args`
- `robot_idx`
- `success`

### Image Columns

- `room_view`
- `top_view`
- `map`
- `agentview_center`
- `agentview_left`
- `agentview_right`
- `wrist`

These are standard Hugging Face `Image()` columns, so the viewer can render them directly.

### Sidecar Episode Metadata

Large JSON is stored once per episode as sidecar files in the dataset repo and referenced by path:

- `adapted_trajectory_path`
- `original_trajectory_path`
- `execution_metadata_path`

This keeps the table flat and viewable while avoiding repeated JSON blobs in parquet.

### Repo Layout

The dataset repo effectively contains:

```text
README.md
data/train-*.parquet
sweep_summary.json
<task>/traj_<idx>/L<layout>_S<style>_sd<seed>/adapted_trajectory.json
<task>/traj_<idx>/L<layout>_S<style>_sd<seed>/original_trajectory.json
<task>/traj_<idx>/L<layout>_S<style>_sd<seed>/trajectory_execution_metadata.json
```

## Measured Sample

The numbers below come from the current sample sweep output:

- base trajectories: `24`
- scene combos in sample: `2`
- total output trajectories: `48`
- total dataset rows after flattening: `1892`
- average steps per trajectory: about `39.4`

## Storage Requirements

All estimates below are based on the measured sample and assume:

- no MP4 videos
- no per-camera `initial.jpg` / `pre_initial_state.jpg`
- `initial_map.png` is still kept

### Per-Trajectory Storage

Local recording output:

- camera JPEGs: `1.90 MB`
- step `map` PNGs: `0.70 MB`
- `initial_map.png`: `0.70 MB`
- JSON metadata: `0.19 MB`
- total local: `3.49 MB`

Cloud storage on Hugging Face:

- parquet data: about `2.62 MB`
- uploaded sidecar JSON: about `0.11 MB`
- total repo storage: `2.73 MB`

Loading back for training with `load_dataset()`:

- download/cache size: `1.86 MB` per trajectory

This training-load number is smaller than full repo storage because `load_dataset()` primarily downloads the parquet shards, not the sidecar JSON files.

### Optional Overheads

If you enable optional outputs, add roughly:

- MP4 videos: `1.50 MB` per trajectory
- per-camera debug `initial` / `pre_initial_state` frames: `0.68 MB` per trajectory

If you also remove `initial_map.png`, local output drops from `3.49 MB` to `2.79 MB` per trajectory.

### Scaling Estimates

#### Local Recording Output

Assumes current defaults plus `initial_map.png`:

| Trajectories | Local Storage |
|---|---:|
| 1,000 | 3.49 GB |
| 100,000 | 349.0 GB |
| 1,000,000 | 3.49 TB |

If you also drop `initial_map.png`:

| Trajectories | Local Storage |
|---|---:|
| 1,000 | 2.79 GB |
| 100,000 | 278.9 GB |
| 1,000,000 | 2.79 TB |

#### Cloud Repo Storage

Full dataset repo size on Hugging Face:

| Trajectories | HF Repo Storage |
|---|---:|
| 1,000 | 2.73 GB |
| 100,000 | 272.8 GB |
| 1,000,000 | 2.73 TB |

#### Loading Back for Training

Approximate cache/download footprint when using `datasets.load_dataset()`:

| Trajectories | Training Download / Cache |
|---|---:|
| 1,000 | 1.86 GB |
| 100,000 | 185.8 GB |
| 1,000,000 | 1.86 TB |

## Sweep Cardinality

`scripts/sweep_trajectories.py` multiplies the base trajectory count by:

```text
len(layouts) * len(styles) * len(seeds)
```

So:

```text
total output trajectories
= base_trajectories * layouts * styles * seeds
```

For the current sample:

```text
24 * layouts * styles * seeds
```

### Example: About 1k Trajectories

To get about `1000` output trajectories from the current `24` base trajectories, target:

```text
layouts * styles * seeds ~= 42
```

A balanced choice is:

- `3` layouts
- `2` styles
- `7` seeds

Which gives:

```text
24 * 3 * 2 * 7 = 1008
```

That corresponds to roughly:

- local storage: `3.52 GB`
- HF repo storage: `2.75 GB`
- training download/cache: `1.87 GB`

## Example Commands

### Single Local Sweep

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output
```

### Sweep Multiple Layouts, Styles, and Seeds

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 42 56 \
  --styles 34 42 \
  --seeds 1 2 3 4 5 6 7
```

### About 1k Trajectories Plus Push To Hub

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output_1k \
  --layouts 11 42 56 \
  --styles 34 42 \
  --seeds 1 2 3 4 5 6 7 \
  --push-to-hub DorianAtSchool/robocasa-trajectories-single
```

### Limit To Specific Tasks

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --tasks hot_dog_setup prepare_coffee
```

### Limit To Specific Trajectory Indices

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --indices 0 1 2
```

### Dry Run

```bash
python scripts/sweep_trajectories.py \
  --input-dir data_generation/task_level/data/image/20260324T031125Z \
  --output-dir tmp/sweep_output \
  --layouts 11 42 56 \
  --styles 34 42 \
  --seeds 1 2 3 4 5 6 7 \
  --dry-run
```

### Push Existing Sweep Output To Hub

```bash
python scripts/push_sweep_to_hub.py \
  --sweep-dir tmp/sweep_output_1k \
  --repo-id DorianAtSchool/robocasa-trajectories-single
```

### Test The Published Dataset

```bash
python scripts/test_hf_dataset.py \
  --repo-id DorianAtSchool/robocasa-trajectories-single
```

### Browse Sweep Output Locally

```bash
.venv/bin/python scripts/visualize_sweep_output.py \
  --sweep-dir tmp/sweep_output_1k
```

## Notes

- If you use `--videos`, local storage increases substantially.
- If you want the smallest local footprint, generate without videos and avoid saving extra debug frames.
- If you only care about training with `load_dataset()`, the relevant disk number is the training download/cache size, not the full repo size.
