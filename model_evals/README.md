# RoboCasa Multi-Agent VLA Evaluation

This folder contains multi-agent VLA evaluation tooling for the canonical RoboCasa repo. It is separate from the core environment implementation and is intended to compare pretrained VLA models under matched one-robot and two-robot conditions.

Runnable backends live under `model_evals/backends/` and share the same rollout, trajectory-spawn, camera, script, and analysis code. Each backend README documents its server environment, checkpoint location, and client command.

## Folder Layout

```text
model_evals/
  common/            # shared RoboCasa rollout, trajectory spawn, cameras, transport
  backends/
    gwp/              # GigaWorld-Policy backend
    pi05/             # OpenPI pi0.5 backend
    rldx/             # RLDX-1 backend
    gr00t_n1_5/       # GR00T N1.5 backend
  scripts/            # shared one-robot vs two-robot batch scripts
  analysis/           # shared plotting and summary tools
```

## Backend Artifacts

Backend-specific runtime artifacts are stored under each backend folder and ignored by git. For GWP, that means:

```text
model_evals/backends/gwp/assets/
model_evals/backends/gwp/ckpts/
model_evals/backends/gwp/logs/
```

Use equivalent `assets/`, `ckpts/`, and `logs/` paths under `pi05/`, `rldx/`, or `gr00t_n1_5/` for the other model families.

## Evaluation Modes

### Registry Task-Set Eval

Use RoboCasa's built-in task sets, such as `atomic_seen`, `composite_seen`, `composite_unseen`, or combinations passed through `client.sh`.

```bash
NUM_TRIALS=10 \
LOG_DIR="$PWD/model_evals/backends/gwp/logs/composite_seen_traj_or_fallback" \
bash model_evals/backends/gwp/client.sh 1 composite_seen \
  --trajectory_init_mode passive_other \
  --trajectory_other_agent agent_1 \
  --missing_trajectory_policy fallback \
  --cameras default
```

If `--trajectory_init_mode` is not `none` and no `--trajectory_init_root` is provided, the client defaults to the diversity-analysis verbalized sampling root:

```text
/home/dorian/Projects/robocasa/data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized
```

For registry eval, the client tries to find a matching trajectory for each RoboCasa task. `--missing_trajectory_policy` controls what happens when no matching trajectory exists:

```text
error     fail that task immediately
skip      skip that task
fallback  place robot1 with semantic fallback spawning
```

### Trajectory Task Eval

Use the task folders present in the trajectory root instead of a RoboCasa registry task set. This only changes task selection; robot spawning is controlled by the required `--trajectory_init_mode` flag.

```bash
NUM_TRIALS=1 \
LOG_DIR="$PWD/model_evals/backends/gwp/logs/trajectory_tasks_verbalized" \
bash model_evals/backends/gwp/client.sh 1 \
  --eval_trajectory_tasks \
  --trajectory_indices 0 1 2 \
  --trajectory_init_mode passive_other \
  --cameras default
```

This evaluates every task directory under the trajectory root for each requested trajectory index. For example, `--trajectory_indices 0 1 2` evaluates `traj_000000.json`, `traj_000001.json`, and `traj_000002.json` for each trajectory task that has those files.

Each trajectory index is logged separately:

```text
logs/.../<TaskName>/traj_000000/<timestamp>/stats.json
logs/.../<TaskName>/traj_000001/<timestamp>/stats.json
```

`NUM_TRIALS` is still the number of repeated closed-loop episodes per task/trajectory-index job.


### Batch Scripts

The batch scripts run experimental conditions separately so one-robot baselines and two-robot conditions never share a log directory.

#### `scripts/run_trajectory_task_conditions.sh`

Evaluates tasks from the trajectory root, not from a RoboCasa registry task set. It runs two conditions for every selected trajectory task/index:

```text
one_robot
two_robot_trajectory
```

Use it when comparing original VLA behavior against trajectory-based robot1 spawning on our generated trajectory tasks.

```bash
NUM_TRIALS=10 \
TRAJECTORY_INDICES="0 1 2" \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/trajectory_tasks_run" \
bash model_evals/scripts/run_trajectory_task_conditions.sh
```

#### Per-Task-Set Scripts

Each task-set script runs one robot vs. two robots for exactly one RoboCasa registry task set. The two-robot condition uses trajectory placement when a matching trajectory exists and semantic fallback placement otherwise.

```text
scripts/run_atomic_seen_conditions.sh
scripts/run_composite_seen_conditions.sh
scripts/run_composite_unseen_conditions.sh
```

Each script writes:

```text
one_robot
two_robot_traj_or_fallback
```

Examples:

```bash
NUM_TRIALS=10 \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/atomic_seen_run" \
bash model_evals/scripts/run_atomic_seen_conditions.sh
```

```bash
NUM_TRIALS=10 \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/composite_seen_run" \
bash model_evals/scripts/run_composite_seen_conditions.sh
```

```bash
NUM_TRIALS=10 \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/composite_unseen_run" \
bash model_evals/scripts/run_composite_unseen_conditions.sh
```

#### `scripts/run_registry_task_set_conditions.sh`

Runs all three registry task-set scripts in sequence for clarity and debugging:

```text
atomic_seen
composite_seen
composite_unseen
```

Each task set gets its own subdirectory under the shared base log directory.

```bash
NUM_TRIALS=10 \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/registry_run" \
bash model_evals/scripts/run_registry_task_set_conditions.sh
```

Common overrides:

```text
BACKEND=gwp
GPU_IDS=1
NUM_TRIALS=10
TRAJECTORY_INDICES="0 1 2"
TRAJECTORY_INDEX=0
TRAJECTORY_INIT_ROOT=/path/to/verbalized
BASE_LOG_DIR=/path/to/logs
```

Set `BACKEND=pi05`, `BACKEND=rldx`, or `BACKEND=gr00t_n1_5` to reuse the same batch scripts with another backend after its server is running.

### Single-Task Debug

Use one env and one trajectory JSON for qualitative inspection.

```bash
NUM_TRIALS=1 \
LOG_DIR="$PWD/model_evals/backends/gwp/logs/debug_deliver_straw" \
bash model_evals/backends/gwp/client.sh 1 \
  --single_task DeliverStraw \
  --trajectory_init_json /path/to/deliver_straw/trajectories/traj_000005.json \
  --trajectory_init_mode passive_other \
  --trajectory_other_agent agent_1 \
  --cameras default
```

## Analysis

Use the analysis script to compare completed one-robot and two-robot condition logs:

```bash
python model_evals/analysis/report_success_comparison.py \
  model_evals/backends/gwp/logs/robot_interference/atomic_seen_5trials
```

It writes `success_comparison.png`, `success_comparison.csv`, and `summary.json` under `<LOG_DIR>/analysis/`. The plot includes success confidence intervals, task-level runtime, aggregate runtime multiplier, and spawn-source counts. See `model_evals/analysis/README.md` for details.

## Robot Spawning

`passive_other` leaves robot0 at the original RoboCasa task spawn and places only robot1 from `agent_1.location` in the trajectory. The placement uses the same occupancy-grid style used by the multi-agent trajectory runner and records placement metadata in `stats.json`.

`all_agents` maps sorted trajectory agents to robot indices and can move robot0. Do not use it for benchmark-style comparisons unless that is intentional.

Fallback spawning is used only when requested with `--missing_trajectory_policy fallback` and no matching trajectory is found. It uses the same occupancy-grid placement backend as trajectory spawning, but chooses robot1's fixture with this priority:

```text
1. final target / receptacle-like object fixture
2. task-keyword fixture, such as dishwasher, sink, cabinet, coffee machine, etc.
3. primary object source fixture, including init_robot_here objects
4. task init_robot_base_ref
5. nearest fixture to robot0's original start
6. central work surface
```

The chosen placement is recorded in `stats.json` with `spawn_source`, for example:

```text
trajectory
fallback_final_target_fixture
fallback_primary_source_fixture
fallback_init_robot_base_ref
fallback_nearest_robot0
fallback_central_work_surface
```

## Cameras

`--cameras default` records a compact diagnostic set:

```text
robot0_agentview_center
robot0_eye_in_hand
robot1_agentview_center
```

Use `--cameras all` for qualitative debugging or `--cameras render` for the legacy single rendered view.
