# Robot-Interference Batch Scripts

These scripts run one-robot and two-robot evaluation conditions as separate jobs so their logs can be compared cleanly. Run them from the repository root while the selected policy inference server is already running.

## Registry Task Sets

Each task-set script writes two condition folders under `BASE_LOG_DIR`:

```text
one_robot
two_robot_traj_or_fallback
```

Available scripts:

```text
run_atomic_seen_conditions.sh
run_composite_seen_conditions.sh
run_composite_unseen_conditions.sh
run_registry_task_set_conditions.sh
```

Example:

```bash
NUM_TRIALS=20 \
BASE_LOG_DIR="$PWD/logs/robot_interference/atomic_seen_20trials" \
bash model_evals/scripts/run_atomic_seen_conditions.sh
```

`run_registry_task_set_conditions.sh` runs atomic seen, composite seen, and composite unseen in sequence, each under its own subdirectory.

## Trajectory Tasks

`run_trajectory_task_conditions.sh` evaluates tasks discovered from the trajectory root rather than from a RoboCasa registry task set. It writes:

```text
one_robot
two_robot_trajectory
```

Example:

```bash
NUM_TRIALS=1 \
TRAJECTORY_INDICES="0 1 2 3 4" \
BASE_LOG_DIR="$PWD/logs/robot_interference/trajectory_tasks_5idx" \
bash model_evals/scripts/run_trajectory_task_conditions.sh
```

## Common Overrides

```text
GPU_IDS=1
NUM_TRIALS=10
TRAJECTORY_INDEX=0
TRAJECTORY_INDICES="0 1 2"
TRAJECTORY_INIT_ROOT=/path/to/verbalized
BASE_LOG_DIR=/path/to/logs
```
