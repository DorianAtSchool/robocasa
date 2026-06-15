#!/bin/bash
# Run RoboCasa VLA evaluation conditions as separate experiments.
#
# Usage examples:
#   bash model_evals/scripts/run_eval_conditions.sh trajectory_tasks
#   bash model_evals/scripts/run_eval_conditions.sh composite_seen
#   bash model_evals/scripts/run_eval_conditions.sh all
#
# Environment overrides:
#   GPU_IDS="1"
#   NUM_TRIALS=10
#   TRAJECTORY_INDICES="0 1 2"
#   BASE_LOG_DIR="$PWD/logs/robot_interference"
#   TASK_SET="composite_seen"
#   TRAJECTORY_INIT_ROOT="/path/to/verbalized"

set -euo pipefail

GPU_IDS="${GPU_IDS:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
TRAJECTORY_INDICES="${TRAJECTORY_INDICES:-0 1 2}"
TASK_SET="${TASK_SET:-composite_seen}"
BACKEND="${BACKEND:-gwp}"
MODE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_TRAJECTORY_INIT_ROOT="/home/dorian/Projects/robocasa/data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized"
TRAJECTORY_INIT_ROOT="${TRAJECTORY_INIT_ROOT:-$DEFAULT_TRAJECTORY_INIT_ROOT}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$PROJECT_ROOT/logs/robot_interference/$(date +%Y%m%d_%H%M%S)}"

run_condition() {
  local condition_name="$1"
  shift
  echo "============================================================"
  echo "Running condition: $condition_name"
  echo "Backend: $BACKEND"
  echo "Log dir: $BASE_LOG_DIR/$condition_name"
  echo "============================================================"
  NUM_TRIALS="$NUM_TRIALS" \
  LOG_DIR="$BASE_LOG_DIR/$condition_name" \
  bash "model_evals/backends/$BACKEND/client.sh" "$GPU_IDS" "$@"
}

run_trajectory_tasks() {
  run_condition trajectory_tasks_one_robot \
    --eval_trajectory_tasks \
    --trajectory_init_root "$TRAJECTORY_INIT_ROOT" \
    --trajectory_indices $TRAJECTORY_INDICES \
    --trajectory_init_mode none \
    --cameras default

  run_condition trajectory_tasks_two_robot_trajectory \
    --eval_trajectory_tasks \
    --trajectory_init_root "$TRAJECTORY_INIT_ROOT" \
    --trajectory_indices $TRAJECTORY_INDICES \
    --trajectory_init_mode passive_other \
    --trajectory_other_agent agent_1 \
    --cameras default
}

run_registry_tasks() {
  run_condition "${TASK_SET}_one_robot" \
    "$TASK_SET" \
    --trajectory_init_mode none \
    --cameras default

  run_condition "${TASK_SET}_two_robot_traj_or_fallback" \
    "$TASK_SET" \
    --trajectory_init_root "$TRAJECTORY_INIT_ROOT" \
    --trajectory_index 0 \
    --trajectory_init_mode passive_other \
    --trajectory_other_agent agent_1 \
    --missing_trajectory_policy fallback \
    --cameras default
}

case "$MODE" in
  trajectory_tasks)
    run_trajectory_tasks
    ;;
  registry|task_set|composite_seen)
    run_registry_tasks
    ;;
  all)
    run_trajectory_tasks
    run_registry_tasks
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Expected: trajectory_tasks | composite_seen | registry | task_set | all" >&2
    exit 2
    ;;
esac

echo "All requested conditions completed. Logs: $BASE_LOG_DIR"
