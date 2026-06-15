#!/bin/bash
# Run composite_unseen evaluations as two separate conditions:
#   1. one robot baseline
#   2. two robots using trajectory init when available, otherwise fallback spawn

set -euo pipefail

GPU_IDS="${GPU_IDS:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
BACKEND="${BACKEND:-gwp}"
TASK_SET="composite_unseen"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_TRAJECTORY_INIT_ROOT="/home/dorian/Projects/robocasa/data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized"
TRAJECTORY_INIT_ROOT="${TRAJECTORY_INIT_ROOT:-$DEFAULT_TRAJECTORY_INIT_ROOT}"
TRAJECTORY_INDEX="${TRAJECTORY_INDEX:-0}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$PROJECT_ROOT/logs/robot_interference/composite_unseen/$(date +%Y%m%d_%H%M%S)}"

run_condition() {
  local condition_name="$1"
  shift
  echo "============================================================"
  echo "composite_unseen condition: $condition_name"
  echo "Backend: $BACKEND"
  echo "Log dir: $BASE_LOG_DIR/$condition_name"
  echo "============================================================"
  NUM_TRIALS="$NUM_TRIALS" \
  LOG_DIR="$BASE_LOG_DIR/$condition_name" \
  bash "model_evals/backends/$BACKEND/client.sh" "$GPU_IDS" "$@"
}

run_condition one_robot \
  "$TASK_SET" \
  --trajectory_init_mode none \
  --cameras default

run_condition two_robot_traj_or_fallback \
  "$TASK_SET" \
  --trajectory_init_root "$TRAJECTORY_INIT_ROOT" \
  --trajectory_index "$TRAJECTORY_INDEX" \
  --trajectory_init_mode passive_other \
  --trajectory_other_agent agent_1 \
  --missing_trajectory_policy fallback \
  --cameras default

echo "composite_unseen conditions complete. Logs: $BASE_LOG_DIR"
