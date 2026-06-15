#!/bin/bash
# RLDX-1 RoboCasa evaluation client.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-"1"}
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RLDX_ROOT="${RLDX_ROOT:-$PROJECT_ROOT/external/rldx-1}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"

if ! python -c "import zmq" >/dev/null 2>&1; then
    echo "ERROR: pyzmq is not installed in the RoboCasa client environment."
    echo "Install it in the active client env, for example:"
    echo "  python -m pip install pyzmq"
    exit 1
fi

RLDX_HOST="${RLDX_HOST:-127.0.0.1}"
SPLIT="${SPLIT:-pretrain}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    TASK_SET="$1"
    shift
else
    TASK_SET="${TASK_SET:-atomic_seen}"
fi
CLIENT_EXTRA_ARGS=("$@")
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
CLIENT_SEED="${SEED:-7}"

INFO_FILE="logs/robocasa_eval/.server_rldx_info"
if [ -f "$INFO_FILE" ]; then
    echo "Using server info: $INFO_FILE"
    source "$INFO_FILE"
    PORT="${PORT:-$BASE_PORT}"
    ACTION_CHUNK="${ACTION_CHUNK:-16}"
    REPLAN_STEPS="${REPLAN_STEPS:-${REPLAN_STEPS:-8}}"
    unset SEED
else
    PORT="${PORT:-5555}"
    ACTION_CHUNK="${ACTION_CHUNK:-16}"
fi

if [[ -z "${LOG_DIR:-}" && -n "${MODEL_PATH:-}" ]]; then
    LOG_DIR="./model_evals/backends/rldx/logs/eval/$(basename "$MODEL_PATH")"
fi
LOG_DIR="${LOG_DIR:-./model_evals/backends/rldx/logs/eval/single/$(date +%m%d_%H%M)}"
mkdir -p "$LOG_DIR"

EXTRA_ROBOT_ARGS=()
if [[ "${EXTRA_ROBOT:-0}" == "1" ]]; then
    EXTRA_ROBOT_ARGS+=(--extra_robot)
fi

echo "============================================================"
echo "  RLDX-1 RoboCasa Client"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Server:       $RLDX_HOST:$PORT"
echo "  Task sets:    $TASK_SET"
echo "  Split:        $SPLIT"
echo "  Trials/task:  $NUM_TRIALS"
echo "  Replan:       every $REPLAN_STEPS steps"
echo "  Action chunk: $ACTION_CHUNK"
echo "  Log dir:      $LOG_DIR"
echo "  Extra robot:  ${EXTRA_ROBOT:-0}"
echo "  Extra args:   ${CLIENT_EXTRA_ARGS[*]:-}"
echo "============================================================"

python -u model_evals/backends/rldx/inference_client.py \
    --host "$RLDX_HOST" \
    --port "$PORT" \
    --task_set $TASK_SET \
    --split "$SPLIT" \
    --num_trials "$NUM_TRIALS" \
    --replan_steps "$REPLAN_STEPS" \
    --action_chunk "$ACTION_CHUNK" \
    --log_dir "$LOG_DIR" \
    --seed "$CLIENT_SEED" \
    "${EXTRA_ROBOT_ARGS[@]}" \
    "${CLIENT_EXTRA_ARGS[@]}"
