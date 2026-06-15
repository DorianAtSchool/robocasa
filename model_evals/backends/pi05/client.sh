#!/bin/bash
# π0.5 / openpi RoboCasa Evaluation Client.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-"1"}
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/external/openpi}"
if [[ -d "$OPENPI_ROOT" ]]; then
    export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client:$OPENPI_ROOT/src:$OPENPI_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"
fi

HOST="${HOST:-0.0.0.0}"
SPLIT="${SPLIT:-pretrain}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    TASK_SET="$1"
    shift
else
    TASK_SET="${TASK_SET:-atomic_seen}"
fi
CLIENT_EXTRA_ARGS=("$@")
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
CLIENT_SEED="${SEED:-7}"

INFO_FILE="logs/robocasa_eval/.server_pi05_info"
if [ -f "$INFO_FILE" ]; then
    echo "Using server info: $INFO_FILE"
    source "$INFO_FILE"
    PORT="${PORT:-$BASE_PORT}"
    ACTION_CHUNK="${ACTION_CHUNK:-50}"
    unset SEED
else
    PORT="${PORT:-8000}"
    ACTION_CHUNK="${ACTION_CHUNK:-50}"
fi

if [[ -z "${LOG_DIR:-}" && -n "${CHECKPOINT:-}" ]]; then
    LOG_DIR="./model_evals/backends/pi05/logs/eval/$(basename "$CHECKPOINT")"
fi
LOG_DIR="${LOG_DIR:-./model_evals/backends/pi05/logs/eval/single/$(date +%m%d_%H%M)}"
mkdir -p "$LOG_DIR"

EXTRA_ROBOT_ARGS=()
if [[ "${EXTRA_ROBOT:-0}" == "1" ]]; then
    EXTRA_ROBOT_ARGS+=(--extra_robot)
fi

echo "============================================================"
echo "  pi05 RoboCasa Client"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Server:       $HOST:$PORT"
echo "  Task sets:    $TASK_SET"
echo "  Split:        $SPLIT"
echo "  Trials/task:  $NUM_TRIALS"
echo "  Replan:       every $REPLAN_STEPS steps"
echo "  Action chunk: $ACTION_CHUNK"
echo "  Log dir:      $LOG_DIR"
echo "  Extra robot:  ${EXTRA_ROBOT:-0}"
echo "  Extra args:   ${CLIENT_EXTRA_ARGS[*]:-}"
echo "============================================================"

python -u model_evals/backends/pi05/inference_client.py \
    --host "$HOST" \
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
