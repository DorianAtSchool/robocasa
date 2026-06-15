#!/bin/bash
# GigaWorld-Policy RoboCasa Evaluation Client (single process, T-shape server)
# Usage:
#   bash client.sh [GPU_IDS] [TASK_SET]
#
# Optional env vars (with defaults):
#   PORT         Server port (default: read from .server_tshape_info, otherwise 16055)
#   HOST         Server host (default: 0.0.0.0)
#   TASK_SET     Space-separated task sets (default: atomic_seen, also $2)
#                Allowed values: atomic_seen | composite_seen | composite_unseen
#   SPLIT        Dataset split (default: pretrain)
#   NUM_TRIALS   Episodes per task (default: 50)
#   REPLAN_STEPS Re-query model every N env steps (default: 20)
#   LOG_DIR      Where to save per-task stats.json + replay videos
#                If unset, derived from CHECKPOINT path when .server_tshape_info exists.
#   SEED         Client-side seed for np.random + robocasa env init (default: 7)
#
# Requires: a running inference_server (e.g. `bash server.sh`).

export CUDA_VISIBLE_DEVICES=${1:-"1"}
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/model_evals/backends/gwp/src:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"

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
REPLAN_STEPS="${REPLAN_STEPS:-20}"
CLIENT_SEED="${SEED:-7}"

INFO_FILE="logs/robocasa_eval/.server_tshape_info"
if [ -f "$INFO_FILE" ]; then
    echo "Using server info: $INFO_FILE"
    source "$INFO_FILE"
    PORT="${PORT:-$BASE_PORT}"
    ACTION_CHUNK="${ACTION_CHUNK:-24}"
    unset SEED
else
    PORT="${PORT:-16055}"
    ACTION_CHUNK="${ACTION_CHUNK:-24}"
fi

if [[ -z "${LOG_DIR:-}" && -n "${CHECKPOINT:-}" ]]; then
    CKPT_DIR="$(dirname "$CHECKPOINT")"
    EXP_NAME="$(basename "$(dirname "$CKPT_DIR")")"
    CKPT_NAME="$(basename "$CKPT_DIR")"
    MODEL_NAME="$(basename "$CHECKPOINT" .pt)"
    LOG_DIR="./logs/eval/${EXP_NAME}/${CKPT_NAME}/${MODEL_NAME}"
fi
LOG_DIR="${LOG_DIR:-./logs/eval/single/$(date +%m%d_%H%M)}"

mkdir -p "$LOG_DIR"

EXTRA_ROBOT_ARGS=()
if [[ "${EXTRA_ROBOT:-0}" == "1" ]]; then
    EXTRA_ROBOT_ARGS+=(--extra_robot)
fi

echo "============================================================"
echo "  GigaWorld-Policy RoboCasa Client"
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

python -u model_evals/backends/gwp/inference_client.py \
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
