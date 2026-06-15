#!/bin/bash
# RLDX-1 RoboCasa inference server.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-"0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

RLDX_ROOT="${RLDX_ROOT:-$PROJECT_ROOT/external/rldx-1}"
MODEL_PATH="${2:-${MODEL_PATH:-RLWRLD/RLDX-1-FT-RC365}}"
RLDX_HOST="${RLDX_HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
DEVICE="${DEVICE:-cuda}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GENERAL_EMBODIMENT}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
COMPILE="${COMPILE:-none}"
RTC_INFERENCE_MODE="${RTC_INFERENCE_MODE:-}"

if [[ ! -d "$RLDX_ROOT" ]]; then
    echo "ERROR: RLDX_ROOT not found: $RLDX_ROOT"
    echo "Clone RLDX-1 into external/rldx-1 or set RLDX_ROOT."
    exit 1
fi
if [[ ! -f "$RLDX_ROOT/rldx/eval/run_rldx_server.py" ]]; then
    echo "ERROR: run_rldx_server.py not found under: $RLDX_ROOT"
    exit 1
fi

export PYTHONPATH="$RLDX_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"

INFO_FILE="logs/robocasa_eval/.server_rldx_info"
mkdir -p "$(dirname "$INFO_FILE")"
{
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "BASE_PORT=${PORT}"
    echo "RLDX_HOST=${RLDX_HOST}"
    echo "NUM_WORKERS=1"
    echo "ACTION_CHUNK=${ACTION_CHUNK}"
    echo "REPLAN_STEPS=${REPLAN_STEPS}"
    echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
} > "$INFO_FILE"

echo "============================================================"
echo "  RLDX-1 RoboCasa Server"
echo "  GPUs:          $CUDA_VISIBLE_DEVICES"
echo "  Model path:    $MODEL_PATH"
echo "  Host/port:     $RLDX_HOST:$PORT"
echo "  Device:        $DEVICE"
echo "  Embodiment:    $EMBODIMENT_TAG"
echo "  Action chunk:  $ACTION_CHUNK"
echo "  Replan steps:  $REPLAN_STEPS"
echo "  Compile:       $COMPILE"
echo "  RLDX_ROOT:     $RLDX_ROOT"
echo "============================================================"
echo "Server info written to: $INFO_FILE"

args=(
    "$RLDX_ROOT/rldx/eval/run_rldx_server.py"
    --model-path "$MODEL_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --device "$DEVICE"
    --use-sim-policy-wrapper
    --host "$RLDX_HOST"
    --port "$PORT"
    --compile "$COMPILE"
)
if [[ -n "$RTC_INFERENCE_MODE" ]]; then
    args+=(--rtc-inference-mode "$RTC_INFERENCE_MODE")
fi

python -u "${args[@]}"
