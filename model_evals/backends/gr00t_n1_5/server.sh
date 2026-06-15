#!/bin/bash
# GR00T N1.5 RoboCasa inference server.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-"0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

GROOT_ROOT="${GROOT_ROOT:-$PROJECT_ROOT/external/Isaac-GR00T}"
MODEL_PATH="${2:-${MODEL_PATH:-$PROJECT_ROOT/model_evals/backends/gr00t_n1_5/ckpts/gr00t_n1-5/multitask_learning/checkpoint-120000}}"
GROOT_HOST="${GROOT_HOST:-127.0.0.1}"
PORT="${PORT:-5556}"
DATA_CONFIG="${DATA_CONFIG:-panda_omron}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
ACTION_CHUNK="${ACTION_CHUNK:-16}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"

if [[ ! -d "$GROOT_ROOT" ]]; then
    echo "ERROR: GROOT_ROOT not found: $GROOT_ROOT"
    echo "Clone the RoboCasa Isaac-GR00T fork into external/Isaac-GR00T or set GROOT_ROOT."
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    echo "Download the leaderboard checkpoint, for example:"
    echo "  hf download robocasa/robocasa365_checkpoints --repo-type model \\"
    echo "    --include 'gr00t_n1-5/multitask_learning/checkpoint-120000/**' \\"
    echo "    --local-dir model_evals/backends/gr00t_n1_5/ckpts"
    exit 1
fi

export PYTHONPATH="$GROOT_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"

INFO_FILE="logs/robocasa_eval/.server_gr00t_n1_5_info"
mkdir -p "$(dirname "$INFO_FILE")"
{
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "BASE_PORT=${PORT}"
    echo "GROOT_HOST=${GROOT_HOST}"
    echo "NUM_WORKERS=1"
    echo "ACTION_CHUNK=${ACTION_CHUNK}"
    echo "REPLAN_STEPS=${REPLAN_STEPS}"
    echo "DATA_CONFIG=${DATA_CONFIG}"
    echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
} > "$INFO_FILE"

echo "============================================================"
echo "  GR00T N1.5 RoboCasa Server"
echo "  GPUs:           $CUDA_VISIBLE_DEVICES"
echo "  Model path:     $MODEL_PATH"
echo "  Host/port:      $GROOT_HOST:$PORT"
echo "  Data config:    $DATA_CONFIG"
echo "  Embodiment:     $EMBODIMENT_TAG"
echo "  Denoise steps:  $DENOISING_STEPS"
echo "  Action chunk:   $ACTION_CHUNK"
echo "  Replan steps:   $REPLAN_STEPS"
echo "  GROOT_ROOT:     $GROOT_ROOT"
echo "============================================================"
echo "Server info written to: $INFO_FILE"

python -u model_evals/backends/gr00t_n1_5/inference_server.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --host "$GROOT_HOST" \
    --port "$PORT" \
    --denoising-steps "$DENOISING_STEPS"
