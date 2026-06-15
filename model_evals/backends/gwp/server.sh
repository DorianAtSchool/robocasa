#!/bin/bash
# GigaWorld-Policy RoboCasa T-shape Inference Server (single GPU)
# Usage:
#   bash server.sh [GPU_IDS] [CHECKPOINT] [ACTION_CHUNK] [SEED]
#
# Required (positional or env var):
#   CHECKPOINT    Path to model.pt checkpoint (also accepted as $2)
#
# Optional env vars (with defaults):
#   STATS_PATH          Path to norm_stats_delta.json
#                       (default: model_evals/backends/gwp/assets/norm_stats_delta.json)
#   MODEL_ID            HF repo id or local path of base Wan2.2 model
#                       (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers)
#   PORT                Server port (default: 16055)
#   ACTION_CHUNK        Action chunk length (default: 24, also $3)
#   NUM_FRAMES          Latent temporal length, must match training (default: 24)
#   NUM_STEPS           Flow-matching denoising steps (default: 10)
#   SEED                Sample-noise seed (default: 42, also $4)
#   TSHAPE_HEAD_INDEX   Index of head view in image list (default: 2 = agentview_right)
#   FPS                 Eval control rate, only stored to .server_tshape_info (default: 20)
#
# Example:
#   bash model_evals/backends/gwp/server.sh 0 ./ckpts/model.pt 24

export CUDA_VISIBLE_DEVICES=${1:-"0"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/model_evals/backends/gwp/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT="${2:-${CHECKPOINT:-$PROJECT_ROOT/model_evals/backends/gwp/ckpts/model.pt}}"
STATS_PATH="${STATS_PATH:-$PROJECT_ROOT/model_evals/backends/gwp/assets/norm_stats_delta.json}"
PORT="${PORT:-16055}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_FRAMES="${NUM_FRAMES:-24}"
ACTION_CHUNK="${3:-${ACTION_CHUNK:-24}}"
SEED="${4:-${SEED:-42}}"
TSHAPE_HEAD_INDEX="${TSHAPE_HEAD_INDEX:-2}"
FPS="${FPS:-20}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: must provide CHECKPOINT (as \$2 or env var)."
    echo "  bash $0 0 model_evals/backends/gwp/ckpts/model.pt 24"
    exit 1
fi
if [[ ! -f "$STATS_PATH" ]]; then
    echo "ERROR: STATS_PATH not found: $STATS_PATH"
    exit 1
fi

echo "============================================================"
echo "  GigaWorld-Policy RoboCasa Server [T-shape]"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  Model:       $MODEL_ID"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Stats:       $STATS_PATH"
echo "  FPS (eval):  $FPS  |  action_chunk: $ACTION_CHUNK (~$(awk "BEGIN{printf \"%.2f\", $ACTION_CHUNK/$FPS}")s per chunk)"
echo "  Port:        $PORT"
echo "  Layout:      T-shape (head_index=$TSHAPE_HEAD_INDEX)"
echo "  Seed:        $SEED"
echo "============================================================"

INFO_FILE="logs/robocasa_eval/.server_tshape_info"
mkdir -p "$(dirname "$INFO_FILE")"
{
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "SERVER_LOG_DIR="
    echo "BASE_PORT=${PORT}"
    echo "NUM_WORKERS=1"
    echo "FPS=${FPS}"
    echo "ACTION_CHUNK=${ACTION_CHUNK}"
    echo "TSHAPE=1"
    echo "SEED=${SEED}"
} > "$INFO_FILE"
echo "Server info written to: $INFO_FILE"

python -u model_evals/backends/gwp/inference_server.py \
    --model_id "$MODEL_ID" \
    --checkpoint_path "$CHECKPOINT" \
    --stats_path "$STATS_PATH" \
    --port "$PORT" \
    --num_steps "$NUM_STEPS" \
    --num_frames "$NUM_FRAMES" \
    --action_chunk "$ACTION_CHUNK" \
    --action_only \
    --dst_size 320 256 \
    --tshape \
    --tshape_head_index "$TSHAPE_HEAD_INDEX" \
    --zero_action_dims 3 \
    --ctrl_mode_dim 4 \
    --seed "$SEED"
