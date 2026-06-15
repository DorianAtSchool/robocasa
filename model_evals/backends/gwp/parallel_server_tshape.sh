#!/bin/bash
# Launch N inference servers using the T-shape image layout.
# Uses --tshape --tshape_head_index 2 --dst_size 320 256, writes shared info
# to .server_tshape_info, and gives every replica the same --seed so
# sample-noise is reproducible across workers.
#
# Usage:
#   bash parallel_server_tshape.sh [CHECKPOINT] [ACTION_CHUNK] [SEED]
#
# Required (positional or env var):
#   CHECKPOINT    Path to model.pt   (also accepted as $1)
#
# Optional env vars (with defaults):
#   STATS_PATH    Path to norm_stats_delta.json
#                 (default: model_evals/backends/gwp/assets/norm_stats_delta.json)
#   MODEL_ID      HF repo id or local path of base Wan2.2 model
#                 (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers)
#   ACTION_CHUNK  Action chunk length, must match training (default: 24, also $2)
#   NUM_FRAMES    Latent temporal length, must match training (default: 24)
#   NUM_STEPS     Flow-matching denoising steps (default: 10)
#   SEED          Sample-noise seed shared across replicas (default: 42, also $3)
#   BASE_PORT     First server port (default: 16055)
#   NUM_WORKERS   Number of replicas (default: 4)
#   GPU_OFFSET    First GPU index used (default: 4; servers use GPU_OFFSET..GPU_OFFSET+NUM_WORKERS-1)
#   TSHAPE_HEAD_INDEX  Index of head view in image list (default: 2 = agentview_right)
#   FPS           Eval control rate, only stored to .server_tshape_info (default: 20)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/model_evals/backends/gwp/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT="${1:-${CHECKPOINT:-$PROJECT_ROOT/model_evals/backends/gwp/ckpts/model.pt}}"
STATS_PATH="${STATS_PATH:-$PROJECT_ROOT/model_evals/backends/gwp/assets/norm_stats_delta.json}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_FRAMES="${NUM_FRAMES:-24}"
ACTION_CHUNK="${2:-${ACTION_CHUNK:-24}}"
SEED="${3:-${SEED:-42}}"
BASE_PORT="${BASE_PORT:-16055}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_OFFSET="${GPU_OFFSET:-4}"
TSHAPE_HEAD_INDEX="${TSHAPE_HEAD_INDEX:-2}"
FPS="${FPS:-20}"

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: must provide CHECKPOINT (as \$1 or env var)."
    echo "  CHECKPOINT=model_evals/backends/gwp/ckpts/model.pt bash $0"
    exit 1
fi
if [[ ! -f "$STATS_PATH" ]]; then
    echo "ERROR: STATS_PATH not found: $STATS_PATH"
    exit 1
fi

TIMESTAMP=$(date +%m%d_%H%M)
SERVER_LOG_DIR="logs/robocasa_eval/server_tshape/${TIMESTAMP}"
mkdir -p "$SERVER_LOG_DIR"

echo "============================================================"
echo "  Launching $NUM_WORKERS servers [T-shape] (GPU $GPU_OFFSET-$((GPU_OFFSET+NUM_WORKERS-1)))"
echo "  Model:       $MODEL_ID"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Stats:       $STATS_PATH"
echo "  FPS (eval):  $FPS  |  action_chunk: $ACTION_CHUNK (~$(awk "BEGIN{printf \"%.2f\", $ACTION_CHUNK/$FPS}")s per chunk)"
echo "  Ports:       $BASE_PORT - $((BASE_PORT + NUM_WORKERS - 1))"
echo "  Layout:      T-shape (head_index=$TSHAPE_HEAD_INDEX)"
echo "  Seed:        $SEED (shared across all $NUM_WORKERS servers)"
echo "  Logs:        $SERVER_LOG_DIR"
echo "============================================================"

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    GPU=$((GPU_OFFSET + i))
    LOG="${SERVER_LOG_DIR}/server_${i}.log"

    echo "  [Server $i] GPU=$GPU  Port=$PORT  Log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU python -u model_evals/backends/gwp/inference_server.py \
        --model_id "$MODEL_ID" \
        --checkpoint_path "$CHECKPOINT" \
        --stats_path "$STATS_PATH" \
        --port $PORT \
        --num_steps $NUM_STEPS \
        --num_frames $NUM_FRAMES \
        --action_chunk $ACTION_CHUNK \
        --action_only \
        --dst_size 320 256 \
        --tshape \
        --tshape_head_index $TSHAPE_HEAD_INDEX \
        --zero_action_dims 3 \
        --ctrl_mode_dim 4 \
        --seed $SEED \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

PID_FILE="${SERVER_LOG_DIR}/pids.txt"
echo "kill ${PIDS[*]}" > "$PID_FILE"
echo "" >> "$PID_FILE"
echo "# Server PIDs (T-shape) - $(date)" >> "$PID_FILE"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "server_${i}: ${PIDS[$i]}" >> "$PID_FILE"
done

echo ""
echo "Server PIDs: ${PIDS[*]}"
echo "PIDs saved to: $PID_FILE"
echo "Waiting for all servers to be ready..."

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    LOG="${SERVER_LOG_DIR}/server_${i}.log"
    while ! grep -q "Server listening" "$LOG" 2>/dev/null; do
        sleep 2
    done
    echo "  [Server $i] Ready."
done

echo ""
echo "All $NUM_WORKERS servers are ready!"
echo "To stop: kill ${PIDS[*]}"
echo ""

INFO_FILE="logs/robocasa_eval/.server_tshape_info"
{
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "SERVER_LOG_DIR=${SERVER_LOG_DIR}"
    echo "BASE_PORT=${BASE_PORT}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "FPS=${FPS}"
    echo "ACTION_CHUNK=${ACTION_CHUNK}"
    echo "TSHAPE=1"
    echo "SEED=${SEED}"
} > "$INFO_FILE"
echo "Server info written to: $INFO_FILE"

wait
