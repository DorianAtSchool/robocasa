#!/bin/bash
# π0.5 / openpi RoboCasa inference server.
#
# Requires the robocasa-benchmark/openpi fork installed or available via
# OPENPI_ROOT. The RoboCasa leaderboard π0.5 submission used:
#   config: pi05_pretrain_human300
#   checkpoint: robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-"0"}
export OPENPI_SKIP_LOCAL_NORM_STATS_FALLBACK=${OPENPI_SKIP_LOCAL_NORM_STATS_FALLBACK:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/external/openpi}"
if [[ -n "$OPENPI_ROOT" ]]; then
    export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client:$OPENPI_ROOT/src:$OPENPI_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"
fi

CHECKPOINT="${2:-${CHECKPOINT:-$PROJECT_ROOT/model_evals/backends/pi05/ckpts/pi05_pretrain_human300/multitask_learning/75000}}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_pretrain_human300}"
PORT="${PORT:-8000}"
ACTION_CHUNK="${ACTION_CHUNK:-50}"
FPS="${FPS:-20}"
SERVE_POLICY_SCRIPT="${SERVE_POLICY_SCRIPT:-}"

if [[ -z "$SERVE_POLICY_SCRIPT" && -n "$OPENPI_ROOT" ]]; then
    SERVE_POLICY_SCRIPT="$OPENPI_ROOT/scripts/serve_policy.py"
fi
if [[ -z "$SERVE_POLICY_SCRIPT" ]]; then
    echo "ERROR: OPENPI_ROOT or SERVE_POLICY_SCRIPT is required."
    echo "  Default expected path: $PROJECT_ROOT/external/openpi"
    exit 1
fi
if [[ ! -f "$SERVE_POLICY_SCRIPT" ]]; then
    echo "ERROR: serve_policy.py not found: $SERVE_POLICY_SCRIPT"
    echo "Clone the RoboCasa openpi fork into external/openpi or set OPENPI_ROOT."
    exit 1
fi

if [[ ! -e "$CHECKPOINT" ]]; then
    echo "ERROR: CHECKPOINT not found: $CHECKPOINT"
    echo "Download the leaderboard checkpoint, for example:"
    echo "  hf download robocasa/robocasa365_checkpoints --repo-type model \\"
    echo "    --include 'pi05_pretrain_human300/multitask_learning/75000/**' \\"
    echo "    --local-dir model_evals/backends/pi05/ckpts"
    exit 1
fi

echo "============================================================"
echo "  pi05 RoboCasa Server"
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  Config:      $POLICY_CONFIG"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Port:        $PORT"
echo "  FPS:         $FPS"
echo "  Action chunk:$ACTION_CHUNK"
echo "  OPENPI_ROOT: ${OPENPI_ROOT:-<installed package>}"
echo "  Serve script:$SERVE_POLICY_SCRIPT"
echo "============================================================"

INFO_FILE="logs/robocasa_eval/.server_pi05_info"
mkdir -p "$(dirname "$INFO_FILE")"
{
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "BASE_PORT=${PORT}"
    echo "NUM_WORKERS=1"
    echo "FPS=${FPS}"
    echo "ACTION_CHUNK=${ACTION_CHUNK}"
    echo "POLICY_CONFIG=${POLICY_CONFIG}"
} > "$INFO_FILE"
echo "Server info written to: $INFO_FILE"

python -u "$SERVE_POLICY_SCRIPT" \
    --port="$PORT" \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$CHECKPOINT"
