#!/bin/bash
# Launch N evaluation clients matching the running T-shape parallel server.
# Reads server connection info from logs/robocasa_eval/.server_tshape_info.
#
# Usage:
#   bash parallel_client.sh [TASK_SET]
#
# Optional env vars (with defaults):
#   TASK_SET     Task set name (default: atomic_seen, also $1)
#                Allowed values: atomic_seen | composite_seen | composite_unseen
#   SPLIT        Dataset split (default: pretrain; legacy "target" also accepted)
#   NUM_TRIALS   Episodes per task (default: 50)
#   REPLAN_STEPS Re-query model every N env steps (default: 20)
#   GPU_OFFSET   First GPU index used (default: 4; clients use GPU_OFFSET..GPU_OFFSET+NUM_WORKERS-1)
#   LOG_DIR      Where to save eval logs.
#                If unset, derived from CHECKPOINT path under ./logs/eval/
#   SEED         Client-side seed for np.random + robocasa env init (default: 7)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/model_evals/backends/gwp/src:$PROJECT_ROOT:$PROJECT_ROOT/robosuite:${PYTHONPATH:-}"

TASK_SET="${1:-${TASK_SET:-atomic_seen}}"
SPLIT="${SPLIT:-pretrain}"
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-20}"
CLIENT_SEED="${SEED:-7}"
GPU_OFFSET="${GPU_OFFSET:-4}"

# Read server info (checkpoint, port, num_workers, fps, action_chunk).
INFO_FILE="logs/robocasa_eval/.server_tshape_info"
if [ ! -f "$INFO_FILE" ]; then
    echo "ERROR: Server info not found: $INFO_FILE"
    echo "Please start parallel_server_tshape.sh first."
    exit 1
fi
echo "Using server info: $INFO_FILE"
source "$INFO_FILE"
FPS="${FPS:-20}"
ACTION_CHUNK="${ACTION_CHUNK:-24}"
# Drop server-side SEED so it doesn't override CLIENT_SEED below.
unset SEED

# Derive LOG_DIR from checkpoint path if not explicitly set.
if [[ -z "${LOG_DIR:-}" ]]; then
    CKPT_DIR="$(dirname "$CHECKPOINT")"
    EXP_NAME="$(basename "$(dirname "$CKPT_DIR")")"
    CKPT_NAME="$(basename "$CKPT_DIR")"
    MODEL_NAME="$(basename "$CHECKPOINT" .pt)"
    LOG_DIR="./logs/eval/${EXP_NAME}/${CKPT_NAME}/${MODEL_NAME}"
fi

TIMESTAMP=$(date +%m%d_%H%M)
CLIENT_LOG_DIR="logs/robocasa_eval/client/${TIMESTAMP}"
mkdir -p "$CLIENT_LOG_DIR" "$LOG_DIR"

echo "============================================================"
echo "  Launching $NUM_WORKERS clients"
echo "  Task set:    $TASK_SET"
echo "  Split:       $SPLIT"
echo "  FPS (eval):  $FPS  |  action_chunk: $ACTION_CHUNK (~$(awk "BEGIN{printf \"%.2f\", $ACTION_CHUNK/$FPS}")s per chunk, must match server)"
echo "  Ports:       $BASE_PORT - $((BASE_PORT + NUM_WORKERS - 1))"
echo "  GPU range:   $GPU_OFFSET - $((GPU_OFFSET + NUM_WORKERS - 1))"
echo "  Log dir:     $LOG_DIR"
echo "  Term logs:   $CLIENT_LOG_DIR"
echo "============================================================"

PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    PORT=$((BASE_PORT + i))
    GPU=$((GPU_OFFSET + i))
    LOG="${CLIENT_LOG_DIR}/client_${i}.log"

    echo "  [Client $i] GPU=$GPU  Port=$PORT  Worker=$i/$NUM_WORKERS  Log=$LOG"

    CUDA_VISIBLE_DEVICES=$GPU python -u model_evals/backends/gwp/inference_client.py \
        --port $PORT \
        --task_set $TASK_SET \
        --split $SPLIT \
        --num_trials $NUM_TRIALS \
        --replan_steps $REPLAN_STEPS \
        --action_chunk $ACTION_CHUNK \
        --log_dir "$LOG_DIR" \
        --worker_id $i \
        --num_workers $NUM_WORKERS \
        --seed $CLIENT_SEED \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

PID_FILE="${CLIENT_LOG_DIR}/pids.txt"
echo "kill ${PIDS[*]}" > "$PID_FILE"
echo "" >> "$PID_FILE"
echo "# Client PIDs - $(date)" >> "$PID_FILE"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "client_${i}: ${PIDS[$i]}" >> "$PID_FILE"
done

echo ""
echo "Client PIDs: ${PIDS[*]}"
echo "PIDs saved to: $PID_FILE"
echo "Logs:         ${CLIENT_LOG_DIR}/client_{0..$((NUM_WORKERS-1))}.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${CLIENT_LOG_DIR}/client_0.log"
echo "  # or watch all:"
echo "  tail -f ${CLIENT_LOG_DIR}/client_*.log"
echo ""

wait

echo ""
echo "============================================================"
echo "  All $NUM_WORKERS workers finished!"
echo "  Per-task stats:  $LOG_DIR/<task>/<timestamp>/stats.json"
echo "============================================================"

# ---------------------------------------------------------------------------
# Aggregate per-task success rates and print a summary table.
# Picks the LATEST stats.json under each task dir (by mtime), so repeated
# runs with the same LOG_DIR won't mix old and new numbers.
# ---------------------------------------------------------------------------
SUMMARY_TS=$(date +%m%d_%H%M)
SUMMARY_FILE="$LOG_DIR/summary_${SUMMARY_TS}.json"

python - "$LOG_DIR" "$SUMMARY_FILE" <<'PY'
import json
import os
import sys
from glob import glob

log_dir = sys.argv[1]
summary_file = sys.argv[2]

rows = []
for task_dir in sorted(glob(os.path.join(log_dir, "*"))):
    if not os.path.isdir(task_dir):
        continue
    task_name = os.path.basename(task_dir)
    stats_paths = glob(os.path.join(task_dir, "*", "stats.json"))
    if not stats_paths:
        continue
    stats_paths.sort(key=os.path.getmtime, reverse=True)
    latest = stats_paths[0]
    try:
        with open(latest) as f:
            s = json.load(f)
        rows.append({
            "task": task_name,
            "num_episodes": int(s.get("num_episodes", 0)),
            "success_rate": float(s.get("success_rate", 0.0)),
            "stats_path": latest,
        })
    except Exception as e:
        print(f"  [WARN] Failed to read {latest}: {e}")

if not rows:
    print("\n[Summary] No stats.json found under:", log_dir)
    sys.exit(0)

name_w = max(len(r["task"]) for r in rows)
name_w = max(name_w, len("Task"))

print("")
print("=" * 60)
print("  Evaluation Summary")
print("=" * 60)
print(f"  {'Task'.ljust(name_w)}  {'N':>4}  {'Success':>8}  {'Rate':>7}")
print(f"  {'-' * name_w}  {'-' * 4}  {'-' * 8}  {'-' * 7}")

total_ep = 0
total_succ = 0
macro_sum = 0.0
for r in rows:
    n = r["num_episodes"]
    sr = r["success_rate"]
    succ = int(round(sr * n))
    total_ep += n
    total_succ += succ
    macro_sum += sr
    print(f"  {r['task'].ljust(name_w)}  {n:>4d}  {succ:>8d}  {sr*100:>6.1f}%")

macro_avg = macro_sum / len(rows) if rows else 0.0
micro_avg = total_succ / total_ep if total_ep else 0.0
print(f"  {'-' * name_w}  {'-' * 4}  {'-' * 8}  {'-' * 7}")
print(f"  {'Macro avg (per-task mean)'.ljust(name_w)}  {'':>4}  {'':>8}  {macro_avg*100:>6.1f}%")
print(f"  {'Micro avg (episode-weighted)'.ljust(name_w)}  {total_ep:>4d}  {total_succ:>8d}  {micro_avg*100:>6.1f}%")
print("=" * 60)

out = {
    "log_dir": log_dir,
    "num_tasks": len(rows),
    "total_episodes": total_ep,
    "total_successes": total_succ,
    "macro_success_rate": macro_avg,
    "micro_success_rate": micro_avg,
    "per_task": rows,
}
os.makedirs(os.path.dirname(summary_file), exist_ok=True)
with open(summary_file, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Summary written to: {summary_file}")
PY
