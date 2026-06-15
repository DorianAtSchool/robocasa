#!/bin/bash
# Run the three RoboCasa registry task-set condition scripts:
#   atomic_seen, composite_seen, composite_unseen.
# Each child script runs one-robot and two-robot trajectory-or-fallback conditions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND="${BACKEND:-gwp}"
REGISTRY_BASE_LOG_DIR="${BASE_LOG_DIR:-$PROJECT_ROOT/logs/robot_interference/registry_task_sets/$(date +%Y%m%d_%H%M%S)}"

echo "Combined registry task-set logs: $REGISTRY_BASE_LOG_DIR"
echo "Backend: $BACKEND"

BASE_LOG_DIR="$REGISTRY_BASE_LOG_DIR/atomic_seen" \
BACKEND="$BACKEND" \
bash model_evals/scripts/run_atomic_seen_conditions.sh

BASE_LOG_DIR="$REGISTRY_BASE_LOG_DIR/composite_seen" \
BACKEND="$BACKEND" \
bash model_evals/scripts/run_composite_seen_conditions.sh

BASE_LOG_DIR="$REGISTRY_BASE_LOG_DIR/composite_unseen" \
BACKEND="$BACKEND" \
bash model_evals/scripts/run_composite_unseen_conditions.sh

echo "All registry task-set conditions complete. Logs: $REGISTRY_BASE_LOG_DIR"
