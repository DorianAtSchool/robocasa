#!/usr/bin/env bash
set -euo pipefail

# Prints the wrapper usage for timestamp-based image post-processing runs.
usage() {
  cat <<'EOF'
Usage: bash scripts/post_process_task_level_images.sh <run_timestamp> [image_cli_args...]

Post-processes every task summary under:
  data_generation/task_level/data/raw/<run_timestamp>/*/summary.json

Writes the copied post-processed trajectories to:
  data_generation/task_level/data/pre_image/<run_timestamp>/*/summary.json

Additional CLI arguments are forwarded to:
  python -m data_generation.task_level.generation.image.cli

Do not pass --dataset or --output-dataset to this wrapper.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

run_timestamp="$1"
shift

for arg in "$@"; do
  if [[ "$arg" == "--dataset" || "$arg" == --dataset=* ]]; then
    echo "Pass the run timestamp as the first argument instead of --dataset." >&2
    exit 1
  fi
  if [[ "$arg" == "--output-dataset" || "$arg" == --output-dataset=* ]]; then
    echo "This wrapper does not support overriding --output-dataset." >&2
    exit 1
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
data_root="${ROBOCASA_TASK_LEVEL_DATA_ROOT:-$repo_root/data_generation/task_level/data}"
run_dir="$data_root/raw/$run_timestamp"

if [[ ! -d "$run_dir" ]]; then
  echo "Raw run directory not found: $run_dir" >&2
  exit 1
fi

# Expand the timestamp into one summary per task while ignoring the run-level
# summary file stored directly under the timestamped directory.
shopt -s nullglob
summary_paths=("$run_dir"/*/summary.json)
shopt -u nullglob

if [[ ${#summary_paths[@]} -eq 0 ]]; then
  echo "No task summary files found under: $run_dir" >&2
  exit 1
fi

for summary_path in "${summary_paths[@]}"; do
  python -m data_generation.task_level.generation.image.cli \
    --dataset "$summary_path" \
    "$@"
done
