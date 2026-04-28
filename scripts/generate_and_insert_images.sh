#!/usr/bin/env bash
set -euo pipefail

# Prints the wrapper usage for timestamp-based task-level sweep runs.
usage() {
  cat <<'EOF'
Usage: bash scripts/generate_and_insert_images.sh <run_timestamp> [--workers N] [sweep_cli_args...]

Sweeps every task trajectory under:
  data_generation/task_level/data/pre_image/<run_timestamp>

Writes the rendered sweep output to:
  data_generation/task_level/data/image/<run_timestamp>

Wrapper arguments:
  -j, --workers N    Maximum parallel trajectory workers (default: 1)
  -v, --verbose      Show the sweep CLI's normal informational output

Additional CLI arguments are forwarded to:
  python scripts/sweep_trajectories.py

By default this wrapper keeps the progress bar visible while suppressing normal
stdout from the sweep run. Pass --verbose to show the full CLI output.

For multi-GPU sweeps, forward GPU args such as:
  --gpu-ids 0 1 2 3
  --procs-per-gpu 2 2 2 2
  --gl-backend egl

For CPU-only sweeps, omit GPU args and forward:
  --gl-backend osmesa

To reduce render cost and VRAM, forward render size args such as:
  --render-width 256
  --render-height 256

Do not pass --input-dir, --output-dir, --workers, --quiet, or --verbose to the
forwarded CLI args.
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

workers="1"
workers_arg_seen="0"
verbose="0"
forwarded_args=()

# Parse wrapper-owned arguments before forwarding the rest to the Python CLI.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--workers)
      if [[ "$workers_arg_seen" == "1" ]]; then
        echo "Pass --workers only once." >&2
        exit 1
      fi
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --workers." >&2
        exit 1
      fi
      workers="$2"
      workers_arg_seen="1"
      shift 2
      ;;
    --workers=*)
      if [[ "$workers_arg_seen" == "1" ]]; then
        echo "Pass --workers only once." >&2
        exit 1
      fi
      workers="${1#*=}"
      workers_arg_seen="1"
      shift
      ;;
    -v|--verbose)
      verbose="1"
      shift
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer." >&2
  exit 1
fi

for arg in "${forwarded_args[@]}"; do
  if [[ "$arg" == "--input-dir" || "$arg" == --input-dir=* ]]; then
    echo "Pass the run timestamp as the first argument instead of --input-dir." >&2
    exit 1
  fi
  if [[ "$arg" == "--output-dir" || "$arg" == --output-dir=* ]]; then
    echo "This wrapper does not support overriding --output-dir." >&2
    exit 1
  fi
  if [[ "$arg" == "-j" || "$arg" == "--workers" || "$arg" == --workers=* ]]; then
    echo "Pass --workers to the wrapper directly instead of forwarding it." >&2
    exit 1
  fi
  if [[ "$arg" == "--quiet" || "$arg" == "--verbose" ]]; then
    echo "Pass verbosity flags to the wrapper directly instead of forwarding them." >&2
    exit 1
  fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
data_root="${ROBOCASA_TASK_LEVEL_DATA_ROOT:-$repo_root/data_generation/task_level/data}"
input_dir="$data_root/pre_image/$run_timestamp"
output_dir="$data_root/image/$run_timestamp"

if [[ ! -d "$input_dir" ]]; then
  echo "Timestamp directory not found under pre_image: $input_dir" >&2
  exit 1
fi

(
  cd "$repo_root"
  python_args=(
    scripts/sweep_trajectories.py
    --input-dir "$input_dir"
    --output-dir "$output_dir"
    --workers "$workers"
  )
  if [[ "$verbose" != "1" ]]; then
    python_args+=(--quiet)
  fi
  python_args+=("${forwarded_args[@]}")
  python "${python_args[@]}"
)
