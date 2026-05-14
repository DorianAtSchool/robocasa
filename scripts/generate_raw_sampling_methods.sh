#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/generate_raw_sampling_methods.sh [options] [-- raw_cli_args...]

Generates raw trajectories for the 52 verified task specs into:
  data_generation/task_level/data/raw/sampling_methods/base
  data_generation/task_level/data/raw/sampling_methods/random
  data_generation/task_level/data/raw/sampling_methods/verbalized
  data_generation/task_level/data/raw/sampling_methods/high_temperature

Defaults generate 20 saved trajectories per verified task for each sampling
method. Existing method directories are resumed in place.
If one sampling method fails, later methods still run; the wrapper exits nonzero
after all methods finish if any method failed.

Options:
  --num-trajectories N       Saved trajectories per task/method (default: 20)
  --verbalized-k N           Trajectories requested per verbalized run (default: 4)
  --max-workers N            Parallel trajectory workers per task (default: 4)
  --max-retries N            Maximum attempts per run (default: 5)
  --model NAME               Vertex model (default: gemini-3-flash-preview)
  --location LOCATION        Vertex location (default: global)
  --thinking-level LEVEL     Gemini 3 thinking level (default: low)
  --base-temperature VALUE   Temperature for base/random/verbalized (default: 0.6)
  --high-temperature VALUE   Temperature for high_temperature (default: 1.0)
  --parallelize-tasks        Run verified tasks concurrently

Arguments after -- are forwarded to:
  python -m data_generation.task_level.generation.raw.cli
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

num_trajectories="32"
verbalized_k="4"
max_workers="4"
max_retries="1"
model="${ROBOCASA_RAW_MODEL:-gemini-3-flash-preview}"
location="${GOOGLE_CLOUD_LOCATION:-global}"
thinking_level="${ROBOCASA_RAW_THINKING_LEVEL:-low}"
base_temperature="0.6"
high_temperature="1.0"
parallelize_tasks="0"
forwarded_args=()

require_option_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1." >&2
    exit 1
  fi
}

ensure_verified_task_resume_dirs() {
  local method_dir="$1"
  python - "$method_dir" <<'PY'
from pathlib import Path
import sys

from data_generation.task_level.tasks.specs import supported_verified_task_names
from data_generation.utils import camel_to_snake_case

method_dir = Path(sys.argv[1])
method_dir.mkdir(parents=True, exist_ok=True)
for task_name in supported_verified_task_names():
    (method_dir / camel_to_snake_case(task_name)).mkdir(parents=True, exist_ok=True)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-trajectories)
      require_option_value "$1" "${2:-}"
      num_trajectories="$2"
      shift 2
      ;;
    --verbalized-k)
      require_option_value "$1" "${2:-}"
      verbalized_k="$2"
      shift 2
      ;;
    --max-workers)
      require_option_value "$1" "${2:-}"
      max_workers="$2"
      shift 2
      ;;
    --max-retries)
      require_option_value "$1" "${2:-}"
      max_retries="$2"
      shift 2
      ;;
    --model)
      require_option_value "$1" "${2:-}"
      model="$2"
      shift 2
      ;;
    --location)
      require_option_value "$1" "${2:-}"
      location="$2"
      shift 2
      ;;
    --thinking-level)
      require_option_value "$1" "${2:-}"
      thinking_level="$2"
      shift 2
      ;;
    --base-temperature)
      require_option_value "$1" "${2:-}"
      base_temperature="$2"
      shift 2
      ;;
    --high-temperature)
      require_option_value "$1" "${2:-}"
      high_temperature="$2"
      shift 2
      ;;
    --parallelize-tasks)
      parallelize_tasks="1"
      shift
      ;;
    --)
      shift
      forwarded_args+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

for positive_integer in num_trajectories verbalized_k max_workers max_retries; do
  value="${!positive_integer}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "--${positive_integer//_/-} must be a positive integer." >&2
    exit 1
  fi
done

if (( num_trajectories % verbalized_k != 0 )); then
  echo "--num-trajectories must be divisible by --verbalized-k." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
data_root="${ROBOCASA_TASK_LEVEL_DATA_ROOT:-$repo_root/data_generation/task_level/data}"
sampling_root="$data_root/raw/sampling_methods"

methods=(base random verbalized high_temperature)
failed_methods=()

for method in "${methods[@]}"; do
  method_dir="$sampling_root/$method"
  summary_path="$method_dir/summary.json"
  runs="$num_trajectories"
  temperature="$base_temperature"
  extra_sampling_args=()

  if [[ "$method" == "verbalized" ]]; then
    runs="$((num_trajectories / verbalized_k))"
    extra_sampling_args+=(--verbalized-k "$verbalized_k")
  fi

  if [[ "$method" == "high_temperature" ]]; then
    temperature="$high_temperature"
  fi

  cli_args=(
    -m data_generation.task_level.generation.raw.cli
    --tasks verified
    --num-runs "$runs"
    --random-start-location true
    --sampling "$method"
    "${extra_sampling_args[@]}"
    --temperature "$temperature"
    --model "$model"
    --location "$location"
    --thinking-level "$thinking_level"
    --max-workers "$max_workers"
    --max-retries "$max_retries"
    --disable-validation
  )

  if [[ "$parallelize_tasks" == "1" ]]; then
    cli_args+=(--parallelize-tasks)
  fi

  if [[ -d "$method_dir" ]]; then
    ensure_verified_task_resume_dirs "$method_dir"
    cli_args+=(--resume "$method_dir")
  else
    cli_args+=(--summary-path "$summary_path")
  fi

  echo "Generating $method into $method_dir"
  if (
    cd "$repo_root"
    python "${cli_args[@]}" "${forwarded_args[@]}"
  ); then
    echo "Completed $method"
  else
    exit_code="$?"
    failed_methods+=("$method:$exit_code")
    echo "Sampling method $method failed with exit code $exit_code; continuing." >&2
  fi
done

if [[ ${#failed_methods[@]} -gt 0 ]]; then
  echo "One or more sampling methods failed:" >&2
  for failed_method in "${failed_methods[@]}"; do
    echo "  $failed_method" >&2
  done
  exit 1
fi
