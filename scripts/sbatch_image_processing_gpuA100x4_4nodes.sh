#!/usr/bin/env bash
#SBATCH --job-name=image-processing
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mnakamura@umass.edu
#SBATCH --time=15:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm-%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch scripts/sbatch_image_processing_gpuA100x4_4nodes.sh <run_timestamp> [sweep_cli_args...]

Runs task-level image generation across 4 Delta gpuA100x4 nodes.
Each node renders a deterministic shard of the run timestamp on its local
4 A100 GPUs.

Shard logs and summaries are written under:
  data_generation/task_level/data/image/<run_timestamp>/.slurm_shards/

After all nodes finish, the wrapper merges the shard summaries into:
  data_generation/task_level/data/image/<run_timestamp>/sweep_summary.json

Example:
  sbatch scripts/sbatch_image_processing_gpuA100x4_4nodes.sh 20260401T000000Z

Optional environment overrides:
  WORKERS_PER_GPU=28
  GPUS_PER_NODE=4

Additional args are forwarded to:
  bash scripts/generate_and_insert_images.sh <run_timestamp> ...
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

workers_per_gpu="${WORKERS_PER_GPU:-28}"
gpus_per_node="${GPUS_PER_NODE:-4}"

if ! [[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer." >&2
  exit 1
fi
if ! [[ "$gpus_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer." >&2
  exit 1
fi

workers=$((workers_per_gpu * gpus_per_node))
gpu_ids=()
procs_per_gpu=()
for ((gpu_id = 0; gpu_id < gpus_per_node; gpu_id += 1)); do
  gpu_ids+=("$gpu_id")
  procs_per_gpu+=("$workers_per_gpu")
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
data_root="${ROBOCASA_TASK_LEVEL_DATA_ROOT:-$repo_root/data_generation/task_level/data}"
output_dir="$data_root/image/$run_timestamp"
shard_dir="$output_dir/.slurm_shards"
mkdir -p "$shard_dir"

node_tasks="${SLURM_NTASKS:-${SLURM_NNODES:-0}}"
if ! [[ "$node_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "Unable to resolve the allocated Slurm task count." >&2
  exit 1
fi

gpu_ids_csv="$(IFS=,; echo "${gpu_ids[*]}")"
procs_per_gpu_csv="$(IFS=,; echo "${procs_per_gpu[*]}")"

echo "Launching $node_tasks shard(s) for run timestamp $run_timestamp"
echo "Per-node workers: $workers (${workers_per_gpu} per GPU across ${gpus_per_node} GPUs)"
echo "Shard artifacts: $shard_dir"

if ! srun --ntasks="$node_tasks" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  repo_root="$1"
  run_timestamp="$2"
  shard_dir="$3"
  workers="$4"
  gpu_ids_csv="$5"
  procs_per_gpu_csv="$6"
  shift 6

  IFS="," read -r -a gpu_ids <<< "$gpu_ids_csv"
  IFS="," read -r -a procs_per_gpu <<< "$procs_per_gpu_csv"

  log_path="$shard_dir/shard${SLURM_PROCID}.log"
  summary_path="$shard_dir/sweep_summary_shard${SLURM_PROCID}.json"

  bash "$repo_root/scripts/generate_and_insert_images.sh" "$run_timestamp" \
    --workers "$workers" \
    --gpu-ids "${gpu_ids[@]}" \
    --procs-per-gpu "${procs_per_gpu[@]}" \
    --gl-backend egl \
    --num-shards "${SLURM_NTASKS}" \
    --shard-index "${SLURM_PROCID}" \
    --summary-path "$summary_path" \
    "$@" \
    >"$log_path" 2>&1
' bash "$repo_root" "$run_timestamp" "$shard_dir" "$workers" "$gpu_ids_csv" "$procs_per_gpu_csv" "$@"; then
  echo "One or more shards failed. Inspect per-shard logs under $shard_dir" >&2
  exit 1
fi

summary_paths=()
for ((rank = 0; rank < node_tasks; rank += 1)); do
  summary_paths+=("$shard_dir/sweep_summary_shard${rank}.json")
done

python "$repo_root/scripts/merge_sweep_summaries.py" \
  --output "$output_dir/sweep_summary.json" \
  "${summary_paths[@]}"

echo "Merged summary: $output_dir/sweep_summary.json"
