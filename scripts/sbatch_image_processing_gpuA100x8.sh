#!/usr/bin/env bash
#SBATCH --job-name=image-processing
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=8
#SBATCH --mem=256g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mnakamura@umass.edu
#SBATCH --time=15:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm-%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch scripts/sbatch_image_processing_gpuA100x8.sh <run_timestamp> [sweep_cli_args...]

Runs task-level image generation on one Delta gpuA100x8 node using all 8 GPUs.

Example:
  sbatch scripts/sbatch_image_processing_gpuA100x8.sh 20260401T000000Z

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

bash scripts/generate_and_insert_images.sh "$run_timestamp" \
  --workers 120 \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --procs-per-gpu 15 15 15 15 15 15 15 15 \
  --gl-backend egl \
  "$@"
