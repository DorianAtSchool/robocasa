#!/usr/bin/env bash
#SBATCH --job-name=gemma4-e4b-it-bc-vlm
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

export UV_CACHE_DIR=/work/hdd/bgjs/mnakamura/.cache/uv
export TOKENIZERS_PARALLELISM=false

python_bin=/work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python
model_path=google/gemma-4-E4B-it
dataset_root=/work/hdd/bgjs/mnakamura/robocasa/data_generation/task_level/data/image/20260413T205634Z
max_length=4096
image_resolution=256

bash training/bc_task_vlm/launch_gh200_test.sh \
--python-bin "${python_bin}" \
--model-path "${model_path}" \
--dataset-root "${dataset_root}" \
--per-device-batch-size 4 \
--grad-accum 2 \
--max-length "${max_length}" \
--image-resolution "${image_resolution}" \
--report-to wandb \
--wandb-mode online \
--wandb-project robocasa-bc-task-vlm \
--run-name gemma4-e4b-it-task-vlm-test \
-- \
--num-workers 8 \
--eval-max-samples 256 \
--eval-steps 500 \
--save-steps 500
