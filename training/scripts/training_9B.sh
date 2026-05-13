#!/usr/bin/env bash
#SBATCH --job-name=qwen35-9b-bc-vlm
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

bash training/bc_task_vlm/launch_gh200_test.sh \
--python-bin /work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python \
--model-path Qwen/Qwen3.5-9B \
--preprocessed-data-dir /work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/preprocessed/task_vlm_bc_v3 \
--per-device-batch-size 4 \
--grad-accum 2 \
--max-length 4096 \
--image-resolution 512 \
--report-to wandb \
--wandb-mode online \
--wandb-project robocasa-bc-task-vlm \
--run-name qwen35-9b-task-vlm-test \
-- \
--num-workers 8 \
--eval-steps 1000 \
--save-steps 1000
