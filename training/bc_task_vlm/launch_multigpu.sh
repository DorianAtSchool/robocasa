#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No CUDA devices detected. Set NUM_PROCESSES explicitly or make GPUs visible." >&2
  exit 1
fi

accelerate launch \
  --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
  --num_processes "${NUM_PROCESSES}" \
  -m training.bc_task_vlm.main \
  "$@"
