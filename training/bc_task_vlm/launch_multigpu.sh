#!/usr/bin/env bash
set -euo pipefail

count_visible_gpus() {
  local gpu_list_var raw_value item count

  for gpu_list_var in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS; do
    raw_value="${!gpu_list_var:-}"
    if [[ -z "${raw_value}" ]]; then
      continue
    fi

    count=0
    IFS=',' read -r -a gpu_ids <<< "${raw_value}"
    for item in "${gpu_ids[@]}"; do
      item="${item//[[:space:]]/}"
      if [[ -n "${item}" && "${item}" != "-1" ]]; then
        count=$((count + 1))
      fi
    done

    printf '%s\n' "${count}"
    return 0
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {count += 1} END {print count + 0}'
    return 0
  fi

  printf '0\n'
}

NUM_PROCESSES="${NUM_PROCESSES:-$(count_visible_gpus)}"

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No GPUs detected from CUDA_VISIBLE_DEVICES/SLURM allocation/nvidia-smi." >&2
  echo "Set NUM_PROCESSES explicitly or make GPUs visible before launching accelerate." >&2
  exit 1
fi

accelerate launch \
  --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
  --num_processes "${NUM_PROCESSES}" \
  -m training.bc_task_vlm.main \
  "$@"
