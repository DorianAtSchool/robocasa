#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training/scripts/calculate_average_tokens_for_tasks.sh --model MODEL [options]

  training/scripts/calculate_average_tokens_for_tasks.sh \
    --model Qwen/Qwen3.5-0.8B \
    --python-bin /work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python

Estimates average multimodal tokens per trajectory for each task using the first
20 trajectories per task by default. The script hides CUDA devices before
loading the processor so the first iteration stays CPU-only. When invoked
outside an existing Slurm allocation, it submits itself to the CPU partition via
`sbatch` by default. Pass `--no-sbatch` to run directly in the current shell.

Required:
  --model MODEL                    Model or processor name/path used for tokenization.

Options:
  --dataset-root PATH              Rendered dataset root.
                                   Default: data_generation/task_level/data/image/20260413T205634Z
  --tasks CSV                      Task list. Default: all supported task directories.
  --trajectories-per-task N        Number of trajectories to sample per task. Default: 20.
  --granularity NAME               decentralized or centralized. Default: decentralized.
  --image-resolution N             Square image resolution estimate. Default: 512.
  --trust-remote-code              Allow remote processor code.
  --local-files-only               Do not download model/processor files.
  --sbatch                         Force Slurm submission, even inside Slurm.
  --no-sbatch                      Run directly in the current shell.
  --sbatch-account NAME            Slurm account. Default: bgjs-delta-cpu.
  --sbatch-partition NAME          Slurm partition. Default: cpu.
  --sbatch-cpus-per-task N         Slurm CPUs per task. Default: 8.
  --sbatch-mem SIZE                Slurm memory request. Default: 64g.
  --sbatch-time HH:MM:SS           Slurm time limit. Default: 02:00:00.
  --sbatch-job-name NAME           Slurm job name. Default: avg-task-tokens.
  --sbatch-output PATH             Slurm stdout path. Default: slurm_logs/%x-%j.out.
  --sbatch-error PATH              Slurm stderr path. Default: slurm_logs/%x-%j.err.
  --python-bin CMD                 Python executable. Default: repo .venv_d, .venv, then python3/python.
  -h, --help                       Show this help text.
EOF
}

original_args=("$@")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_repo_root="$(cd -- "${script_dir}/../.." && pwd)"
repo_root="${ROBOCASA_AVG_TOKENS_REPO_ROOT:-}"
if [[ -z "${repo_root}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/training/scripts/calculate_average_tokens_for_tasks.sh" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
  fi
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="${default_repo_root}"
fi
script_path="${repo_root}/training/scripts/calculate_average_tokens_for_tasks.sh"
if [[ ! -f "${script_path}" ]]; then
  script_path="${script_dir}/calculate_average_tokens_for_tasks.sh"
fi

choose_default_python() {
  local candidate

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  for candidate in \
    "${repo_root}/.venv_d/bin/python" \
    "${repo_root}/.venv/bin/python" \
    python3 \
    python
  do
    if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' python
}

python_bin="$(choose_default_python)"
model_name_or_path=""
submit_via_sbatch="auto"
sbatch_account="${AVG_TOKENS_SBATCH_ACCOUNT:-bgjs-delta-cpu}"
sbatch_partition="${AVG_TOKENS_SBATCH_PARTITION:-cpu}"
sbatch_cpus_per_task="${AVG_TOKENS_SBATCH_CPUS_PER_TASK:-8}"
sbatch_mem="${AVG_TOKENS_SBATCH_MEM:-64g}"
sbatch_time="${AVG_TOKENS_SBATCH_TIME:-02:00:00}"
sbatch_job_name="${AVG_TOKENS_SBATCH_JOB_NAME:-avg-task-tokens}"
sbatch_output="${AVG_TOKENS_SBATCH_OUTPUT:-slurm_logs/%x-%j.out}"
sbatch_error="${AVG_TOKENS_SBATCH_ERROR:-slurm_logs/%x-%j.err}"
python_args=(
  --dataset-root "${repo_root}/data_generation/task_level/data/image/20260413T205634Z"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-name-or-path)
      model_name_or_path="$2"
      shift 2
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --sbatch)
      submit_via_sbatch="true"
      shift
      ;;
    --no-sbatch)
      submit_via_sbatch="false"
      shift
      ;;
    --sbatch-account)
      sbatch_account="$2"
      shift 2
      ;;
    --sbatch-partition)
      sbatch_partition="$2"
      shift 2
      ;;
    --sbatch-cpus-per-task)
      sbatch_cpus_per_task="$2"
      shift 2
      ;;
    --sbatch-mem)
      sbatch_mem="$2"
      shift 2
      ;;
    --sbatch-time)
      sbatch_time="$2"
      shift 2
      ;;
    --sbatch-job-name)
      sbatch_job_name="$2"
      shift 2
      ;;
    --sbatch-output)
      sbatch_output="$2"
      shift 2
      ;;
    --sbatch-error)
      sbatch_error="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      python_args+=("$@")
      break
      ;;
    *)
      python_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${model_name_or_path}" ]]; then
  echo "error: --model MODEL is required." >&2
  echo >&2
  usage >&2
  exit 2
fi

resolve_repo_path() {
  local raw_path="$1"

  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi

  printf '%s/%s\n' "${repo_root}" "${raw_path}"
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

cd "${repo_root}"

if ! "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  fail "Python 3.10 or newer is required for task-VLM scripts, but ${python_bin} is too old or not runnable. Pass --python-bin /work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python."
fi

sbatch_output="$(resolve_repo_path "${sbatch_output}")"
sbatch_error="$(resolve_repo_path "${sbatch_error}")"

should_submit_via_sbatch="false"
if [[ -z "${ROBOCASA_AVG_TOKENS_IN_SBATCH:-}" ]]; then
  if [[ "${submit_via_sbatch}" == "true" ]]; then
    should_submit_via_sbatch="true"
  elif [[ "${submit_via_sbatch}" != "false" && -z "${SLURM_JOB_ID:-}" ]]; then
    should_submit_via_sbatch="true"
  fi
fi

if [[ "${should_submit_via_sbatch}" == "true" ]]; then
  command -v sbatch >/dev/null 2>&1 || fail "sbatch not found in PATH."
  mkdir -p "$(dirname "${sbatch_output}")" "$(dirname "${sbatch_error}")"

  submit_cmd=(
    sbatch
    --account "${sbatch_account}"
    --partition "${sbatch_partition}"
    --nodes 1
    --ntasks 1
    --cpus-per-task "${sbatch_cpus_per_task}"
    --mem "${sbatch_mem}"
    --time "${sbatch_time}"
    --job-name "${sbatch_job_name}"
    --chdir "${repo_root}"
    --output "${sbatch_output}"
    --error "${sbatch_error}"
    --export "ALL,ROBOCASA_AVG_TOKENS_IN_SBATCH=1,ROBOCASA_AVG_TOKENS_REPO_ROOT=${repo_root}"
    "${script_path}"
    "${original_args[@]}"
    --no-sbatch
  )

  echo "Submitting average-token estimate job via sbatch"
  echo "Slurm account: ${sbatch_account}"
  echo "Slurm partition: ${sbatch_partition}"
  echo "Slurm cpus-per-task: ${sbatch_cpus_per_task}"
  echo "Slurm memory: ${sbatch_mem}"
  echo "Slurm time: ${sbatch_time}"
  echo "Slurm job name: ${sbatch_job_name}"
  echo "Slurm stdout: ${sbatch_output}"
  echo "Slurm stderr: ${sbatch_error}"
  echo "Python executable: ${python_bin}"
  printf 'Running:'
  printf ' %q' "${submit_cmd[@]}"
  printf '\n'

  "${submit_cmd[@]}"
  exit $?
fi

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

exec "${python_bin}" -m training.bc_task_vlm.calculate_average_tokens_for_tasks \
  --model-name-or-path "${model_name_or_path}" \
  "${python_args[@]}"
