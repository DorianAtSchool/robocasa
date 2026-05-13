#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training/scripts/launch_cpu_preprocess.sh [script options] [-- extra preprocess args]

This script builds a reusable task-VLM preprocessed artifact on a CPU node.
When invoked outside an existing Slurm allocation, it submits itself to the
Delta CPU partition via `sbatch` by default. Pass `--no-sbatch` to run it
directly in the current shell instead.

Script options:
  --dataset-root PATH              Rendered dataset root containing task directories.
  --output-dir PATH                Local preprocessed artifact directory.
  --run-name NAME                  Output-name override when --output-dir is omitted.
  --example-build-workers N        CPU worker processes for example building.
  --training-samples-cache-dir DIR Raw serialized-example cache directory.
  --no-use-example-cache           Disable the raw serialized-example cache.
  --train-tasks CSV                Training task list.
  --val-tasks CSV                  Validation task list. Default: same_as_train.
  --pretokenize                    Pretokenize with the configured processor. Default: enabled.
  --no-pretokenize                 Skip artifact pretokenization.
  --processor-name-or-path REF     Processor used for pretokenization. Default: Qwen/Qwen3.5-0.8B.
  --max-length N                   Pretokenization sequence cap. Default: 4096.
  --pretokenize-batch-size N       Processor batch size for pretokenization. Default: 1.
  --pretokenize-flush-interval N   Serialized examples per shard disk flush. Default: 400.
  --artifact-image-size N           Resize artifact image copies to NxN pixels. Source dataset is unchanged.
  --resume-existing-artifact-images Reuse an existing images/ directory in the output artifact.
  --skip-existing-artifact-image-validation Rebuild artifact image relpaths without per-file checks.
  --trust-remote-code              Pass trust_remote_code through to the pretokenization processor.
  --push-to-hub REPO               Optional Hugging Face dataset repo id.
  --hub-revision REV               Optional Hugging Face revision or branch name.
  --hub-private                    Create or update the Hub dataset repo as private.
  --hub-data-dir PATH              Optional subdirectory inside the Hub dataset repo.
  --sbatch                         Force Slurm submission, even if already inside Slurm.
  --no-sbatch                      Run directly in the current shell. Default: auto-submit when outside Slurm.
  --sbatch-account NAME            Slurm account for CPU preprocess jobs. Default: bgjs-delta-cpu.
  --sbatch-partition NAME          Slurm partition for CPU preprocess jobs. Default: cpu.
  --sbatch-cpus-per-task N         Slurm CPUs per task. Default: 8, or the explicit --example-build-workers value.
  --sbatch-mem SIZE                Slurm memory request. Default: 128g.
  --sbatch-time HH:MM:SS           Slurm time limit. Default: 08:00:00.
  --sbatch-job-name NAME           Slurm job name. Default: run name.
  --sbatch-output PATH             Slurm stdout path. Default: slurm_logs/%x-%j.out.
  --sbatch-error PATH              Slurm stderr path. Default: slurm_logs/%x-%j.err.
  --python-bin CMD                 Python executable for preflight checks. Default: python.
  -h, --help                       Show this help text.

Anything after `--` is forwarded to `python -m training.bc_task_vlm.preprocess`.
EOF
}

original_args=("$@")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_repo_root="$(cd -- "${script_dir}/../.." && pwd)"
repo_root="${ROBOCASA_CPU_PREPROCESS_REPO_ROOT:-}"
if [[ -z "${repo_root}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/training/scripts/launch_cpu_preprocess.sh" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${SLURM_SUBMIT_DIR}/training/bc_task_vlm/launch_cpu_preprocess.sh" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
  fi
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="${default_repo_root}"
fi
script_path="${repo_root}/training/scripts/launch_cpu_preprocess.sh"
if [[ ! -f "${script_path}" ]]; then
  legacy_script_path="${repo_root}/training/bc_task_vlm/launch_cpu_preprocess.sh"
  if [[ -f "${legacy_script_path}" ]]; then
    script_path="${legacy_script_path}"
  else
    script_path="${script_dir}/launch_cpu_preprocess.sh"
  fi
fi
run_ts="$(date -u +%Y%m%dT%H%M%SZ)"

default_workers="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "${default_workers}" ]]; then
  default_workers="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '16\n')"
fi

dataset_root="data_generation/task_level/data/image/20260413T205634Z"
run_name="bc-task-vlm-preprocessed-${run_ts}"
output_dir=""
example_build_workers="${EXAMPLE_BUILD_WORKERS:-${default_workers}}"
example_build_workers_explicit="false"
training_samples_cache_dir=""
use_example_cache="true"
train_tasks="hot_dog_setup,prepare_sandwich_station,prepare_cheese_station,prepare_sausage_cheese"
val_tasks="same_as_train"
pretokenize="true"
processor_name_or_path="${PROCESSOR_NAME_OR_PATH:-Qwen/Qwen3.5-0.8B}"
max_length="${MAX_LENGTH:-4096}"
pretokenize_batch_size="${PRETOKENIZE_BATCH_SIZE:-1}"
pretokenize_flush_interval="${PRETOKENIZE_FLUSH_INTERVAL:-400}"
artifact_image_size="${ARTIFACT_IMAGE_SIZE:-}"
resume_existing_artifact_images="false"
skip_existing_artifact_image_validation="false"
trust_remote_code="false"
push_to_hub=""
hub_revision=""
hub_private="false"
hub_data_dir=""
submit_via_sbatch="auto"
sbatch_account="${CPU_PREPROCESS_SBATCH_ACCOUNT:-bgjs-delta-cpu}"
sbatch_partition="${CPU_PREPROCESS_SBATCH_PARTITION:-cpu}"
sbatch_cpus_per_task="${CPU_PREPROCESS_SBATCH_CPUS_PER_TASK:-8}"
if [[ -n "${CPU_PREPROCESS_SBATCH_CPUS_PER_TASK:-}" ]]; then
  sbatch_cpus_per_task_explicit="true"
else
  sbatch_cpus_per_task_explicit="false"
fi
sbatch_mem="${CPU_PREPROCESS_SBATCH_MEM:-128g}"
sbatch_time="${CPU_PREPROCESS_SBATCH_TIME:-08:00:00}"
sbatch_job_name="${CPU_PREPROCESS_SBATCH_JOB_NAME:-}"
sbatch_output="${CPU_PREPROCESS_SBATCH_OUTPUT:-slurm_logs/%x-%j.out}"
sbatch_error="${CPU_PREPROCESS_SBATCH_ERROR:-slurm_logs/%x-%j.err}"
python_bin="${PYTHON_BIN:-python}"
forwarded_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      dataset_root="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --run-name)
      run_name="$2"
      shift 2
      ;;
    --example-build-workers)
      example_build_workers="$2"
      example_build_workers_explicit="true"
      shift 2
      ;;
    --training-samples-cache-dir)
      training_samples_cache_dir="$2"
      shift 2
      ;;
    --no-use-example-cache)
      use_example_cache="false"
      shift
      ;;
    --train-tasks)
      train_tasks="$2"
      shift 2
      ;;
    --val-tasks)
      val_tasks="$2"
      shift 2
      ;;
    --pretokenize)
      pretokenize="true"
      shift
      ;;
    --no-pretokenize)
      pretokenize="false"
      shift
      ;;
    --processor-name-or-path)
      processor_name_or_path="$2"
      shift 2
      ;;
    --max-length)
      max_length="$2"
      shift 2
      ;;
    --pretokenize-batch-size)
      pretokenize_batch_size="$2"
      shift 2
      ;;
    --pretokenize-flush-interval)
      pretokenize_flush_interval="$2"
      shift 2
      ;;
    --artifact-image-size)
      artifact_image_size="$2"
      shift 2
      ;;
    --resume-existing-artifact-images)
      resume_existing_artifact_images="true"
      shift
      ;;
    --skip-existing-artifact-image-validation)
      resume_existing_artifact_images="true"
      skip_existing_artifact_image_validation="true"
      shift
      ;;
    --trust-remote-code)
      trust_remote_code="true"
      shift
      ;;
    --push-to-hub)
      push_to_hub="$2"
      shift 2
      ;;
    --hub-revision)
      hub_revision="$2"
      shift 2
      ;;
    --hub-private)
      hub_private="true"
      shift
      ;;
    --hub-data-dir)
      hub_data_dir="$2"
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
      sbatch_cpus_per_task_explicit="true"
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
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      forwarded_args+=("$@")
      break
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${output_dir}" ]]; then
  output_dir="training/bc_task_vlm/preprocessed/${run_name}"
fi

if [[ -z "${sbatch_job_name}" ]]; then
  sbatch_job_name="${run_name}"
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

dependency_install_hint() {
  local module_name="$1"
  printf '%s is not importable from %s. Install the task-VLM preprocessing extras into that interpreter, for example:\n  uv pip install --python %q -r training/bc_task_vlm/requirements-preprocess.txt' \
    "${module_name}" \
    "${python_bin}" \
    "${python_bin}"
}

cd "${repo_root}"

dataset_root="$(resolve_repo_path "${dataset_root}")"
output_dir="$(resolve_repo_path "${output_dir}")"
if [[ -n "${training_samples_cache_dir}" ]]; then
  training_samples_cache_dir="$(resolve_repo_path "${training_samples_cache_dir}")"
fi
sbatch_output="$(resolve_repo_path "${sbatch_output}")"
sbatch_error="$(resolve_repo_path "${sbatch_error}")"

if [[ "${example_build_workers_explicit}" == "true" && "${sbatch_cpus_per_task_explicit}" != "true" ]]; then
  sbatch_cpus_per_task="${example_build_workers}"
fi

should_submit_via_sbatch="false"
if [[ -z "${ROBOCASA_CPU_PREPROCESS_IN_SBATCH:-}" ]]; then
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
    --export "ALL,ROBOCASA_CPU_PREPROCESS_IN_SBATCH=1,ROBOCASA_CPU_PREPROCESS_REPO_ROOT=${repo_root}"
    "${script_path}"
    --no-sbatch
    "${original_args[@]}"
  )

  echo "Submitting CPU preprocessing job via sbatch"
  echo "Slurm account: ${sbatch_account}"
  echo "Slurm partition: ${sbatch_partition}"
  echo "Slurm cpus-per-task: ${sbatch_cpus_per_task}"
  echo "Slurm memory: ${sbatch_mem}"
  echo "Slurm time: ${sbatch_time}"
  echo "Slurm job name: ${sbatch_job_name}"
  echo "Slurm stdout: ${sbatch_output}"
  echo "Slurm stderr: ${sbatch_error}"
  printf 'Running:'
  printf ' %q' "${submit_cmd[@]}"
  printf '\n'

  "${submit_cmd[@]}"
  exit $?
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Running inside existing Slurm allocation ${SLURM_JOB_ID}"
  if [[ -n "${SLURM_JOB_NAME:-}" ]]; then
    echo "Slurm runtime job name: ${SLURM_JOB_NAME}"
  fi
  if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    echo "Slurm runtime partition: ${SLURM_JOB_PARTITION}"
  fi
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Slurm runtime nodes: ${SLURM_JOB_NODELIST}"
  fi
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Slurm runtime cpus-per-task: ${SLURM_CPUS_PER_TASK}"
  fi
  if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
    echo "Slurm runtime mem-per-node: ${SLURM_MEM_PER_NODE}"
  elif [[ -n "${SLURM_MEM_PER_CPU:-}" ]]; then
    echo "Slurm runtime mem-per-cpu: ${SLURM_MEM_PER_CPU}"
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "Slurm submit dir: ${SLURM_SUBMIT_DIR}"
  fi
  if [[ -n "${ROBOCASA_CPU_PREPROCESS_IN_SBATCH:-}" ]]; then
    echo "Resolved repo root: ${repo_root}"
    echo "Slurm requested account: ${sbatch_account}"
    echo "Slurm requested partition: ${sbatch_partition}"
    echo "Slurm requested cpus-per-task: ${sbatch_cpus_per_task}"
    echo "Slurm requested memory: ${sbatch_mem}"
    echo "Slurm requested time: ${sbatch_time}"
    echo "Slurm requested job name: ${sbatch_job_name}"
    echo "Slurm requested stdout: ${sbatch_output}"
    echo "Slurm requested stderr: ${sbatch_error}"
  fi
fi

[[ -d "${dataset_root}" ]] || fail "Dataset root does not exist: ${dataset_root}"
echo "Resolved repo root: ${repo_root}"
command -v "${python_bin}" >/dev/null 2>&1 || fail "Python executable not found: ${python_bin}"
if [[ -n "${artifact_image_size}" ]]; then
  [[ "${artifact_image_size}" =~ ^[0-9]+$ ]] || fail "--artifact-image-size must be a positive integer."
  [[ "${artifact_image_size}" -ge 1 ]] || fail "--artifact-image-size must be a positive integer."
fi
"${python_bin}" -c "import datasets" >/dev/null 2>&1 || fail \
  "$(dependency_install_hint "datasets")"
if [[ "${pretokenize}" == "true" ]]; then
  "${python_bin}" -c "import transformers, PIL, torchvision" >/dev/null 2>&1 || fail \
    "$(dependency_install_hint "transformers, Pillow, and torchvision")"
elif [[ -n "${artifact_image_size}" ]]; then
  "${python_bin}" -c "import PIL" >/dev/null 2>&1 || fail \
    "$(dependency_install_hint "Pillow")"
fi
if [[ -n "${push_to_hub}" ]]; then
  "${python_bin}" -c "import huggingface_hub" >/dev/null 2>&1 || fail \
    "$(dependency_install_hint "huggingface_hub")"
fi

cmd=(
  "${python_bin}"
  -m training.bc_task_vlm.preprocess
  --dataset-root "${dataset_root}"
  --train-tasks "${train_tasks}"
  --val-tasks "${val_tasks}"
  --output-dir "${output_dir}"
  --example-build-workers "${example_build_workers}"
)

if [[ "${use_example_cache}" == "false" ]]; then
  cmd+=(--no-use-example-cache)
fi

if [[ -n "${training_samples_cache_dir}" ]]; then
  cmd+=(--training-samples-cache-dir "${training_samples_cache_dir}")
fi

if [[ -n "${push_to_hub}" ]]; then
  cmd+=(--push-to-hub "${push_to_hub}")
fi

if [[ -n "${hub_revision}" ]]; then
  cmd+=(--hub-revision "${hub_revision}")
fi

if [[ "${hub_private}" == "true" ]]; then
  cmd+=(--hub-private)
fi

if [[ -n "${hub_data_dir}" ]]; then
  cmd+=(--hub-data-dir "${hub_data_dir}")
fi

if [[ "${pretokenize}" == "true" ]]; then
  cmd+=(--pretokenize)
  cmd+=(--processor-name-or-path "${processor_name_or_path}")
  cmd+=(--pretokenize-batch-size "${pretokenize_batch_size}")
  cmd+=(--pretokenize-flush-interval "${pretokenize_flush_interval}")
  if [[ -n "${max_length}" ]]; then
    cmd+=(--max-length "${max_length}")
  fi
  if [[ -n "${artifact_image_size}" ]]; then
    cmd+=(--artifact-image-size "${artifact_image_size}")
  fi
  if [[ "${resume_existing_artifact_images}" == "true" ]]; then
    cmd+=(--resume-existing-artifact-images)
  fi
  if [[ "${skip_existing_artifact_image_validation}" == "true" ]]; then
    cmd+=(--skip-existing-artifact-image-validation)
  fi
  if [[ "${trust_remote_code}" == "true" ]]; then
    cmd+=(--trust-remote-code)
  fi
elif [[ -n "${artifact_image_size}" ]]; then
  cmd+=(--artifact-image-size "${artifact_image_size}")
  if [[ "${resume_existing_artifact_images}" == "true" ]]; then
    cmd+=(--resume-existing-artifact-images)
  fi
  if [[ "${skip_existing_artifact_image_validation}" == "true" ]]; then
    cmd+=(--skip-existing-artifact-image-validation)
  fi
fi

cmd+=("${forwarded_args[@]}")

echo "Dataset root: ${dataset_root}"
echo "Output dir: ${output_dir}"
echo "Example build workers: ${example_build_workers}"
echo "Pretokenize: ${pretokenize}"
if [[ "${pretokenize}" == "true" ]]; then
  echo "Pretokenization processor: ${processor_name_or_path}"
  echo "Pretokenization max length: ${max_length}"
  echo "Pretokenization batch size: ${pretokenize_batch_size}"
fi
if [[ -n "${artifact_image_size}" ]]; then
  echo "Artifact image size: ${artifact_image_size}x${artifact_image_size}"
fi
if [[ "${resume_existing_artifact_images}" == "true" ]]; then
  echo "Resume existing artifact images: true"
fi
if [[ "${skip_existing_artifact_image_validation}" == "true" ]]; then
  echo "Skip existing artifact image validation: true"
fi
if [[ -n "${training_samples_cache_dir}" ]]; then
  echo "Training samples cache dir: ${training_samples_cache_dir}"
fi
if [[ -n "${push_to_hub}" ]]; then
  echo "Push to HF dataset: ${push_to_hub}"
fi
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
