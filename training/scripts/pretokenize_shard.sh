#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training/scripts/pretokenize_shard.sh --output-dir PATH [options]

Runs one task-VLM pretokenization shard. In a Slurm array, the shard index
defaults to SLURM_ARRAY_TASK_ID.

Options:
  --dataset-root PATH              Rendered dataset root.
  --output-dir PATH                Preprocessed artifact output directory.
  --shard-dir PATH                 Shard output directory. Default: OUTPUT/pretokenized_shards.
  --shard-index N                  Zero-based shard index. Default: SLURM_ARRAY_TASK_ID.
  --num-shards N                   Total shard count. Default: 128.
  --train-tasks CSV                Training task list.
  --val-tasks CSV                  Validation task list. Default: same_as_train.
  --example-build-workers N        Raw example build workers. Default: 1.
  --training-samples-cache-dir DIR Raw serialized-example cache directory.
  --no-use-example-cache           Disable the raw serialized-example cache.
  --processor-name-or-path REF     Processor used for pretokenization.
  --max-length N                   Pretokenization sequence cap. Default: 4096.
  --artifact-image-size N          Artifact image size used by the staged images.
  --pretokenize-batch-size N       Processor batch size. Default: 4.
  --pretokenize-flush-interval N   Serialized examples per shard disk flush. Default: 400.
  --trust-remote-code              Pass trust_remote_code to the processor.
  --python-bin CMD                 Python executable. Default: python.
  -h, --help                       Show this help text.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${ROBOCASA_PRETOKENIZE_REPO_ROOT:-${ROBOCASA_CPU_PREPROCESS_REPO_ROOT:-}}"
if [[ -z "${repo_root}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/training/scripts/pretokenize_shard.sh" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
  fi
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="$(cd -- "${script_dir}/../.." && pwd)"
fi
dataset_root="data_generation/task_level/data/image/20260413T205634Z"
output_dir=""
shard_dir=""
shard_index="${SLURM_ARRAY_TASK_ID:-}"
num_shards="${PRETOKENIZE_NUM_SHARDS:-128}"
train_tasks="hot_dog_setup,prepare_sandwich_station,prepare_cheese_station,prepare_sausage_cheese"
val_tasks="same_as_train"
example_build_workers="${SLURM_CPUS_PER_TASK:-1}"
training_samples_cache_dir=""
use_example_cache="true"
processor_name_or_path="${PROCESSOR_NAME_OR_PATH:-Qwen/Qwen3.5-0.8B}"
max_length="${MAX_LENGTH:-4096}"
artifact_image_size="${ARTIFACT_IMAGE_SIZE:-}"
pretokenize_batch_size="${PRETOKENIZE_BATCH_SIZE:-4}"
pretokenize_flush_interval="${PRETOKENIZE_FLUSH_INTERVAL:-400}"
trust_remote_code="false"
python_bin="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --shard-dir) shard_dir="$2"; shift 2 ;;
    --shard-index) shard_index="$2"; shift 2 ;;
    --num-shards) num_shards="$2"; shift 2 ;;
    --train-tasks) train_tasks="$2"; shift 2 ;;
    --val-tasks) val_tasks="$2"; shift 2 ;;
    --example-build-workers) example_build_workers="$2"; shift 2 ;;
    --training-samples-cache-dir) training_samples_cache_dir="$2"; shift 2 ;;
    --no-use-example-cache) use_example_cache="false"; shift ;;
    --processor-name-or-path) processor_name_or_path="$2"; shift 2 ;;
    --max-length) max_length="$2"; shift 2 ;;
    --artifact-image-size) artifact_image_size="$2"; shift 2 ;;
    --pretokenize-batch-size) pretokenize_batch_size="$2"; shift 2 ;;
    --pretokenize-flush-interval) pretokenize_flush_interval="$2"; shift 2 ;;
    --trust-remote-code) trust_remote_code="true"; shift ;;
    --python-bin) python_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "${output_dir}" ]] || { echo "--output-dir is required" >&2; exit 1; }
[[ -n "${shard_index}" ]] || { echo "--shard-index or SLURM_ARRAY_TASK_ID is required" >&2; exit 1; }
if [[ -z "${shard_dir}" ]]; then
  shard_dir="${output_dir}/pretokenized_shards"
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=(
  "${python_bin}" -m training.bc_task_vlm.preprocess
  --dataset-root "${dataset_root}"
  --train-tasks "${train_tasks}"
  --val-tasks "${val_tasks}"
  --output-dir "${output_dir}"
  --example-build-workers "${example_build_workers}"
  --pretokenize
  --processor-name-or-path "${processor_name_or_path}"
  --max-length "${max_length}"
  --pretokenize-batch-size "${pretokenize_batch_size}"
  --pretokenize-flush-interval "${pretokenize_flush_interval}"
  --pretokenize-shard-index "${shard_index}"
  --pretokenize-num-shards "${num_shards}"
  --pretokenize-shard-output-dir "${shard_dir}"
  --resume-existing-artifact-images unchecked
)

if [[ "${use_example_cache}" == "false" ]]; then
  cmd+=(--no-use-example-cache)
fi
if [[ -n "${training_samples_cache_dir}" ]]; then
  cmd+=(--training-samples-cache-dir "${training_samples_cache_dir}")
fi
if [[ -n "${artifact_image_size}" ]]; then
  cmd+=(--artifact-image-size "${artifact_image_size}")
fi
if [[ "${trust_remote_code}" == "true" ]]; then
  cmd+=(--trust-remote-code)
fi

printf 'Running pretokenization shard:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
