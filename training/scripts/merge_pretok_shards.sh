#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training/scripts/merge_pretok_shards.sh --output-dir PATH [options]

Merges task-VLM pretokenization shard Arrow files into the final Hugging Face
DatasetDict artifact.

Options:
  --dataset-root PATH              Rendered dataset root.
  --output-dir PATH                Preprocessed artifact output directory.
  --shard-dir PATH                 Shard output directory. Default: OUTPUT/pretokenized_shards.
  --num-shards N                   Total shard count. Default: 128.
  --train-tasks CSV                Training task list.
  --val-tasks CSV                  Validation task list. Default: same_as_train.
  --example-build-workers N        Raw example build workers. Default: 1.
  --training-samples-cache-dir DIR Raw serialized-example cache directory.
  --no-use-example-cache           Disable the raw serialized-example cache.
  --artifact-image-size N          Artifact image size used by the staged images.
  --merge-flush-interval N         Final artifact rows per disk flush. Default: 400.
  --push-to-hub REPO               Optional Hugging Face dataset repo id.
  --hub-revision REV               Optional Hugging Face revision or branch name.
  --hub-private                    Create or update the Hub dataset repo as private.
  --hub-data-dir PATH              Optional subdirectory inside the Hub dataset repo.
  --python-bin CMD                 Python executable. Default: python.
  -h, --help                       Show this help text.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${ROBOCASA_PRETOKENIZE_REPO_ROOT:-${ROBOCASA_CPU_PREPROCESS_REPO_ROOT:-}}"
if [[ -z "${repo_root}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/training/scripts/merge_pretok_shards.sh" ]]; then
    repo_root="${SLURM_SUBMIT_DIR}"
  fi
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="$(cd -- "${script_dir}/../.." && pwd)"
fi
dataset_root="data_generation/task_level/data/image/20260413T205634Z"
output_dir=""
shard_dir=""
num_shards="${PRETOKENIZE_NUM_SHARDS:-128}"
train_tasks="hot_dog_setup,prepare_sandwich_station,prepare_cheese_station,prepare_sausage_cheese"
val_tasks="same_as_train"
example_build_workers="${SLURM_CPUS_PER_TASK:-1}"
training_samples_cache_dir=""
use_example_cache="true"
artifact_image_size="${ARTIFACT_IMAGE_SIZE:-}"
merge_flush_interval="${PRETOKENIZE_MERGE_FLUSH_INTERVAL:-400}"
push_to_hub=""
hub_revision=""
hub_private="false"
hub_data_dir=""
python_bin="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --shard-dir) shard_dir="$2"; shift 2 ;;
    --num-shards) num_shards="$2"; shift 2 ;;
    --train-tasks) train_tasks="$2"; shift 2 ;;
    --val-tasks) val_tasks="$2"; shift 2 ;;
    --example-build-workers) example_build_workers="$2"; shift 2 ;;
    --training-samples-cache-dir) training_samples_cache_dir="$2"; shift 2 ;;
    --no-use-example-cache) use_example_cache="false"; shift ;;
    --artifact-image-size) artifact_image_size="$2"; shift 2 ;;
    --merge-flush-interval) merge_flush_interval="$2"; shift 2 ;;
    --push-to-hub) push_to_hub="$2"; shift 2 ;;
    --hub-revision) hub_revision="$2"; shift 2 ;;
    --hub-private) hub_private="true"; shift ;;
    --hub-data-dir) hub_data_dir="$2"; shift 2 ;;
    --python-bin) python_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "${output_dir}" ]] || { echo "--output-dir is required" >&2; exit 1; }
if [[ -z "${shard_dir}" ]]; then
  shard_dir="${output_dir}/pretokenized_shards"
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=(
  "${python_bin}" -m training.bc_task_vlm.merge_pretokenized_shards
  --dataset-root "${dataset_root}"
  --train-tasks "${train_tasks}"
  --val-tasks "${val_tasks}"
  --output-dir "${output_dir}"
  --shard-dir "${shard_dir}"
  --num-shards "${num_shards}"
  --example-build-workers "${example_build_workers}"
  --merge-flush-interval "${merge_flush_interval}"
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

printf 'Running pretokenization shard merge:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
