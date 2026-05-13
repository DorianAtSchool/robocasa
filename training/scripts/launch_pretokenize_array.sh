#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training/scripts/launch_pretokenize_array.sh --output-dir PATH [options]

Submits a Slurm array for task-VLM pretokenization shards, then submits a merge
job with afterok dependency on the array. The output directory should already
contain the staged images/ directory when --artifact-image-size is used.

Options:
  --dataset-root PATH              Rendered dataset root.
  --output-dir PATH                Preprocessed artifact output directory.
  --shard-dir PATH                 Shard output directory. Default: OUTPUT/pretokenized_shards.
  --num-shards N                   Total shard count. Default: 128.
  --train-tasks CSV                Training task list.
  --val-tasks CSV                  Validation task list. Default: same_as_train.
  --example-build-workers N        Raw example build workers per shard. Default: cpus-per-task.
  --training-samples-cache-dir DIR Raw serialized-example cache directory.
  --no-use-example-cache           Disable the raw serialized-example cache.
  --processor-name-or-path REF     Processor used for pretokenization.
  --max-length N                   Pretokenization sequence cap. Default: 4096.
  --artifact-image-size N          Artifact image size used by the staged images.
  --pretokenize-batch-size N       Processor batch size. Default: 4.
  --pretokenize-flush-interval N   Serialized examples per shard disk flush. Default: 400.
  --merge-flush-interval N         Final artifact rows per merge disk flush. Default: 400.
  --pretokenize-workers N          Reserved for future local workers; accepted for CLI compatibility.
  --trust-remote-code              Pass trust_remote_code to the processor.
  --sbatch-account NAME            Slurm account. Default: bgjs-delta-cpu.
  --sbatch-partition NAME          Slurm partition. Default: cpu.
  --cpus-per-task N                Slurm CPUs per task. Default: 4.
  --mem SIZE                       Slurm memory. Default: 32g.
  --time HH:MM:SS                  Slurm time limit. Default: 01:00:00.
  --python-bin CMD                 Python executable. Default: python.
  -h, --help                       Show this help text.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
dataset_root="data_generation/task_level/data/image/20260413T205634Z"
output_dir=""
shard_dir=""
num_shards="${PRETOKENIZE_NUM_SHARDS:-128}"
train_tasks="hot_dog_setup,prepare_sandwich_station,prepare_cheese_station,prepare_sausage_cheese"
val_tasks="same_as_train"
example_build_workers=""
training_samples_cache_dir=""
use_example_cache="true"
processor_name_or_path="${PROCESSOR_NAME_OR_PATH:-Qwen/Qwen3.5-0.8B}"
max_length="${MAX_LENGTH:-4096}"
artifact_image_size="${ARTIFACT_IMAGE_SIZE:-}"
pretokenize_batch_size="${PRETOKENIZE_BATCH_SIZE:-4}"
pretokenize_flush_interval="${PRETOKENIZE_FLUSH_INTERVAL:-400}"
merge_flush_interval="${PRETOKENIZE_MERGE_FLUSH_INTERVAL:-400}"
pretokenize_workers="${PRETOKENIZE_WORKERS:-1}"
trust_remote_code="false"
sbatch_account="${CPU_PREPROCESS_SBATCH_ACCOUNT:-bgjs-delta-cpu}"
sbatch_partition="${CPU_PREPROCESS_SBATCH_PARTITION:-cpu}"
cpus_per_task="${PRETOKENIZE_CPUS_PER_TASK:-4}"
mem="${PRETOKENIZE_MEM:-32g}"
time_limit="${PRETOKENIZE_TIME:-01:00:00}"
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
    --processor-name-or-path) processor_name_or_path="$2"; shift 2 ;;
    --max-length) max_length="$2"; shift 2 ;;
    --artifact-image-size) artifact_image_size="$2"; shift 2 ;;
    --pretokenize-batch-size) pretokenize_batch_size="$2"; shift 2 ;;
    --pretokenize-flush-interval) pretokenize_flush_interval="$2"; shift 2 ;;
    --merge-flush-interval) merge_flush_interval="$2"; shift 2 ;;
    --pretokenize-workers) pretokenize_workers="$2"; shift 2 ;;
    --trust-remote-code) trust_remote_code="true"; shift ;;
    --sbatch-account) sbatch_account="$2"; shift 2 ;;
    --sbatch-partition) sbatch_partition="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --python-bin) python_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "${output_dir}" ]] || { echo "--output-dir is required" >&2; exit 1; }
if [[ -z "${shard_dir}" ]]; then
  shard_dir="${output_dir}/pretokenized_shards"
fi
if [[ -z "${example_build_workers}" ]]; then
  example_build_workers="${cpus_per_task}"
fi
if [[ "${pretokenize_workers}" != "1" ]]; then
  echo "Warning: --pretokenize-workers is accepted but local worker fanout is not implemented yet." >&2
fi

mkdir -p "${repo_root}/slurm_logs"
array_max=$((num_shards - 1))

common_args=(
  --dataset-root "${dataset_root}"
  --output-dir "${output_dir}"
  --shard-dir "${shard_dir}"
  --num-shards "${num_shards}"
  --train-tasks "${train_tasks}"
  --val-tasks "${val_tasks}"
  --example-build-workers "${example_build_workers}"
  --python-bin "${python_bin}"
)
if [[ "${use_example_cache}" == "false" ]]; then
  common_args+=(--no-use-example-cache)
fi
if [[ -n "${training_samples_cache_dir}" ]]; then
  common_args+=(--training-samples-cache-dir "${training_samples_cache_dir}")
fi
if [[ -n "${artifact_image_size}" ]]; then
  common_args+=(--artifact-image-size "${artifact_image_size}")
fi
merge_args=(
  "${common_args[@]}"
  --merge-flush-interval "${merge_flush_interval}"
)

shard_args=(
  "${common_args[@]}"
  --processor-name-or-path "${processor_name_or_path}"
  --max-length "${max_length}"
  --pretokenize-batch-size "${pretokenize_batch_size}"
  --pretokenize-flush-interval "${pretokenize_flush_interval}"
)
if [[ "${trust_remote_code}" == "true" ]]; then
  shard_args+=(--trust-remote-code)
fi

cd "${repo_root}"
export_pythonpath="${repo_root}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export_pythonpath="${repo_root}:${PYTHONPATH}"
fi
pretok_job_id="$(
  sbatch --parsable \
    --account "${sbatch_account}" \
    --partition "${sbatch_partition}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${cpus_per_task}" \
    --mem "${mem}" \
    --time "${time_limit}" \
    --job-name "bc-task-vlm-pretok" \
    --chdir "${repo_root}" \
    --array "0-${array_max}" \
    --export "ALL,ROBOCASA_PRETOKENIZE_REPO_ROOT=${repo_root},PYTHONPATH=${export_pythonpath}" \
    --output "slurm_logs/%x-%A_%a.out" \
    --error "slurm_logs/%x-%A_%a.err" \
    "${script_dir}/pretokenize_shard.sh" \
    "${shard_args[@]}"
)"

merge_job_id="$(
  sbatch --parsable \
    --account "${sbatch_account}" \
    --partition "${sbatch_partition}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task "${cpus_per_task}" \
    --mem "${mem}" \
    --time "${time_limit}" \
    --job-name "bc-task-vlm-pretok-merge" \
    --chdir "${repo_root}" \
    --dependency "afterok:${pretok_job_id}" \
    --export "ALL,ROBOCASA_PRETOKENIZE_REPO_ROOT=${repo_root},PYTHONPATH=${export_pythonpath}" \
    --output "slurm_logs/%x-%j.out" \
    --error "slurm_logs/%x-%j.err" \
    "${script_dir}/merge_pretok_shards.sh" \
    "${merge_args[@]}"
)"

echo "Pretokenization array job id: ${pretok_job_id}"
echo "Merge job id: ${merge_job_id}"
echo "Output dir: ${output_dir}"
echo "Shard dir: ${shard_dir}"
echo "Shard logs: ${repo_root}/slurm_logs/bc-task-vlm-pretok-${pretok_job_id}_<array>.{out,err}"
echo "Merge logs: ${repo_root}/slurm_logs/bc-task-vlm-pretok-merge-${merge_job_id}.{out,err}"
