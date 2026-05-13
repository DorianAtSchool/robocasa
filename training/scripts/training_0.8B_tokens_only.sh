#!/usr/bin/env bash
#SBATCH --job-name=qwen35-08b-bc-token-only
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

export UV_CACHE_DIR=/work/hdd/bgjs/mnakamura/.cache/uv
export TOKENIZERS_PARALLELISM=false

python_bin=/work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python
model_path=Qwen/Qwen3.5-0.8B
dataset_root=/work/hdd/bgjs/mnakamura/robocasa/data_generation/task_level/data/image/20260413T205634Z
preprocessed_root=/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/preprocessed/task_vlm_same_task_qwen35_08b_pretok
pretokenized_shard_dir="${preprocessed_root}/pretokenized_shards"
pretokenized_num_shards=12
max_length=4096
image_resolution=256

shards_match_pretokenized_defaults() {
  "${python_bin}" - <<'PY' \
    "${pretokenized_shard_dir}" \
    "${model_path}" \
    "${max_length}" \
    "${image_resolution}" \
    "${pretokenized_num_shards}"
import json
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
model_path = sys.argv[2]
max_length = int(sys.argv[3])
image_resolution = int(sys.argv[4])
num_shards = int(sys.argv[5])

if not shard_dir.is_dir():
    raise SystemExit(1)

def normalize(metadata):
    normalized = dict(metadata)
    if "image_resolution" not in normalized:
        min_pixels = normalized.get("image_min_pixels")
        max_pixels = normalized.get("image_max_pixels")
        if min_pixels == max_pixels and min_pixels is not None:
            resolution = int(int(min_pixels) ** 0.5)
            if resolution * resolution == int(min_pixels):
                normalized["image_resolution"] = resolution
    normalized.pop("image_min_pixels", None)
    normalized.pop("image_max_pixels", None)
    return normalized

required_values = {
    "processor_name_or_path": model_path,
    "max_length": max_length,
    "image_resolution": image_resolution,
    "trust_remote_code": False,
}

expected_pretokenization = None
for shard_index in range(num_shards):
    manifest_path = shard_dir / f"manifest-{shard_index:06d}.json"
    train_path = shard_dir / f"train-{shard_index:06d}.arrow"
    validation_path = shard_dir / f"validation-{shard_index:06d}.arrow"
    if not manifest_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        raise SystemExit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("shard_index") != shard_index or manifest.get("num_shards") != num_shards:
        raise SystemExit(1)
    pretokenization = manifest.get("pretokenization")
    if not isinstance(pretokenization, dict):
        raise SystemExit(1)
    pretokenization = normalize(pretokenization)
    for field_name, expected_value in required_values.items():
        if pretokenization.get(field_name) != expected_value:
            raise SystemExit(1)
    if expected_pretokenization is None:
        expected_pretokenization = pretokenization
    elif pretokenization != expected_pretokenization:
        raise SystemExit(1)
PY
}

if ! shards_match_pretokenized_defaults; then
  echo "Pretokenized shards in ${pretokenized_shard_dir} are missing or do not match the 0.8B shard-native defaults." >&2
  echo "Regenerate shards or update training/scripts/training_0.8B_tokens_only.sh to point at the intended shard set." >&2
  exit 1
fi

bash training/bc_task_vlm/launch_gh200_test.sh \
--python-bin "${python_bin}" \
--model-path "${model_path}" \
--dataset-root "${dataset_root}" \
--pretokenized-shard-dir "${pretokenized_shard_dir}" \
--pretokenized-num-shards "${pretokenized_num_shards}" \
--per-device-batch-size 6 \
--grad-accum 2 \
--max-length "${max_length}" \
--image-resolution "${image_resolution}" \
--report-to wandb \
--wandb-mode online \
--wandb-project robocasa-bc-task-vlm \
--run-name qwen35-08b-task-vlm-token-only-test \
-- \
--num-workers 8 \
--eval-max-samples 1024 \
--eval-steps 500 \
--save-steps 500 \
--pretokenized-shard-manifest-only \
--pretokenized-drop-visual-inputs \
--no-gradient-checkpointing
