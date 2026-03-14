#!/usr/bin/env bash
# Run the semantic HotDogSetup demo plan across multiple seeds for one layout and style.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-osmesa}"
LAYOUT="${LAYOUT:-11}"
STYLE="${STYLE:-34}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
FPS="${FPS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/tmp/hotdog_seeds_layout${LAYOUT}_style${STYLE}}"

# Seeds are not a fixed RoboCasa catalog; these are just a useful default sweep.
SEED_VALUES=(
  42
  100
  200
  300
  400
)

mkdir -p "${OUTPUT_ROOT}"

for seed in "${SEED_VALUES[@]}"; do
  run_dir="${OUTPUT_ROOT}/layout${LAYOUT}_style${STYLE}_seed${seed}"

  echo "Running HotDogSetup for layout=${LAYOUT} style=${STYLE} seed=${seed}"
  MUJOCO_GL="${MUJOCO_GL_BACKEND}" "${PYTHON_BIN}" -m robocasa.utils.sim_tool_executor \
    --task HotDogSetup \
    --robots 2 \
    --layout "${LAYOUT}" \
    --style "${STYLE}" \
    --seed "${seed}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --demo-plan cooperative_hotdog_setup \
    --output-dir "${run_dir}"
done

echo "Saved outputs under ${OUTPUT_ROOT}"
