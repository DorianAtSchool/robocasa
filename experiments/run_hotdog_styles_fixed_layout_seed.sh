#!/usr/bin/env bash
# Run the semantic HotDogSetup demo plan across multiple styles for one layout and seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-osmesa}"
LAYOUT="${LAYOUT:-11}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
FPS="${FPS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/tmp/hotdog_styles_layout${LAYOUT}_seed${SEED}}"

# These styles were validated previously with the hotdog semantic template.
STYLE_IDS=(
  34
  7
  42
  12
  20
)

mkdir -p "${OUTPUT_ROOT}"

for style in "${STYLE_IDS[@]}"; do
  run_dir="${OUTPUT_ROOT}/layout${LAYOUT}_style${style}_seed${SEED}"

  echo "Running HotDogSetup for layout=${LAYOUT} style=${style} seed=${SEED}"
  MUJOCO_GL="${MUJOCO_GL_BACKEND}" "${PYTHON_BIN}" -m robocasa.utils.sim_tool_executor \
    --task HotDogSetup \
    --robots 2 \
    --layout "${LAYOUT}" \
    --style "${style}" \
    --seed "${SEED}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --demo-plan cooperative_hotdog_setup \
    --output-dir "${run_dir}"
done

echo "Saved outputs under ${OUTPUT_ROOT}"
