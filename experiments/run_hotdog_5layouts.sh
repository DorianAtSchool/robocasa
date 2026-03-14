#!/usr/bin/env bash
# Run the semantic HotDogSetup demo plan across five validated layout/style pairs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-osmesa}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
FPS="${FPS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/tmp/hotdog_5layouts}"

mkdir -p "${OUTPUT_ROOT}"

LAYOUT_STYLE_PAIRS=(
  "11 34"
  "24 7"
  "56 42"
  "8 12"
  "15 20"
)

for pair in "${LAYOUT_STYLE_PAIRS[@]}"; do
  read -r layout style <<< "${pair}"
  run_dir="${OUTPUT_ROOT}/layout${layout}_style${style}"

  echo "Running HotDogSetup for layout=${layout} style=${style}"
  MUJOCO_GL="${MUJOCO_GL_BACKEND}" "${PYTHON_BIN}" -m robocasa.utils.sim_tool_executor \
    --task HotDogSetup \
    --robots 2 \
    --layout "${layout}" \
    --style "${style}" \
    --seed "${SEED}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fps "${FPS}" \
    --demo-plan cooperative_hotdog_setup \
    --output-dir "${run_dir}"
done

echo "Saved outputs under ${OUTPUT_ROOT}"
