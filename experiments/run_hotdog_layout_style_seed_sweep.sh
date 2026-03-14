#!/usr/bin/env bash
# Run the semantic HotDogSetup demo plan across a user-specified sweep of
# layouts, styles, and seeds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MUJOCO_GL_BACKEND="${MUJOCO_GL_BACKEND:-osmesa}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
FPS="${FPS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/tmp/hotdog_layout_style_seed_sweep}"

LAYOUTS_CSV="${LAYOUTS:-}"
STYLES_CSV="${STYLES:-}"
SEEDS_CSV="${SEEDS:-}"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_hotdog_layout_style_seed_sweep.sh \
    --layouts 11,24,56 \
    --styles 34,7,42 \
    --seeds 42,100,200

Options:
  --layouts   Comma-separated layout ids
  --styles    Comma-separated style ids
  --seeds     Comma-separated seed values
  --output    Output root directory
  --help      Show this message

Environment overrides:
  PYTHON_BIN
  MUJOCO_GL_BACKEND
  WIDTH
  HEIGHT
  FPS
  OUTPUT_ROOT
  LAYOUTS
  STYLES
  SEEDS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layouts)
      LAYOUTS_CSV="${2:-}"
      shift 2
      ;;
    --styles)
      STYLES_CSV="${2:-}"
      shift 2
      ;;
    --seeds)
      SEEDS_CSV="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LAYOUTS_CSV}" || -z "${STYLES_CSV}" || -z "${SEEDS_CSV}" ]]; then
  echo "You must provide --layouts, --styles, and --seeds." >&2
  usage >&2
  exit 1
fi

IFS=',' read -r -a LAYOUT_IDS <<< "${LAYOUTS_CSV}"
IFS=',' read -r -a STYLE_IDS <<< "${STYLES_CSV}"
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS_CSV}"

mkdir -p "${OUTPUT_ROOT}"

for layout in "${LAYOUT_IDS[@]}"; do
  for style in "${STYLE_IDS[@]}"; do
    for seed in "${SEED_VALUES[@]}"; do
      run_dir="${OUTPUT_ROOT}/layout${layout}_style${style}_seed${seed}"

      echo "Running HotDogSetup for layout=${layout} style=${style} seed=${seed}"
      MUJOCO_GL="${MUJOCO_GL_BACKEND}" "${PYTHON_BIN}" -m robocasa.utils.sim_tool_executor \
        --task HotDogSetup \
        --robots 2 \
        --layout "${layout}" \
        --style "${style}" \
        --seed "${seed}" \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --fps "${FPS}" \
        --demo-plan cooperative_hotdog_setup \
        --output-dir "${run_dir}"
    done
  done
done

echo "Saved outputs under ${OUTPUT_ROOT}"
