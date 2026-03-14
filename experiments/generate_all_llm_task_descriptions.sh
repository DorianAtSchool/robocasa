#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/experiments/llm_task_descriptions}"
ROBOTS="${ROBOTS:-2}"
SEED="${SEED:-42}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-240}"
GL_BACKEND="${MUJOCO_GL:-osmesa}"

mkdir -p "$OUTPUT_DIR"

mapfile -t TASKS < <(
  cd "$ROOT_DIR" && \
  python -m robocasa.scripts.generate_llm_task_descriptions --list-tasks
)

SUCCESS_LOG="$OUTPUT_DIR/successes.txt"
FAIL_LOG="$OUTPUT_DIR/failures.txt"
: > "$SUCCESS_LOG"
: > "$FAIL_LOG"

for TASK_NAME in "${TASKS[@]}"; do
  TASK_OUTPUT_DIR="$OUTPUT_DIR/$TASK_NAME"
  mkdir -p "$TASK_OUTPUT_DIR"

  echo "Generating description for $TASK_NAME"
  if (
    cd "$ROOT_DIR" && \
    MUJOCO_GL="$GL_BACKEND" python -m robocasa.scripts.generate_llm_task_descriptions \
      --task "$TASK_NAME" \
      --robots "$ROBOTS" \
      --seed "$SEED" \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --gl-backend "$GL_BACKEND" \
      --output-dir "$TASK_OUTPUT_DIR"
  ); then
    echo "$TASK_NAME" >> "$SUCCESS_LOG"
  else
    echo "$TASK_NAME" >> "$FAIL_LOG"
    echo "Failed to generate description for $TASK_NAME"
  fi
done

echo "Descriptions written to $OUTPUT_DIR"
echo "Successes logged in $SUCCESS_LOG"
echo "Failures logged in $FAIL_LOG"
