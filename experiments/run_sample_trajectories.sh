#!/usr/bin/env bash
# Run realistic sample trajectories and save visual observations.
# Outputs images + scene.json to experiments/trajectory_outputs/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/trajectory_outputs"
mkdir -p "$OUTPUT_DIR"

CONDA_ENV="${CONDA_ENV:-robocasa}"

conda run -n "$CONDA_ENV" python "${SCRIPT_DIR}/run_sample_trajectories.py" \
    --output-dir "$OUTPUT_DIR"
