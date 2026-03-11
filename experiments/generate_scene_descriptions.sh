#!/usr/bin/env bash
# Generate scene descriptions for several task+layout combinations.
# Outputs JSON files to experiments/scene_descriptions/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/scene_descriptions"
mkdir -p "$OUTPUT_DIR"

CONDA_ENV="${CONDA_ENV:-robocasa}"

# Task / layout / style / seed combinations to generate
declare -a CONFIGS=(
    "Kitchen|11|34|42"
    "PrepareCoffee|11|34|42"
    "PickPlaceCounterToCabinet|11|34|100"
    "PickPlaceCounterToSink|11|34|200"
    "OpenSingleDoor|11|34|300"
    "MicrowaveThawing|11|34|400"
    "PrepareCoffee|5|10|42"
    "PickPlaceCounterToCabinet|5|10|99"
)

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r task layout style seed <<< "$config"
    outfile="${OUTPUT_DIR}/${task}_layout${layout}_style${style}_seed${seed}.json"
    echo "Generating: ${task} (layout=${layout}, style=${style}, seed=${seed})"

    conda run -n "$CONDA_ENV" python -c "
import os, json, sys
os.environ['MUJOCO_GL'] = 'osmesa'
from robocasa.utils.trajectory_runner import TrajectoryRunner

try:
    runner = TrajectoryRunner(
        task_name='${task}',
        robots=2,
        layout=${layout},
        style=${style},
        seed=${seed},
    )
    scene = runner.get_scene_description()
    with open('${outfile}', 'w') as f:
        json.dump(scene, f, indent=2)
    runner.close()
    print(f'  -> {\"${outfile}\"}')
    print(f'     fixtures={len(scene[\"fixtures\"])}, objects={len(scene[\"objects\"])}')
except Exception as e:
    print(f'  FAILED: {e}', file=sys.stderr)
    sys.exit(1)
"
done

echo ""
echo "Done. Scene descriptions saved to: ${OUTPUT_DIR}/"
ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | while read f; do
    echo "  $(basename "$f") ($(wc -c < "$f") bytes)"
done
