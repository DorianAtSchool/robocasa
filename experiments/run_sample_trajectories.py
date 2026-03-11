#!/usr/bin/env python3
"""
Run several realistic multi-agent trajectories and save visual observations.

Each trajectory uses real fixture IDs extracted from the scene.
The script builds the env, discovers fixtures, then runs a hand-crafted
trajectory that exercises move_object, interact, and communicate primitives.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from robocasa.utils.trajectory_runner import TrajectoryRunner


def make_prepare_coffee_trajectory(scene: dict) -> dict:
    """PrepareCoffee: agent_0 fetches mug from cabinet, agent_1 starts machine."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    # Find relevant fixtures
    cabinet = _find_fixture(fixtures, "cabinet")
    counter = _find_fixture(fixtures, "counter", near=cabinet)
    coffee_machine = _find_fixture(fixtures, "coffee_machine")

    # Find the mug
    mug = _find_object(objects, "mug") or _find_object(objects, type_contains="obj")

    if not all([cabinet, counter, coffee_machine, mug]):
        raise RuntimeError(
            f"Cannot build PrepareCoffee trajectory. "
            f"cabinet={cabinet}, counter={counter}, coffee_machine={coffee_machine}, mug={mug}"
        )

    return {
        "task": "PrepareCoffee",
        "steps": [
            {
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll get the mug from the cabinet and put it on the counter."},
            },
            {
                "action": "communicate",
                "args": {"to": "agent_0", "message": "Got it. I'll start the coffee machine once the mug is in position."},
            },
            {
                "action": "interact",
                "args": {"fixture": cabinet, "action": "open"},
            },
            {
                "action": "move_object",
                "args": {"object": mug, "from": cabinet, "to": counter},
            },
            {
                "action": "move_object",
                "args": {"object": mug, "from": counter, "to": coffee_machine},
            },
            {
                "action": "interact",
                "args": {"fixture": coffee_machine, "action": "turn_on"},
            },
        ],
    }


def make_pick_place_cabinet_trajectory(scene: dict) -> dict:
    """PickPlaceCounterToCabinet: agent_0 opens cabinet, agent_1 moves object."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    cabinet = _find_fixture(fixtures, "cabinet")
    counter = _find_fixture(fixtures, "counter")
    obj = _first_object(objects)

    if not all([cabinet, counter, obj]):
        raise RuntimeError(
            f"Cannot build PickPlace trajectory. "
            f"cabinet={cabinet}, counter={counter}, obj={obj}"
        )

    return {
        "task": "PickPlaceCounterToCabinet",
        "steps": [
            {
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll open the cabinet. You move the object in."},
            },
            {
                "action": "communicate",
                "args": {"to": "agent_0", "message": "Ready when you are."},
            },
            {
                "action": "interact",
                "args": {"fixture": cabinet, "action": "open"},
            },
            {
                "action": "move_object",
                "args": {"object": obj, "from": counter, "to": cabinet},
            },
            {
                "action": "interact",
                "args": {"fixture": cabinet, "action": "close"},
            },
        ],
    }


def make_microwave_thawing_trajectory(scene: dict) -> dict:
    """MicrowaveThawing: agent_0 gets food, agent_1 operates microwave."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    microwave = _find_fixture(fixtures, "microwave")
    counter = _find_fixture(fixtures, "counter")
    obj = _first_object(objects)

    if not all([microwave, counter, obj]):
        raise RuntimeError(
            f"Cannot build MicrowaveThawing trajectory. "
            f"microwave={microwave}, counter={counter}, obj={obj}"
        )

    return {
        "task": "MicrowaveThawing",
        "steps": [
            {
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll move the food to the microwave. You start it."},
            },
            {
                "action": "communicate",
                "args": {"to": "agent_0", "message": "OK, I'll open the microwave and start it."},
            },
            {
                "action": "interact",
                "args": {"fixture": microwave, "action": "open"},
            },
            {
                "action": "move_object",
                "args": {"object": obj, "from": counter, "to": microwave},
            },
            {
                "action": "interact",
                "args": {"fixture": microwave, "action": "close"},
            },
            {
                "action": "interact",
                "args": {"fixture": microwave, "action": "turn_on"},
            },
        ],
    }


def make_kitchen_explore_trajectory(scene: dict) -> dict:
    """Bare Kitchen: just interact with fixtures (no objects)."""
    fixtures = scene["fixtures"]

    cabinet = _find_fixture(fixtures, "cabinet")
    fridge = _find_fixture(fixtures, "fridge")
    drawer = _find_fixture(fixtures, "drawer")

    steps = [
        {
            "action": "communicate",
            "args": {"to": "agent_1", "message": "Let's explore the kitchen. I'll check the cabinet and fridge."},
        },
        {
            "action": "communicate",
            "args": {"to": "agent_0", "message": "I'll open the drawers."},
        },
    ]

    if cabinet:
        steps.append({"action": "interact", "args": {"fixture": cabinet, "action": "open"}})
    if fridge:
        steps.append({"action": "interact", "args": {"fixture": fridge, "action": "open"}})
    if drawer:
        steps.append({"action": "interact", "args": {"fixture": drawer, "action": "open"}})
    if cabinet:
        steps.append({"action": "interact", "args": {"fixture": cabinet, "action": "close"}})
    if fridge:
        steps.append({"action": "interact", "args": {"fixture": fridge, "action": "close"}})

    return {"task": "KitchenExplore", "steps": steps}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_fixture(fixtures: dict, type_contains: str, near: str | None = None) -> str | None:
    """Find a fixture ID whose type contains the given string."""
    candidates = [
        fid for fid, info in fixtures.items()
        if type_contains in info["fixture_type"]
    ]
    if not candidates:
        return None
    if near is None:
        return candidates[0]
    # Pick the one nearest to 'near' fixture
    near_pos = fixtures.get(near, {}).get("position", [0, 0, 0])
    candidates.sort(
        key=lambda fid: sum(
            (a - b) ** 2
            for a, b in zip(fixtures[fid]["position"][:2], near_pos[:2])
        )
    )
    return candidates[0]


def _find_object(objects: dict, type_contains: str | None = None) -> str | None:
    """Find an object ID by type substring."""
    for oid, info in objects.items():
        if type_contains and type_contains in info["object_type"]:
            return oid
    return None


def _first_object(objects: dict) -> str | None:
    """Return the first non-distractor object, or any object."""
    for oid in objects:
        if not oid.startswith("distr"):
            return oid
    return next(iter(objects), None)


# ---------------------------------------------------------------------------
# Scenario configs: (task_name, layout, style, seed, trajectory_builder)
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("PrepareCoffee", 11, 34, 42, make_prepare_coffee_trajectory),
    ("PickPlaceCounterToCabinet", 11, 34, 100, make_pick_place_cabinet_trajectory),
    ("MicrowaveThawing", 11, 34, 400, make_microwave_thawing_trajectory),
    ("Kitchen", 11, 34, 42, make_kitchen_explore_trajectory),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="experiments/trajectory_outputs")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--cameras", type=str, nargs="*", default=None,
        help="Camera names to render. Default: auto-detect.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for task_name, layout, style, seed, builder_fn in SCENARIOS:
        run_name = f"{task_name}_layout{layout}_seed{seed}"
        run_dir = output_root / run_name
        print(f"\n{'='*60}")
        print(f"Scenario: {run_name}")
        print(f"{'='*60}")

        try:
            runner = TrajectoryRunner(
                task_name=task_name,
                robots=2,
                layout=layout,
                style=style,
                seed=seed,
                camera_names=args.cameras,
                render_width=args.width,
                render_height=args.height,
            )

            scene = runner.get_scene_description()
            print(f"  Task lang: {scene.get('task', 'N/A')}")
            print(f"  Fixtures: {len(scene['fixtures'])}")
            print(f"  Objects:  {len(scene['objects'])}")

            trajectory = builder_fn(scene)
            print(f"  Steps:    {len(trajectory['steps'])}")

            # Save the trajectory definition
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "trajectory.json", "w") as f:
                json.dump(trajectory, f, indent=2)

            result = runner.run(trajectory)
            result.save(run_dir)

            print(f"  Saved to: {run_dir}/")
            print(f"  Images:   {1 + len(result.steps) * 2} sets x {len(result.initial_obs)} cameras")

            runner.close()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll done. Output: {output_root}/")


if __name__ == "__main__":
    main()
