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
    """PrepareCoffee: agent_0 fetches mug, delivers it to the coffee machine
    area for agent_1, then moves away so agent_1 can start the machine."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    coffee_machine = _find_fixture(fixtures, "coffee_machine")
    # Counter nearest to the coffee machine — where agent_0 will place the mug
    counter = _find_fixture(fixtures, "counter", near=coffee_machine)
    cabinet = _find_fixture(fixtures, "cabinet")

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
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll get the mug and bring it to the counter by the coffee machine."},
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {"to": "agent_0", "message": "Got it. I'll head to the coffee machine and wait."},
            },
            # Both navigate to their starting positions
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": coffee_machine},
            },
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": cabinet},
            },
            {
                "agent_id": "agent_0",
                "action": "interact",
                "args": {"fixture": cabinet, "action": "open"},
            },
            {
                "agent_id": "agent_1",
                "action": "wait",
                "args": {},
            },
            # agent_0 hands off mug: teleports it in front of agent_1, then stands beside agent_1
            {
                "agent_id": "agent_0",
                "action": "move_away",
                "args": {"object": mug, "to_agent": "agent_1"},
            },
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {"to": "agent_1", "message": "Mug is here. Go ahead."},
            },
            # agent_1 moves mug from counter to coffee machine
            {
                "agent_id": "agent_1",
                "action": "move_object",
                "args": {"object": mug, "from": counter, "to": coffee_machine},
            },
            {
                "agent_id": "agent_1",
                "action": "interact",
                "args": {"fixture": coffee_machine, "action": "turn_on"},
            },
        ],
    }


def make_pick_place_cabinet_trajectory(scene: dict) -> dict:
    """PickPlaceCounterToCabinet: agent_1 picks object from counter, agent_0
    opens cabinet. agent_1 delivers the object and moves away, agent_0 closes."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    cabinet = _find_fixture(fixtures, "cabinet")
    # Counter nearest to the cabinet — where the object should be
    counter = _find_fixture(fixtures, "counter", near=cabinet)
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
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll open the cabinet. Grab the object from the counter and place it inside."},
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {"to": "agent_0", "message": "Ready. I'll wait for you to open it."},
            },
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": cabinet},
            },
            {
                "agent_id": "agent_0",
                "action": "interact",
                "args": {"fixture": cabinet, "action": "open"},
            },
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": counter},
            },
            # agent_1 hands off object in front of agent_0 and stands beside them
            {
                "agent_id": "agent_1",
                "action": "move_away",
                "args": {"object": obj, "to_agent": "agent_0"},
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {"to": "agent_0", "message": "Object is here. Placing it in the cabinet."},
            },
            # agent_0 puts object from counter into cabinet
            {
                "agent_id": "agent_0",
                "action": "move_object",
                "args": {"object": obj, "from": counter, "to": cabinet},
            },
            {
                "agent_id": "agent_0",
                "action": "interact",
                "args": {"fixture": cabinet, "action": "close"},
            },
        ],
    }


def make_microwave_thawing_trajectory(scene: dict) -> dict:
    """MicrowaveThawing: agent_0 brings food from a counter near the microwave,
    delivers it, then moves away so agent_1 can close and start the microwave."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    microwave = _find_fixture(fixtures, "microwave")
    # Counter nearest to the microwave
    counter = _find_fixture(fixtures, "counter", near=microwave)
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
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {"to": "agent_1", "message": "I'll bring the food to the microwave. Open the door for me."},
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {"to": "agent_0", "message": "On it. I'll close the door and start it after."},
            },
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": microwave},
            },
            {
                "agent_id": "agent_1",
                "action": "interact",
                "args": {"fixture": microwave, "action": "open"},
            },
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": counter},
            },
            # agent_0 hands off food in front of agent_1 and stands beside them
            {
                "agent_id": "agent_0",
                "action": "move_away",
                "args": {"object": obj, "to_agent": "agent_1"},
            },
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {"to": "agent_1", "message": "Food's here. Close up and start it."},
            },
            # agent_1 puts food from counter into the microwave, then closes and starts
            {
                "agent_id": "agent_1",
                "action": "move_object",
                "args": {"object": obj, "from": counter, "to": microwave},
            },
            {
                "agent_id": "agent_1",
                "action": "interact",
                "args": {"fixture": microwave, "action": "close"},
            },
            {
                "agent_id": "agent_1",
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
            "agent_id": "agent_0",
            "action": "communicate",
            "args": {"to": "agent_1", "message": "Let's explore the kitchen. I'll check the cabinet and fridge."},
        },
        {
            "agent_id": "agent_1",
            "action": "communicate",
            "args": {"to": "agent_0", "message": "I'll check the drawers."},
        },
    ]

    if cabinet:
        steps.append({"agent_id": "agent_0", "action": "navigate", "args": {"fixture": cabinet}})
        steps.append({"agent_id": "agent_0", "action": "interact", "args": {"fixture": cabinet, "action": "open"}})
    if drawer:
        steps.append({"agent_id": "agent_1", "action": "navigate", "args": {"fixture": drawer}})
        steps.append({"agent_id": "agent_1", "action": "interact", "args": {"fixture": drawer, "action": "open"}})
    if fridge:
        steps.append({"agent_id": "agent_0", "action": "navigate", "args": {"fixture": fridge}})
        steps.append({"agent_id": "agent_0", "action": "interact", "args": {"fixture": fridge, "action": "open"}})
    if cabinet:
        steps.append({"agent_id": "agent_0", "action": "navigate", "args": {"fixture": cabinet}})
        steps.append({"agent_id": "agent_0", "action": "interact", "args": {"fixture": cabinet, "action": "close"}})

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

            # Save scene description in the run folder
            with open(run_dir / "scene.json", "w") as f:
                json.dump(result.scene, f, indent=2)

            # Save only per-agent views (for VLM training)
            agent_views = result.split_by_agent()
            for agent_id, agent_traj in agent_views.items():
                agent_dir = run_dir / agent_id
                agent_traj.save(agent_dir)

            print(f"  Saved to: {run_dir}/")
            print(f"  Agent views: {list(agent_views.keys())}")

            runner.close()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll done. Output: {output_root}/")


if __name__ == "__main__":
    main()
