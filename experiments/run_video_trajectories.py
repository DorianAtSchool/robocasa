#!/usr/bin/env python3
"""
Generate multi-camera trajectory videos with atomic-task playback.

Builds a two-robot trajectory generically for any task that is actually
chainable under the available atomic transition graph. Robots are visually
distinguished (robot1 is recoloured amber) and never overlap because
clearing ``navigate`` steps are explicit in the trajectory.

Usage:
    python experiments/run_video_trajectories.py
    python experiments/run_video_trajectories.py --output-dir /tmp/traj_videos
"""

import argparse
from collections import deque
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from robocasa.models.fixtures import FixtureType


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from run_sample_trajectories.py)
# ---------------------------------------------------------------------------

def _find_fixture(
    fixtures: dict, type_contains: str, near: str | None = None
) -> str | None:
    """Find a fixture ID whose type contains the given string."""
    candidates = [
        fid
        for fid, info in fixtures.items()
        if type_contains in info["fixture_type"]
    ]
    if not candidates:
        return None
    if near is None:
        return candidates[0]
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


def _find_fixture_away_from(
    fixtures: dict, avoid_fixtures: list[str], min_type: str = "counter"
) -> str | None:
    """Find a fixture with good distance from the listed fixture IDs.

    Prefers counters/surfaces that are far from all avoidance points.
    """
    avoid_positions = []
    for fid in avoid_fixtures:
        if fid and fid in fixtures:
            pos = fixtures[fid].get("position", [0, 0, 0])
            avoid_positions.append(np.array(pos[:2]))

    if not avoid_positions:
        return _find_fixture(fixtures, min_type)

    candidates = [
        fid
        for fid, info in fixtures.items()
        if min_type in info["fixture_type"]
    ]
    if not candidates:
        # Fall back to any fixture
        candidates = list(fixtures.keys())

    def min_dist_to_avoid(fid):
        pos = np.array(fixtures[fid]["position"][:2])
        return min(float(np.linalg.norm(pos - ap)) for ap in avoid_positions)

    # Sort by distance (farthest first), return the farthest
    candidates.sort(key=min_dist_to_avoid, reverse=True)
    return candidates[0] if candidates else None


_FIXTURE_ALIASES = (
    ("coffee machine dispenser", "coffee_machine"),
    ("coffee machine", "coffee_machine"),
    ("microwave", "microwave"),
    ("toaster oven", "toaster_oven"),
    ("stand mixer", "stand_mixer"),
    ("blender", "blender"),
    ("cabinet", "cabinet"),
    ("drawer", "drawer"),
    ("counter", "counter"),
    ("sink", "sink"),
    ("stove", "stove"),
    ("oven", "oven"),
    ("fridge", "fridge"),
)

_FIXTURE_ENUMS = {
    "coffee_machine": FixtureType.COFFEE_MACHINE,
    "microwave": FixtureType.MICROWAVE,
    "toaster_oven": FixtureType.TOASTER_OVEN,
    "stand_mixer": FixtureType.STAND_MIXER,
    "blender": FixtureType.BLENDER,
    "cabinet": FixtureType.CABINET,
    "drawer": FixtureType.DRAWER,
    "counter": FixtureType.COUNTER,
    "sink": FixtureType.SINK,
    "stove": FixtureType.STOVE,
    "oven": FixtureType.OVEN,
    "fridge": FixtureType.FRIDGE,
}

_MOVE_EDGE_SPECS = (
    ("cabinet", "counter", "PickPlaceCabinetToCounter"),
    ("counter", "cabinet", "PickPlaceCounterToCabinet"),
    ("counter", "sink", "PickPlaceCounterToSink"),
    ("sink", "counter", "PickPlaceSinkToCounter"),
    ("counter", "microwave", "PickPlaceCounterToMicrowave"),
    ("microwave", "counter", "PickPlaceMicrowaveToCounter"),
    ("counter", "oven", "PickPlaceCounterToOven"),
    ("counter", "stove", "PickPlaceCounterToStove"),
    ("stove", "counter", "PickPlaceStoveToCounter"),
    ("counter", "toaster_oven", "PickPlaceCounterToToasterOven"),
    ("toaster_oven", "counter", "PickPlaceToasterOvenToCounter"),
    ("counter", "blender", "PickPlaceCounterToBlender"),
    ("counter", "drawer", "PickPlaceCounterToDrawer"),
    ("drawer", "counter", "PickPlaceDrawerToCounter"),
    ("counter", "coffee_machine", "CoffeeSetupMug"),
)

_INTERACT_TASKS = {
    ("open", "cabinet"): "OpenCabinet",
    ("close", "cabinet"): "CloseCabinet",
    ("turn_on", "microwave"): "TurnOnMicrowave",
    ("turn_on", "coffee_machine"): "StartCoffeeMachine",
}


def _canonical_fixture_type(text: str | None) -> str | None:
    """Map a fixture description to a canonical semantic type."""
    if not text:
        return None
    lowered = text.lower().replace("-", " ").replace("_", " ")
    for alias, canonical in _FIXTURE_ALIASES:
        if alias in lowered:
            return canonical
    return None


def _fixture_obj_to_id(runner, fixture_obj) -> str | None:
    """Map a fixture instance back to the runner's fixture id."""
    if fixture_obj is None:
        return None
    for fixture_id, candidate in runner._fixtures.items():
        if candidate is fixture_obj:
            return fixture_id
    fixture_name = getattr(fixture_obj, "name", None)
    if fixture_name is not None:
        for fixture_id, candidate in runner._fixtures.items():
            if getattr(candidate, "name", None) == fixture_name:
                return fixture_id
    return None


def _resolve_related_fixture_id(runner, semantic_type: str, ref_fixture_id: str) -> str | None:
    """Resolve a fixture of the requested type that is semantically tied to ref_fixture_id."""
    fixture_enum = _FIXTURE_ENUMS.get(semantic_type)
    ref_fixture = runner._fixtures.get(ref_fixture_id)
    if fixture_enum is None or ref_fixture is None:
        return None
    try:
        related = runner.env.get_fixture(fixture_enum, ref=ref_fixture)
    except Exception:
        return None
    return _fixture_obj_to_id(runner, related)


def _extract_destination_fixture_type(task_lang: str) -> str | None:
    """Extract the destination fixture type from the task language."""
    patterns = (
        r"(?:place|move|transport|carry)[^.]*?(?:into|in|onto|on|under|to|next to) the ([a-z _-]+?)(?:\.|,| and| then|$)",
        r"(?:put)[^.]*?(?:into|in|onto|on|under|to|next to) the ([a-z _-]+?)(?:\.|,| and| then|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, task_lang)
        if match:
            semantic = _canonical_fixture_type(match.group(1))
            if semantic is not None:
                return semantic
    return None


def _extract_source_fixture_type(task_lang: str) -> str | None:
    """Extract the source fixture type from the task language."""
    patterns = (
        r"(?:pick|pick up|move|transport|carry)[^.]*? from the ([a-z _-]+?)(?:\.|,| and| then| to| into| in| onto| on| under|$)",
        r"(?:take|get|grab)[^.]*? from the ([a-z _-]+?)(?:\.|,| and| then| to| into| in| onto| on| under|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, task_lang)
        if match:
            semantic = _canonical_fixture_type(match.group(1))
            if semantic is not None:
                return semantic
    return None


def _extract_interaction_action(task_lang: str, dest_semantic: str | None) -> tuple[str, str] | None:
    """Extract a terminal interaction like turning on an appliance."""
    if "turn on the microwave" in task_lang:
        return ("turn_on", "microwave")
    if "turn on the coffee machine" in task_lang:
        return ("turn_on", "coffee_machine")
    if "press the start button" in task_lang and dest_semantic in {"microwave", "coffee_machine"}:
        return ("turn_on", dest_semantic)
    return None


def _choose_primary_object_id(runner, scene: dict, task_lang: str) -> str | None:
    """Choose the main task object from graspable object configs and language."""
    object_cfgs = runner.env.get_ep_meta().get("object_cfgs", [])
    candidates = []
    for cfg in object_cfgs:
        obj_name = cfg.get("name")
        if not obj_name or obj_name not in scene["objects"]:
            continue
        if obj_name.startswith("distr") or obj_name == "container":
            continue
        if cfg.get("graspable"):
            candidates.append(obj_name)

    if not candidates:
        return _first_object(
            {
                oid: info
                for oid, info in scene["objects"].items()
                if oid != "container" and not oid.startswith("distr")
            }
        )

    if len(candidates) == 1:
        return candidates[0]

    scored = []
    for obj_id in candidates:
        obj_type = scene["objects"][obj_id]["object_type"].lower().replace("_", " ")
        score = 0
        for token in obj_type.split():
            if token and token in task_lang:
                score += 1
        if obj_id.lower() in task_lang:
            score += 2
        scored.append((score, obj_id))
    scored.sort(reverse=True)
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        raise RuntimeError(
            f"Ambiguous primary object for task language {task_lang!r}: {candidates}"
        )
    return scored[0][1]


def _build_atomic_move_graph(runner, scene: dict) -> dict[str, list[tuple[str, str]]]:
    """Build the directed graph of actual fixture-to-fixture atomic transitions."""
    fixture_semantics = {
        fixture_id: _canonical_fixture_type(info["fixture_type"])
        for fixture_id, info in scene["fixtures"].items()
    }
    graph = {fixture_id: [] for fixture_id in scene["fixtures"]}

    for src_semantic, dst_semantic, atomic_task in _MOVE_EDGE_SPECS:
        if dst_semantic == "counter":
            for src_fixture_id, semantic in fixture_semantics.items():
                if semantic != src_semantic:
                    continue
                dst_fixture_id = _resolve_related_fixture_id(
                    runner,
                    "counter",
                    src_fixture_id,
                )
                if dst_fixture_id:
                    graph[src_fixture_id].append((dst_fixture_id, atomic_task))
        elif src_semantic == "counter":
            for dst_fixture_id, semantic in fixture_semantics.items():
                if semantic != dst_semantic:
                    continue
                src_fixture_id = _resolve_related_fixture_id(
                    runner,
                    "counter",
                    dst_fixture_id,
                )
                if src_fixture_id:
                    graph[src_fixture_id].append((dst_fixture_id, atomic_task))

    return graph


def _find_atomic_path(
    runner,
    scene: dict,
    start_semantic: str,
    target_semantic: str,
) -> tuple[str, list[tuple[str, str, str]], str] | tuple[None, None, None]:
    """Find a shortest atomic move path on actual fixtures."""
    graph = _build_atomic_move_graph(runner, scene)
    fixture_semantics = {
        fixture_id: _canonical_fixture_type(info["fixture_type"])
        for fixture_id, info in scene["fixtures"].items()
    }
    goal_fixtures = {
        fixture_id
        for fixture_id, semantic in fixture_semantics.items()
        if semantic == target_semantic
    }
    start_fixtures = [
        fixture_id
        for fixture_id, semantic in fixture_semantics.items()
        if semantic == start_semantic
    ]
    if not start_fixtures:
        return None, None, None
    if not goal_fixtures:
        return None, None, None

    queue = deque((start_fixture_id, start_fixture_id, []) for start_fixture_id in start_fixtures)
    visited = set(start_fixtures)

    while queue:
        root_fixture_id, fixture_id, path = queue.popleft()
        if fixture_id in goal_fixtures and path:
            return root_fixture_id, path, fixture_id
        for next_fixture_id, atomic_task in graph.get(fixture_id, []):
            if next_fixture_id in visited:
                continue
            visited.add(next_fixture_id)
            queue.append(
                (
                    root_fixture_id,
                    next_fixture_id,
                    path + [(fixture_id, next_fixture_id, atomic_task)],
                )
            )

    return None, None, None


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------

def make_prepare_coffee_video_trajectory(scene: dict) -> dict:
    """PrepareCoffee decomposed into atomic tasks plus clearance navigations."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    cabinet = _find_fixture(fixtures, "cabinet")
    coffee_machine = _find_fixture(fixtures, "coffee_machine")
    coffee_counter = _find_fixture(fixtures, "counter", near=coffee_machine)
    mug = _find_object(objects, "mug") or _first_object(objects)

    # Find a counter/fixture away from the action areas for clearing
    clear_fixture = _find_fixture_away_from(
        fixtures, [cabinet, coffee_machine]
    )

    if not all([cabinet, coffee_counter, coffee_machine, mug]):
        raise RuntimeError(
            f"Cannot build PrepareCoffee trajectory. "
            f"cabinet={cabinet}, "
            f"coffee_counter={coffee_counter}, "
            f"coffee_machine={coffee_machine}, mug={mug}"
        )

    return {
        "task": "PrepareCoffee",
        "steps": [
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I'll get the mug from the cabinet.",
                },
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I'll wait by the counter.",
                },
            },
            # agent_1 clears space by moving to a counter away from the cabinet
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": clear_fixture},
            },
            # agent_0 navigates to the cabinet
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": cabinet},
            },
            # agent_0 opens the cabinet
            {
                "agent_id": "agent_0",
                "action": "interact",
                "args": {"fixture": cabinet, "action": "open"},
                "atomic_task": "OpenCabinet",
            },
            # There is no single atomic cabinet -> coffee machine transfer.
            # Decompose it into cabinet -> nearby counter, then counter -> machine.
            {
                "agent_id": "agent_0",
                "action": "move_object",
                "args": {"object": mug, "to": coffee_counter},
                "atomic_task": "PickPlaceCabinetToCounter",
            },
            {
                "agent_id": "agent_0",
                "action": "move_object",
                "args": {"object": mug, "to": coffee_machine},
                "atomic_task": "CoffeeSetupMug",
            },
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "Mug placed. Moving away.",
                },
            },
            # agent_0 clears space so agent_1 can approach the coffee machine
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": clear_fixture},
            },
            # agent_1 navigates to the coffee machine
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": coffee_machine},
            },
            # agent_1 turns on the coffee machine
            {
                "agent_id": "agent_1",
                "action": "interact",
                "args": {"fixture": coffee_machine, "action": "turn_on"},
                "atomic_task": "StartCoffeeMachine",
            },
        ],
    }


def make_microwave_thawing_video_trajectory(scene: dict) -> dict:
    """MicrowaveThawing split across two robots with a clean atomic handoff."""
    fixtures = scene["fixtures"]
    objects = scene["objects"]

    microwave = _find_fixture(fixtures, "microwave")
    counter = _find_fixture(fixtures, "counter", near=microwave)
    food = _first_object({oid: info for oid, info in objects.items() if oid != "container"})
    clear_fixture = _find_fixture_away_from(fixtures, [microwave, counter])

    if not all([microwave, counter, food, clear_fixture]):
        raise RuntimeError(
            f"Cannot build MicrowaveThawing trajectory. "
            f"microwave={microwave}, counter={counter}, food={food}, clear={clear_fixture}"
        )

    return {
        "task": "MicrowaveThawing",
        "steps": [
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I'll load the food into the microwave.",
                },
            },
            {
                "agent_id": "agent_1",
                "action": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I'll wait clear, then start it.",
                },
            },
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": clear_fixture},
            },
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": counter},
            },
            {
                "agent_id": "agent_0",
                "action": "move_object",
                "args": {"object": food, "to": microwave},
                "atomic_task": "PickPlaceCounterToMicrowave",
            },
            {
                "agent_id": "agent_0",
                "action": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "Food is loaded. You can start the microwave.",
                },
            },
            {
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": clear_fixture},
            },
            {
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": microwave},
            },
            {
                "agent_id": "agent_1",
                "action": "interact",
                "args": {"fixture": microwave, "action": "turn_on"},
                "atomic_task": "TurnOnMicrowave",
            },
        ],
    }


def make_chainable_video_trajectory(runner, scene: dict) -> dict:
    """Build a two-robot trajectory for any task chainable by the atomic graph."""
    task_lang = (scene.get("task") or "").lower()
    if not task_lang:
        raise RuntimeError("Scene is missing task language; cannot infer chainable plan.")

    object_id = _choose_primary_object_id(runner, scene, task_lang)
    if object_id is None:
        raise RuntimeError("Could not infer the primary task object.")

    start_semantic = _extract_source_fixture_type(task_lang)
    if start_semantic is None:
        object_location = scene["objects"][object_id]["location"]
        if object_location in scene["fixtures"]:
            start_semantic = _canonical_fixture_type(
                scene["fixtures"][object_location]["fixture_type"]
            )
    if start_semantic is None:
        raise RuntimeError(
            f"Could not infer a source fixture from task language: {scene['task']!r}"
        )

    dest_semantic = _extract_destination_fixture_type(task_lang)
    interaction = _extract_interaction_action(task_lang, dest_semantic)
    if dest_semantic is None and interaction is not None:
        dest_semantic = interaction[1]
    if dest_semantic is None:
        raise RuntimeError(
            f"Could not infer a fixture destination from task language: {scene['task']!r}"
        )

    start_fixture_id, move_path, final_fixture_id = _find_atomic_path(
        runner,
        scene,
        start_semantic,
        dest_semantic,
    )
    if move_path is None or start_fixture_id is None:
        raise RuntimeError(
            "Task is not chainable under the current atomic transition graph. "
            f"source={start_semantic} target={dest_semantic}"
        )

    clear_fixture = _find_fixture_away_from(
        scene["fixtures"],
        [start_fixture_id, final_fixture_id],
    )
    if clear_fixture is None:
        raise RuntimeError("Could not find a clearance fixture for the idle robot.")

    steps = [
        {
            "agent_id": "agent_0",
            "action": "communicate",
            "args": {
                "to": "agent_1",
                "message": f"I'll handle the {object_id} transfer.",
            },
        },
        {
            "agent_id": "agent_1",
            "action": "communicate",
            "args": {
                "to": "agent_0",
                "message": "I'll stay clear and take over if an appliance interaction is needed.",
            },
        },
        {
            "agent_id": "agent_1",
            "action": "navigate",
            "args": {"fixture": clear_fixture},
        },
        {
            "agent_id": "agent_0",
            "action": "navigate",
            "args": {"fixture": start_fixture_id},
        },
    ]

    if start_semantic == "cabinet":
        steps.append(
            {
                "agent_id": "agent_0",
                "action": "interact",
                "args": {"fixture": start_fixture_id, "action": "open"},
                "atomic_task": "OpenCabinet",
            }
        )

    current_fixture_id = start_fixture_id
    for from_fixture_id, next_fixture_id, atomic_task in move_path:
        steps.append(
            {
                "agent_id": "agent_0",
                "action": "move_object",
                "args": {
                    "object": object_id,
                    "from": from_fixture_id,
                    "to": next_fixture_id,
                },
                "atomic_task": atomic_task,
            }
        )
        current_fixture_id = next_fixture_id

    if interaction is not None:
        interact_action, interact_semantic = interaction
        if _canonical_fixture_type(scene["fixtures"][current_fixture_id]["fixture_type"]) != interact_semantic:
            raise RuntimeError(
                f"Final fixture {current_fixture_id!r} does not match interaction target {interact_semantic!r}"
            )
        steps.extend(
            [
                {
                    "agent_id": "agent_0",
                    "action": "communicate",
                    "args": {
                        "to": "agent_1",
                        "message": "Transfer complete. Your turn on the appliance.",
                    },
                },
                {
                    "agent_id": "agent_0",
                    "action": "navigate",
                    "args": {"fixture": clear_fixture},
                },
                {
                    "agent_id": "agent_1",
                    "action": "navigate",
                    "args": {"fixture": current_fixture_id},
                },
                {
                    "agent_id": "agent_1",
                    "action": "interact",
                    "args": {"fixture": current_fixture_id, "action": interact_action},
                    "atomic_task": _INTERACT_TASKS[(interact_action, interact_semantic)],
                },
            ]
        )

    return {
        "task": type(getattr(runner.env, "env", runner.env)).__name__,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory videos with atomic-task playback"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/trajectory_videos",
    )
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--task",
        type=str,
        default="MicrowaveThawing",
    )
    args = parser.parse_args()

    from robocasa.utils.video_trajectory_runner import VideoTrajectoryRunner

    scenarios = [(args.task, args.seed)]

    output_root = Path(args.output_dir)

    for task_name, seed in scenarios:
        run_name = f"{task_name}_layout{args.layout}_seed{seed}"
        run_dir = output_root / run_name

        print(f"\n{'=' * 60}")
        print(f"Scenario: {run_name}")
        print(f"{'=' * 60}")

        try:
            runner = VideoTrajectoryRunner(
                task_name=task_name,
                robots=2,
                layout=args.layout,
                style=args.style,
                seed=seed,
                render_width=args.width,
                render_height=args.height,
                fps=args.fps,
                output_dir=str(run_dir),
            )

            scene = runner.get_scene_description()
            print(f"  Task lang: {scene.get('task', 'N/A')}")
            print(f"  Fixtures:  {len(scene['fixtures'])}")
            print(f"  Objects:   {len(scene['objects'])}")

            # Use task-specific builders when available, fall back to chainable
            if task_name == "PrepareCoffee":
                trajectory = make_prepare_coffee_video_trajectory(scene)
            elif task_name == "MicrowaveThawing":
                trajectory = make_microwave_thawing_video_trajectory(scene)
            else:
                trajectory = make_chainable_video_trajectory(runner, scene)
            print(f"  Steps:     {len(trajectory['steps'])}")

            metadata = runner.run(trajectory)
            print(f"  Frames:    {metadata['total_frames']}")
            print(f"  Output:    {run_dir}/")

            # List generated files
            for f in sorted(run_dir.iterdir()):
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"    {f.name} ({size_kb:.1f} KB)")

            runner.close()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. Output: {output_root}/")


if __name__ == "__main__":
    main()
