#!/usr/bin/env python3
"""
Generate two-robot HotDogSetup trajectory videos.

Two modes:
  --mode teleport   Teleport robots and objects per step (fast, visual-only).
  --mode demo       Split a real single-robot demo across two robots, replaying
                    actual arm motion from the dataset.

HotDogSetup has 3 subtasks:
  1. hotdog_bun:  counter → plate on dining table  (agent_0)
  2. condiment:   cabinet → near plate              (agent_0)
  3. sausage:     fridge  → plate on dining table   (agent_1)

Agent 1 is "teleported in" near the fridge to handle the sausage subtask
concurrently / sequentially while agent 0 handles bun + condiment.

Usage:
    python experiments/run_hotdog_trajectories.py --mode teleport
    python experiments/run_hotdog_trajectories.py --mode demo
    python experiments/run_hotdog_trajectories.py --mode demo --episode 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_fixture(fixtures: dict, type_contains: str, near: str | None = None) -> str | None:
    type_lower = type_contains.lower()
    candidates = [
        fid for fid, info in fixtures.items()
        if type_lower in info["fixture_type"].lower()
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


def _find_fixture_away_from(fixtures: dict, avoid: list[str], prefer_type: str = "counter") -> str | None:
    avoid_pos = []
    for fid in avoid:
        if fid and fid in fixtures:
            avoid_pos.append(np.array(fixtures[fid]["position"][:2]))
    if not avoid_pos:
        return _find_fixture(fixtures, prefer_type)
    candidates = [fid for fid, info in fixtures.items() if prefer_type.lower() in info["fixture_type"].lower()]
    if not candidates:
        candidates = list(fixtures.keys())
    candidates.sort(
        key=lambda fid: min(
            float(np.linalg.norm(np.array(fixtures[fid]["position"][:2]) - ap))
            for ap in avoid_pos
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _first_object(objects: dict) -> str | None:
    for oid in objects:
        if not oid.startswith("distr") and oid != "container":
            return oid
    return next(iter(objects), None)


# ---------------------------------------------------------------------------
# Mode 1: Teleport trajectory
# ---------------------------------------------------------------------------

def _resolve_objects(objects: dict) -> dict[str, str | None]:
    """Map semantic names to object IDs for the HotDogSetup task."""
    result = {"bun": None, "sausage": None, "condiment": None, "plate": None}
    for oid, info in objects.items():
        otype = info["object_type"].lower()
        if "bun" in otype or "hotdog_bun" in oid:
            result["bun"] = oid
        elif "sausage" in otype or "sausage" in oid:
            result["sausage"] = oid
        elif any(k in otype for k in ("condiment", "ketchup", "mustard")) or oid == "condiment":
            result["condiment"] = oid
        elif "plate" in otype and "container" not in oid:
            result["plate"] = oid
    return result


def make_teleport_trajectory(scene: dict) -> dict:
    """Build a two-robot HotDogSetup trajectory using teleportation.

    Objects and robots are teleported per step.  No demo playback.
    """
    fixtures = scene["fixtures"]
    objects = scene["objects"]
    obj = _resolve_objects(objects)

    # Resolve fixtures — the task uses: counter (bun), cabinet (condiment),
    # fridge (sausage), dining table (destination plate).
    # The scene's _setup_scene already opens the cabinet door.
    cabinet = _find_fixture(fixtures, "cabinet")
    counter = _find_fixture(fixtures, "counter", near=cabinet)
    fridge = _find_fixture(fixtures, "fridge")
    dining = _find_fixture(fixtures, "dining") or _find_fixture(fixtures, "counter")

    if not all([cabinet, counter, fridge, dining]):
        raise RuntimeError(
            f"Missing fixtures: cabinet={cabinet}, counter={counter}, "
            f"fridge={fridge}, dining={dining}"
        )

    steps = [
        # --- Communication ---
        {
            "agent_id": "agent_0",
            "action": "communicate",
            "args": {"to": "agent_1", "message": "I'll get the bun and condiment. You grab the sausage from the fridge."},
        },
        {
            "agent_id": "agent_1",
            "action": "communicate",
            "args": {"to": "agent_0", "message": "Got it. Heading to the fridge."},
        },

        # --- Agent 1: go to fridge and open it ---
        {
            "agent_id": "agent_1",
            "action": "navigate",
            "args": {"fixture": fridge},
        },
        {
            "agent_id": "agent_1",
            "action": "teleport_interact",
            "args": {"fixture": fridge, "action": "open"},
        },

        # --- Agent 0: go to counter (bun is here) ---
        {
            "agent_id": "agent_0",
            "action": "navigate",
            "args": {"fixture": counter},
        },

        # --- Agent 0: bun → dining table ---
        {
            "agent_id": "agent_0",
            "action": "teleport_object",
            "args": {"object": obj["bun"], "from": counter, "to": dining},
        },

        # --- Agent 1: sausage → dining table ---
        {
            "agent_id": "agent_1",
            "action": "teleport_object",
            "args": {"object": obj["sausage"], "from": fridge, "to": dining},
        },

        # --- Agent 0: navigate to cabinet for condiment ---
        {
            "agent_id": "agent_0",
            "action": "navigate",
            "args": {"fixture": cabinet},
        },

        # --- Agent 0: condiment → dining table ---
        {
            "agent_id": "agent_0",
            "action": "teleport_object",
            "args": {"object": obj["condiment"], "from": cabinet, "to": dining},
        },

        # --- Wrap up ---
        {
            "agent_id": "agent_0",
            "action": "communicate",
            "args": {"to": "agent_1", "message": "All done. Hot dog is set up!"},
        },
    ]

    # Drop steps whose object resolved to None
    steps = [s for s in steps if s["args"].get("object") is not None or s["action"] != "teleport_object"]

    return {"task": "HotDogSetup", "steps": steps}


# ---------------------------------------------------------------------------
# Mode 2: Demo-split trajectory
# ---------------------------------------------------------------------------

def _segment_demo(states: np.ndarray, nav_threshold: float = 0.01, gap: int = 5):
    """Split a demo into stationary action segments separated by navigation.

    Returns a list of (start_frame, end_frame) for each stationary segment
    where the robot's arm is active and the base is mostly still.
    """
    base_xy = states[:, 1:3]
    diffs = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)

    nav_mask = diffs > nav_threshold
    nav_mask = np.append(nav_mask, False)

    segments = []
    in_segment = False
    seg_start = 0

    for i in range(len(states)):
        if not nav_mask[i]:
            if not in_segment:
                seg_start = i
                in_segment = True
        else:
            if in_segment and (i - seg_start) > gap:
                segments.append((seg_start, i))
            in_segment = False

    if in_segment and (len(states) - seg_start) > gap:
        segments.append((seg_start, len(states)))

    return segments


def _find_gripper_events(states: np.ndarray, threshold: float = 0.005, min_gap: int = 20):
    """Find pickup (close) and place (open) gripper events."""
    gripper = states[:, 12:14]
    grip_width = gripper[:, 0] - gripper[:, 1]
    grip_diff = np.diff(grip_width)

    pickups = []
    prev = -min_gap
    for f in np.where(grip_diff < -threshold)[0]:
        if f - prev > min_gap:
            pickups.append(int(f))
        prev = f

    places = []
    prev = -min_gap
    for f in np.where(grip_diff > threshold)[0]:
        if f - prev > min_gap:
            places.append(int(f))
        prev = f

    return pickups, places


def _find_fridge_trip_start(
    states: np.ndarray,
    segments: list[tuple[int, int]],
    pickups: list[int],
) -> int | None:
    """Find the segment index where the fridge trip begins.

    The HotDogSetup demo does bun (counter), condiment (cabinet), then
    sausage (fridge).  The fridge trip starts when the robot navigates to
    a new base position for the last pickup+place cycle.

    We identify it by finding the segment containing the last pickup event
    (sausage grab) and then looking backward for the first segment at the
    same base position cluster — that's where the fridge trip begins
    (the robot navigated there).
    """
    if len(pickups) < 2:
        return None

    base_xy = states[:, 1:3]
    last_pickup = pickups[-1]

    # Find which segment contains the last pickup
    fridge_seg = None
    for i, (s, e) in enumerate(segments):
        if s <= last_pickup < e:
            fridge_seg = i
            break
    if fridge_seg is None:
        return None

    # The fridge base position cluster
    fridge_base = base_xy[segments[fridge_seg][0]:segments[fridge_seg][1]].mean(axis=0)

    # Walk backward: include all earlier segments at a similar base position
    # (the robot may have multiple stationary segments at the fridge)
    trip_start = fridge_seg
    for i in range(fridge_seg - 1, -1, -1):
        seg_base = base_xy[segments[i][0]:segments[i][1]].mean(axis=0)
        dist = float(np.linalg.norm(seg_base - fridge_base))
        if dist < 1.0:
            trip_start = i
        else:
            break

    return trip_start



def _load_episode_meta(ep_idx: int) -> dict:
    """Load the ep_meta.json for a HotDogSetup episode."""
    from robocasa.utils.video_trajectory_runner import _get_demo_dataset_path

    ds_path, _ = _get_demo_dataset_path("HotDogSetup")
    meta_file = ds_path / "extras" / f"episode_{ep_idx:06d}" / "ep_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No episode metadata at {meta_file}")
    import json as _json
    return _json.loads(meta_file.read_text())



def build_demo_split_trajectory(scene: dict | None, ep_idx: int = 0) -> tuple[dict, int, int]:
    """Build a two-robot trajectory by splitting a real HotDogSetup demo.

    Returns (trajectory_dict, layout_id, style_id) so the caller can
    create the env with the correct layout/style matching the demo episode.

    Agent 0 replays all pre-fridge demo segments (bun + condiment subtasks).
    Agent 1 gets the fridge trip: navigate to fridge, open fridge, grab
    sausage, navigate to dining table, place sausage.
    """
    from robocasa.utils.video_trajectory_runner import _get_demo_dataset_path
    import robocasa.utils.lerobot_utils as LU

    # Load episode metadata to get layout/style and fixture refs
    ep_meta = _load_episode_meta(ep_idx)
    layout_id = ep_meta["layout_id"]
    style_id = ep_meta["style_id"]
    fixture_refs = ep_meta["fixture_refs"]
    print(f"  Demo episode {ep_idx}: layout={layout_id}, style={style_id}")
    print(f"  Fixture refs: {fixture_refs}")

    ds_path, _ = _get_demo_dataset_path("HotDogSetup")
    states = LU.get_episode_states(ds_path, ep_idx)
    print(f"  Demo frames: {len(states)}")

    segments = _segment_demo(states)
    base_xy = states[:, 1:3]
    print(f"  Stationary segments: {len(segments)}")
    for i, (s, e) in enumerate(segments):
        arm = states[s:e, 5:12]
        act = float(np.max(np.abs(np.diff(arm, axis=0)))) if len(arm) > 1 else 0.0
        print(f"    seg {i}: frames {s}-{e} ({e-s} frames), arm_activity={act:.4f}")

    pickups, places = _find_gripper_events(states)
    print(f"  Pickups at frames: {pickups}")
    print(f"  Places at frames: {places}")

    # Use fixture_refs from the demo episode — these are the actual fixture IDs
    # in the demo's kitchen layout. Since we'll create the env with the same
    # layout/style, these fixture IDs will exist in our env too.
    fridge_id = fixture_refs.get("fridge")
    cabinet_id = fixture_refs.get("cabinet")
    counter_id = fixture_refs.get("counter")
    dining_id = fixture_refs.get("dining_table")

    print(f"  Using fixture refs: counter={counter_id}, cabinet={cabinet_id}, "
          f"fridge={fridge_id}, dining={dining_id}")

    # Find where the fridge trip starts
    fridge_trip_start = _find_fridge_trip_start(states, segments, pickups)
    print(f"  Fridge trip starts at segment: {fridge_trip_start}")

    if fridge_trip_start is None:
        fridge_trip_start = len(segments) - 1

    # HotDogSetup demo visits fixtures in order:
    #   counter (bun start), cabinet (condiment), counter (back for bun),
    #   dining_table (place bun), [then fridge trip]
    pre_fridge_fixtures = [counter_id, cabinet_id, counter_id, dining_id]

    steps = [
        {
            "agent_id": "agent_0",
            "action": "communicate",
            "args": {"to": "agent_1", "message": "I'll handle bun and condiment. You get the sausage from the fridge."},
        },
        {
            "agent_id": "agent_1",
            "action": "communicate",
            "args": {"to": "agent_0", "message": "On it."},
        },
    ]

    # --- Agent 1: navigate to fridge and open it ---
    if fridge_id:
        steps.append({
            "agent_id": "agent_1",
            "action": "navigate",
            "args": {"fixture": fridge_id},
        })
        steps.append({
            "agent_id": "agent_1",
            "action": "interact",
            "args": {"fixture": fridge_id, "action": "open"},
        })

    # --- Agent 0: replay pre-fridge segments ---
    fix_idx = 0
    for seg_idx in range(fridge_trip_start):
        seg_start, seg_end = segments[seg_idx]

        arm = states[seg_start:seg_end, 5:12]
        arm_activity = float(np.max(np.abs(np.diff(arm, axis=0)))) if len(arm) > 1 else 0.0
        if arm_activity < 0.005:
            continue

        target_fix = pre_fridge_fixtures[fix_idx] if fix_idx < len(pre_fridge_fixtures) else dining_id
        fix_idx += 1
        if target_fix:
            steps.append({
                "agent_id": "agent_0",
                "action": "navigate",
                "args": {"fixture": target_fix},
            })

        steps.append({
            "agent_id": "agent_0",
            "action": "demo_clip",
            "args": {
                "task": "HotDogSetup",
                "episode": ep_idx,
                "start_frame": seg_start,
                "end_frame": seg_end,
            },
        })

    # --- Agent 1: replay fridge segments (sausage retrieval) ---
    fridge_seg_count = 0
    for seg_idx in range(fridge_trip_start, len(segments)):
        seg_start, seg_end = segments[seg_idx]

        arm = states[seg_start:seg_end, 5:12]
        arm_activity = float(np.max(np.abs(np.diff(arm, axis=0)))) if len(arm) > 1 else 0.0
        if arm_activity < 0.005:
            continue

        # Last segment navigates to dining table for placement
        is_last = (seg_idx == len(segments) - 1)
        if is_last and dining_id:
            steps.append({
                "agent_id": "agent_1",
                "action": "navigate",
                "args": {"fixture": dining_id},
            })

        steps.append({
            "agent_id": "agent_1",
            "action": "demo_clip",
            "args": {
                "task": "HotDogSetup",
                "episode": ep_idx,
                "start_frame": seg_start,
                "end_frame": seg_end,
            },
        })
        fridge_seg_count += 1

    steps.append({
        "agent_id": "agent_0",
        "action": "communicate",
        "args": {"to": "agent_1", "message": "Hot dog setup complete!"},
    })

    return {"task": "HotDogSetup", "steps": steps}, layout_id, style_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-robot HotDogSetup trajectories")
    parser.add_argument("--mode", choices=["teleport", "demo"], default="demo")
    parser.add_argument("--output-dir", type=str, default="experiments/trajectory_videos")
    parser.add_argument("--layout", type=int, default=None, help="Kitchen layout (default: task default)")
    parser.add_argument("--style", type=int, default=None, help="Kitchen style (default: task default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0, help="Demo episode index (demo mode only)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    from robocasa.utils.video_trajectory_runner import VideoTrajectoryRunner

    print(f"\n{'=' * 60}")
    print(f"HotDogSetup — mode={args.mode}")
    print(f"{'=' * 60}")

    try:
        layout = args.layout
        style = args.style

        if args.mode == "demo":
            # For demo mode, we must match the episode's layout/style so that
            # fixture positions align with the recorded arm motions.
            # Build trajectory first to extract layout/style from ep_meta.
            trajectory, ep_layout, ep_style = build_demo_split_trajectory(
                scene=None,  # scene not needed yet
                ep_idx=args.episode,
            )
            # Use episode layout/style unless explicitly overridden
            if layout is None:
                layout = ep_layout
            if style is None:
                style = ep_style
            print(f"  Using layout={layout}, style={style} (from episode metadata)")

        run_name = f"HotDogSetup_{args.mode}_seed{args.seed}"
        if args.mode == "demo":
            run_name += f"_ep{args.episode}"
        run_dir = Path(args.output_dir) / run_name

        runner = VideoTrajectoryRunner(
            task_name="HotDogSetup",
            robots=2,
            layout=layout,
            style=style,
            seed=args.seed,
            render_width=args.width,
            render_height=args.height,
            fps=args.fps,
            output_dir=str(run_dir),
        )

        scene = runner.get_scene_description()
        print(f"  Task lang: {scene.get('task', 'N/A')}")
        print(f"  Fixtures:  {len(scene['fixtures'])}")
        print(f"  Objects:   {len(scene['objects'])}")

        if args.mode == "teleport":
            trajectory = make_teleport_trajectory(scene)

        print(f"  Steps:     {len(trajectory['steps'])}")

        metadata = runner.run(trajectory)
        print(f"  Frames:    {metadata['total_frames']}")
        print(f"  Output:    {run_dir}/")

        for f in sorted(run_dir.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name} ({size_kb:.1f} KB)")

        runner.close()

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nDone. Output: {run_dir}/")


if __name__ == "__main__":
    main()
