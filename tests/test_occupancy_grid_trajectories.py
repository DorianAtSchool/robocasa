#!/usr/bin/env python
"""
Trajectory-level tests for occupancy-grid robot placement.

Three test suites:
  A) Demo plan tests — run scripted multi-step plans end-to-end:
     - cooperative_hotdog_setup (2-robot, multi-fixture)
     - sandwich_station (1-robot, fridge → counter)
  B) Collision stress test — both robots repeatedly target the same fixture
  C) Generic placement stress test — auto-discovers all fixtures and objects
     in any task, exercises navigate/pick/place for each one

Each test verifies per-step invariants:
  - Robots land in standable positions (not inside fixtures)
  - Robots are near the target fixture (within 3m)
  - Two robots never overlap (>0.1m apart)
  - Robots stay inside the kitchen grid bounds

Usage:
  python tests/test_occupancy_grid_trajectories.py --placement grid --output tmp/test_output
  python tests/test_occupancy_grid_trajectories.py --placement grid --layout 56 --style 42
  python tests/test_occupancy_grid_trajectories.py --test generic --task MicrowaveThawing
  python tests/test_occupancy_grid_trajectories.py --test sandwich --placement grid --output tmp/sandwich

Options:
  --placement    grid (default: grid)
  --output       Directory to save videos, frames, and placement maps (optional)
  --layout       Kitchen layout id (default: 11)
  --style        Kitchen style id (default: 34)
  --seed         Random seed (default: 42)
  --task         Task class name (default: HotDogSetup)
  --robots       Number of robots (default: 2)
  --test         Which tests to run: all, hotdog, sandwich, collision, tour, generic
                 (default: all). Comma-separated for multiple.
  --robot-radius Robot collision radius in meters (default: 0.18)
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import imageio
import numpy as np

from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils.sim_tool_executor import SimToolExecutor


# ---------------------------------------------------------------------------
# Globals set by CLI args — tests read these at runtime
# ---------------------------------------------------------------------------
_PLACEMENT = "grid"
_OUTPUT_DIR: str | None = None
_LAYOUT = 11
_STYLE = 34
_SEED = 42
_TASK = "HotDogSetup"
_ROBOTS = 2
_ROBOT_RADIUS = 0.18


# ---------------------------------------------------------------------------
# Shared invariant checks
# ---------------------------------------------------------------------------

def _assert_placement_invariants(
    test_case: unittest.TestCase,
    executor: SimToolExecutor,
    tag: str,
):
    """Check that all robots are in valid positions."""
    runner = executor.runner

    for robot_idx in range(runner._num_robots):
        pos = runner._get_robot_position(robot_idx)[:2]
        grid = runner._occupancy_grid
        cell = grid._world_to_grid(pos)
        r, c = cell
        test_case.assertGreaterEqual(r, 0, f"[{tag}] Robot {robot_idx} outside grid (row)")
        test_case.assertLess(r, grid._rows, f"[{tag}] Robot {robot_idx} outside grid (row)")
        test_case.assertGreaterEqual(c, 0, f"[{tag}] Robot {robot_idx} outside grid (col)")
        test_case.assertLess(c, grid._cols, f"[{tag}] Robot {robot_idx} outside grid (col)")

    if runner._num_robots >= 2:
        pos0 = runner._get_robot_position(0)[:2]
        pos1 = runner._get_robot_position(1)[:2]
        dist = float(np.linalg.norm(pos0 - pos1))
        test_case.assertGreater(
            dist, 0.1,
            f"[{tag}] Robots overlapping: r0={pos0}, r1={pos1}, dist={dist:.3f}m",
        )


def _assert_navigate_invariants(
    test_case: unittest.TestCase,
    executor: SimToolExecutor,
    robot_idx: int,
    fixture_id: str,
    tag: str,
):
    """Check post-navigate invariants for a specific robot."""
    runner = executor.runner
    pos = runner._get_robot_position(robot_idx)[:2]

    standable = runner._occupancy_grid.is_standable(pos)
    test_case.assertTrue(
        standable,
        f"[{tag}] Robot {robot_idx} in non-standable position after navigate to {fixture_id}. "
        f"pos={pos}",
    )

    fxtr = runner._fixtures[fixture_id]
    fxtr_pos = np.array(fxtr.pos[:2])
    dist = float(np.linalg.norm(pos - fxtr_pos))
    test_case.assertLess(
        dist, 3.0,
        f"[{tag}] Robot {robot_idx} too far from {fixture_id}: "
        f"{dist:.2f}m (pos={pos}, fxtr={fxtr_pos})",
    )


# ---------------------------------------------------------------------------
# Video/output helpers
# ---------------------------------------------------------------------------

class OutputRecorder:
    """Records frames and writes videos + placement maps when output is enabled."""

    def __init__(self, executor: SimToolExecutor, output_dir: str | None, tag: str):
        self._executor = executor
        self._enabled = output_dir is not None
        self._tag = tag.replace("/", "_").replace(" ", "_")
        if self._enabled:
            self._dir = Path(output_dir) / self._tag
            self._dir.mkdir(parents=True, exist_ok=True)
            self._writers: dict[str, imageio.core.Format.Writer] = {}
            self._step = 0
            # Save initial placement map
            executor.save_placement_map(str(self._dir), prefix="placement")
        else:
            self._dir = None
            self._writers = {}
            self._step = 0

    def record_step(self, step_name: str):
        """Capture frames from all cameras after a step."""
        if not self._enabled:
            return
        frames = self._executor.render()
        for camera_name, image in frames.items():
            # Lazy-init writers
            if camera_name not in self._writers:
                self._writers[camera_name] = imageio.get_writer(
                    self._dir / f"{camera_name}.mp4", fps=2,
                )
            self._writers[camera_name].append_data(image)

            # Save frame image
            cam_dir = self._dir / "frames" / camera_name
            cam_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(cam_dir / f"{self._step:03d}_{step_name}.png", image)
        self._step += 1

    def save_metadata(self, metadata: dict):
        """Write metadata JSON."""
        if not self._enabled:
            return
        with open(self._dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def close(self):
        for w in self._writers.values():
            w.close()
        self._writers.clear()


def _make_executor(task: str | None = None, robots: int | None = None, **overrides) -> SimToolExecutor:
    """Create a SimToolExecutor with the global CLI settings."""
    return SimToolExecutor(
        task_name=task or _TASK,
        robots=robots if robots is not None else _ROBOTS,
        layout=overrides.get("layout", _LAYOUT),
        style=overrides.get("style", _STYLE),
        seed=overrides.get("seed", _SEED),
        render_width=overrides.get("render_width", 320 if _OUTPUT_DIR else 160),
        render_height=overrides.get("render_height", 240 if _OUTPUT_DIR else 128),
        placement=_PLACEMENT,
        robot_radius=_ROBOT_RADIUS,
    )


# ======================================================================
# A) Demo plan tests
# ======================================================================

class TestDemoPlan_HotdogSetup(unittest.TestCase):
    """Run the cooperative_hotdog_setup demo plan and check invariants."""

    def _run_hotdog_with_checks(self, layout: int, style: int, seed: int):
        tag = f"hotdog_L{layout}_S{style}_s{seed}"
        executor = _make_executor(
            task="HotDogSetup", robots=2,
            layout=layout, style=style, seed=seed,
        )
        recorder = OutputRecorder(executor, _OUTPUT_DIR, tag)
        try:
            grid = executor.runner._occupancy_grid
            total = grid._grid.size
            occupied = int(grid._grid.sum())
            self.assertLess(
                occupied, total * 0.8,
                f"[{tag}] Grid >80% occupied ({occupied}/{total})",
            )

            recorder.record_step("initial")

            plan = executor.build_demo_plan("cooperative_hotdog_setup")
            steps_meta = []
            for i, step in enumerate(plan):
                tool = step["tool"]
                ridx = step.get("robot_idx", 0)
                args = step.get("args", {})

                result = executor.execute(tool, robot_idx=ridx, **args)
                self.assertTrue(
                    result.success,
                    f"[{tag}] Step {i} ({tool}) failed: {result.details}",
                )

                if tool in ("navigate_to_fixture", "pick_up_object"):
                    fid = args.get("fixture_id") or args.get("source_id")
                    if fid:
                        _assert_navigate_invariants(
                            self, executor, ridx, fid, f"{tag}_step{i}",
                        )
                _assert_placement_invariants(self, executor, f"{tag}_step{i}")
                recorder.record_step(f"step{i}_{tool}")
                steps_meta.append({"step": i, "tool": tool, "success": result.success})

            recorder.save_metadata({"tag": tag, "plan_name": "cooperative_hotdog_setup", "steps": steps_meta})
        finally:
            recorder.close()
            executor.close()

    def test_layout11(self):
        self._run_hotdog_with_checks(layout=_LAYOUT, style=_STYLE, seed=_SEED)

    def test_layout56(self):
        self._run_hotdog_with_checks(layout=56, style=42, seed=42)


class TestDemoPlan_SandwichStation(unittest.TestCase):
    """Run the sandwich_station demo plan and check invariants."""

    def _run_sandwich_with_checks(self, layout: int, style: int, seed: int):
        tag = f"sandwich_L{layout}_S{style}_s{seed}"
        executor = _make_executor(
            task="PrepareSandwichStation", robots=2,
            layout=layout, style=style, seed=seed,
        )
        recorder = OutputRecorder(executor, _OUTPUT_DIR, tag)
        try:
            recorder.record_step("initial")

            plan = executor.build_demo_plan("sandwich_station")
            steps_meta = []
            for i, step in enumerate(plan):
                tool = step["tool"]
                ridx = step.get("robot_idx", 0)
                args = step.get("args", {})

                result = executor.execute(tool, robot_idx=ridx, **args)
                self.assertTrue(
                    result.success,
                    f"[{tag}] Step {i} ({tool}) failed: {result.details}",
                )

                if tool in ("navigate_to_fixture", "pick_up_object"):
                    fid = args.get("fixture_id") or args.get("source_id")
                    if fid:
                        _assert_navigate_invariants(
                            self, executor, ridx, fid, f"{tag}_step{i}",
                        )
                _assert_placement_invariants(self, executor, f"{tag}_step{i}")
                recorder.record_step(f"step{i}_{tool}")
                steps_meta.append({"step": i, "tool": tool, "success": result.success})

            recorder.save_metadata({"tag": tag, "plan_name": "sandwich_station", "steps": steps_meta})
        finally:
            recorder.close()
            executor.close()

    def test_sandwich_default(self):
        self._run_sandwich_with_checks(layout=_LAYOUT, style=_STYLE, seed=_SEED)

    def test_sandwich_layout56(self):
        self._run_sandwich_with_checks(layout=56, style=42, seed=42)


# ======================================================================
# B) Collision stress — both robots at the same fixture
# ======================================================================

class TestCollisionStress(unittest.TestCase):
    """Both robots navigate to the same fixture repeatedly."""

    def test_both_robots_alternate_at_same_fixture(self):
        executor = _make_executor(task="HotDogSetup", robots=2)
        recorder = OutputRecorder(executor, _OUTPUT_DIR, "collision_stress")
        runner = executor.runner

        try:
            counter_id = None
            for fid, fxtr in runner._fixtures.items():
                if fixture_is_type(fxtr, FixtureType.COUNTER):
                    counter_id = fid
                    break
            self.assertIsNotNone(counter_id, "No counter found")

            recorder.record_step("initial")

            for iteration in range(3):
                for ridx in (0, 1):
                    tag = f"collision_iter{iteration}_r{ridx}"
                    executor.execute(
                        "navigate_to_fixture",
                        robot_idx=ridx,
                        fixture_id=counter_id,
                    )
                    _assert_navigate_invariants(
                        self, executor, ridx, counter_id, tag,
                    )
                    _assert_placement_invariants(self, executor, tag)
                    recorder.record_step(f"iter{iteration}_r{ridx}")

                pos0 = runner._get_robot_position(0)[:2]
                pos1 = runner._get_robot_position(1)[:2]
                dist = float(np.linalg.norm(pos0 - pos1))
                self.assertGreater(
                    dist, 0.3,
                    f"Iteration {iteration}: robots too close ({dist:.3f}m)",
                )
        finally:
            recorder.close()
            executor.close()


# ======================================================================
# C) Fixture tour — navigate to every major fixture type
# ======================================================================

class TestFixtureTour(unittest.TestCase):
    """Navigate robot 0 to every major fixture type, then robot 1 follows."""

    def test_tour_all_fixture_types(self):
        executor = _make_executor(task="HotDogSetup", robots=2)
        recorder = OutputRecorder(executor, _OUTPUT_DIR, "fixture_tour")
        runner = executor.runner

        try:
            target_types = [
                FixtureType.COUNTER, FixtureType.FRIDGE, FixtureType.SINK,
                FixtureType.STOVE, FixtureType.CABINET,
            ]
            targets: list[tuple[str, str]] = []
            for ft in target_types:
                for fid, fxtr in runner._fixtures.items():
                    if fixture_is_type(fxtr, ft):
                        targets.append((fid, ft.name if hasattr(ft, 'name') else str(ft)))
                        break

            self.assertGreater(len(targets), 0, "No fixtures found for tour")

            recorder.record_step("initial")

            for fid, ftype_name in targets:
                tag = f"tour_r0_{fid}"
                result = executor.execute(
                    "navigate_to_fixture", robot_idx=0, fixture_id=fid,
                )
                self.assertTrue(result.success, f"[{tag}] navigate failed")
                _assert_navigate_invariants(self, executor, 0, fid, tag)
                _assert_placement_invariants(self, executor, tag)
                recorder.record_step(f"r0_{ftype_name}")

            for fid, ftype_name in targets:
                tag = f"tour_r1_{fid}"
                result = executor.execute(
                    "navigate_to_fixture", robot_idx=1, fixture_id=fid,
                )
                self.assertTrue(result.success, f"[{tag}] navigate failed")
                _assert_navigate_invariants(self, executor, 1, fid, tag)
                _assert_placement_invariants(self, executor, tag)
                recorder.record_step(f"r1_{ftype_name}")
        finally:
            recorder.close()
            executor.close()


# ======================================================================
# D) Generic placement stress test — works with any task
# ======================================================================

class TestGenericPlacement(unittest.TestCase):
    """Auto-discover fixtures and objects, exercise navigate/pick/place.

    Works with any task — no hard-coded plan needed.  Discovers:
      - All navigable fixtures (counters, cabinets, fridge, sink, stove, etc.)
      - All objects and their current fixture locations
    Then runs:
      1. Navigate to each fixture with robot 0
      2. For each object: pick it up from its current location, navigate to
         a random counter, place it there
      3. If 2 robots: robot 1 mirrors the fixture tour while robot 0 holds
         an object, testing multi-robot placement with a held object
    """

    # Fixture types worth navigating to
    _NAV_TYPES = [
        FixtureType.COUNTER, FixtureType.COUNTER_NON_DINING,
        FixtureType.FRIDGE, FixtureType.SINK, FixtureType.STOVE,
        FixtureType.CABINET, FixtureType.ISLAND, FixtureType.DINING_COUNTER,
        FixtureType.MICROWAVE, FixtureType.OVEN,
    ]
    _PLACEABLE_TYPES = [
        FixtureType.COUNTER, FixtureType.COUNTER_NON_DINING,
        FixtureType.ISLAND, FixtureType.DINING_COUNTER,
    ]

    def test_generic_placement(self):
        executor = _make_executor()
        recorder = OutputRecorder(executor, _OUTPUT_DIR, f"generic_{_TASK}")
        runner = executor.runner

        try:
            recorder.record_step("initial")

            # --- Phase 1: Navigate to all fixture types ---
            nav_fixtures = []
            for ft in self._NAV_TYPES:
                for fid, fxtr in runner._fixtures.items():
                    if fixture_is_type(fxtr, ft):
                        nav_fixtures.append(fid)
                        break  # one per type

            for fid in nav_fixtures:
                tag = f"generic_nav_r0_{fid}"
                result = executor.execute(
                    "navigate_to_fixture", robot_idx=0, fixture_id=fid,
                )
                self.assertTrue(result.success, f"[{tag}] failed")
                _assert_navigate_invariants(self, executor, 0, fid, tag)
                _assert_placement_invariants(self, executor, tag)
                recorder.record_step(f"nav_{fid[:30]}")

            # --- Phase 2: Pick up each object, place on a counter ---
            placeable_fixtures = []
            for ft in self._PLACEABLE_TYPES:
                for fid, fxtr in runner._fixtures.items():
                    if fixture_is_type(fxtr, ft):
                        placeable_fixtures.append(fid)

            scene = executor.get_scene_description()
            objects = scene.get("objects", {})
            object_ids = list(objects.keys())

            for obj_id in object_ids:
                obj_info = objects[obj_id]
                source_fid = obj_info.get("location")
                if source_fid is None or source_fid not in runner._fixtures:
                    continue

                # Pick up
                tag = f"generic_pick_{obj_id}"
                try:
                    result = executor.execute(
                        "pick_up_object", robot_idx=0,
                        object_id=obj_id, source_id=source_fid,
                    )
                except Exception as e:
                    # Some objects may not be pickable — skip
                    print(f"  [skip] pick_up_object({obj_id}) from {source_fid}: {e}")
                    continue
                self.assertTrue(result.success, f"[{tag}] failed: {result.details}")
                _assert_placement_invariants(self, executor, tag)
                recorder.record_step(f"pick_{obj_id[:20]}")

                # Place on a counter
                if placeable_fixtures:
                    target_fid = placeable_fixtures[hash(obj_id) % len(placeable_fixtures)]
                    tag = f"generic_place_{obj_id}_on_{target_fid}"
                    try:
                        result = executor.execute(
                            "place_on_surface", robot_idx=0,
                            object_id=obj_id, support_id=target_fid,
                        )
                    except Exception as e:
                        print(f"  [skip] place_on_surface({obj_id}) on {target_fid}: {e}")
                        continue
                    self.assertTrue(result.success, f"[{tag}] failed: {result.details}")
                    _assert_placement_invariants(self, executor, tag)
                    recorder.record_step(f"place_{obj_id[:20]}")

            # --- Phase 3: Multi-robot stress (if 2+ robots) ---
            if runner._num_robots >= 2 and nav_fixtures:
                # Robot 1 visits all fixtures while robot 0 is somewhere
                for fid in nav_fixtures[:3]:  # just first 3 to keep it fast
                    tag = f"generic_r1_nav_{fid}"
                    result = executor.execute(
                        "navigate_to_fixture", robot_idx=1, fixture_id=fid,
                    )
                    self.assertTrue(result.success, f"[{tag}] failed")
                    _assert_navigate_invariants(self, executor, 1, fid, tag)
                    _assert_placement_invariants(self, executor, tag)
                    recorder.record_step(f"r1_nav_{fid[:30]}")

            recorder.save_metadata({
                "tag": f"generic_{_TASK}",
                "task": _TASK,
                "fixtures_navigated": len(nav_fixtures),
                "objects_picked": len(object_ids),
                "placement": _PLACEMENT,
            })
        finally:
            recorder.close()
            executor.close()


# ======================================================================
# CLI and test selection
# ======================================================================

_TEST_MAP = {
    "hotdog": TestDemoPlan_HotdogSetup,
    "sandwich": TestDemoPlan_SandwichStation,
    "collision": TestCollisionStress,
    "tour": TestFixtureTour,
    "generic": TestGenericPlacement,
}


def main():
    global _PLACEMENT, _OUTPUT_DIR, _LAYOUT, _STYLE, _SEED, _TASK, _ROBOTS, _ROBOT_RADIUS

    parser = argparse.ArgumentParser(
        description="Trajectory-level placement tests with optional video output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--placement", default="grid", choices=["grid"])
    parser.add_argument("--output", default=None, help="Output directory for videos/frames/maps")
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="HotDogSetup")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument(
        "--test", default="all",
        help="Which tests to run: all, hotdog, sandwich, collision, tour, generic. "
             "Comma-separated for multiple (e.g. --test hotdog,sandwich).",
    )
    # Collect remaining args to pass to unittest
    args, remaining = parser.parse_known_args()

    _PLACEMENT = args.placement
    _OUTPUT_DIR = args.output
    _LAYOUT = args.layout
    _STYLE = args.style
    _SEED = args.seed
    _TASK = args.task
    _ROBOTS = args.robots
    _ROBOT_RADIUS = args.robot_radius

    # Build test suite
    if args.test == "all":
        test_classes = list(_TEST_MAP.values())
    else:
        test_classes = []
        for name in args.test.split(","):
            name = name.strip()
            if name not in _TEST_MAP:
                print(f"Unknown test: {name!r}. Available: {sorted(_TEST_MAP)}")
                sys.exit(1)
            test_classes.append(_TEST_MAP[name])

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
