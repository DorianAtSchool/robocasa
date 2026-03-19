import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.scripts.generate_llm_task_descriptions import (
    build_compact_task_context,
    render_llm_prompt,
)
import robocasa.utils.object_utils as OU
from robocasa.utils.placement import (  # noqa: E402
    MAX_FRONT_WORKING_LATERAL_OFFSET,
    get_face_center,
    get_face_order,
    get_front_alignment_metrics,
    get_fixture_aabb,
)
from robocasa.utils.sim_tool_executor import SimToolExecutor  # noqa: E402
from robocasa.utils.sim_tool_executor import _is_approach_center  # noqa: E402
from robocasa.utils.sim_tool_specs import SIM_TOOL_SPEC_BY_NAME  # noqa: E402


def _objects_intersect(executor: SimToolExecutor, object_a: str, object_b: str) -> bool:
    obj_a = executor._require_object(object_a)
    obj_b = executor._require_object(object_b)
    pos_a, quat_a_wxyz = executor._get_object_pose(object_a)
    pos_b, quat_b_wxyz = executor._get_object_pose(object_b)
    return OU.objs_intersect(
        obj_a,
        pos_a,
        T.convert_quat(quat_a_wxyz, to="xyzw"),
        obj_b,
        pos_b,
        T.convert_quat(quat_b_wxyz, to="xyzw"),
    )


class TestSimToolExecutor(unittest.TestCase):
    def test_every_tool_spec_has_executor_method(self):
        executor = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            for tool_name in SIM_TOOL_SPEC_BY_NAME:
                self.assertTrue(callable(getattr(executor, tool_name, None)), tool_name)
        finally:
            executor.close()

    def test_compact_context_and_prompt_generation(self):
        context = build_compact_task_context(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            width=160,
            height=128,
        )
        self.assertEqual(context["task_name"], "HotDogSetup")
        self.assertIn("instruction", context)
        self.assertIn("objects", context)
        self.assertIn("fixtures", context)
        self.assertTrue(any(obj["object_id"] == "plate" for obj in context["objects"]))
        self.assertTrue(any(fx["fixture_type"] == "fridge" for fx in context["fixtures"]))

        prompt_text = render_llm_prompt(context)
        self.assertIn("Task:", prompt_text)
        self.assertIn("Tools:", prompt_text)
        self.assertIn("hotdog_bun", prompt_text)
        self.assertIn("dining_dining_group", prompt_text)
        self.assertNotIn("O1=", prompt_text)
        self.assertNotIn("F1=", prompt_text)

    def test_hotdog_demo_plan_renders_and_places_on_plate(self):
        executor = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            plan = executor.build_demo_plan("cooperative_hotdog_setup")
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                metadata = executor.run_tool_plan(plan, output_dir=output_dir, fps=1)

                self.assertIn("top_view", metadata["cameras"])
                self.assertIn("room_view", metadata["cameras"])
                self.assertIn("robot0_eye_in_hand", metadata["cameras"])
                self.assertIn("robot1_eye_in_hand", metadata["cameras"])
                self.assertTrue((output_dir / "top_view.mp4").exists())
                self.assertTrue((output_dir / "room_view.mp4").exists())
                self.assertTrue((output_dir / "metadata.json").exists())

                robot_indices = {step["robot_idx"] for step in metadata["steps"]}
                tools = {step["tool"] for step in metadata["steps"]}
                self.assertEqual(robot_indices, {0, 1})
                self.assertIn("communicate", tools)
                self.assertIn("wait", tools)

                with open(output_dir / "metadata.json", "r") as f:
                    saved_metadata = json.load(f)
                self.assertEqual(saved_metadata["cameras"], metadata["cameras"])

                plate = executor._require_object("plate")
                plate_body_id = executor.env.obj_body_id["plate"]
                plate_points = plate.get_bbox_points(
                    trans=executor.env.sim.data.body_xpos[plate_body_id].copy(),
                    rot=T.convert_quat(
                        executor.env.sim.data.body_xquat[plate_body_id].copy(),
                        to="xyzw",
                    ),
                )
                plate_top_z = max(point[2] for point in plate_points)

                for object_id in ("hotdog_bun", "sausage"):
                    body_id = executor.env.obj_body_id[object_id]
                    object_z = float(executor.env.sim.data.body_xpos[body_id][2])
                    self.assertLess(abs(object_z - plate_top_z), 0.08, object_id)
        finally:
            executor.close()

    def test_hotdog_demo_plan_builds_on_alt_layout(self):
        executor = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            plan = executor.build_demo_plan("cooperative_hotdog_setup")
            fixture_ids = [
                step["args"]["fixture_id"]
                for step in plan
                if step["tool"] == "navigate_to_fixture" and "fixture_id" in step["args"]
            ]
            place_steps = [step for step in plan if step["tool"] == "place_on_object"]
            self.assertIn("island_island_group_1", fixture_ids)
            self.assertTrue(any("fridge" in fixture_id for fixture_id in fixture_ids))
            self.assertTrue(all("anchor_fixture_id" in step["args"] for step in place_steps))
        finally:
            executor.close()

    def test_hotdog_template_grounds_differently_across_layouts(self):
        executor_a = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        executor_b = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            template_a = executor_a.build_demo_plan_template("cooperative_hotdog_setup")
            template_b = executor_b.build_demo_plan_template("cooperative_hotdog_setup")
            self.assertEqual(template_a, template_b)

            grounded_a = executor_a.ground_plan_template(template_a)
            grounded_b = executor_b.ground_plan_template(template_b)

            fixture_ids_a = {
                step["args"]["fixture_id"]
                for step in grounded_a
                if step["tool"] == "navigate_to_fixture" and "fixture_id" in step["args"]
            }
            fixture_ids_b = {
                step["args"]["fixture_id"]
                for step in grounded_b
                if step["tool"] == "navigate_to_fixture" and "fixture_id" in step["args"]
            }

            self.assertIn("dining_dining_group", fixture_ids_a)
            self.assertIn("island_island_group_1", fixture_ids_b)
            self.assertNotEqual(fixture_ids_a, fixture_ids_b)

            place_steps_a = [step for step in grounded_a if step["tool"] == "place_on_object"]
            place_steps_b = [step for step in grounded_b if step["tool"] == "place_on_object"]
            self.assertTrue(
                all(step["args"]["anchor_fixture_id"] == "dining_dining_group" for step in place_steps_a)
            )
            self.assertTrue(
                all(step["args"]["anchor_fixture_id"] == "island_island_group_1" for step in place_steps_b)
            )
        finally:
            executor_a.close()
            executor_b.close()

    def test_object_aware_navigation_moves_robot_closer_to_plate(self):
        executor_fixture = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
        )
        executor_object = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            plate_body_id = executor_fixture.env.obj_body_id["plate"]
            plate_xy_fixture = executor_fixture.env.sim.data.body_xpos[plate_body_id][:2].copy()
            plate_xy_object = executor_object.env.sim.data.body_xpos[
                executor_object.env.obj_body_id["plate"]
            ][:2].copy()

            executor_fixture.runner._move_robot_near_fixture(0, "island_island_group_1")
            robot_xy_fixture = executor_fixture.runner._get_robot_position(0)[:2]
            dist_fixture = float(((robot_xy_fixture - plate_xy_fixture) ** 2).sum() ** 0.5)

            executor_object.runner._move_robot_near_fixture(
                0,
                "island_island_group_1",
                ref_object_id="plate",
            )
            robot_xy_object = executor_object.runner._get_robot_position(0)[:2]
            dist_object = float(((robot_xy_object - plate_xy_object) ** 2).sum() ** 0.5)

            self.assertLessEqual(dist_object, dist_fixture)
        finally:
            executor_fixture.close()
            executor_object.close()


class TestToolsFunctional(unittest.TestCase):
    """Functional tests that call individual tools against a live simulation."""

    _executor = None

    @classmethod
    def setUpClass(cls):
        cls._executor = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        cls._scene = cls._executor.get_scene_description()

    @classmethod
    def tearDownClass(cls):
        if cls._executor is not None:
            cls._executor.close()

    # -- helpers --

    def _object_pos(self, object_id):
        pos, _ = self._executor._get_object_pose(object_id)
        return pos.copy()

    def _fixture_pos(self, fixture_id):
        return np.array(
            self._scene["fixtures"][fixture_id]["position"], dtype=float
        )

    def _robot_pos(self, robot_idx=0):
        return self._executor.runner._get_robot_position(robot_idx)

    # -- navigate_to_fixture --

    def test_navigate_moves_robot_near_fixture(self):
        # Pick any fixture from the scene
        fixture_id = next(iter(self._scene["fixtures"]))
        result = self._executor.navigate_to_fixture(fixture_id, robot_idx=0)
        self.assertTrue(result.success)
        robot_xy = self._robot_pos(0)[:2]
        fixture_xy = self._fixture_pos(fixture_id)[:2]
        dist = float(np.linalg.norm(robot_xy - fixture_xy))
        self.assertLess(dist, 2.0, "Robot should be within 2m of fixture")

    # -- pick_up_object --

    def test_pick_up_marks_object_as_held(self):
        scene = self._executor.get_scene_description()
        obj_id = "hotdog_bun"
        source = scene["objects"][obj_id]["location"]
        result = self._executor.pick_up_object(obj_id, source, robot_idx=0)
        self.assertTrue(result.success)
        self.assertEqual(self._executor._held_objects.get(0), obj_id)

    # -- place_on_surface --

    def test_place_on_surface_moves_object(self):
        obj_id = "hotdog_bun"
        # Ensure it's picked up first
        scene = self._executor.get_scene_description()
        source = scene["objects"][obj_id]["location"]
        self._executor.pick_up_object(obj_id, source, robot_idx=0)

        # Find a counter to place on
        target_fixture = None
        for fid, info in self._scene["fixtures"].items():
            if info.get("can_place_objects") and "counter" in info.get("fixture_type", ""):
                target_fixture = fid
                break
        self.assertIsNotNone(target_fixture)

        result = self._executor.place_on_surface(obj_id, target_fixture, robot_idx=0)
        self.assertTrue(result.success)
        self.assertNotIn(0, self._executor._held_objects)

    # -- place_on_object --

    def test_place_on_object_stacks(self):
        # Pick up sausage, place on plate
        scene = self._executor.get_scene_description()
        source = scene["objects"]["sausage"]["location"]
        self._executor.pick_up_object("sausage", source, robot_idx=1)

        plate_pos_before = self._object_pos("plate")
        result = self._executor.place_on_object("sausage", "plate", robot_idx=1)
        self.assertTrue(result.success)

        sausage_z = self._object_pos("sausage")[2]
        plate_z = self._object_pos("plate")[2]
        self.assertGreater(sausage_z, plate_z - 0.01,
                           "Sausage should be at or above plate level")

    # -- place_next_to --

    def test_place_next_to_puts_object_nearby(self):
        scene = self._executor.get_scene_description()
        source = scene["objects"]["hotdog_bun"]["location"]
        self._executor.pick_up_object("hotdog_bun", source, robot_idx=0)

        result = self._executor.place_next_to("hotdog_bun", "plate", robot_idx=0)
        self.assertTrue(result.success)

        bun_xy = self._object_pos("hotdog_bun")[:2]
        plate_xy = self._object_pos("plate")[:2]
        dist = float(np.linalg.norm(bun_xy - plate_xy))
        self.assertLess(dist, 0.5, "Objects should be close together")
        self.assertGreater(dist, 0.01, "Objects should not overlap exactly")

    # -- place_under (generic / non-dispenser) --

    def test_place_under_generic_aligns_xy(self):
        """place_under a non-dispenser fixture should align XY under it."""
        scene = self._executor.get_scene_description()
        # Find any non-placeable, non-dispenser fixture (cabinet, hood, etc.)
        ref_fixture = None
        for fid, info in scene["fixtures"].items():
            ftype = info.get("fixture_type", "")
            if not info.get("can_place_objects", False) and ftype not in (
                "coffee_machine", "sink", ""
            ):
                ref_fixture = fid
                break

        if ref_fixture is None:
            self.skipTest("No non-placeable non-dispenser fixture in this layout")

        obj_id = "hotdog_bun"
        source = scene["objects"][obj_id]["location"]
        self._executor.pick_up_object(obj_id, source, robot_idx=0)

        result = self._executor.place_under(obj_id, ref_fixture, robot_idx=0)
        self.assertTrue(result.success)

        obj_xy = self._object_pos(obj_id)[:2]
        fixture_xy = self._fixture_pos(ref_fixture)[:2]
        xy_dist = float(np.linalg.norm(obj_xy - fixture_xy))
        self.assertLess(xy_dist, 0.05, "Object XY should align under fixture")

    # -- execute dispatch --

    def test_execute_dispatches_to_correct_method(self):
        result = self._executor.execute("wait", robot_idx=0)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "wait")

    def test_execute_rejects_unknown_tool(self):
        with self.assertRaises(ValueError):
            self._executor.execute("nonexistent_tool")

    # -- communicate / wait --

    def test_communicate_returns_success(self):
        result = self._executor.communicate(to="agent_1", message="hello")
        self.assertTrue(result.success)
        self.assertEqual(result.details["message"], "hello")

    def test_wait_returns_success(self):
        result = self._executor.wait(robot_idx=0)
        self.assertTrue(result.success)


class TestPlaceUnderDispenser(unittest.TestCase):
    """Test place_under with dispenser fixtures (CoffeeMachine, Sink)."""

    def test_place_under_coffee_machine(self):
        executor = SimToolExecutor(
            task_name="CoffeeSetupMug",
            robots=1,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            scene = executor.get_scene_description()

            # Find the coffee machine fixture
            coffee_fixture = None
            for fid, info in scene["fixtures"].items():
                if "coffee" in info.get("fixture_type", "").lower():
                    coffee_fixture = fid
                    break
            self.assertIsNotNone(coffee_fixture, "No coffee machine in scene")

            # Find the mug object
            mug_id = None
            for oid in scene["objects"]:
                if "mug" in oid or oid == "obj":
                    mug_id = oid
                    break
            self.assertIsNotNone(mug_id, "No mug object in scene")

            # Pick up and place under coffee machine
            source = scene["objects"][mug_id]["location"]
            executor.pick_up_object(mug_id, source, robot_idx=0)
            result = executor.place_under(mug_id, coffee_fixture, robot_idx=0)
            self.assertTrue(result.success)

            # Verify position is at the receptacle_place_site
            from robocasa.models.fixtures.coffee_machine import CoffeeMachine
            fixture_obj = executor.runner._fixtures[coffee_fixture]
            self.assertIsInstance(fixture_obj, CoffeeMachine)

            site_name = f"{fixture_obj.naming_prefix}receptacle_place_site"
            site_id = executor.env.sim.model.site_name2id(site_name)
            expected_pos = executor.env.sim.data.site_xpos[site_id].copy()

            actual_pos, _ = executor._get_object_pose(mug_id)
            dist = float(np.linalg.norm(actual_pos - expected_pos))
            self.assertLess(dist, 0.01, "Mug should be at dispenser site")
        finally:
            executor.close()

    def test_place_under_sink(self):
        executor = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )
        try:
            scene = executor.get_scene_description()

            # Find the sink fixture
            sink_fixture = None
            for fid, info in scene["fixtures"].items():
                if "sink" in info.get("fixture_type", "").lower():
                    sink_fixture = fid
                    break

            if sink_fixture is None:
                self.skipTest("No sink fixture in this layout")

            from robocasa.models.fixtures.sink import Sink
            fixture_obj = executor.runner._fixtures[sink_fixture]
            if not isinstance(fixture_obj, Sink):
                self.skipTest("Sink fixture is not a Sink instance")

            # Pick up an object and place under sink
            obj_id = "hotdog_bun"
            source = scene["objects"][obj_id]["location"]
            executor.pick_up_object(obj_id, source, robot_idx=0)
            result = executor.place_under(obj_id, sink_fixture, robot_idx=0)
            self.assertTrue(result.success)

            # Verify position is at the water site
            water_site_name = fixture_obj.water_site.get("name")
            site_id = executor.env.sim.model.site_name2id(water_site_name)
            expected_pos = executor.env.sim.data.site_xpos[site_id].copy()

            actual_pos, _ = executor._get_object_pose(obj_id)
            dist = float(np.linalg.norm(actual_pos - expected_pos))
            self.assertLess(dist, 0.01, "Object should be at water site")
        finally:
            executor.close()


class TestEnclosingFixtureFrontAlignment(unittest.TestCase):
    """Front-alignment tests for enclosing fixtures."""

    def _make_executor(self):
        return SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )

    def _assert_front_aligned(self, executor, fixture_id: str, robot_idx: int):
        fixture = executor.runner._fixtures[fixture_id]
        pos = executor.runner._get_robot_position(robot_idx)[:2]
        target_xy = executor.runner._get_fixture_front_target_xy(fixture_id)
        metrics = get_front_alignment_metrics(fixture, pos, target_xy=target_xy)
        self.assertIsNotNone(metrics)

        self.assertTrue(
            bool(metrics["on_front_face"] and metrics["within_span"]),
            f"Robot not on front working face: pos={pos}",
        )
        self.assertLessEqual(
            float(metrics["lateral_offset"]),
            MAX_FRONT_WORKING_LATERAL_OFFSET,
            f"Robot too far from front working line: {pos}",
        )

    def test_pick_up_object_from_fridge_approaches_front_center(self):
        executor = self._make_executor()
        try:
            scene = executor.get_scene_description()
            source_id = scene["objects"]["sausage"]["location"]
            fixture = executor.runner._fixtures[source_id]
            self.assertTrue(_is_approach_center(fixture))

            result = executor.pick_up_object("sausage", source_id, robot_idx=1)

            self.assertTrue(result.success)
            self._assert_front_aligned(executor, source_id, robot_idx=1)
        finally:
            executor.close()

    def test_object_anchor_helper_uses_front_center_for_enclosing_fixture(self):
        executor = self._make_executor()
        try:
            fixture_id = executor._move_robot_near_object_anchor(
                0,
                "sausage",
                preferred_fixture_types=["fridge"],
            )
            fixture = executor.runner._fixtures[fixture_id]
            self.assertTrue(_is_approach_center(fixture))
            self._assert_front_aligned(executor, fixture_id, robot_idx=0)
        finally:
            executor.close()

    def test_pick_up_object_recenters_off_center_fridge_pose(self):
        executor = self._make_executor()
        try:
            scene = executor.get_scene_description()
            source_id = scene["objects"]["sausage"]["location"]
            fixture = executor.runner._fixtures[source_id]

            executor.runner._move_robot_near_fixture(1, source_id, require_front=True)
            centered_pos = executor.runner._get_robot_position(1)[:2].copy()
            aabb = get_fixture_aabb(fixture)
            self.assertIsNotNone(aabb)
            fmin, fmax = aabb
            front_face = get_face_order(fixture)[0]
            front_target = executor.runner._get_fixture_front_target_xy(source_id)
            if front_target is None:
                front_target = get_face_center(front_face, fmin, fmax)

            off_center_pos = centered_pos.copy()
            target_offset = MAX_FRONT_WORKING_LATERAL_OFFSET + 0.08
            if front_face in {"neg_y", "pos_y"}:
                off_center_pos[0] = min(front_target[0] + target_offset, fmax[0] - 0.02)
                if abs(off_center_pos[0] - front_target[0]) <= MAX_FRONT_WORKING_LATERAL_OFFSET:
                    self.skipTest("Fixture front span too narrow for off-center recenter test")
            else:
                off_center_pos[1] = min(front_target[1] + target_offset, fmax[1] - 0.02)
                if abs(off_center_pos[1] - front_target[1]) <= MAX_FRONT_WORKING_LATERAL_OFFSET:
                    self.skipTest("Fixture front span too narrow for off-center recenter test")

            executor.runner._set_robot_pose(1, off_center_pos, 0.0)
            self.assertFalse(executor._robot_near_fixture(1, source_id))

            result = executor.pick_up_object("sausage", source_id, robot_idx=1)

            self.assertTrue(result.success)
            self._assert_front_aligned(executor, source_id, robot_idx=1)
        finally:
            executor.close()


class TestCollisionAwareObjectPlacement(unittest.TestCase):
    def _make_executor(self):
        return SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )

    def test_place_on_surface_avoids_existing_object_on_support(self):
        executor = self._make_executor()
        try:
            scene = executor.get_scene_description()
            support_id = scene["objects"]["plate"]["location"]
            executor.runner.move_object("plate", support_id)

            bun_source = scene["objects"]["hotdog_bun"]["location"]
            executor.pick_up_object("hotdog_bun", bun_source, robot_idx=0)
            result = executor.place_on_surface("hotdog_bun", support_id, robot_idx=0)

            self.assertTrue(result.success)
            self.assertEqual(executor._get_scene_object_location("hotdog_bun"), support_id)
            self.assertFalse(
                _objects_intersect(executor, "hotdog_bun", "plate"),
                "Collision-aware surface placement should avoid the existing plate",
            )
        finally:
            executor.close()

    def test_place_under_generic_avoids_blocker(self):
        executor = self._make_executor()
        try:
            scene = executor.get_scene_description()
            reference_fixture_id = None
            for fixture_id, info in scene["fixtures"].items():
                fixture_type = info.get("fixture_type", "")
                if not info.get("can_place_objects", False) and fixture_type not in (
                    "coffee_machine",
                    "sink",
                    "",
                ):
                    reference_fixture_id = fixture_id
                    break

            if reference_fixture_id is None:
                self.skipTest("No generic reference fixture for place_under test")

            support_fixture_id = executor._find_placeable_surface_near_fixture(reference_fixture_id)
            reference_xy = np.asarray(
                scene["fixtures"][reference_fixture_id]["position"][:2],
                dtype=float,
            )
            blocker_target = executor.runner._compute_object_target_pos(
                executor.runner._fixtures[support_fixture_id],
                object_id="plate",
                preferred_xy=reference_xy,
            )
            plate_quat = executor._get_object_pose("plate")[1]
            executor._set_object_pose("plate", blocker_target, plate_quat)
            executor.runner._set_object_location("plate", support_fixture_id)

            bun_source = scene["objects"]["hotdog_bun"]["location"]
            executor.pick_up_object("hotdog_bun", bun_source, robot_idx=0)
            result = executor.place_under("hotdog_bun", reference_fixture_id, robot_idx=0)

            self.assertTrue(result.success)
            self.assertEqual(executor._get_scene_object_location("hotdog_bun"), support_fixture_id)
            self.assertFalse(
                _objects_intersect(executor, "hotdog_bun", "plate"),
                "Generic place_under should slide to a nearby free pose when center is blocked",
            )

            bun_xy = executor._get_object_pose("hotdog_bun")[0][:2]
            self.assertLess(
                float(np.linalg.norm(bun_xy - reference_xy)),
                0.35,
                "place_under should stay near the reference fixture even after collision avoidance",
            )
        finally:
            executor.close()


class TestReceptacleCarrySemantics(unittest.TestCase):
    def _make_executor(self):
        return SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            render_width=160,
            render_height=128,
        )

    def _stage_sausage_on_plate(self, executor: SimToolExecutor):
        scene = executor.get_scene_description()
        sausage_source = scene["objects"]["sausage"]["location"]
        executor.pick_up_object("sausage", sausage_source, robot_idx=1)
        result = executor.place_on_object("sausage", "plate", robot_idx=1)
        self.assertTrue(result.success)

    def test_pick_up_plate_carries_contents(self):
        executor = self._make_executor()
        try:
            self._stage_sausage_on_plate(executor)
            plate_before = executor._get_object_pose("plate")[0].copy()
            sausage_before = executor._get_object_pose("sausage")[0].copy()

            plate_source = executor._get_scene_object_location("plate")
            result = executor.pick_up_object("plate", plate_source, robot_idx=0)

            self.assertTrue(result.success)
            plate_after = executor._get_object_pose("plate")[0].copy()
            sausage_after = executor._get_object_pose("sausage")[0].copy()
            self.assertTrue(
                np.allclose(
                    sausage_after - sausage_before,
                    plate_after - plate_before,
                    atol=2e-2,
                ),
                "Picking up a receptacle should move its contents with it",
            )
        finally:
            executor.close()

    def test_runner_move_object_carries_contents(self):
        executor = self._make_executor()
        try:
            self._stage_sausage_on_plate(executor)
            scene = executor.get_scene_description()
            plate_source = executor._get_scene_object_location("plate")
            target_fixture_id = None
            for fixture_id, info in scene["fixtures"].items():
                if fixture_id == plate_source or not info.get("can_place_objects", False):
                    continue
                if "counter" in info.get("fixture_type", "") or "island" in info.get("fixture_type", ""):
                    target_fixture_id = fixture_id
                    break

            self.assertIsNotNone(target_fixture_id)

            plate_before = executor._get_object_pose("plate")[0].copy()
            sausage_before = executor._get_object_pose("sausage")[0].copy()
            executor.runner.move_object("plate", target_fixture_id)
            plate_after = executor._get_object_pose("plate")[0].copy()
            sausage_after = executor._get_object_pose("sausage")[0].copy()

            self.assertTrue(
                np.allclose(
                    sausage_after - sausage_before,
                    plate_after - plate_before,
                    atol=2e-2,
                ),
                "Runner-level receptacle moves should carry contained contents",
            )
            self.assertEqual(executor._get_scene_object_location("plate"), target_fixture_id)
            self.assertEqual(executor._get_scene_object_location("sausage"), target_fixture_id)
        finally:
            executor.close()


class TestFrontRetryUnit(unittest.TestCase):
    def test_safe_compute_object_target_pos_falls_back_to_legacy_sampling(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"island": object()}
        executor.runner._compute_object_target_pos = MagicMock(
            side_effect=[AssertionError(), np.array([1.0, 2.0, 3.0])]
        )

        target = executor._safe_compute_object_target_pos(
            "island",
            "condiment",
            preferred_xy=np.array([9.0, 8.0]),
        )

        self.assertTrue(np.allclose(target, np.array([9.0, 8.0, 3.0])))
        self.assertEqual(executor.runner._compute_object_target_pos.call_count, 2)

    @patch("robocasa.utils.sim_tool_executor.get_front_alignment_metrics")
    @patch("robocasa.utils.sim_tool_executor._is_approach_center")
    def test_robot_near_fixture_requires_bounded_front_gap(
        self,
        mock_is_approach_center,
        mock_get_front_alignment_metrics,
    ):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(pos=np.array([0.0, 0.0, 0.0]))
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge": fixture}
        executor.runner._get_robot_position.return_value = np.array([0.0, 0.0, 0.0])
        executor.runner._get_fixture_front_target_xy.return_value = np.array([0.0, 0.0])

        mock_is_approach_center.return_value = True
        mock_get_front_alignment_metrics.return_value = {
            "on_front_face": True,
            "within_span": True,
            "lateral_offset": 0.0,
            "front_gap": 1.2,
        }

        self.assertFalse(executor._robot_near_fixture(0, "fridge"))

    def test_front_retry_retries_after_clearing_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock(side_effect=[False, True])
        executor._clear_fixture_blockers = MagicMock(return_value=True)

        placed = executor._move_robot_near_fixture_with_retries(
            1,
            "fridge",
            require_front=True,
        )

        self.assertTrue(placed)
        executor._clear_fixture_blockers.assert_called_once_with(1, "fridge")
        self.assertEqual(executor.runner._move_robot_near_fixture.call_count, 2)

    def test_non_front_retry_does_not_clear_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock(return_value=False)
        executor._clear_fixture_blockers = MagicMock(return_value=True)

        placed = executor._move_robot_near_fixture_with_retries(
            0,
            "counter",
            require_front=False,
        )

        self.assertFalse(placed)
        executor._clear_fixture_blockers.assert_not_called()
        executor.runner._move_robot_near_fixture.assert_called_once()


if __name__ == "__main__":
    unittest.main()
