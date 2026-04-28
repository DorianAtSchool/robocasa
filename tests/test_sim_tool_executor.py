import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.utils.generate_llm_task_descriptions import (
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
import robocasa.utils.trajectory_runner as trajectory_runner_module
from robocasa.utils.trajectory_runner import TrajectoryRunner  # noqa: E402
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
    def test_configure_mujoco_gl_backend_updates_cached_binding_choice(self):
        original_gl = os.environ.get("MUJOCO_GL")
        original_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        fake_binding_utils = SimpleNamespace(_MUJOCO_GL="egl")
        observed_gl = None
        observed_egl_device = None

        try:
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"
            with patch.dict(
                sys.modules,
                {"robosuite.utils.binding_utils": fake_binding_utils},
            ):
                configured_backend = (
                    trajectory_runner_module._configure_mujoco_gl_backend("osmesa")
                )
                observed_gl = os.environ.get("MUJOCO_GL")
                observed_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        finally:
            if original_gl is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = original_gl
            if original_egl_device is None:
                os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            else:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = original_egl_device

        self.assertEqual(configured_backend, "osmesa")
        self.assertEqual(observed_gl, "osmesa")
        self.assertIsNone(observed_egl_device)
        self.assertEqual(fake_binding_utils._MUJOCO_GL, "osmesa")

    def test_restore_baseline_state_resets_runtime_and_model_state(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(_turned_on=True, _num_steps_on=3)
        model = SimpleNamespace(
            site_rgba=np.ones((2, 4)),
            site_size=np.ones((2, 3)),
            geom_rgba=np.ones((3, 4)),
        )
        sim = MagicMock()
        sim.model = model
        env = SimpleNamespace(
            sim=sim,
            update_sites=MagicMock(),
            update_state=MagicMock(),
        )
        runner = SimpleNamespace(
            _fixtures={"coffee_machine": fixture},
            _object_locations={"mug": "counter_mutated"},
            _scene={"objects": {"mug": {"location": "counter_mutated"}}},
        )

        executor.env = env
        executor.runner = runner
        executor._held_objects = {0: "mug"}
        executor._baseline_sim_state = np.array([1.0, 2.0, 3.0])
        executor._baseline_model_site_rgba = np.zeros((2, 4))
        executor._baseline_model_site_size = np.full((2, 3), 0.5)
        executor._baseline_model_geom_rgba = np.full((3, 4), 0.25)
        executor._baseline_fixture_runtime_state = {
            "coffee_machine": {"_turned_on": False, "_num_steps_on": 0}
        }
        executor._baseline_object_locations = {"mug": "counter_clean"}
        executor._baseline_scene = {"objects": {"mug": {"location": "counter_clean"}}}

        executor.restore_baseline_state()

        sim.set_state_from_flattened.assert_called_once()
        np.testing.assert_array_equal(
            sim.set_state_from_flattened.call_args.args[0],
            np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(sim.forward.call_count, 2)
        env.update_sites.assert_called_once()
        env.update_state.assert_called_once()
        self.assertEqual(executor._held_objects, {})
        self.assertFalse(fixture._turned_on)
        self.assertEqual(fixture._num_steps_on, 0)
        self.assertEqual(runner._object_locations["mug"], "counter_clean")
        self.assertEqual(
            runner._scene["objects"]["mug"]["location"],
            "counter_clean",
        )
        np.testing.assert_array_equal(model.site_rgba, np.zeros((2, 4)))
        np.testing.assert_array_equal(model.site_size, np.full((2, 3), 0.5))
        np.testing.assert_array_equal(model.geom_rgba, np.full((3, 4), 0.25))

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
        self.assertTrue(
            any(fx["fixture_type"] == "fridge" for fx in context["fixtures"])
        )

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
                self.assertNotIn("wait", tools)

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
                if step["tool"] == "navigate_to_fixture"
                and "fixture_id" in step["args"]
            ]
            place_steps = [step for step in plan if step["tool"] == "place_on_object"]
            self.assertIn("island_island_group_1", fixture_ids)
            self.assertTrue(any("fridge" in fixture_id for fixture_id in fixture_ids))
            self.assertTrue(
                all("anchor_fixture_id" in step["args"] for step in place_steps)
            )
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
                if step["tool"] == "navigate_to_fixture"
                and "fixture_id" in step["args"]
            }
            fixture_ids_b = {
                step["args"]["fixture_id"]
                for step in grounded_b
                if step["tool"] == "navigate_to_fixture"
                and "fixture_id" in step["args"]
            }

            self.assertIn("dining_dining_group", fixture_ids_a)
            self.assertIn("island_island_group_1", fixture_ids_b)
            self.assertNotEqual(fixture_ids_a, fixture_ids_b)

            place_steps_a = [
                step for step in grounded_a if step["tool"] == "place_on_object"
            ]
            place_steps_b = [
                step for step in grounded_b if step["tool"] == "place_on_object"
            ]
            self.assertTrue(
                all(
                    step["args"]["anchor_fixture_id"] == "dining_dining_group"
                    for step in place_steps_a
                )
            )
            self.assertTrue(
                all(
                    step["args"]["anchor_fixture_id"] == "island_island_group_1"
                    for step in place_steps_b
                )
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
            plate_xy_fixture = executor_fixture.env.sim.data.body_xpos[plate_body_id][
                :2
            ].copy()
            plate_xy_object = executor_object.env.sim.data.body_xpos[
                executor_object.env.obj_body_id["plate"]
            ][:2].copy()

            executor_fixture.runner._move_robot_near_fixture(0, "island_island_group_1")
            robot_xy_fixture = executor_fixture.runner._get_robot_position(0)[:2]
            dist_fixture = float(
                ((robot_xy_fixture - plate_xy_fixture) ** 2).sum() ** 0.5
            )

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
        return np.array(self._scene["fixtures"][fixture_id]["position"], dtype=float)

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

    def test_handled_cookware_syncs_handle_to_eef(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {0: "pan"}
        executor._HELD_Z_OFFSET = 0.0
        executor._scene_object_tokens = MagicMock(return_value={"pan"})
        executor._get_robot_eef_pos = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0], dtype=float)
        )
        executor._get_object_pose = MagicMock(
            return_value=(
                np.zeros(3, dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._set_object_pose = MagicMock()

        bbox = np.array(
            [
                [-0.10, -0.10, -0.02],
                [-0.10, 0.10, 0.02],
                [0.50, -0.10, -0.02],
                [0.50, 0.10, 0.02],
            ],
            dtype=float,
        )
        fake_obj = SimpleNamespace(get_bbox_points=MagicMock(return_value=bbox))
        executor._require_object = MagicMock(return_value=fake_obj)

        offset = executor._held_pose_offset("pan")
        executor._held_object_offsets = {0: offset}
        executor._sync_held_object(0)

        target_pos = executor._set_object_pose.call_args.args[1]
        np.testing.assert_allclose(target_pos, np.array([0.575, 2.0, 3.0]))

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
            if info.get("can_place_objects") and "counter" in info.get(
                "fixture_type", ""
            ):
                try:
                    self._executor.runner._compute_object_target_pos(
                        self._executor.runner._fixtures[fid],
                        object_id=obj_id,
                    )
                except RuntimeError:
                    continue
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
        self.assertGreater(
            sausage_z, plate_z - 0.01, "Sausage should be at or above plate level"
        )

    def test_place_on_object_spreads_multiple_items_without_overlap(self):
        scene = self._executor.get_scene_description()

        condiment_source = scene["objects"]["condiment"]["location"]
        self._executor.pick_up_object("condiment", condiment_source, robot_idx=0)
        condiment_result = self._executor.place_on_object("condiment", "plate", robot_idx=0)
        self.assertTrue(condiment_result.success)

        sausage_source = scene["objects"]["sausage"]["location"]
        self._executor.pick_up_object("sausage", sausage_source, robot_idx=1)
        sausage_result = self._executor.place_on_object("sausage", "plate", robot_idx=1)
        self.assertTrue(sausage_result.success)

        condiment_pos = self._object_pos("condiment")
        sausage_pos = self._object_pos("sausage")
        self.assertGreater(
            float(np.linalg.norm(condiment_pos[:2] - sausage_pos[:2])),
            0.01,
        )
        self.assertFalse(
            _objects_intersect(self._executor, "condiment", "sausage"),
            "Objects placed on the same support should avoid sibling collisions",
        )

    def test_place_on_object_keeps_existing_item_stable_when_adding_second_item(self):
        scene = self._executor.get_scene_description()

        condiment_source = scene["objects"]["condiment"]["location"]
        self._executor.pick_up_object("condiment", condiment_source, robot_idx=0)
        self.assertTrue(
            self._executor.place_on_object("condiment", "plate", robot_idx=0).success
        )
        condiment_pos_before = self._object_pos("condiment").copy()

        sausage_source = scene["objects"]["sausage"]["location"]
        self._executor.pick_up_object("sausage", sausage_source, robot_idx=1)
        self.assertTrue(
            self._executor.place_on_object("sausage", "plate", robot_idx=1).success
        )
        condiment_pos_after = self._object_pos("condiment").copy()

        self.assertLess(
            float(np.linalg.norm(condiment_pos_after[:2] - condiment_pos_before[:2])),
            0.02,
            "Adding a new supported item should not repack previously placed items",
        )

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

    def test_place_next_to_tries_alternate_offsets_on_same_support(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_reference_target = MagicMock(
            return_value=("toaster_oven_main_group", True)
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {
            "toaster_oven_main_group": SimpleNamespace(pos=np.array([1.0, 1.0, 0.0])),
            "counter_main": object(),
        }
        executor._reference_extent_xy = MagicMock(return_value=np.array([0.4, 0.3]))
        executor._find_placeable_surface_near_fixture = MagicMock(
            return_value="counter_main"
        )
        executor._require_fixture = MagicMock()
        executor._candidate_xy_next_to_reference = MagicMock(
            return_value=[np.array([0.0, 0.0]), np.array([0.5, 0.0])]
        )
        executor._safe_compute_object_target_pos = MagicMock(
            side_effect=[
                RuntimeError("blocked"),
                np.array([0.45, 0.02, 0.9]),
            ]
        )
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value=set())
        executor.runner._move_robot_near_fixture = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._held_objects = {0: "ingredient_bowl"}
        executor._settle_scene = MagicMock()

        result = executor.place_next_to(
            "ingredient_bowl",
            reference_fixture_id="toaster_oven_main_group",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            executor._safe_compute_object_target_pos.call_args_list[0].args[:2],
            ("counter_main", "ingredient_bowl"),
        )
        self.assertEqual(
            executor._safe_compute_object_target_pos.call_args_list[1].args[:2],
            ("counter_main", "ingredient_bowl"),
        )
        move_args, move_kwargs = executor.runner._move_robot_near_fixture.call_args
        self.assertEqual(move_args[:2], (0, "counter_main"))
        np.testing.assert_allclose(
            move_kwargs["ref_pos_override"],
            np.array([0.45, 0.02]),
        )
        executor.runner._set_object_location.assert_called_once_with(
            "ingredient_bowl",
            "counter_main",
        )

    def test_place_next_to_ignores_countertop_appliance_reference_fixture_collision(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_reference_target = MagicMock(
            return_value=("toaster_oven_main_group", True)
        )
        executor.runner = MagicMock()
        toaster_fixture = SimpleNamespace(pos=np.array([1.0, 1.0, 0.0]))
        executor.runner._fixtures = {
            "toaster_oven_main_group": toaster_fixture,
            "counter_main": object(),
        }
        executor._reference_extent_xy = MagicMock(return_value=np.array([0.4, 0.3]))
        executor._find_placeable_surface_near_fixture = MagicMock(
            return_value="counter_main"
        )
        executor._require_fixture = MagicMock()
        executor._get_fixture_type_name = MagicMock(
            return_value="toaster_oven"
        )
        executor._resolve_fixture_site_id = MagicMock(return_value=None)
        executor._candidate_xy_next_to_reference = MagicMock(
            return_value=[np.array([0.2, 0.0])]
        )
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value=set())
        executor.runner._world_to_fixture_local = MagicMock(
            side_effect=[
                np.array([0.0, 0.0], dtype=float),
                np.array([0.1, 0.05], dtype=float),
                np.array([0.1, 0.05], dtype=float),
            ]
        )
        executor.runner._move_robot_near_fixture = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._held_objects = {0: "ingredient_bowl"}
        executor._settle_scene = MagicMock()

        executor._safe_compute_object_target_pos = MagicMock(
            return_value=np.array([0.25, 0.01, 0.9])
        )
        result = executor.place_next_to(
            "ingredient_bowl",
            reference_fixture_id="toaster_oven_main_group",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        ignored_fixture_ids = executor._safe_compute_object_target_pos.call_args.kwargs[
            "ignored_fixture_ids"
        ]
        self.assertEqual(ignored_fixture_ids, {"toaster_oven_main_group"})

    def test_candidate_xy_next_to_reference_biases_toward_counter_side_of_stool(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        support_fixture = SimpleNamespace()
        executor._require_fixture = MagicMock(return_value=support_fixture)
        executor.runner._world_to_fixture_local = MagicMock(
            side_effect=lambda _fixture, point: np.array(
                [float(point[0]), float(point[1]), 0.0],
                dtype=float,
            )
        )
        executor.runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, point: np.array(
                [float(point[0]), float(point[1]), 0.0],
                dtype=float,
            )
        )
        executor._fixture_local_bounds = MagicMock(
            return_value=(
                np.array([-1.0, -1.0], dtype=float),
                np.array([1.0, 1.0], dtype=float),
            )
        )
        executor._get_fixture_type_name = MagicMock(return_value="stool")
        executor._reference_extent_xy_in_support_local = MagicMock(
            return_value=np.array([0.3, 0.3], dtype=float)
        )

        preferred_positions = executor._candidate_xy_next_to_reference(
            support_fixture_id="dining_counter_main",
            reference_xy=np.array([0.0, -0.9], dtype=float),
            reference_extent_xy=np.array([0.3, 0.3], dtype=float),
            reference_id="stool_main",
            reference_is_fixture=True,
        )

        self.assertGreater(len(preferred_positions), 0)
        self.assertGreater(
            float(preferred_positions[0][1]),
            -0.9,
            "Preferred placement should move inward toward the counter",
        )
        self.assertLess(abs(float(preferred_positions[0][0])), 0.2)

    def test_candidate_xy_next_to_reference_preserves_stool_lateral_coordinate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        support_fixture = SimpleNamespace()
        executor._require_fixture = MagicMock(return_value=support_fixture)
        executor.runner._world_to_fixture_local = MagicMock(
            side_effect=lambda _fixture, point: np.array(
                [float(point[0]), float(point[1]), 0.0],
                dtype=float,
            )
        )
        executor.runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, point: np.array(
                [float(point[0]), float(point[1]), 0.0],
                dtype=float,
            )
        )
        executor._fixture_local_bounds = MagicMock(
            return_value=(
                np.array([-0.7, -0.2], dtype=float),
                np.array([0.7, 0.2], dtype=float),
            )
        )
        executor._get_fixture_type_name = MagicMock(return_value="stool")
        executor._reference_extent_xy_in_support_local = MagicMock(
            return_value=np.array([0.3, 0.3], dtype=float)
        )

        preferred_positions = executor._candidate_xy_next_to_reference(
            support_fixture_id="dining_counter_main",
            reference_xy=np.array([-0.55, -0.335], dtype=float),
            reference_extent_xy=np.array([0.3, 0.3], dtype=float),
            reference_id="stool_main",
            reference_is_fixture=True,
        )

        self.assertGreater(len(preferred_positions), 0)
        self.assertAlmostEqual(float(preferred_positions[0][0]), -0.55, delta=0.02)
        self.assertGreater(float(preferred_positions[0][1]), -0.335)

    def test_place_next_to_aligns_directional_objects_perpendicular_to_offset_axis(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_reference_target = MagicMock(return_value=("plate_main", False))
        executor._default_relative_position_next_to_object = MagicMock(return_value=None)
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )
        executor._reference_extent_xy = MagicMock(return_value=np.array([0.2, 0.2]))
        executor._get_scene_object_location = MagicMock(return_value="counter_main")
        executor._require_fixture = MagicMock(return_value=SimpleNamespace())
        executor._resolve_fixture_site_id = MagicMock(return_value=None)
        executor._candidate_xy_next_to_reference = MagicMock(
            return_value=[np.array([1.25, 1.0])]
        )
        executor._ignored_fixture_ids_for_adjacent_reference = MagicMock(return_value=set())
        executor._safe_compute_object_target_pos = MagicMock(
            return_value=np.array([1.25, 1.0, 0.9])
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._world_to_fixture_local = MagicMock(
            side_effect=[
                np.array([1.0, 1.0], dtype=float),
                np.array([1.25, 1.0], dtype=float),
            ]
        )
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._fixture_inward_world_axis = MagicMock(
            return_value=np.array([0.0, 1.0, 0.0], dtype=float)
        )
        executor._aligned_flat_quat_for_world_axis = MagicMock(
            return_value=np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._held_objects = {0: "baguette"}
        executor._held_object_offsets = {}
        executor._settle_scene = MagicMock()

        result = executor.place_next_to(
            "baguette",
            reference_object_id="plate_main",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._fixture_inward_world_axis.assert_called_once_with(
            "counter_main",
            "y",
        )
        aligned_args = executor._aligned_flat_quat_for_world_axis.call_args
        np.testing.assert_allclose(
            aligned_args.args[1],
            np.array([0.0, -1.0, 0.0], dtype=float),
        )
        self.assertFalse(aligned_args.kwargs["preserve_direction"])
        self.assertFalse(aligned_args.kwargs["allow_target_sign_flip"])
        pose_args = executor._set_object_pose.call_args.args
        np.testing.assert_allclose(pose_args[1], np.array([1.25, 1.0, 0.9]))
        np.testing.assert_allclose(pose_args[2], np.array([1.0, 0.0, 0.0, 0.0]))

    def test_place_next_to_defaults_knife_to_right_of_plate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_reference_target = MagicMock(return_value=("plate_main", False))
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: {
                "butter_knife": {"butter", "knife"},
                "plate_main": {"plate"},
            }.get(object_id, set())
        )
        executor._get_object_pose = MagicMock(
            return_value=(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._reference_extent_xy = MagicMock(return_value=np.array([0.2, 0.2]))
        executor._get_scene_object_location = MagicMock(return_value="counter_main")
        executor._require_fixture = MagicMock(return_value=SimpleNamespace())
        executor._resolve_fixture_site_id = MagicMock(return_value=None)
        executor._candidate_xy_next_to_reference = MagicMock(
            return_value=[np.array([1.25, 1.0])]
        )
        executor._ignored_fixture_ids_for_adjacent_reference = MagicMock(return_value=set())
        executor._safe_compute_object_target_pos = MagicMock(
            return_value=np.array([1.25, 1.0, 0.9])
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._world_to_fixture_local = MagicMock(
            side_effect=[
                np.array([1.0, 1.0], dtype=float),
                np.array([1.25, 1.0], dtype=float),
            ]
        )
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._fixture_inward_world_axis = MagicMock(
            return_value=np.array([0.0, 1.0, 0.0], dtype=float)
        )
        executor._aligned_flat_quat_for_world_axis = MagicMock(
            return_value=np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._held_objects = {0: "butter_knife"}
        executor._held_object_offsets = {}
        executor._settle_scene = MagicMock()

        result = executor.place_next_to(
            "butter_knife",
            reference_object_id="plate_main",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        self.assertEqual(
            executor._candidate_xy_next_to_reference.call_args.kwargs[
                "relative_position"
            ],
            "right",
        )

    # -- place_under (generic / non-dispenser) --

    def test_place_under_generic_aligns_xy(self):
        """place_under a non-dispenser fixture should align XY under it."""
        scene = self._executor.get_scene_description()
        # Find any non-placeable, non-dispenser fixture (cabinet, hood, etc.)
        ref_fixture = None
        for fid, info in scene["fixtures"].items():
            ftype = info.get("fixture_type", "")
            if not info.get("can_place_objects", False) and ftype not in (
                "coffee_machine",
                "sink",
                "",
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
        support_fixture_id = self._executor._find_placeable_surface_near_fixture(
            ref_fixture
        )
        expected_xy = self._executor._project_xy_onto_fixture(
            support_fixture_id,
            self._fixture_pos(ref_fixture)[:2],
        )
        xy_dist = float(np.linalg.norm(obj_xy - expected_xy))
        self.assertLess(
            xy_dist,
            0.5,
            "Object XY should stay near the projected support target below the fixture",
        )

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
                if (
                    abs(off_center_pos[0] - front_target[0])
                    <= MAX_FRONT_WORKING_LATERAL_OFFSET
                ):
                    self.skipTest(
                        "Fixture front span too narrow for off-center recenter test"
                    )
            else:
                off_center_pos[1] = min(front_target[1] + target_offset, fmax[1] - 0.02)
                if (
                    abs(off_center_pos[1] - front_target[1])
                    <= MAX_FRONT_WORKING_LATERAL_OFFSET
                ):
                    self.skipTest(
                        "Fixture front span too narrow for off-center recenter test"
                    )

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
            self.assertEqual(
                executor._get_scene_object_location("hotdog_bun"), support_id
            )
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

            support_fixture_id = executor._find_placeable_surface_near_fixture(
                reference_fixture_id
            )
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
            result = executor.place_under(
                "hotdog_bun", reference_fixture_id, robot_idx=0
            )

            self.assertTrue(result.success)
            self.assertEqual(
                executor._get_scene_object_location("hotdog_bun"), support_fixture_id
            )
            self.assertFalse(
                _objects_intersect(executor, "hotdog_bun", "plate"),
                "Generic place_under should slide to a nearby free pose when center is blocked",
            )

            bun_xy = executor._get_object_pose("hotdog_bun")[0][:2]
            expected_xy = executor._project_xy_onto_fixture(
                support_fixture_id,
                reference_xy,
            )
            self.assertLess(
                float(np.linalg.norm(bun_xy - expected_xy)),
                0.5,
                "place_under should stay near the projected support target even after collision avoidance",
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
                if fixture_id == plate_source or not info.get(
                    "can_place_objects", False
                ):
                    continue
                if "counter" in info.get("fixture_type", "") or "island" in info.get(
                    "fixture_type", ""
                ):
                    target_fixture_id = fixture_id
                    try:
                        executor.runner._compute_object_target_pos(
                            executor.runner._fixtures[fixture_id],
                            object_id="plate",
                            ignored_object_ids={"sausage"},
                        )
                    except RuntimeError:
                        continue
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
            self.assertEqual(
                executor.runner._object_locations.get("plate"), target_fixture_id
            )
            self.assertEqual(
                executor.runner._object_locations.get("sausage"), target_fixture_id
            )
        finally:
            executor.close()


class TestFrontRetryUnit(unittest.TestCase):
    def test_safe_compute_object_target_pos_forwards_support_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"island": object()}
        executor._find_contained_objects = MagicMock(return_value=[])
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )

        target = executor._safe_compute_object_target_pos(
            "island",
            "condiment",
            preferred_xy=np.array([9.0, 8.0]),
            support_site_id="rear_left",
        )

        self.assertTrue(np.allclose(target, np.array([1.0, 2.0, 3.0])))
        executor.runner._compute_object_target_pos.assert_called_once()
        _, kwargs = executor.runner._compute_object_target_pos.call_args
        self.assertEqual(kwargs["site_id"], "rear_left")
        self.assertTrue(np.allclose(kwargs["preferred_xy"], np.array([9.0, 8.0])))

    def test_safe_compute_object_target_pos_retries_same_site_without_switching(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"blender": object()}
        executor._find_contained_objects = MagicMock(return_value=[])
        executor.runner._compute_object_target_pos = MagicMock(
            side_effect=[
                RuntimeError("no candidate"),
                np.array([0.1, 0.2, 0.3]),
            ]
        )

        target = executor._safe_compute_object_target_pos(
            "blender",
            "tomato",
            preferred_xy=np.array([1.0, 2.0]),
            support_site_id="bowl",
            ignored_fixture_ids={"counter_main"},
        )

        self.assertTrue(np.allclose(target, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(2, executor.runner._compute_object_target_pos.call_count)
        first_call = executor.runner._compute_object_target_pos.call_args_list[0].kwargs
        second_call = executor.runner._compute_object_target_pos.call_args_list[1].kwargs
        self.assertEqual(first_call["site_id"], "bowl")
        self.assertEqual(second_call["site_id"], "bowl")
        self.assertTrue(np.allclose(first_call["preferred_xy"], np.array([1.0, 2.0])))
        self.assertIsNone(second_call["preferred_xy"])
        self.assertEqual(first_call["ignored_fixture_ids"], {"counter_main"})
        self.assertEqual(second_call["ignored_fixture_ids"], {"counter_main"})

    def test_safe_compute_object_target_pos_ignores_contained_children(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter": object()}
        executor._find_contained_objects = MagicMock(return_value=["slice_a", "slice_b"])
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([0.0, 0.0, 0.0])
        )

        executor._safe_compute_object_target_pos(
            "counter",
            "ingredient_bowl",
            ignored_object_ids={"plate"},
        )

        _, kwargs = executor.runner._compute_object_target_pos.call_args
        self.assertEqual(
            kwargs["ignored_object_ids"],
            {"plate", "slice_a", "slice_b"},
        )

    def test_explicit_site_candidate_generation_keeps_compact_region(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "mug": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.20, 0.20, 0.10], dtype=float),
                "xy_radius": 0.09,
            }
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.06, 0.06),
                    }
                }
            )
        )

        regions = runner._get_fixture_reset_regions(
            target_fixture,
            min_size=np.array([0.20, 0.20, 0.10], dtype=float),
            site_id="rack",
        )
        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "mug",
            site_id="rack",
        )

        self.assertEqual(1, len(regions))
        self.assertTrue(candidates)

    def test_explicit_site_candidate_generation_uses_fixture_aligned_extent(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "baguette": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.22, 0.22, 0.06], dtype=float),
                "xy_radius": 0.11,
            }
        )
        runner._get_object_fixture_xy_half_extent = MagicMock(
            return_value=np.array([0.04, 0.01], dtype=float)
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            rot=None,
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.12, 0.06),
                    }
                }
            )
        )

        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "baguette",
            site_id="rack",
        )

        self.assertGreater(len(candidates), 1)

    def test_explicit_site_candidate_generation_samples_full_site_span(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "bread": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.24, 0.08, 0.06], dtype=float),
                "xy_radius": 0.12,
            }
        )
        runner._get_object_fixture_xy_half_extent = MagicMock(
            return_value=np.array([0.12, 0.03], dtype=float)
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            rot=None,
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.10, 0.04),
                    }
                }
            )
        )

        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "bread",
            site_id="rack",
        )

        self.assertGreater(len(candidates), 1)
        xs = sorted({round(float(candidate[0]), 4) for candidate in candidates})
        self.assertGreater(len(xs), 1)

    def test_resolve_joint_name_normalizes_timer_knob(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(
            name="toaster_oven_main_group",
            _joint_names={"time": "toaster_oven_knob_time_joint"},
        )

        resolved = executor._resolve_joint_name(fixture, "timer_knob")

        self.assertEqual(resolved, "toaster_oven_knob_time_joint")

    def test_support_object_prefers_upright_for_deep_cup_like_receptacles(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "cup_main": {"object_type": "glass_cup"},
                    "straw_main": {"object_type": "straw"},
                }
            }
        )
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": np.array([0.18, 0.01, 0.01], dtype=float)}
        )

        self.assertTrue(executor._support_object_prefers_upright("cup_main", "straw_main"))

    def test_support_object_prefers_upright_from_support_object_id_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "pot_main": {"object_type": ""},
                    "straw_main": {"object_type": "straw"},
                }
            }
        )
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": np.array([0.18, 0.01, 0.01], dtype=float)}
        )

        self.assertTrue(executor._support_object_prefers_upright("pot_main", "straw_main"))

    def test_support_object_floor_clearance_is_tighter_for_bowl_like_supports(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={"objects": {"bowl_main": {"object_type": "mixing_bowl"}}}
        )

        clearance = executor._support_object_floor_clearance(
            "bowl_main",
            {"extent_z": 0.10},
        )

        self.assertLessEqual(clearance, 0.01)
        self.assertGreaterEqual(clearance, 0.002)

    def test_quat_align_vectors_rotates_source_onto_target(self):
        quat_xyzw = SimToolExecutor._quat_align_vectors(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

        rotated = T.quat2mat(quat_xyzw) @ np.array([1.0, 0.0, 0.0], dtype=float)

        np.testing.assert_allclose(rotated, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_object_is_upright_checks_dominant_axis_alignment(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._dominant_object_local_axis = MagicMock(
            return_value=np.array([1.0, 0.0, 0.0], dtype=float)
        )

        upright_quat_xyzw = SimToolExecutor._quat_align_vectors(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        upright_quat_wxyz = T.convert_quat(upright_quat_xyzw, to="wxyz")

        self.assertTrue(executor._object_is_upright("straw_main", upright_quat_wxyz))
        self.assertFalse(
            executor._object_is_upright(
                "straw_main",
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )

    def test_fixture_site_slot_positions_front_bias_interior_row(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._get_fixture_reset_regions = MagicMock(
            return_value=[
                {
                    "name": "shelf_0",
                    "offset": (0.0, 0.0, 0.0),
                    "size": (0.40, 0.20, 0.10),
                }
            ]
        )
        executor.runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        executor._require_fixture = MagicMock(return_value=object())
        executor._get_fixture_type_name = MagicMock(return_value="cabinet")

        positions = executor._fixture_site_slot_positions("cab_main", "shelf_0", count=2)

        self.assertEqual(len(positions), 2)
        self.assertLess(float(positions[0][1]), 0.0)
        self.assertLess(float(positions[1][1]), 0.0)

    def test_support_object_slot_positions_use_wider_spread_for_plate_like_supports(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_geometry = MagicMock(
            return_value={
                "extent_x": 0.30,
                "extent_y": 0.24,
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "is_concave": False,
                "rot_xy": np.eye(2, dtype=float),
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        positions = executor._support_object_slot_positions("plate_main", count=2)

        self.assertEqual(len(positions), 2)
        self.assertGreater(
            float(np.linalg.norm(np.asarray(positions[0]) - np.asarray(positions[1]))),
            0.13,
        )
        self.assertLess(float(positions[0][0]), 0.0)
        self.assertGreater(float(positions[1][0]), 0.0)

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

    def test_place_in_receptacle_uses_front_retry_for_front_required_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(
            side_effect=[("fridge", "shelf_0"), ("fridge", "shelf_0")]
        )
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._preferred_xy_for_fixture_target = MagicMock(
            return_value=np.array([1.0, 2.0], dtype=float)
        )
        executor._incoming_fixture_site_preference = MagicMock(return_value=None)
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value=set())
        executor._compute_object_target_pose = MagicMock(
            return_value=(
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._fixture_requires_front_approach = MagicMock(return_value=True)
        executor._surface_fixture_prefers_front_approach = MagicMock(return_value=False)
        executor._move_robot_near_fixture_with_retries = MagicMock(return_value=True)
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {0: "bowl"}
        executor._held_object_offsets = {0: np.zeros(3, dtype=float)}
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge": object()}
        executor.runner._set_object_location = MagicMock()
        executor.env = SimpleNamespace(objects={"bowl": object()})

        result = executor.place_in_receptacle(
            "bowl",
            receptacle_id="fridge",
            target_site_id="shelf_0",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._move_robot_near_fixture_with_retries.assert_called_once()
        self.assertTrue(
            executor._move_robot_near_fixture_with_retries.call_args.kwargs[
                "require_front"
            ]
        )

    def test_partitioned_fixture_requires_explicit_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"cab_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "cab_main": {"fixture_type": "cabinet_double_door"},
                }
            }
        )
        executor.get_support_sites = MagicMock(
            return_value=["shelf_1", "shelf_2"]
        )
        executor._resolve_fixture_site_id = MagicMock(return_value="shelf_1")

        with self.assertRaises(ValueError):
            executor._require_explicit_site_if_needed(
                "cab_main",
                None,
                action_name="place_in_receptacle",
            )

    def test_normalize_target_site_drops_generic_surface_geom(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {"fixture_type": "counter_non_dining"},
                    "stove_main": {"fixture_type": "stove"},
                }
            }
        )
        executor._resolve_fixture_site_id = MagicMock(
            side_effect=lambda fixture_id, site_id: site_id
        )

        self.assertIsNone(
            executor._normalize_target_site_id_for_placement("counter_main", "geom_0")
        )
        self.assertEqual(
            executor._normalize_target_site_id_for_placement("stove_main", "front_right_burner"),
            "burner",
        )

    def test_default_fixture_surface_preference_avoids_child_fixture_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=[])
        executor._fixture_surface_blocker_positions = MagicMock(
            return_value=[np.array([0.0, 0.0], dtype=float)]
        )
        executor._surface_fixture_slot_positions = MagicMock(
            return_value=[
                np.array([0.0, 0.0], dtype=float),
                np.array([0.35, 0.0], dtype=float),
            ]
        )
        executor._preferred_xy_for_fixture_target = MagicMock(
            return_value=np.array([0.0, 0.0], dtype=float)
        )

        preferred_xy = executor._default_fixture_surface_preference(
            "counter_main",
            "bowl_main",
        )

        np.testing.assert_allclose(preferred_xy, np.array([0.35, 0.0], dtype=float))

    def test_default_fixture_surface_preference_keeps_duplicate_cubes_compact(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=["sugar_cube_1"])
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._scene_object_tokens = MagicMock(return_value={"sugar", "cube"})
        executor._require_fixture = MagicMock(
            return_value=SimpleNamespace(rot=0.0)
        )
        executor._project_xy_onto_fixture = MagicMock(
            side_effect=lambda fixture_id, world_xy: np.asarray(world_xy, dtype=float)
        )
        executor._fixture_surface_blocker_positions = MagicMock()
        executor._surface_fixture_slot_positions = MagicMock()

        preferred_xy = executor._default_fixture_surface_preference(
            "dining_counter",
            "sugar_cube_2",
        )

        np.testing.assert_allclose(preferred_xy, np.array([1.08, 2.0], dtype=float))
        executor._surface_fixture_slot_positions.assert_not_called()

    def test_default_fixture_surface_preference_anchors_first_cube_near_plate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=["plate"])

        def object_pose(object_id):
            return (
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )

        def object_tokens(object_id):
            if object_id == "sugar_cube_1":
                return {"sugar", "cube"}
            return {object_id}

        executor._get_object_pose = MagicMock(side_effect=object_pose)
        executor._scene_object_tokens = MagicMock(side_effect=object_tokens)
        executor._require_fixture = MagicMock(return_value=SimpleNamespace(rot=0.0))
        executor._project_xy_onto_fixture = MagicMock(
            side_effect=lambda fixture_id, world_xy: np.asarray(world_xy, dtype=float)
        )
        executor._fixture_surface_blocker_positions = MagicMock()
        executor._surface_fixture_slot_positions = MagicMock()

        preferred_xy = executor._default_fixture_surface_preference(
            "dining_counter",
            "sugar_cube_1",
        )

        np.testing.assert_allclose(preferred_xy, np.array([1.08, 2.0], dtype=float))
        executor._surface_fixture_slot_positions.assert_not_called()

    def test_place_on_surface_uses_surface_slot_preference_and_target_pose(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_placement_semantics = MagicMock(return_value="surface")
        executor._default_fixture_surface_preference = MagicMock(
            return_value=np.array([0.25, 0.1], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value={"wall_back"})
        target_pos = np.array([0.4, 0.15, 0.9], dtype=float)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        executor._compute_object_target_pose = MagicMock(
            return_value=(target_pos, target_quat)
        )
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._held_objects = {0: "condiment"}
        executor._settle_scene = MagicMock()

        result = executor.place_on_surface("condiment", "counter_main", robot_idx=0)

        self.assertTrue(result.success)
        executor._default_fixture_surface_preference.assert_called_once_with(
            "counter_main",
            "condiment",
        )
        np.testing.assert_allclose(
            executor._compute_object_target_pose.call_args.kwargs["preferred_xy"],
            np.array([0.25, 0.1], dtype=float),
        )
        executor._set_object_pose.assert_called_once_with(
            "condiment",
            target_pos,
            target_quat,
        )

    def test_ignored_fixture_ids_for_placement_ignores_structural_runner_fixtures(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {
            "counter_main": object(),
            "box_left_group": object(),
            "fridge_housing_right_group": object(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {
                        "fixture_type": "counter_non_dining",
                        "parent_fixture": None,
                    }
                }
            }
        )

        ignored = executor._ignored_fixture_ids_for_placement("counter_main")

        self.assertIn("box_left_group", ignored)
        self.assertIn("fridge_housing_right_group", ignored)

    def test_ignored_fixture_ids_for_placement_ignores_box_structures_without_box_token(self):
        class Box:
            pass

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {
            "counter_main": object(),
            "counter_stack_main_group_base": Box(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {
                        "fixture_type": "counter_non_dining",
                        "parent_fixture": None,
                    }
                }
            }
        )

        ignored = executor._ignored_fixture_ids_for_placement("counter_main")

        self.assertIn("counter_stack_main_group_base", ignored)

    def test_safe_compute_object_target_pos_ignores_supported_descendants(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([0.2, -0.1, 0.9], dtype=float)
        )
        executor._iter_supported_descendants = MagicMock(
            return_value=["child_a", "child_b"]
        )
        executor._find_contained_objects = MagicMock(return_value=["child_c"])

        target = executor._safe_compute_object_target_pos(
            "counter_main",
            "bowl_main",
            ignored_object_ids={"existing"},
        )

        np.testing.assert_allclose(target, np.array([0.2, -0.1, 0.9], dtype=float))
        self.assertEqual(
            executor.runner._compute_object_target_pos.call_args.kwargs[
                "ignored_object_ids"
            ],
            {"existing", "child_a", "child_b", "child_c"},
        )

    def test_resolve_fixture_site_id_maps_unique_family_match(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        fixture = MagicMock()
        fixture.get_reset_regions.return_value = {
            "shelf_0": {
                "offset": (0.0, 0.0, 0.0),
                "size": (0.2, 0.2, 0.1),
            }
        }
        executor._require_fixture = MagicMock(return_value=fixture)

        self.assertEqual(
            executor._resolve_fixture_site_id("cab_main", "shelf_1"),
            "shelf_0",
        )

    def test_place_on_surface_rejects_receptacle_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(return_value=("cab_main", "shelf_1"))
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")

        with self.assertRaises(ValueError):
            executor.place_on_surface(
                "mug",
                target_id="cab_main",
                target_site_id="shelf_1",
                robot_idx=0,
            )

    def test_place_in_receptacle_rejects_surface_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={})
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor._resolve_support_target = MagicMock(
            return_value=("counter_main", None)
        )
        executor._fixture_placement_semantics = MagicMock(return_value="surface")

        with self.assertRaises(ValueError):
            executor.place_in_receptacle(
                "mug",
                target_id="counter_main",
                robot_idx=0,
            )

    def test_place_in_receptacle_accepts_support_site_identifier(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={})
        executor.runner = MagicMock()
        executor.runner._fixtures = {"blender_main": object()}
        executor._resolve_support_target = MagicMock(
            return_value=("blender_main", "bowl")
        )
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._preferred_xy_for_fixture_target = MagicMock(return_value=None)
        executor._compute_object_target_pose = MagicMock(
            return_value=(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()

        result = executor.place_in_receptacle(
            "mug",
            receptacle_id="blender_bowl",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._resolve_support_target.assert_any_call("blender_bowl", None)
        executor.runner._set_object_location.assert_called_once_with("mug", "blender_main")

    def test_place_in_receptacle_prefers_object_receptacle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"chicken": object(), "serving_tray": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._get_scene_object_location = MagicMock(return_value="table_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()

        result = executor.place_in_receptacle(
            "chicken",
            receptacle_id="serving_tray",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._resolve_support_target.assert_not_called()
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call(
                    "chicken",
                    "serving_tray",
                    relative_position=None,
                ),
                unittest.mock.call(
                    "chicken",
                    "serving_tray",
                    relative_position=None,
                ),
            ],
        )
        self.assertGreaterEqual(executor.runner._set_object_location.call_count, 2)

    def test_place_in_receptacle_keeps_explicit_bowl_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"bowl": object(), "tray": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._scene_object_tokens = MagicMock(return_value={"bowl"})
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._get_scene_object_location = MagicMock(return_value="counter_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_accepts_child_without_settle = MagicMock(return_value=False)
        executor._settle_scene = MagicMock()
        executor.place_next_to = MagicMock()

        result = executor.place_in_receptacle(
            "bowl",
            receptacle_id="tray",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor.place_next_to.assert_not_called()
        executor._place_on_object_center.assert_called_once_with(
            "bowl",
            "tray",
            relative_position=None,
        )

    def test_place_in_receptacle_skips_final_generic_settle_for_upright_insertions(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"straw": object(), "glass_cup": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._get_scene_object_location = MagicMock(return_value="dining_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=True)
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()

        result = executor.place_in_receptacle(
            "straw",
            receptacle_id="glass_cup",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._settle_and_reseat_supported_object.assert_called_once_with(
            "straw",
            "glass_cup",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_yogurt_prefers_upright_on_surface(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"yogurt"})
        executor._get_object_placement_metadata = MagicMock()

        self.assertTrue(executor._object_prefers_upright_on_surface("yogurt_1"))
        executor._get_object_placement_metadata.assert_not_called()

    def test_load_initial_state_skips_explicit_site_check_when_object_already_on_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"milk": {"location": "fridge_main"}},
                "object_placements": {"milk": "fridge_main"},
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value="shelf_1")
        executor._current_support_object = MagicMock(return_value=None)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge_main": object()}
        executor._resolve_support_target = MagicMock(return_value=("fridge_main", None))
        executor._require_explicit_site_if_needed = MagicMock(
            side_effect=AssertionError("should not be called")
        )

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "fridge_main", "held_object": None}},
                "objects": {"milk": {"location": "fridge_main", "object_type": "milk"}},
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_not_called()

    def test_load_initial_state_uses_target_site_id_with_fixture_location(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"mug": {"location": "cab_main"}},
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._incoming_fixture_site_preference = MagicMock(
            return_value=np.array([0.12, -0.04], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"cab_main": object()}
        executor._resolve_support_target = MagicMock(return_value=("cab_main", "shelf_0"))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "cab_main", "held_object": None}},
                "objects": {
                    "mug": {
                        "location": "cab_main",
                        "target_site_id": "shelf_0",
                        "object_type": "mug",
                    }
                },
                "fixtures": {"cab_main": {"fixture_type": "cabinet"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._resolve_support_target.assert_any_call("cab_main", "shelf_0")
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("mug", "cab_main"))
        np.testing.assert_allclose(
            kwargs["preferred_xy"],
            np.array([0.12, -0.04], dtype=float),
        )
        self.assertEqual(kwargs["target_site_id"], "shelf_0")
        self.assertFalse(kwargs["settle"])

    def test_load_initial_state_uses_default_site_when_partitioned_fixture_is_unspecified(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"baguette_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"baguette_main": {"location": "fridge_main"}},
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._incoming_fixture_site_preference = MagicMock(
            return_value=np.array([0.08, -0.03], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            return_value=(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("fridge_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_requires_explicit_site = MagicMock(return_value=True)
        executor._default_support_site_for_unspecified_fixture = MagicMock(
            return_value="shelf_1"
        )
        executor._normalize_target_site_id_for_placement = MagicMock(side_effect=lambda fixture_id, site_id: site_id)
        executor._repair_supported_children_vertical_gaps = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "fridge_main", "held_object": None}},
                "objects": {
                    "baguette_main": {
                        "location": "fridge_main",
                        "object_type": "baguette",
                    }
                },
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._default_support_site_for_unspecified_fixture.assert_called_once_with(
            "fridge_main",
            incoming_object_id="baguette_main",
        )
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("baguette_main", "fridge_main"))
        self.assertEqual(kwargs["target_site_id"], "shelf_1")

    def test_load_initial_state_preserves_support_object_with_contained_children(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"bowl_main": object(), "berry_1": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {"bowl_main": "counter_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "bowl_main": {
                        "location": "counter_main",
                        "object_type": "bowl",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_not_called()
        self.assertEqual(
            summary["placement_events"][0]["reason"],
            "already_on_target_fixture",
        )

    def test_load_initial_state_reseats_surface_object_when_pose_needs_reset(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"fork_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"fork_main": {"location": "counter_main"}},
                "object_placements": {"fork_main": "counter_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value="geom_0")
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=True)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            return_value=(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._normalize_target_site_id_for_placement = MagicMock(return_value=None)
        executor._fixture_requires_explicit_site = MagicMock(return_value=False)
        executor._default_fixture_surface_preference = MagicMock(
            return_value=np.array([0.25, 0.0], dtype=float)
        )

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "fork_main": {
                        "location": "counter_main",
                        "object_type": "fork",
                    }
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("fork_main", "counter_main"))
        np.testing.assert_allclose(
            kwargs["preferred_xy"],
            np.array([0.25, 0.0], dtype=float),
        )
        self.assertIsNone(kwargs["target_site_id"])
        self.assertFalse(kwargs["settle"])

    def test_current_support_object_detects_supported_item(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_parents = {}
        executor.env = SimpleNamespace(objects={"plate": object(), "sugar_cube_1": object()})
        executor._find_objects_on_support = MagicMock(
            side_effect=lambda support_object_id, exclude=None: ["sugar_cube_1"]
            if support_object_id == "plate"
            else []
        )

        self.assertEqual(executor._current_support_object("sugar_cube_1"), "plate")

    def test_load_initial_state_does_not_skip_object_resting_on_other_object(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"sugar_cube_1": {"location": "dining_main"}},
                "object_placements": {"sugar_cube_1": "dining_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value="plate_main")
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"dining_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("dining_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "dining_main", "held_object": None}},
                "objects": {
                    "sugar_cube_1": {
                        "location": "dining_main",
                        "object_type": "sugar_cube",
                    }
                },
                "fixtures": {"dining_main": {"fixture_type": "dining_counter"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_called_once_with(
            "sugar_cube_1",
            "dining_main",
            preferred_xy=None,
            target_site_id=None,
            settle=False,
        )

    def test_load_initial_state_reseats_supported_objects_after_settle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.24]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": True, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "bowl_main": {
                        "location": "counter_main",
                        "object_type": "bowl",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("berry_1", "bowl_main"),
                unittest.mock.call(
                    "berry_1",
                    "bowl_main",
                    relative_position=None,
                ),
            ],
        )
        self.assertEqual(executor._settle_scene.call_count, 1)
        executor._set_object_pose.assert_not_called()

    def test_load_initial_state_places_nested_supports_parent_first(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {
            "tray_main": object(),
            "bowl_main": object(),
            "berry_1": object(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "tray_main": {"location": "counter_main"},
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "tray_main": {
                        "location": "counter_main",
                        "object_type": "tray",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                    "bowl_main": {
                        "location": "tray_main",
                        "object_type": "bowl",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("bowl_main", "tray_main"),
                unittest.mock.call("berry_1", "bowl_main"),
            ],
        )

    def test_fixture_pose_needs_reset_for_sideways_upright_object_in_receptacle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._object_prefers_upright_on_surface = MagicMock(return_value=True)
        executor._object_is_upright = MagicMock(return_value=False)

        needs_reset = executor._fixture_pose_needs_reset(
            "condiment",
            "cab_main",
            site_id="shelf_0",
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

        self.assertTrue(needs_reset)

    def test_settle_and_reseat_supported_object_does_not_raise_settled_z(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.26]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("straw_main", "cup_main")

        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "straw_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.18]))
        np.testing.assert_allclose(pose_args[2], np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(executor._settle_scene.call_count, 2)

    def test_support_object_slot_positions_spread_two_items_across_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 0.4,
                "support_tokens": {"tray"},
                "is_concave": False,
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        slot_positions = executor._support_object_slot_positions(
            "tray_main",
            count=2,
        )

        self.assertEqual(len(slot_positions), 2)
        self.assertLess(float(slot_positions[0][0]), 0.0)
        self.assertGreater(float(slot_positions[1][0]), 0.0)
        self.assertAlmostEqual(abs(float(slot_positions[0][0])), 0.22)
        self.assertAlmostEqual(abs(float(slot_positions[1][0])), 0.22)
        self.assertAlmostEqual(float(slot_positions[0][1]), 0.0)
        self.assertAlmostEqual(float(slot_positions[1][1]), 0.0)

    def test_supported_children_should_balance_bulky_pair_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"tray_main": object(), "mug": object(), "kettle": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: {object_id}
        )

        self.assertTrue(
            executor._supported_children_should_balance_slots(
                "tray_main",
                ["mug", "kettle"],
            )
        )

    def test_first_bulky_tray_item_uses_reserved_pair_slot(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._iter_direct_supported_children = MagicMock(return_value=[])
        executor._find_objects_on_support = MagicMock(return_value=[])
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(return_value={"kettle"})
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 0.4,
                "support_tokens": {"tray"},
                "is_concave": False,
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        preferred = executor._incoming_support_slot_preference("tray_main", "kettle")

        np.testing.assert_allclose(preferred, np.array([-0.22, 0.0], dtype=float))

    def test_supported_children_does_not_balance_flat_meats_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"tray_main": object(), "meat1": object(), "meat2": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(return_value={"meat"})

        self.assertFalse(
            executor._supported_children_should_balance_slots(
                "tray_main",
                ["meat1", "meat2"],
            )
        )

    def test_supported_children_balances_hotdog_pair_on_plate_for_portion_task(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._task_name = "PortionHotDogs"
        executor.env = SimpleNamespace(
            objects={"plate1": object(), "hotdog_bun1": object(), "sausage1": object()}
        )
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: (
                {"bun", "hotdog"} if "bun" in object_id else {"sausage"}
            )
        )

        self.assertTrue(
            executor._supported_children_should_balance_slots(
                "plate1",
                ["hotdog_bun1", "sausage1"],
            )
        )

    def test_portion_hotdogs_second_plate_item_uses_inboard_opposite_slot(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._task_name = "PortionHotDogs"
        executor._iter_direct_supported_children = MagicMock(return_value=["hotdog_bun1"])
        executor._find_objects_on_support = MagicMock(return_value=["hotdog_bun1"])
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: (
                {"bun", "hotdog"} if "bun" in object_id else {"sausage"}
            )
        )
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 1.0,
                "support_tokens": {"plate"},
                "is_concave": False,
            }
        )
        executor._support_object_anchor_axes_xy = MagicMock(
            return_value=(
                np.array([1.0, 0.0], dtype=float),
                np.array([0.0, 1.0], dtype=float),
            )
        )
        executor._get_object_pose = MagicMock(
            return_value=(np.array([-0.16, -0.05, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )

        preferred = executor._incoming_support_slot_preference("plate1", "sausage1")

        self.assertIsNotNone(preferred)
        self.assertGreater(float(preferred[0]), 0.0)
        self.assertLess(float(preferred[0]), 0.2)
        self.assertGreater(float(preferred[1]), 0.0)

    def test_settle_and_reseat_supported_object_reapplies_upright_pose_after_tip(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=True)
        executor._object_is_upright = MagicMock(return_value=False)
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("straw_main", "cup_main")

        self.assertEqual(executor._place_on_object_center.call_count, 2)

    def test_settle_and_reseat_supported_object_lowers_flat_support_gap_after_settle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(side_effect=[0.02, 0.0])
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("bun_main", "plate_main")

        self.assertEqual(executor._settle_scene.call_count, 3)
        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "bun_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.16]))

    def test_settle_and_reseat_supported_object_does_not_raise_sunk_deep_support_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": True, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=-0.03)
        executor._iter_direct_supported_children = MagicMock(return_value=[])
        executor._supported_children_need_repack = MagicMock(return_value=False)
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.15]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("sausage_main", "bowl_main")

        executor._set_object_pose.assert_not_called()
        executor._repair_supported_children_vertical_gaps.assert_not_called()
        executor._settle_scene.assert_not_called()

    def test_repair_supported_children_vertical_gaps_lowers_floating_deep_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=0.03)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repair_supported_children_vertical_gaps(
            "bowl_main",
            ["bun_main"],
            support_geometry={"bottom_z": 0.0, "top_z": 0.2, "extent_z": 0.1},
        )

        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "bun_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.15]))

    def test_repair_supported_children_vertical_gaps_does_not_raise_sunken_deep_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=-0.03)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repair_supported_children_vertical_gaps(
            "bowl_main",
            ["bun_main"],
            support_geometry={"bottom_z": 0.0, "top_z": 0.2, "extent_z": 0.1},
        )

        executor._set_object_pose.assert_not_called()

    def test_find_objects_on_support_accepts_children_below_bowl_center(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(
            obj_body_id={"bowl_main": 0},
            objects={"bowl_main": object(), "bun_main": object()},
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos=np.array([[0.0, 0.0, 0.30]], dtype=float)
                )
            ),
        )
        executor._support_object_geometry = MagicMock(
            return_value={"top_z": 0.38, "bottom_z": 0.22, "extent_z": 0.16}
        )
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._get_object_pose = MagicMock(return_value=(np.array([0.01, 0.0, 0.24]), np.array([1.0, 0.0, 0.0, 0.0])))

        bowl = SimpleNamespace(horizontal_radius=0.10)
        bun = SimpleNamespace(horizontal_radius=0.04)
        executor.env.objects = {"bowl_main": bowl, "bun_main": bun}

        contained = executor._find_objects_on_support("bowl_main")

        self.assertEqual(contained, ["bun_main"])

    def test_repack_supported_children_uses_size_aware_object_center_placement(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"big_child": object(), "small_child": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._require_object = MagicMock(
            side_effect=lambda object_id: SimpleNamespace(
                horizontal_radius=0.08 if object_id == "big_child" else 0.04
            )
        )
        executor._place_on_object_center = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repack_supported_children(
            "bowl_main",
            ["small_child", "big_child"],
        )

        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("big_child", "bowl_main"),
                unittest.mock.call("small_child", "bowl_main"),
            ],
        )

    def test_support_object_placement_family_uses_shallow_receptacle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"plate"})

        family = executor._support_object_placement_family(
            "plate_main",
            {"is_concave": False},
        )

        self.assertEqual(family, "shallow_receptacle")

    def test_support_object_placement_family_uses_deep_receptacle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"bowl"})

        family = executor._support_object_placement_family(
            "bowl_main",
            {"is_concave": False},
        )

        self.assertEqual(family, "deep_receptacle")

    def test_object_prefers_upright_on_surface_for_bottle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"bottle"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.05, 0.05, 0.20]}
        )

        self.assertTrue(executor._object_prefers_upright_on_surface("bottle_main"))

    def test_object_prefers_upright_on_surface_rejects_flat_food_shapes(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"sausage"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.13, 0.19, 0.19]}
        )

        self.assertFalse(executor._object_prefers_upright_on_surface("sausage_main"))

    def test_object_prefers_upright_on_surface_rejects_baguette_and_cutlery_profiles(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"baguette"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.19198154, 0.06302114, 0.04465542]}
        )
        self.assertFalse(executor._object_prefers_upright_on_surface("baguette_main"))

        executor._scene_object_tokens = MagicMock(return_value={"fork"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.04250569, 0.25490584, 0.07001984]}
        )
        self.assertFalse(executor._object_prefers_upright_on_surface("fork_main"))

    def test_object_prefers_upright_on_surface_rejects_bun_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.12495396, 0.18144512, 0.14319583]}
        )

        self.assertFalse(executor._object_prefers_upright_on_surface("hotdog_bun1"))

    def test_object_has_directional_long_axis_for_bun_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.12495396, 0.18144512, 0.14319583]}
        )

        self.assertTrue(executor._object_has_directional_long_axis("hotdog_bun1"))

    def test_object_is_flat_top_up_on_surface_rejects_inverted_mesh_normal(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._flat_alignment_up_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )

        self.assertTrue(
            executor._object_is_flat_top_up_on_surface(
                "hotdog_bun1",
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        self.assertFalse(
            executor._object_is_flat_top_up_on_surface(
                "hotdog_bun1",
                np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
            )
        )

    def test_object_requires_top_up_flat_on_support_for_hotdog_items_on_plate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        self.assertTrue(
            executor._object_requires_top_up_flat_on_support(
                "hotdog_bun1",
                "plate1",
            )
        )

        executor._support_object_tokens = MagicMock(return_value={"tray"})
        self.assertFalse(
            executor._object_requires_top_up_flat_on_support(
                "hotdog_bun1",
                "tray1",
            )
        )

    def test_support_object_candidate_quaternions_uses_single_sign_for_top_up_items(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._object_requires_top_up_flat_on_support = MagicMock(return_value=True)
        executor._object_prefers_upright_on_surface = MagicMock(return_value=False)
        executor._support_object_tokens = MagicMock(return_value=set())
        executor._object_has_directional_long_axis = MagicMock(return_value=False)
        executor._thinnest_object_local_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )
        executor._quat_align_vectors = MagicMock(
            return_value=np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        )
        executor._object_is_flat_top_up_on_surface = MagicMock(return_value=True)
        executor._object_bbox_metadata_for_quat = MagicMock(
            return_value={"spans": np.array([0.2, 0.1, 0.05], dtype=float)}
        )

        quats = executor._support_object_candidate_quaternions(
            "hotdog_bun1",
            "plate1",
        )

        self.assertTrue(quats)
        self.assertEqual(executor._quat_align_vectors.call_count, 1)

    def test_flat_alignment_up_axis_flips_utensil_mesh_normal(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._thinnest_object_local_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )
        executor._scene_object_tokens = MagicMock(return_value={"fork"})

        np.testing.assert_allclose(
            executor._flat_alignment_up_axis("fork_main"),
            np.array([0.0, 0.0, -1.0], dtype=float),
        )

        executor._scene_object_tokens = MagicMock(return_value={"baguette"})
        np.testing.assert_allclose(
            executor._flat_alignment_up_axis("baguette_main"),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

    @patch("robocasa.utils.sim_tool_executor.get_fixture_aabb")
    def test_open_sliding_part_marks_swept_volume_and_moves_children(
        self,
        mock_get_fixture_aabb,
    ):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = MagicMock()
        executor._require_fixture = MagicMock(return_value=fixture)
        executor._robot_near_fixture = MagicMock(return_value=True)
        executor._move_robot_near_fixture_with_retries = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._resolve_joint_name = MagicMock(return_value="top_drawer_slidejoint")
        executor._sliding_joint_open_fraction = MagicMock(return_value=0.55)
        executor._set_named_joint = MagicMock()
        executor._iter_direct_supported_children = MagicMock(return_value=["cup_main"])
        executor._held_by_robot = MagicMock(return_value=None)
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([0.10, 0.0, 0.20], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.env = SimpleNamespace(objects={"cup_main": object()})
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor.runner._occupancy_grid = MagicMock()
        executor.runner._occupancy_grid.is_free = MagicMock(return_value=False)
        executor.runner._get_robot_position = MagicMock(
            return_value=np.array([0.25, 0.0, 0.0], dtype=float)
        )
        executor.runner._move_robot_near_fixture = MagicMock()

        mock_get_fixture_aabb.side_effect = [
            (np.array([0.0, -0.1]), np.array([0.2, 0.1])),
            (np.array([0.0, -0.1]), np.array([0.35, 0.1])),
        ]

        result = executor.open_sliding_part("drawer_main", "drawer", robot_idx=0)

        self.assertTrue(result.success)
        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "cup_main")
        np.testing.assert_allclose(
            pose_args[1],
            np.array([0.25, 0.0, 0.20], dtype=float),
        )
        executor.runner._occupancy_grid.mark_world_aabb_occupied.assert_called_once()
        executor.runner._move_robot_near_fixture.assert_called_once_with(
            0,
            "drawer_main",
            require_front=True,
        )

    def test_sliding_joint_open_fraction_keeps_drawers_partial(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)

        self.assertAlmostEqual(
            executor._sliding_joint_open_fraction("top_drawer_slidejoint"),
            0.55,
        )
        self.assertAlmostEqual(
            executor._sliding_joint_open_fraction(
                "toaster_oven_rack_joint",
                part_id="sliding",
            ),
            1.0,
        )

    def test_support_object_prefers_upright_for_straw_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"cup"})
        executor._scene_object_tokens = MagicMock(return_value={"straw"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.01, 0.01, 0.20]}
        )

        self.assertTrue(executor._support_object_prefers_upright("cup_main", "straw_main"))

    def test_supported_object_vertical_gap_uses_interior_floor_for_bowl_family(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._support_object_floor_clearance = MagicMock(return_value=0.01)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.95]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        fake_obj = SimpleNamespace(
            get_bbox_points=lambda trans, rot: np.array(
                [
                    [-0.02, -0.02, trans[2] - 0.05],
                    [0.02, 0.02, trans[2] + 0.05],
                ],
                dtype=float,
            )
        )
        executor._require_object = MagicMock(return_value=fake_obj)

        gap = executor._supported_object_vertical_gap(
            "bun_main",
            "bowl_main",
            support_geometry={
                "top_z": 1.0,
                "bottom_z": 0.9,
                "extent_z": 0.1,
                "placement_family": "deep_receptacle",
                "is_concave": False,
            },
        )

        self.assertAlmostEqual(gap, -0.024)

    def test_bowl_support_children_skip_settle_and_repack(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"bowl"})
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor.runner = SimpleNamespace(_set_object_location=MagicMock())
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()

        executor._settle_and_reseat_supported_object("bun_main", "bowl_main")

        executor._place_on_object_center.assert_called_once_with(
            "bun_main",
            "bowl_main",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_cutting_board_support_children_skip_settle_and_repack(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"cutting", "board"})
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor.runner = SimpleNamespace(_set_object_location=MagicMock())
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()

        executor._settle_and_reseat_supported_object(
            "sausage_main",
            "cutting_board_main",
        )

        executor._place_on_object_center.assert_called_once_with(
            "sausage_main",
            "cutting_board_main",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_resolve_fixture_site_id_accepts_dispenser_alias_for_bottom_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._get_fixture_reset_regions = MagicMock(return_value={"bottom": {}})
        executor.get_support_sites = MagicMock(return_value=["bottom"])

        site_id = executor._resolve_fixture_site_id(
            "coffee_machine_left_group",
            "coffee_machine_dispenser",
        )

        self.assertEqual(site_id, "bottom")

    def test_resolve_support_target_prefers_fixture_tokens_for_ambiguous_sites(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(
            _fixtures={
                "dishwasher_left_group": SimpleNamespace(name="dishwasher_left_group"),
                "toaster_oven_main_group": SimpleNamespace(name="toaster_oven_main_group"),
            }
        )
        executor._get_fixture_type_name = MagicMock(
            side_effect=lambda fixture_id: {
                "dishwasher_left_group": "dishwasher",
                "toaster_oven_main_group": "toaster_oven",
            }[fixture_id]
        )
        executor._resolve_fixture_site_id = MagicMock(return_value="rack")

        fixture_id, site_id = executor._resolve_support_target("toaster_oven_rack")

        self.assertEqual(fixture_id, "toaster_oven_main_group")
        self.assertEqual(site_id, "rack")

    def test_find_placeable_surface_near_stove_prefers_adjacent_counter(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "stove_1_main_group": {
                        "fixture_type": "stove",
                        "can_place_objects": True,
                        "position": [2.0, 0.0, 0.0],
                    },
                    "counter_1_main_group": {
                        "fixture_type": "counter",
                        "can_place_objects": True,
                        "position": [1.8, 0.0, 0.0],
                    },
                }
            }
        )

        support_id = executor._find_placeable_surface_near_fixture(
            "stove_1_main_group"
        )

        self.assertEqual(support_id, "counter_1_main_group")


class TestTrajectoryRunnerPlacementSites(unittest.TestCase):
    def test_explicit_site_disables_fallback_region(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = object()
        fixture = SimpleNamespace(
            get_reset_regions=lambda env: {
                "front_left": {"offset": (0.0, 0.0, 0.0), "size": (0.2, 0.2, 0.1)}
            },
            sample_reset_region=lambda env, min_size=None: {
                "offset": (1.0, 1.0, 0.0),
                "size": (0.1, 0.1, 0.1),
            },
        )

        regions = runner._get_fixture_reset_regions(
            fixture,
            site_id="rear_right",
        )

        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
