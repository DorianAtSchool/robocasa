import json
import os
from pathlib import Path
import tempfile
import unittest

import robosuite.utils.transform_utils as T

os.environ.setdefault("MUJOCO_GL", "osmesa")

from robocasa.scripts.generate_llm_task_descriptions import (  # noqa: E402
    build_compact_task_context,
    render_llm_prompt,
)
from robocasa.utils.sim_tool_executor import SimToolExecutor  # noqa: E402
from robocasa.utils.sim_tool_specs import SIM_TOOL_SPEC_BY_NAME  # noqa: E402


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
            gl_backend="osmesa",
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
            gl_backend="osmesa",
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
            gl_backend="osmesa",
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
            gl_backend="osmesa",
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
            gl_backend="osmesa",
        )
        executor_b = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
            gl_backend="osmesa",
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
            gl_backend="osmesa",
        )
        executor_object = SimToolExecutor(
            task_name="HotDogSetup",
            robots=2,
            layout=56,
            style=42,
            seed=42,
            render_width=160,
            render_height=128,
            gl_backend="osmesa",
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


if __name__ == "__main__":
    unittest.main()
