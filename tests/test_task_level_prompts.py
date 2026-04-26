from __future__ import annotations

import unittest

from data_generation.task_level.pipeline.few_shot import FewShotExample
from data_generation.task_level.pipeline.prompts.spec_generation import (
    build_spec_generation_prompt,
)
from data_generation.task_level.tasks.shared.prompting import make_task_prompt_builder


class TaskLevelPromptTests(unittest.TestCase):
    def test_spec_generation_prompt_mentions_distinct_instances_and_sites(self):
        prompt = build_spec_generation_prompt(
            task_name="ExampleTask",
            source_python="class ExampleTask: pass",
            source_module="example.module",
            metadata={},
            examples=(
                FewShotExample(
                    task_name="Example",
                    source_python="class Example: pass",
                    spec_json='{"spec_version": 1}',
                ),
            ),
        )

        self.assertIn("Different symbolic objects must correspond to different physical", prompt)
        self.assertIn("concrete support-site id", prompt)
        self.assertIn("fallback to another", prompt)

    def test_spec_generation_prompt_detects_setup_bowls_proximity_pattern(self):
        prompt = build_spec_generation_prompt(
            task_name="SetupBowls",
        source_python="""
class SetupBowls:
    def _check_success(self):
        assigned_stools = set()
        for bowl_name, bowl_pos in bowl_positions.items():
            closest_stool_idx = None
            for idx, stool_pos in enumerate(self.stool_positions):
                bowl_x, bowl_y = OU.transform_global_to_local(0, 0, 0)
                stool_x, stool_y = OU.transform_global_to_local(0, 0, 0)
                x_dist = abs(bowl_x - stool_x)
                y_dist = abs(bowl_y - stool_y)
            bowl_on_counter = OU.check_obj_any_counter_contact(self, bowl_name)
""",
            source_module="robocasa.environments.kitchen.composite.setting_the_table.setup_bowls",
            metadata={
                "obj_configs": [
                    {"name": "bowl1", "obj_groups": "bowl"},
                    {"name": "bowl2", "obj_groups": "bowl"},
                ],
                "fixture_refs": [
                    {"name": "cabinet", "fixture_type": "cabinet"},
                    {"name": "stool1", "fixture_type": "stool"},
                    {"name": "stool2", "fixture_type": "stool"},
                    {"name": "dining_counter", "fixture_type": "counter"},
                ],
            },
            examples=(),
        )

        self.assertIn("Detected task pattern", prompt)
        self.assertIn("unique XY proximity assignment", prompt)
        self.assertIn("Concrete inferred pattern for SetupBowls", prompt)
        self.assertIn("use `place_next_to`", prompt)
        self.assertIn("reference_fixture_id` set to a distinct stool", prompt)
        self.assertIn("Do not add final `object_at_location` goals", prompt)

    def test_spec_generation_prompt_detects_generic_unique_assignment_pattern(self):
        prompt = build_spec_generation_prompt(
            task_name="SetupWineGlasses",
            source_python="""
class SetupWineGlasses:
    def _check_success(self):
        assigned_plates = set()
        for wine_glass_name in wine_glass_names:
            best_candidate = None
            best_score = float("inf")
            for idx, plate_name in enumerate(plate_names):
                plate_x, plate_y = OU.transform_global_to_local(0, 0, 0)
                wine_glass_x, wine_glass_y = OU.transform_global_to_local(0, 0, 0)
                x_dist = abs(wine_glass_x - plate_x)
                y_dist = abs(wine_glass_y - plate_y)
            wine_glass_on_table = OU.check_obj_fixture_contact(self, wine_glass_name, self.dining_counter)
""",
            source_module="example.setup_wine_glasses",
            metadata={
                "obj_configs": [
                    {"name": "wine_glass1", "obj_groups": "wine_glass"},
                    {"name": "wine_glass2", "obj_groups": "wine_glass"},
                    {"name": "plate1", "obj_groups": "plate"},
                    {"name": "plate2", "obj_groups": "plate"},
                ],
                "fixture_refs": [
                    {"name": "dining_counter", "fixture_type": "counter"},
                ],
            },
            examples=(),
        )

        self.assertIn("unique XY proximity assignment", prompt)
        self.assertIn("`reference_object_id` for movable anchors", prompt)
        self.assertIn("Candidate movable object symbols", prompt)

    def test_spec_generation_prompt_detects_generic_simple_proximity_pattern(self):
        prompt = build_spec_generation_prompt(
            task_name="SetupButterPlate",
            source_python="""
class SetupButterPlate:
    def _check_success(self):
        xy_dist = np.linalg.norm(knife_pos[:2] - plate_pos[:2])
        knife_near_plate = xy_dist <= 0.3
        knife_on_counter = OU.check_obj_any_counter_contact(self, "butter_knife")
""",
            source_module="example.setup_butter_plate",
            metadata={
                "obj_configs": [
                    {"name": "butter_knife", "obj_groups": "knife"},
                    {"name": "plate", "obj_groups": "plate"},
                ],
                "fixture_refs": [
                    {"name": "counter", "fixture_type": "counter"},
                ],
            },
            examples=(),
        )

        self.assertIn("object proximity plus support/contact constraint", prompt)
        self.assertIn("use `place_next_to`", prompt)
        self.assertIn("Do not represent proximity by leaving the object at its source", prompt)

    def test_spec_generation_prompt_detects_setup_bowls_with_generic_stool_metadata(self):
        prompt = build_spec_generation_prompt(
            task_name="setupbowls",
            source_python="class SetupBowls: pass",
            source_module="robocasa.environments.kitchen.composite.setting_the_table.setup_bowls",
            metadata={
                "obj_configs": [
                    {"name": "bowl1", "obj_groups": "bowl"},
                    {"name": "bowl2", "obj_groups": "bowl"},
                ],
                "fixture_refs": [
                    {"name": "cabinet", "fixture_type": "cabinet"},
                    {"name": "stool", "fixture_type": "stool"},
                ],
            },
            examples=(),
        )

        self.assertIn("Passive anchors: stool1, stool2.", prompt)
        self.assertIn("Allowed tools should include `place_next_to`", prompt)

    def test_task_prompt_builder_mentions_exact_site_and_distinct_objects(self):
        prompt_builder = make_task_prompt_builder(
            composite_task="ExampleTask",
            task_goal="Stage two bowls on the stove.",
            initial_state={
                "agents": {
                    "agent_0": {"location": "counter_main", "held_object": None},
                    "agent_1": {"location": "counter_main", "held_object": None},
                },
                "objects": {
                    "bowl1": {"object_type": "bowl", "location": "counter_main"},
                    "bowl2": {"object_type": "bowl", "location": "counter_main"},
                },
                "fixtures": {
                    "counter_main": {"fixture_type": "counter_non_dining"},
                    "stove_main": {
                        "fixture_type": "stove",
                        "support_sites": {
                            "front_left_burner": {"site_type": "support"},
                            "front_right_burner": {"site_type": "support"},
                        },
                    },
                },
            },
            allowed_tool_specs={
                "communicate": {"tool_args": ["to", "message"]},
                "navigate_to_fixture": {"tool_args": ["fixture_id"]},
                "place_on_surface": {
                    "tool_args": ["object_id"],
                    "tool_arg_any_of": [["target_id", "support_id"]],
                    "optional_tool_args": ["target_site_id", "relative_position"],
                },
            },
            non_communicate_tool_names=("navigate_to_fixture", "place_on_surface"),
        )

        prompt = prompt_builder("variation-1")

        self.assertIn("distinct physical instances", prompt)
        self.assertIn("target_site_id", prompt)
        self.assertIn("Do not silently switch to a different site", prompt)

    def test_task_prompt_builder_mentions_canonical_placement_arg_styles_and_effect_constraints(self):
        prompt_builder = make_task_prompt_builder(
            composite_task="CoffeeTask",
            task_goal="Place the mug under the coffee machine and start it.",
            initial_state={
                "agents": {
                    "agent_0": {"location": "counter", "held_object": None},
                    "agent_1": {"location": "counter", "held_object": None},
                },
                "objects": {
                    "mug": {"object_type": "mug", "location": "counter"},
                },
                "fixtures": {
                    "counter": {"fixture_type": "counter"},
                    "coffee_machine": {
                        "fixture_type": "coffee_machine",
                        "support_sites": {
                            "dispenser_site": {"site_type": "support"},
                        },
                    },
                },
                "machine_state": {
                    "coffee_machine": {"turned_on": False},
                },
            },
            allowed_tool_specs={
                "communicate": {"tool_args": ["to", "message"]},
                "navigate_to_fixture": {"tool_args": ["fixture_id"]},
                "place_under": {
                    "tool_args": ["object_id"],
                    "tool_arg_any_of": [["target_id", "reference_fixture_id"]],
                    "optional_tool_args": [
                        "target_id",
                        "reference_fixture_id",
                        "target_site_id",
                    ],
                },
                "press_button": {"tool_args": ["target_id", "control_id"]},
            },
            non_communicate_tool_names=("navigate_to_fixture", "place_under", "press_button"),
            task_effects=(
                {
                    "kind": "set_machine_flag_on_action",
                    "tool": "press_button",
                    "required_object_locations": [
                        {"object_id": "mug", "location": "dispenser_site"},
                    ],
                },
            ),
        )

        prompt = prompt_builder("variation-2")

        self.assertIn(
            "For place_under, prefer reference_fixture_id for the fixture",
            prompt,
        )
        self.assertIn(
            "Only use press_button after mug is already at dispenser_site.",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
