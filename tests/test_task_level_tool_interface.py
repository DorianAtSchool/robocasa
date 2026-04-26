from __future__ import annotations

import unittest

from data_generation.task_level.grounding_specs import (
    _resolve_nearest_placeable_surface,
)
from data_generation.task_level.subatomic_tool_specs import (
    TASK_LEVEL_ALLOWED_TOOL_SPECS,
)
from data_generation.task_level.tasks.shared.schema import build_task_response_schema
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


class TaskLevelToolInterfaceTests(unittest.TestCase):
    def test_place_on_surface_spec_includes_optional_site_and_relative_args(self):
        spec = TASK_LEVEL_ALLOWED_TOOL_SPECS["place_on_surface"]

        self.assertEqual(["object_id"], spec["tool_args"])
        self.assertEqual([["target_id", "support_id"]], spec["tool_arg_any_of"])
        self.assertEqual(
            ["target_id", "support_id", "target_site_id", "relative_position"],
            spec["optional_tool_args"],
        )

    def test_place_next_to_spec_includes_all_reference_variants(self):
        spec = TASK_LEVEL_ALLOWED_TOOL_SPECS["place_next_to"]

        self.assertEqual(["object_id"], spec["tool_args"])
        self.assertEqual(
            [["reference_id", "reference_object_id", "reference_fixture_id"]],
            spec["tool_arg_any_of"],
        )
        self.assertEqual(
            [
                "reference_id",
                "reference_object_id",
                "reference_fixture_id",
                "target_site_id",
                "relative_position",
            ],
            spec["optional_tool_args"],
        )

    def test_response_schema_includes_optional_tool_args(self):
        schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs={
                "place_on_surface": TASK_LEVEL_ALLOWED_TOOL_SPECS["place_on_surface"],
            },
            min_steps=1,
        )

        args_properties = schema["properties"]["steps"]["items"]["properties"]["args"][
            "properties"
        ]
        self.assertIn("target_site_id", args_properties)
        self.assertIn("relative_position", args_properties)

    def test_nearest_placeable_surface_prefers_parent_fixture(self):
        scene = {
            "fixtures": {
                "blender": {
                    "position": [1.0, 1.0, 0.0],
                    "fixture_type": "blender",
                    "can_place_objects": False,
                    "parent_fixture": "prep_counter",
                },
                "prep_counter": {
                    "position": [1.0, 1.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
                "dining_counter": {
                    "position": [1.1, 1.0, 0.0],
                    "fixture_type": "dining_counter",
                    "can_place_objects": True,
                },
            }
        }
        resolved = {"blender_fixture": {"resolved_id": "blender"}}
        spec = {
            "anchor_fixture_symbol": "blender_fixture",
            "preferred_fixture_types": ["counter_non_dining", "dining_counter"],
        }

        result = _resolve_nearest_placeable_surface(
            "support_surface",
            spec,
            scene,
            resolved,
        )

        self.assertEqual("prep_counter", result["resolved_id"])
        self.assertEqual(1.0, result["confidence"])

    def test_nearest_placeable_surface_rejects_ambiguous_close_candidates(self):
        scene = {
            "fixtures": {
                "coffee_machine": {
                    "position": [0.0, 0.0, 0.0],
                    "fixture_type": "coffee_machine",
                    "can_place_objects": False,
                },
                "counter_a": {
                    "position": [0.2, 0.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
                "counter_b": {
                    "position": [0.28, 0.0, 0.0],
                    "fixture_type": "counter_non_dining",
                    "can_place_objects": True,
                },
            }
        }
        resolved = {"coffee_fixture": {"resolved_id": "coffee_machine"}}
        spec = {
            "anchor_fixture_symbol": "coffee_fixture",
            "preferred_fixture_types": ["counter_non_dining"],
        }

        result = _resolve_nearest_placeable_surface(
            "support_surface",
            spec,
            scene,
            resolved,
        )

        self.assertIsNone(result["resolved_id"])
        self.assertEqual(["counter_a", "counter_b"], result["candidates"])

    def test_counter_family_excludes_dining_counter_and_island(self):
        self.assertEqual(
            {"counter", "counter_non_dining", "counter_non_corner"},
            TrajectoryAdapter._FIXTURE_TYPE_FAMILIES["counter"],
        )


if __name__ == "__main__":
    unittest.main()
