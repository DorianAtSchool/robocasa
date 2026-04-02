from __future__ import annotations

import unittest

from data_generation.task_level.grounding_specs import (
    build_grounding_map_for_task,
    build_resolved_grounding_payload,
    resolve_grounding_map,
)
from data_generation.task_level.tasks import get_task_definition
from data_generation.task_level.tasks.specs import load_all_task_specs, load_task_spec
from data_generation.task_level.tasks.shared.types import TaskInstance

HOT_DOG_SETUP_INITIAL_STATE = load_task_spec("HotDogSetup").initial_state
PREPARE_COFFEE_INITIAL_STATE = load_task_spec("PrepareCoffee").initial_state
PREPARE_SANDWICH_STATION_INITIAL_STATE = load_task_spec(
    "PrepareSandwichStation"
).initial_state


class TaskLevelGroundingTests(unittest.TestCase):
    def test_every_task_spec_example_validates_and_persists_grounding(self):
        generation_usage = {"total_cost_usd": 0.0}

        for task_spec in load_all_task_specs():
            task_definition = get_task_definition(task_spec.composite_task)
            self.assertIsNotNone(task_definition)
            validator = task_definition.validator_factory(
                TaskInstance(initial_state=task_spec.initial_state)
            )
            validation = validator.validate(task_spec.example_trajectory)
            trajectory_record = task_definition.build_trajectory_record(
                task_spec.example_trajectory,
                validation,
                "traj_spec_example",
                generation_usage,
                TaskInstance(initial_state=task_spec.initial_state),
            )

            grounding_map = trajectory_record.get("grounding_map")
            self.assertIsInstance(grounding_map, dict)
            self.assertEqual(grounding_map["map_kind"], "scene_agnostic")
            self.assertEqual(
                grounding_map["composite_task"],
                task_spec.composite_task,
            )
            self.assertTrue(grounding_map["symbols"])

    def test_task_record_builders_persist_scene_agnostic_grounding_map(self):
        candidate = {"steps": []}
        validation = {"is_valid": True}
        generation_usage = {"total_cost_usd": 0.0}

        trajectory_records = [
            get_task_definition("HotDogSetup").build_trajectory_record(
                candidate,
                validation,
                "traj_000001",
                generation_usage,
                TaskInstance(initial_state=HOT_DOG_SETUP_INITIAL_STATE),
            ),
            get_task_definition("PrepareCoffee").build_trajectory_record(
                candidate,
                validation,
                "traj_000002",
                generation_usage,
                TaskInstance(initial_state=PREPARE_COFFEE_INITIAL_STATE),
            ),
            get_task_definition("PrepareSandwichStation").build_trajectory_record(
                candidate,
                validation,
                "traj_000003",
                generation_usage,
                TaskInstance(initial_state=PREPARE_SANDWICH_STATION_INITIAL_STATE),
            ),
        ]

        for trajectory_record in trajectory_records:
            grounding_map = trajectory_record.get("grounding_map")
            self.assertIsInstance(grounding_map, dict)
            self.assertEqual(grounding_map["map_kind"], "scene_agnostic")
            self.assertEqual(
                grounding_map["composite_task"],
                trajectory_record["composite_task"],
            )
            self.assertTrue(grounding_map["symbols"])

    def test_symbolic_task_record_omits_raw_scene_metadata(self):
        candidate = {"steps": []}
        validation = {"is_valid": True}
        generation_usage = {"total_cost_usd": 0.0}
        task_instance = TaskInstance(initial_state=PREPARE_COFFEE_INITIAL_STATE)

        trajectory_record = get_task_definition("PrepareCoffee").build_trajectory_record(
            candidate,
            validation,
            "traj_000010",
            generation_usage,
            task_instance,
        )

        self.assertIn("grounding_map", trajectory_record)
        self.assertNotIn("grounding_mode", trajectory_record)
        self.assertNotIn("scene_config", trajectory_record)
        self.assertNotIn("scene_summary", trajectory_record)
        self.assertNotIn("layout", trajectory_record)
        self.assertNotIn("style", trajectory_record)
        self.assertNotIn("seed", trajectory_record)
        self.assertNotIn("robots", trajectory_record)

    def test_hot_dog_grounding_map_resolves_across_scene_instances(self):
        grounding_map = build_grounding_map_for_task(
            "HotDogSetup",
            HOT_DOG_SETUP_INITIAL_STATE,
        )
        first_scene = {
            "fixtures": {
                "counter_2_main_group": {
                    "fixture_type": "counter_non_dining",
                    "position": [3.0, -1.0, 0.0],
                    "can_place_objects": True,
                },
                "fridge_right_group": {
                    "fixture_type": "fridge",
                    "position": [4.0, -1.0, 0.0],
                    "can_place_objects": True,
                },
                "stack_3_main_group_2": {
                    "fixture_type": "cabinet",
                    "position": [2.8, -1.0, 0.0],
                    "can_place_objects": True,
                },
                "dining_dining_group": {
                    "fixture_type": "dining_counter",
                    "position": [-1.1, -2.0, 0.0],
                    "can_place_objects": True,
                },
            },
            "objects": {
                "hotdog_bun": {
                    "object_type": "hotdog_bun",
                    "location": "counter_2_main_group",
                },
                "sausage": {
                    "object_type": "sausage",
                    "location": "fridge_right_group",
                },
                "condiment": {
                    "object_type": "condiment_bottle",
                    "location": "stack_3_main_group_2",
                },
                "plate": {
                    "object_type": "plate",
                    "location": "dining_dining_group",
                },
            },
        }
        second_scene = {
            "fixtures": {
                "counter_alpha": {
                    "fixture_type": "counter_non_dining",
                    "position": [1.0, 0.0, 0.0],
                    "can_place_objects": True,
                },
                "fridge_main_group": {
                    "fixture_type": "fridge",
                    "position": [4.0, 0.0, 0.0],
                    "can_place_objects": True,
                },
                "cab_3_main_group": {
                    "fixture_type": "cabinet",
                    "position": [2.0, 1.5, 0.0],
                    "can_place_objects": True,
                },
                "island_island_group": {
                    "fixture_type": "island",
                    "position": [0.0, 3.0, 0.0],
                    "can_place_objects": True,
                },
            },
            "objects": {
                "hotdog_bun_live": {
                    "object_type": "hotdog_bun",
                    "location": "counter_alpha",
                },
                "sausage_live": {
                    "object_type": "sausage",
                    "location": "fridge_main_group",
                },
                "condiment_live": {
                    "object_type": "condiment_bottle",
                    "location": "cab_3_main_group",
                },
                "plate_live": {
                    "object_type": "plate",
                    "location": "island_island_group",
                },
            },
        }

        first_resolution = resolve_grounding_map(grounding_map, first_scene)
        second_resolution = resolve_grounding_map(grounding_map, second_scene)

        self.assertTrue(first_resolution["is_complete"])
        self.assertEqual(
            first_resolution["resolved_symbols"]["bun_source_fixture"]["resolved_id"],
            "counter_2_main_group",
        )
        self.assertEqual(
            first_resolution["resolved_symbols"]["serving_surface"]["resolved_id"],
            "dining_dining_group",
        )
        self.assertTrue(second_resolution["is_complete"])
        self.assertEqual(
            second_resolution["resolved_symbols"]["bun_source_fixture"]["resolved_id"],
            "counter_alpha",
        )
        self.assertEqual(
            second_resolution["resolved_symbols"]["serving_surface"]["resolved_id"],
            "island_island_group",
        )

    def test_prepare_coffee_payload_builds_grounding_map_for_legacy_trajectory(self):
        legacy_initial_state = {
            "agents": {
                "agent_0": {"location": "cabinet_1", "held_object": None},
                "agent_1": {"location": "cabinet_1", "held_object": None},
            },
            "objects": {
                "mug_1": {
                    "object_type": "mug",
                    "location": "cabinet_1",
                }
            },
            "fixtures": {
                "cabinet_1": {
                    "fixture_type": "cabinet",
                    "parts": {
                        "door": {
                            "part_type": "hinged_part",
                            "state": "closed",
                        }
                    },
                },
                "counter_1": {"fixture_type": "counter"},
                "coffee_machine_1": {
                    "fixture_type": "coffee_machine",
                    "controls": {
                        "start_button": {
                            "control_type": "button",
                        }
                    },
                },
            },
            "machine_state": {
                "coffee_machine_1": {
                    "started": False,
                    "dispenser_id": "coffee_machine_dispenser",
                }
            },
        }
        legacy_trajectory = {
            "trajectory_id": "traj_000123",
            "composite_task": "PrepareCoffee",
            "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
            "initial_state": legacy_initial_state,
            "steps": [],
            "validation": {"is_valid": True},
            "generation_usage": {"total_cost_usd": 0.0},
        }
        scene = {
            "fixtures": {
                "cab_main": {
                    "fixture_type": "cabinet",
                    "position": [0.0, 0.0, 0.0],
                    "can_place_objects": True,
                },
                "coffee_machine_left_group": {
                    "fixture_type": "coffee_machine",
                    "position": [1.0, 1.0, 0.0],
                    "can_place_objects": True,
                },
                "counter_near_machine": {
                    "fixture_type": "counter_non_dining",
                    "position": [1.2, 1.0, 0.0],
                    "can_place_objects": True,
                },
                "counter_far": {
                    "fixture_type": "counter_non_dining",
                    "position": [6.0, 6.0, 0.0],
                    "can_place_objects": True,
                },
            },
            "objects": {
                "mug_main": {
                    "object_type": "mug",
                    "location": "cab_main",
                }
            },
        }

        payload = build_resolved_grounding_payload(
            legacy_trajectory,
            scene,
            scene_config={"layout": 11, "style": 34, "seed": 42, "robots": 2},
        )

        self.assertEqual(payload["grounding_map"]["composite_task"], "PrepareCoffee")
        self.assertIn("mug", payload["grounding_map"]["symbols"])
        self.assertTrue(payload["resolved_grounding"]["is_complete"])
        self.assertEqual(
            payload["resolved_grounding"]["resolved_symbols"]["coffee_machine"][
                "resolved_id"
            ],
            "coffee_machine_left_group",
        )
        self.assertEqual(
            payload["resolved_grounding"]["resolved_symbols"]["staging_surface"][
                "resolved_id"
            ],
            "counter_near_machine",
        )


if __name__ == "__main__":
    unittest.main()
