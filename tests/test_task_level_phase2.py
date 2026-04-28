from __future__ import annotations

import unittest

from data_generation.task_level.pipeline.phase2 import (
    _collect_goal_consistency_errors,
    _validate_fixture_references,
)
from data_generation.task_level.pipeline.sim_normalization import (
    FixtureSimulationMetadata,
)


class Phase2FixtureReferenceValidationTests(unittest.TestCase):
    def test_goal_consistency_rejects_source_goal_contradicted_by_trajectory(self):
        payload = {
            "initial_state": {
                "objects": {
                    "bowl1": {"object_type": "bowl", "location": "cabinet"},
                },
                "fixtures": {
                    "cabinet": {"fixture_type": "cabinet"},
                    "counter": {"fixture_type": "counter"},
                },
            },
            "goal_conditions": [
                {
                    "kind": "object_at_location",
                    "object_id": "bowl1",
                    "location": "cabinet",
                }
            ],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 2,
                        "tool": "place_on_surface",
                        "args": {"object_id": "bowl1", "target_id": "counter"},
                    }
                ]
            },
        }

        errors = _collect_goal_consistency_errors(payload)

        self.assertIn("keeps 'bowl1' at its initial location 'cabinet'", errors[0])
        self.assertIn("later places it at 'counter'", errors[0])

    def test_goal_consistency_rejects_proximity_flag_set_by_surface_placement(self):
        payload = {
            "initial_state": {
                "objects": {
                    "bowl1": {"object_type": "bowl", "location": "cabinet"},
                },
                "fixtures": {
                    "cabinet": {"fixture_type": "cabinet"},
                    "dining_counter": {"fixture_type": "counter"},
                    "stool1": {"fixture_type": "stool"},
                },
            },
            "goal_conditions": [
                {
                    "kind": "machine_flag_true",
                    "machine_path": ["bowl1_placed_at_stool"],
                }
            ],
            "task_effects": [
                {
                    "kind": "set_machine_flag_on_action",
                    "tool": "place_on_surface",
                    "args": {"object_id": "bowl1", "target_id": "dining_counter"},
                    "machine_path": ["bowl1_placed_at_stool"],
                    "value": True,
                }
            ],
            "example_trajectory": {"steps": []},
        }

        errors = _collect_goal_consistency_errors(payload)

        self.assertIn("proximity machine flag", errors[0])
        self.assertIn("must be set by place_next_to", errors[0])

    def test_validate_fixture_references_rejects_known_bad_names(self):
        payload = {
            "initial_state": {
                "objects": {
                    "mug": {"object_type": "mug", "location": "counter"},
                    "pan": {"object_type": "pan", "location": "mystery_place"},
                },
                "fixtures": {
                    "counter": {"fixture_type": "counter_non_dining"},
                    "cabinet": {"fixture_type": "cabinet"},
                    "stove": {"fixture_type": "stove"},
                },
            },
            "task_effects": [
                {
                    "tool": "place_under",
                    "args": {
                        "object_id": "mug",
                        "reference_fixture_id": "cabinet",
                        "control_id": "start_button",
                    },
                }
            ],
            "example_trajectory": {
                "steps": [
                    {
                        "tool": "open_hinged_part",
                        "args": {"target_id": "cabinet", "part_id": "left_door"},
                    },
                    {
                        "tool": "place_on_surface",
                        "args": {
                            "object_id": "mug",
                            "target_id": "stove",
                            "target_site_id": "stove_burner",
                        },
                    },
                    {
                        "tool": "place_under",
                        "args": {
                            "object_id": "mug",
                            "reference_fixture_id": "cabinet",
                            "control_id": "unexpected_control",
                        },
                    },
                ]
            },
        }
        fixture_metadata = {
            "cabinet": FixtureSimulationMetadata(
                symbol_id="cabinet",
                concrete_id="cabinet_main",
                fixture_type="cabinet",
                part_ids=("door",),
                control_ids=(),
                support_site_ids=("shelf_0",),
            ),
            "stove": FixtureSimulationMetadata(
                symbol_id="stove",
                concrete_id="stove_main",
                fixture_type="stove",
                part_ids=(),
                control_ids=("knob_front_left",),
                support_site_ids=(
                    "front_left",
                    "front_right",
                    "rear_left",
                    "rear_right",
                ),
            ),
        }

        errors = _validate_fixture_references(payload, fixture_metadata)
        error_text = "\n".join(errors)

        self.assertIn("location 'mystery_place'", error_text)
        self.assertIn("part_id 'left_door'", error_text)
        self.assertIn("target_site_id 'stove_burner'", error_text)
        self.assertIn("place_under does not declare args ['control_id']", error_text)

    def test_validate_fixture_references_allows_declared_support_site_locations(self):
        payload = {
            "initial_state": {
                "objects": {
                    "pan": {
                        "object_type": "pan",
                        "location": "rear_left",
                    }
                },
                "fixtures": {
                    "stove": {
                        "fixture_type": "stove",
                        "support_sites": {
                            "rear_left": {"site_type": "support"},
                        },
                    }
                },
            },
            "task_effects": [],
            "example_trajectory": {"steps": []},
        }
        fixture_metadata = {
            "stove": FixtureSimulationMetadata(
                symbol_id="stove",
                concrete_id="stove_main",
                fixture_type="stove",
                part_ids=(),
                control_ids=("knob_rear_left",),
                support_site_ids=("rear_left",),
            )
        }

        self.assertEqual(_validate_fixture_references(payload, fixture_metadata), [])


if __name__ == "__main__":
    unittest.main()
