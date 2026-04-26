import unittest

from data_generation.task_level.pipeline.sim_normalization import (
    FixtureSimulationMetadata,
    _normalize_token_for_fixture,
)


class TestSimNormalization(unittest.TestCase):
    def test_normalize_timer_knob_to_time_control(self):
        fixture_metadata = {
            "toaster_oven": FixtureSimulationMetadata(
                symbol_id="toaster_oven",
                concrete_id="toaster_oven_main_group",
                fixture_type="toaster_oven",
                part_ids=(),
                control_ids=("time", "temperature", "function", "doneness"),
                support_site_ids=(),
            )
        }
        errors: list[str] = []

        resolved = _normalize_token_for_fixture(
            "timer_knob",
            fixture_id="toaster_oven",
            fixture_metadata_by_symbol=fixture_metadata,
            kind="control",
            location_aliases=None,
            errors=errors,
            context="trajectory.steps[0]",
        )

        self.assertEqual(resolved, "time")
        self.assertEqual(errors, [])

    def test_normalize_timer_knob_to_timer_control(self):
        fixture_metadata = {
            "oven": FixtureSimulationMetadata(
                symbol_id="oven",
                concrete_id="oven_main_group",
                fixture_type="oven",
                part_ids=(),
                control_ids=("timer", "temperature"),
                support_site_ids=(),
            )
        }
        errors: list[str] = []

        resolved = _normalize_token_for_fixture(
            "timer_knob",
            fixture_id="oven",
            fixture_metadata_by_symbol=fixture_metadata,
            kind="control",
            location_aliases=None,
            errors=errors,
            context="trajectory.steps[0]",
        )

        self.assertEqual(resolved, "timer")
        self.assertEqual(errors, [])
