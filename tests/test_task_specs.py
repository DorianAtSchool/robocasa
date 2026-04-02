from __future__ import annotations

import unittest

from data_generation.task_level.tasks import get_task_definition
from data_generation.task_level.tasks.specs import load_all_task_specs, load_task_spec


class TaskSpecTests(unittest.TestCase):
    def test_load_all_task_specs_returns_current_specs(self):
        spec_names = {spec.composite_task for spec in load_all_task_specs()}
        self.assertTrue(
            {"HotDogSetup", "PrepareCoffee", "PrepareSandwichStation"}.issubset(
                spec_names
            )
        )

    def test_hot_dog_setup_spec_builds_runtime_definition(self):
        spec = load_task_spec("HotDogSetup")
        task_definition = get_task_definition("HotDogSetup")

        self.assertIsNotNone(task_definition)
        self.assertEqual(task_definition.composite_task, spec.composite_task)
        self.assertEqual(
            spec.preflight_token_estimate.prompt_tokens,
            task_definition.preflight_token_estimate.prompt_tokens,
        )
        self.assertEqual(
            spec.preflight_token_estimate.output_tokens,
            task_definition.preflight_token_estimate.output_tokens,
        )
        self.assertTrue(spec.grounding["symbols"])
        self.assertTrue(spec.example_trajectory["steps"])

    def test_prepare_coffee_spec_builds_runtime_definition(self):
        spec = load_task_spec("PrepareCoffee")
        task_definition = get_task_definition("PrepareCoffee")

        self.assertIsNotNone(task_definition)
        self.assertEqual(task_definition.composite_task, spec.composite_task)
        self.assertEqual(
            spec.preflight_token_estimate.prompt_tokens,
            task_definition.preflight_token_estimate.prompt_tokens,
        )
        self.assertEqual(
            spec.preflight_token_estimate.output_tokens,
            task_definition.preflight_token_estimate.output_tokens,
        )
        self.assertTrue(spec.grounding["symbols"])
        self.assertTrue(spec.example_trajectory["steps"])

    def test_prepare_sandwich_station_spec_builds_runtime_definition(self):
        spec = load_task_spec("PrepareSandwichStation")
        task_definition = get_task_definition("PrepareSandwichStation")

        self.assertIsNotNone(task_definition)
        self.assertEqual(task_definition.composite_task, spec.composite_task)
        self.assertEqual(
            spec.preflight_token_estimate.prompt_tokens,
            task_definition.preflight_token_estimate.prompt_tokens,
        )
        self.assertEqual(
            spec.preflight_token_estimate.output_tokens,
            task_definition.preflight_token_estimate.output_tokens,
        )
        self.assertTrue(spec.grounding["symbols"])
        self.assertTrue(spec.example_trajectory["steps"])
