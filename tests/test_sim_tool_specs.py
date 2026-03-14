import unittest

from robocasa.utils.sim_tool_specs import (
    SIM_TOOL_SPEC_BY_NAME,
    SIM_TOOL_SPECS,
    get_sim_tool_spec,
    get_sim_tool_specs,
)


class TestSimToolSpecs(unittest.TestCase):
    def test_expected_tool_names_exist(self):
        expected_names = {
            "close_hinged_part",
            "close_sliding_part",
            "communicate",
            "navigate_to_fixture",
            "open_hinged_part",
            "open_sliding_part",
            "pick_up_object",
            "place_in_receptacle",
            "place_on_object",
            "place_on_surface",
            "place_under_dispenser",
            "press_button",
            "press_lever",
            "set_rotary_control",
            "wait",
        }
        self.assertEqual({spec["name"] for spec in SIM_TOOL_SPECS}, expected_names)
        self.assertEqual(set(SIM_TOOL_SPEC_BY_NAME.keys()), expected_names)

    def test_required_parameter_names_match(self):
        expected_parameters = {
            "close_hinged_part": ["target_id", "part_id"],
            "close_sliding_part": ["target_id", "part_id"],
            "communicate": ["to", "message"],
            "navigate_to_fixture": ["fixture_id"],
            "open_hinged_part": ["target_id", "part_id"],
            "open_sliding_part": ["target_id", "part_id"],
            "pick_up_object": ["object_id", "source_id"],
            "place_in_receptacle": ["object_id", "receptacle_id"],
            "place_on_object": ["object_id", "support_object_id"],
            "place_on_surface": ["object_id", "support_id"],
            "place_under_dispenser": ["object_id", "dispenser_id"],
            "press_button": ["target_id", "control_id"],
            "press_lever": ["target_id", "control_id"],
            "set_rotary_control": ["target_id", "control_id", "goal"],
            "wait": [],
        }
        for tool_name, expected in expected_parameters.items():
            spec = SIM_TOOL_SPEC_BY_NAME[tool_name]
            self.assertEqual(
                [param["name"] for param in spec["parameters"]],
                expected,
            )
            self.assertTrue(all(param["required"] for param in spec["parameters"]))
            self.assertTrue(all(param["type"] == "string" for param in spec["parameters"]))

    def test_getters_return_defensive_copies(self):
        all_specs = get_sim_tool_specs()
        all_specs[0]["name"] = "mutated"
        self.assertNotEqual(SIM_TOOL_SPECS[0]["name"], "mutated")

        single_spec = get_sim_tool_spec("navigate_to_fixture")
        single_spec["parameters"][0]["name"] = "mutated_fixture_id"
        self.assertEqual(
            SIM_TOOL_SPEC_BY_NAME["navigate_to_fixture"]["parameters"][0]["name"],
            "fixture_id",
        )


if __name__ == "__main__":
    unittest.main()
