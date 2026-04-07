from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np

from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


class FakeExecutor:
    def __init__(self):
        self.loaded_state = None
        self.executed_steps = []
        self.scene = {
            "fixtures": {
                "cab_main": {"fixture_type": "cabinet_double_door"},
                "counter_main": {"fixture_type": "counter_non_dining"},
                "coffee_machine_main": {"fixture_type": "coffee_machine"},
            },
            "objects": {
                "mug_main": {"object_type": "mug"},
            },
        }

    def get_scene_description(self):
        return self.scene

    def _parse_agent_idx(self, agent_id):
        if isinstance(agent_id, int):
            return agent_id
        return int(str(agent_id).replace("agent_", ""))

    def load_initial_state(self, initial_state):
        self.loaded_state = initial_state
        return {"loaded": True}

    def execute(self, tool_name, robot_idx=0, **kwargs):
        self.executed_steps.append((tool_name, robot_idx, kwargs))
        return SimpleNamespace(
            success=True, details={"tool_name": tool_name, "args": kwargs}
        )


class TestTrajectoryAdapter(unittest.TestCase):
    def test_adapt_resolves_ids_and_rewrites_image_steps(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "trajectory_id": "traj_1",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "staging_surface", "held_object": None},
                    "agent_1": {"location": "coffee_machine", "held_object": None},
                },
                "objects": {
                    "mug": {"object_type": "mug", "location": "mug_source_fixture"},
                },
                "fixtures": {
                    "mug_source_fixture": {"fixture_type": "cabinet"},
                    "staging_surface": {"fixture_type": "counter"},
                    "coffee_machine": {"fixture_type": "coffee_machine"},
                },
                "machine_state": {
                    "coffee_machine": {
                        "started": False,
                        "dispenser_id": "coffee_machine_dispenser",
                    }
                },
            },
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "get_env_image",
                    "args": {"view": "top_view"},
                    "image_path": "images/traj_1/0_top.png",
                },
                {
                    "step": 1,
                    "agent": "agent_0",
                    "tool": "place_under_dispenser",
                    "args": {
                        "object_id": "mug",
                        "dispenser_id": "coffee_machine_dispenser",
                    },
                },
                {
                    "step": 2,
                    "agent": "agent_1",
                    "tool": "get_agent_image",
                    "args": {"view": "agentview_right"},
                    "image_path": "images/traj_1/2_right.png",
                },
            ],
        }

        adapted = adapter.adapt(trajectory, output_dir="tmp/output")

        self.assertEqual(
            adapted["initial_state"]["agents"]["agent_0"]["location"],
            "counter_main",
        )
        self.assertEqual(
            adapted["initial_state"]["objects"]["mug_main"]["location"],
            "cab_main",
        )
        self.assertEqual(adapted["tool_calls"][0]["tool"], "get_image")
        self.assertEqual(adapted["tool_calls"][0]["args"]["views"], ["top_view"])
        self.assertEqual(
            adapted["tool_calls"][0]["args"]["image_paths"],
            [str(Path("tmp/output") / "images/traj_1/0_top.png")],
        )
        self.assertEqual(adapted["tool_calls"][1]["tool"], "place_under")
        self.assertEqual(
            adapted["tool_calls"][1]["args"]["reference_fixture_id"],
            "coffee_machine_main",
        )
        self.assertEqual(adapted["tool_calls"][2]["tool"], "get_image")
        self.assertEqual(adapted["tool_calls"][2]["args"]["views"], ["agentview_right"])
        self.assertEqual(adapted["tool_calls"][2]["args"]["agent_id"], "agent_1")
        self.assertTrue(adapted["resolution_log"])

    def test_fixture_family_matching_handles_specific_scene_types(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)

        self.assertEqual(
            adapter._fixture_candidates_for_type("cabinet"),
            ["cab_main"],
        )
        self.assertEqual(
            adapter._fixture_candidates_for_type("counter"),
            ["counter_main"],
        )

    def test_sim_ground_truth_prefers_object_type_as_object_id(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
                "dining_main": {"fixture_type": "dining_table"},
            },
            "objects": {
                "hotdog_bun_container": {"object_type": "plate"},
                "plate": {"object_type": "plate"},
            },
            "fixture_refs": {
                "dining_table": "dining_main",
            },
            "object_placements": {
                "hotdog_bun_container": "counter_main",
                "plate": "dining_main",
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        adapter._apply_sim_ground_truth(
            {
                "objects": {
                    "serving_plate": {
                        "object_type": "plate",
                        "location": "serving_surface",
                    }
                },
                "fixtures": {
                    "serving_surface": {"fixture_type": "dining_table"},
                },
            }
        )

        self.assertEqual(adapter._object_aliases["serving_plate"], "plate")

    def test_execute_loads_state_then_runs_steps(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "staging_surface", "held_object": None}
                },
                "objects": {
                    "mug": {"object_type": "mug", "location": "mug_source_fixture"}
                },
                "fixtures": {
                    "mug_source_fixture": {"fixture_type": "cabinet"},
                    "staging_surface": {"fixture_type": "counter"},
                },
            },
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "communicate",
                    "args": {"to": "agent_1", "message": "hello"},
                }
            ],
        }

        metadata = adapter.execute(trajectory)

        self.assertTrue(metadata["load_initial_state"]["loaded"])
        self.assertIsNotNone(executor.loaded_state)
        self.assertEqual(len(executor.executed_steps), 1)
        self.assertEqual(executor.executed_steps[0][0], "communicate")


class TestSimToolExecutorObservationHelpers(unittest.TestCase):
    def _make_executor(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(
            render_height=4,
            render_width=5,
            _render_room_view=lambda: np.full((4, 5, 3), 7, dtype=np.uint8),
            _render_top_view=lambda: np.full((4, 5, 3), 9, dtype=np.uint8),
        )
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(
                render=lambda height, width, camera_name: np.full(
                    (height, width, 3), 13, dtype=np.uint8
                )
            )
        )
        return executor

    def test_get_image_saves_requested_env_view(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "top.png"
            result = executor.get_image(
                views=["top_view"],
                image_paths=[str(image_path)],
            )

            self.assertTrue(image_path.exists())
            self.assertEqual(result.details["camera_name"], "top_view")

    def test_get_image_supports_agentview_right(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "right.png"
            result = executor.get_image(
                views=["agentview_right"],
                image_paths=[str(image_path)],
                agent_id="agent_1",
            )

            self.assertTrue(image_path.exists())
            self.assertEqual(result.details["camera_name"], "robot1_agentview_right")

    def test_get_image_supports_multiple_views_and_map(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "top.png"
            map_path = Path(tmpdir) / "map.png"

            def _fake_save_map_image(requested_path):
                requested_path = Path(requested_path)
                requested_path.write_bytes(b"map")
                return requested_path

            executor._save_map_image = _fake_save_map_image
            result = executor.get_image(
                views=["top_view", "map"],
                image_paths=[str(top_path), str(map_path)],
            )

            self.assertTrue(top_path.exists())
            self.assertTrue(map_path.exists())
            self.assertEqual(result.details["views"], ["top_view", "map"])
            self.assertEqual(result.details["camera_names"], ["top_view", "map"])
            self.assertEqual(
                result.details["image_paths"],
                [str(top_path), str(map_path)],
            )


class TestSimToolExecutorLoadInitialState(unittest.TestCase):
    def test_load_initial_state_repositions_agents_and_holds_objects(self):
        move_calls = []
        navigate_calls = []
        machine_calls = []
        location_updates = []
        synced = []

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor.runner = SimpleNamespace(
            _fixtures={"counter_main": object(), "cab_main": object()},
            get_scene_description=lambda: {"object_placements": {}},
            move_object=lambda object_id, location: move_calls.append(
                (object_id, location)
            ),
            _set_object_location=lambda object_id, fixture_id: location_updates.append(
                (object_id, fixture_id)
            ),
        )
        executor.open_hinged_part = lambda target_id, part_id: None
        executor.close_hinged_part = lambda target_id, part_id: None
        executor.open_sliding_part = lambda target_id, part_id: None
        executor.close_sliding_part = lambda target_id, part_id: None
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(),
            objects={},
            obj_body_id={},
        )
        executor._set_fixture_machine_state = (
            lambda fixture_id, started: machine_calls.append((fixture_id, started))
        )
        executor.navigate_to_fixture = (
            lambda fixture_id, robot_idx=0: navigate_calls.append(
                (robot_idx, fixture_id)
            )
        )
        executor._require_object = lambda object_id: object_id
        executor._sync_held_object = lambda robot_idx: synced.append(robot_idx)
        executor._parse_agent_idx = lambda agent_id: int(
            str(agent_id).replace("agent_", "")
        )

        summary = executor.load_initial_state(
            {
                "agents": {
                    "agent_0": {"location": "counter_main", "held_object": "mug_main"},
                    "agent_1": {"location": "cab_main", "held_object": None},
                },
                "objects": {
                    "mug_main": {"location": "cab_main"},
                },
                "fixtures": {},
                "machine_state": {
                    "counter_main": {"started": False},
                },
            }
        )

        self.assertEqual(move_calls, [])
        self.assertEqual(navigate_calls, [(0, "counter_main"), (1, "cab_main")])
        self.assertEqual(machine_calls, [("counter_main", False)])
        self.assertEqual(executor._held_objects, {0: "mug_main"})
        self.assertEqual(synced, [0])
        self.assertEqual(location_updates, [("mug_main", "counter_main")])
        self.assertTrue(summary["loaded"])

    def test_load_initial_state_applies_fixture_part_states(self):
        opened = []
        closed = []

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor.runner = SimpleNamespace(
            _fixtures={"fridge_main": object(), "cab_main": object()},
            get_scene_description=lambda: {"object_placements": {}},
            move_object=lambda object_id, location, preferred_xy=None: None,
            _set_object_location=lambda object_id, fixture_id: None,
        )
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(forward=lambda: None),
            objects={},
            obj_body_id={},
        )
        executor._parse_agent_idx = lambda agent_id: 0
        executor._set_fixture_machine_state = lambda fixture_id, started: None
        executor._sync_held_object = lambda robot_idx: None
        executor._require_object = lambda object_id: object_id
        executor._set_fixture_part_state = lambda fixture_id, part_id, state: (
            opened.append((fixture_id, part_id, state))
            if state == "open"
            else closed.append((fixture_id, part_id, state))
        )

        executor.load_initial_state(
            {
                "agents": {},
                "objects": {},
                "fixtures": {
                    "fridge_main": {"parts": {"hinged": {"state": "closed"}}},
                    "cab_main": {"parts": {"hinged": {"state": "open"}}},
                },
                "machine_state": {},
            }
        )

        self.assertEqual(opened, [("cab_main", "hinged", "open")])
        self.assertEqual(closed, [("fridge_main", "hinged", "closed")])


class TestSimToolExecutorFixtureStateSync(unittest.TestCase):
    def test_open_hinged_part_forwards_sim_state(self):
        fixture = SimpleNamespace(open_door=MagicMock())
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(_fixtures={"fridge_main": fixture})
        executor.env = SimpleNamespace(sim=SimpleNamespace(forward=MagicMock()))
        executor._robot_near_fixture = lambda robot_idx, fixture_id: True
        executor._move_robot_near_fixture_with_retries = MagicMock()
        executor._sync_held_object = MagicMock()

        result = executor.open_hinged_part("fridge_main", "hinged", robot_idx=0)

        fixture.open_door.assert_called_once_with(env=executor.env)
        executor.env.sim.forward.assert_called_once_with()
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
