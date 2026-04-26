from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from unittest import mock

from data_generation.task_level.pipeline.models import TaskAnalysis
from data_generation.task_level.pipeline.phase1 import (
    _ThreadLocalGenerationClient,
    _postprocess_spec_payload,
    SpecGenerationResult,
    generate_specs,
    run_phase1,
)
from data_generation.task_level.runtime.client import BaseGenerationClient


class _RecordingClient(BaseGenerationClient):
    def __init__(self, client_id: int, barrier: threading.Barrier | None = None):
        self.client_id = client_id
        self.barrier = barrier

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict | None,
        temperature: float,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> int:
        if self.barrier is not None:
            self.barrier.wait(timeout=1.0)
        return self.client_id


class Phase1GenerationTests(unittest.TestCase):
    def test_thread_local_generation_client_reuses_one_client_per_thread(self):
        created_client_ids: list[int] = []
        next_client_id = 0
        barrier = threading.Barrier(2)

        def _factory() -> _RecordingClient:
            nonlocal next_client_id
            client = _RecordingClient(
                next_client_id,
                barrier=barrier if next_client_id > 0 else None,
            )
            created_client_ids.append(next_client_id)
            next_client_id += 1
            return client

        client = _ThreadLocalGenerationClient(_factory)

        main_thread_first = client.generate(
            model="test",
            prompt="prompt",
            response_schema=None,
            temperature=0.0,
        )
        main_thread_second = client.generate(
            model="test",
            prompt="prompt",
            response_schema=None,
            temperature=0.0,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            worker_results = list(
                executor.map(
                    lambda _: client.generate(
                        model="test",
                        prompt="prompt",
                        response_schema=None,
                        temperature=0.0,
                    ),
                    range(2),
                )
            )

        self.assertEqual(main_thread_first, main_thread_second)
        self.assertEqual(3, len(created_client_ids))
        self.assertEqual(3, len({main_thread_first, *worker_results}))

    def test_generate_specs_emits_heartbeat_while_waiting_on_slow_task(self):
        candidates = [
            TaskAnalysis(
                task_name="FastTask",
                module_path="tests.fast_task",
                file_path=__file__,
                activity="brewing",
            ),
            TaskAnalysis(
                task_name="SlowTask",
                module_path="tests.slow_task",
                file_path=__file__,
                activity="brewing",
            ),
        ]
        heartbeat_events: list[tuple[int, int, list[tuple[str, float]], int]] = []

        def _fake_generate_one_spec(
            candidate: TaskAnalysis,
            *,
            client: BaseGenerationClient,
            model: str,
            temperature: float,
            examples: tuple[object, ...],
            repair_context: object | None = None,
        ) -> SpecGenerationResult:
            if candidate.task_name == "SlowTask":
                time.sleep(0.08)
            else:
                time.sleep(0.01)
            return SpecGenerationResult(
                task_name=candidate.task_name,
                module_path=candidate.module_path,
                batch=candidate.batch,
                spec_payload={"composite_task": candidate.task_name},
                error=None,
            )

        with mock.patch(
            "data_generation.task_level.pipeline.phase1._generate_one_spec",
            side_effect=_fake_generate_one_spec,
        ):
            results = generate_specs(
                candidates,
                client=_RecordingClient(0),
                model="test",
                examples=(),
                workers=2,
                heartbeat_callback=lambda completed, total, pending_status, pending_count: heartbeat_events.append(
                    (completed, total, pending_status, pending_count)
                ),
                heartbeat_interval_sec=0.02,
            )

        self.assertEqual(2, len(results))
        self.assertTrue(heartbeat_events)
        self.assertTrue(
            any(
                pending_count == 1 and pending_status[0][0] == "SlowTask"
                for _, _, pending_status, pending_count in heartbeat_events
            )
        )

    def test_run_phase1_skips_sim_normalization_by_default(self):
        candidate = TaskAnalysis(
            task_name="TaskA",
            module_path="tests.task_a",
            file_path=__file__,
            activity="prep",
        )
        result = SpecGenerationResult(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            spec_payload={"composite_task": candidate.task_name},
            error=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with mock.patch(
                "data_generation.task_level.pipeline.phase1.generate_specs",
                return_value=[result],
            ):
                with mock.patch(
                    "data_generation.task_level.pipeline.phase1.load_few_shot_examples",
                    return_value=(),
                ):
                    with mock.patch(
                        "data_generation.task_level.pipeline.phase1.normalize_spec_payload_against_simulation"
                    ) as mocked_normalize:
                        run_phase1(
                            [candidate],
                            output_dir=output_dir,
                            client=_RecordingClient(0),
                        )

        mocked_normalize.assert_not_called()

    def test_run_phase1_applies_sim_normalization_when_enabled(self):
        candidate = TaskAnalysis(
            task_name="TaskA",
            module_path="tests.task_a",
            file_path=__file__,
            activity="prep",
        )
        result = SpecGenerationResult(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            spec_payload={"composite_task": candidate.task_name},
            error=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with mock.patch(
                "data_generation.task_level.pipeline.phase1.generate_specs",
                return_value=[result],
            ):
                with mock.patch(
                    "data_generation.task_level.pipeline.phase1.load_few_shot_examples",
                    return_value=(),
                ):
                    with mock.patch(
                        "data_generation.task_level.pipeline.phase1.normalize_spec_payload_against_simulation",
                        return_value=({"composite_task": candidate.task_name}, []),
                    ) as mocked_normalize:
                        run_phase1(
                            [candidate],
                            output_dir=output_dir,
                            client=_RecordingClient(0),
                            sim_normalization=True,
                        )

        mocked_normalize.assert_called_once()

    def test_postprocess_spec_payload_canonicalizes_generic_hinged_part_ids(self):
        payload = {
            "allowed_tool_specs": {
                "open_hinged_part": {
                    "allowed_part_ids": ["left_door", "door"],
                }
            },
            "initial_state": {
                "fixtures": {
                    "cabinet": {
                        "parts": {
                            "left_door": {"part_type": "hinged_part", "state": "closed"},
                            "right_door": {"part_type": "hinged_part", "state": "closed"},
                        }
                    }
                }
            },
            "goal_conditions": [
                {
                    "kind": "fixture_part_state",
                    "fixture_id": "cabinet",
                    "part_id": "door_left",
                    "state": "closed",
                }
            ],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "open_hinged_part",
                        "args": {"target_id": "cabinet", "part_id": "right_door"},
                        "reasoning": "open the cabinet",
                    }
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        fixture_parts = normalized["initial_state"]["fixtures"]["cabinet"]["parts"]
        self.assertEqual(list(fixture_parts), ["hinged"])
        self.assertEqual(
            normalized["allowed_tool_specs"]["open_hinged_part"]["allowed_part_ids"],
            ["hinged"],
        )
        self.assertEqual(
            normalized["goal_conditions"][0]["part_id"],
            "hinged",
        )
        self.assertEqual(
            normalized["example_trajectory"]["steps"][0]["args"]["part_id"],
            "hinged",
        )

    def test_postprocess_spec_payload_does_not_insert_navigation_before_give_space(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {},
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "give_space",
                        "args": {"fixture_id": "counter"},
                        "reasoning": "Bad model output: agent is not at the counter.",
                    }
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        steps = normalized["example_trajectory"]["steps"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["tool"], "give_space")

    def test_postprocess_spec_payload_inserts_give_space_for_existing_fixture_blocker(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "fridge"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                    "cheese": {"object_type": "cheese", "location": "fridge"},
                    "cutting_board": {
                        "object_type": "cutting_board",
                        "location": "counter",
                    },
                },
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "navigate_to_fixture",
                        "args": {"fixture_id": "counter"},
                        "reasoning": "Move to the counter.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_0",
                        "tool": "place_on_object",
                        "args": {
                            "object_id": "sausage",
                            "support_object_id": "cutting_board",
                        },
                        "reasoning": "Place sausage on the board.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_1",
                        "tool": "navigate_to_fixture",
                        "args": {"fixture_id": "counter"},
                        "reasoning": "Move to the counter for cheese.",
                    },
                    {
                        "step": 3,
                        "agent": "agent_1",
                        "tool": "place_on_object",
                        "args": {
                            "object_id": "cheese",
                            "support_object_id": "cutting_board",
                        },
                        "reasoning": "Place cheese on the board.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        steps = normalized["example_trajectory"]["steps"]
        self.assertEqual(
            [(step["agent"], step["tool"]) for step in steps],
            [
                ("agent_0", "navigate_to_fixture"),
                ("agent_0", "place_on_object"),
                ("agent_0", "give_space"),
                ("agent_1", "navigate_to_fixture"),
                ("agent_1", "place_on_object"),
            ],
        )
        self.assertEqual(steps[2]["args"], {"fixture_id": "counter"})

    def test_postprocess_spec_payload_ignores_top_level_allowed_fields(self):
        payload = {
            "allowed_tool_specs": {
                "allowed_object_ids": ["sausage"],
            },
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                },
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertIn("communicate", normalized["allowed_tool_specs"])
        self.assertIn("pick_up_object", normalized["allowed_tool_specs"])
        self.assertNotIn("allowed_object_ids", normalized["allowed_tool_specs"])

    def test_postprocess_spec_payload_coerces_list_objects_and_fixtures(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": [
                    {"fixture_id": "fridge", "fixture_type": "fridge"},
                    {"id": "counter", "fixture_type": "counter"},
                ],
                "objects": [
                    {
                        "role": "sausage",
                        "object_type": "sausage",
                        "location": "fridge",
                    },
                ],
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["initial_state"]["objects"],
            {"sausage": {"object_type": "sausage", "location": "fridge"}},
        )
        self.assertEqual(
            normalized["initial_state"]["fixtures"],
            {
                "fridge": {"fixture_type": "fridge", "parts": {}, "controls": {}},
                "counter": {"fixture_type": "counter", "parts": {}, "controls": {}},
            },
        )

    def test_postprocess_spec_payload_uses_type_as_list_item_id_fallback(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": [
                    {"fixture_type": "fridge"},
                    {"fixture_type": "counter"},
                ],
                "objects": [
                    {"object_type": "sausage", "location": "fridge"},
                    {"object_type": "sausage", "location": "fridge"},
                ],
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            list(normalized["initial_state"]["objects"]),
            ["sausage", "sausage_2"],
        )
        self.assertEqual(
            list(normalized["initial_state"]["fixtures"]),
            ["fridge", "counter"],
        )

    def test_postprocess_spec_payload_coerces_list_agents(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": [
                    {"location": "fridge"},
                    {"agent": "agent_1", "location": "counter"},
                ],
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                },
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["initial_state"]["agents"],
            {
                "agent_0": {"location": "fridge"},
                "agent_1": {"location": "counter"},
            },
        )

    def test_postprocess_spec_payload_coerces_string_initial_public_state(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_public_state": "The sausage starts in the fridge.",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                },
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["initial_public_state"],
            {"summary": "The sausage starts in the fridge."},
        )

    def test_postprocess_spec_payload_coerces_grounding_list_and_rule_string(self):
        payload = {
            "allowed_tool_specs": {},
            "extra_execution_rules": "Place the sausage on the plate.",
            "grounding": [
                {
                    "arg": "sausage",
                    "method": "object_by_type",
                    "object_type": "sausage",
                },
                {
                    "arg": "fridge",
                    "method": "source_fixture_for_object",
                    "object_id": "sausage",
                },
            ],
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                },
            },
            "goal_conditions": [],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["extra_execution_rules"],
            ["Place the sausage on the plate."],
        )
        self.assertEqual(
            normalized["grounding"],
            {
                "legacy_symbol_aliases": {},
                "symbols": {
                    "sausage": {
                        "resolver": "object_by_type",
                        "object_type": "sausage",
                        "entity_type": "object",
                    },
                    "fridge": {
                        "entity_type": "fixture",
                        "fixture_type": "fridge",
                        "object_id": "sausage",
                        "resolver": "unique_fixture_type",
                    },
                },
            },
        )

    def test_postprocess_spec_payload_coerces_fixture_parts_and_step_agent_id(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "fridge"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "fridge": {
                        "fixture_type": "fridge",
                        "parts": [
                            {"part_id": "hinged", "state": "closed"},
                        ],
                    },
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "sausage": {"object_type": "sausage", "location": "fridge"},
                },
            },
            "goal_conditions": [],
            "task_preconditions": [
                {
                    "kind": "fixture_part_state_required_for_pickup",
                    "fixture_id": "fridge",
                    "part_id": "hinged",
                    "tool": "pick_up_object",
                    "source_id": "fridge",
                    "required_state": "open",
                }
            ],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent_id": "agent_0",
                        "tool": "communicate",
                        "args": {"to": "agent_1", "message": "I will get sausage."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 1,
                        "agent_id": "agent_1",
                        "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I will wait."},
                        "reasoning": "Coordinate.",
                    },
                    {
                        "step": 2,
                        "agent_id": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "sausage", "source_id": "fridge"},
                        "reasoning": "Pick up sausage.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["initial_state"]["fixtures"]["fridge"]["parts"],
            {"hinged": {"state": "closed"}},
        )
        self.assertEqual(
            [
                step.get("agent")
                for step in normalized["example_trajectory"]["steps"]
            ],
            ["agent_0", "agent_1", "agent_0"],
        )
        self.assertNotIn("agent_id", normalized["example_trajectory"]["steps"][0])

    def test_postprocess_rewrites_contradictory_source_location_goal(self):
        payload = {
            "allowed_tool_specs": {},
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "cabinet"},
                    "agent_1": {"location": "counter"},
                },
                "fixtures": {
                    "cabinet": {"fixture_type": "cabinet"},
                    "counter": {"fixture_type": "counter"},
                },
                "objects": {
                    "bowl1": {"object_type": "bowl", "location": "cabinet"},
                },
            },
            "goal_conditions": [
                {"kind": "object_at_location", "object_id": "bowl1", "location": "cabinet"}
            ],
            "task_preconditions": [],
            "task_effects": [],
            "example_trajectory": {
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "bowl1", "source_id": "cabinet"},
                        "reasoning": "Pick up bowl.",
                    },
                    {
                        "step": 1,
                        "agent": "agent_0",
                        "tool": "navigate_to_fixture",
                        "args": {"fixture_id": "counter"},
                        "reasoning": "Move to counter.",
                    },
                    {
                        "step": 2,
                        "agent": "agent_0",
                        "tool": "place_on_surface",
                        "args": {"object_id": "bowl1", "target_id": "counter"},
                        "reasoning": "Place bowl on counter.",
                    },
                ]
            },
        }

        normalized = _postprocess_spec_payload(payload)

        self.assertEqual(
            normalized["goal_conditions"][0],
            {"kind": "object_at_location", "object_id": "bowl1", "location": "counter"},
        )


if __name__ == "__main__":
    unittest.main()
