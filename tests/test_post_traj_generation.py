import json
from pathlib import Path
import tempfile
import unittest

from data_generation.task_level.post_traj_generation import (
    POST_PROCESS_VALIDATION_ERROR,
    POST_PROCESS_VALIDATION_ERROR_TYPE,
    post_process_dataset,
    post_process_trajectory,
)
from data_generation.task_level.tasks import (
    PrepareCoffeeValidator,
    TaskSemanticValidationError,
    TrajectoryStructureValidationError,
)


def make_sample_trajectory():
    """Builds a compact saved trajectory fixture for post-processing tests."""

    return {
        "trajectory_id": "traj_000000",
        "composite_task": "PrepareCoffee",
        "agents": [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ],
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to_agent_id": "agent_1",
                    "message": "I will start the task.",
                },
                "reasoning": "We should coordinate before acting.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to_agent_id": "agent_0",
                    "message": "I will stay ready for handoff.",
                },
                "reasoning": "I should confirm the plan.",
            },
            {
                "step": 2,
                "agent": "agent_0",
                "tool": "navigate_to_fixture",
                "args": {"fixture_id": "cabinet_1"},
                "reasoning": "I need to reach the cabinet.",
            },
            {
                "step": 3,
                "agent": "agent_0",
                "tool": "pick_up_object",
                "args": {
                    "object_id": "mug_1",
                    "source_id": "cabinet_1",
                },
                "reasoning": "I should pick up the mug.",
            },
        ],
        "validation": {"signature": "old-signature"},
    }


class PostTrajectoryGenerationTests(unittest.TestCase):
    def test_post_process_trajectory_wraps_actions_with_get_image(self):
        trajectory = post_process_trajectory(make_sample_trajectory())

        self.assertEqual(
            [step["tool"] for step in trajectory["steps"]],
            [
                "get_image",
                "communicate",
                "communicate",
                "get_image",
                "navigate_to_fixture",
                "get_image",
                "get_image",
                "pick_up_object",
                "get_image",
            ],
        )
        self.assertEqual(
            [step["step"] for step in trajectory["steps"]],
            list(range(9)),
        )
        self.assertEqual(
            trajectory["steps"][0]["args"],
            {"camera_view": "top_view"},
        )
        self.assertEqual(
            trajectory["steps"][0]["image_path"],
            "trajectories/images/traj_000000/0_top_view_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][3]["args"],
            {"camera_view": "base_camera"},
        )
        self.assertEqual(
            trajectory["steps"][3]["image_path"],
            "trajectories/images/traj_000000/3_base_camera_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][0]["reasoning"],
            "I need an initial top-view image before the task begins.",
        )
        self.assertEqual(
            trajectory["steps"][3]["reasoning"],
            "I need a base-camera image to observe the current scene.",
        )
        self.assertEqual(
            trajectory["steps"][4]["reasoning"],
            "I need to reach the cabinet.",
        )
        self.assertFalse(trajectory["validation"]["is_valid"])
        self.assertEqual(trajectory["validation"]["checks"], [])
        self.assertIsNone(trajectory["validation"]["final_state"])
        self.assertEqual(
            trajectory["validation"]["error_type"],
            POST_PROCESS_VALIDATION_ERROR_TYPE,
        )
        self.assertEqual(
            trajectory["validation"]["error"],
            POST_PROCESS_VALIDATION_ERROR,
        )
        self.assertIsNone(trajectory["validation"]["step"])
        self.assertNotEqual(trajectory["validation"]["signature"], "old-signature")

    def test_post_process_trajectory_keeps_rewritten_steps_schema_valid(self):
        trajectory = post_process_trajectory(make_sample_trajectory())
        validator = PrepareCoffeeValidator()

        try:
            validator.validate(
                {
                    "agents": trajectory["agents"],
                    "steps": trajectory["steps"],
                }
            )
        except TaskSemanticValidationError:
            pass
        except TrajectoryStructureValidationError as exc:
            self.fail(f"Rewritten steps should remain schema-valid: {exc}")

    def test_post_process_trajectory_is_idempotent(self):
        first_pass = post_process_trajectory(make_sample_trajectory())
        second_pass = post_process_trajectory(first_pass)

        self.assertEqual(second_pass["steps"], first_pass["steps"])
        self.assertEqual(
            second_pass["validation"]["signature"],
            first_pass["validation"]["signature"],
        )
        self.assertEqual(
            second_pass["validation"]["error_type"],
            POST_PROCESS_VALIDATION_ERROR_TYPE,
        )

    def test_post_process_trajectory_inserts_canonical_agents_when_missing(self):
        trajectory = make_sample_trajectory()
        trajectory.pop("agents")

        processed = post_process_trajectory(trajectory)

        self.assertEqual(
            processed["agents"],
            [{"agent": "agent_0"}, {"agent": "agent_1"}],
        )

    def test_post_process_dataset_updates_summary_sidecars_in_place(self):
        summary_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "num_trajectories": 1,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "trajectory_directory": "trajectories",
            "trajectory_files": [
                {
                    "trajectory_id": "traj_000000",
                    "path": "trajectories/traj_000000.json",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "prepare_coffee_trajectories.json"
            trajectory_path = Path(tmpdir) / "trajectories" / "traj_000000.json"
            dataset_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(make_sample_trajectory(), indent=2),
                encoding="utf-8",
            )

            processed_count = post_process_dataset(dataset_path, disable_progress=True)

            self.assertEqual(processed_count, 1)
            updated_trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_trajectory["steps"][0]["tool"], "get_image")
            self.assertEqual(
                updated_trajectory["steps"][0]["image_path"],
                "trajectories/images/traj_000000/0_top_view_agent_0.png",
            )
            self.assertEqual(
                json.loads(dataset_path.read_text(encoding="utf-8"))["trajectory_files"],
                summary_payload["trajectory_files"],
            )


if __name__ == "__main__":
    unittest.main()
