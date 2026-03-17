import json
from pathlib import Path
import tempfile
import unittest

from data_generation.task_level.generation.image import (
    POST_PROCESS_VALIDATION_ERROR,
    POST_PROCESS_VALIDATION_ERROR_TYPE,
    parse_args,
    post_process_dataset,
    post_process_trajectory,
    resolve_output_dataset_path,
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
                    "to": "agent_1",
                    "message": "I will start the task.",
                },
                "reasoning": "We should coordinate before acting.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
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
    def test_post_process_trajectory_v1_wraps_actions_with_get_image(self):
        trajectory = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v1",
        )

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
            "images/traj_000000/0_top_view_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][3]["args"],
            {"camera_view": "base_camera"},
        )
        self.assertEqual(
            trajectory["steps"][3]["image_path"],
            "images/traj_000000/3_base_camera_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][0]["reasoning"],
            "I need an initial top-view image before the task begins.",
        )
        self.assertEqual(
            trajectory["steps"][3]["reasoning"],
            (
                "I need a base-camera image to observe the current scene before "
                "I execute navigate_to_fixture."
            ),
        )
        self.assertEqual(
            trajectory["steps"][4]["reasoning"],
            "I need to reach the cabinet.",
        )
        self.assertEqual(
            trajectory["steps"][5]["reasoning"],
            (
                "I need a base-camera image to observe the current scene after "
                "I executed navigate_to_fixture."
            ),
        )
        self.assertEqual(
            trajectory["steps"][6]["reasoning"],
            (
                "I need a base-camera image to observe the current scene before "
                "I execute pick_up_object."
            ),
        )
        self.assertEqual(
            trajectory["steps"][8]["reasoning"],
            (
                "I need a base-camera image to observe the current scene after "
                "I executed pick_up_object."
            ),
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

    def test_post_process_trajectory_v2_splits_env_and_agent_image_tools(self):
        trajectory = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v2",
        )

        self.assertEqual(
            [step["tool"] for step in trajectory["steps"]],
            [
                "get_env_image",
                "get_env_image",
                "communicate",
                "communicate",
                "get_agent_image",
                "get_agent_image",
                "get_agent_image",
                "navigate_to_fixture",
                "get_agent_image",
                "get_agent_image",
                "get_agent_image",
                "get_agent_image",
                "get_agent_image",
                "pick_up_object",
                "get_agent_image",
                "get_agent_image",
            ],
        )
        self.assertEqual(
            [step["step"] for step in trajectory["steps"]],
            list(range(16)),
        )
        self.assertEqual(trajectory["steps"][0]["args"], {"view": "top_view"})
        self.assertEqual(trajectory["steps"][1]["args"], {"view": "room_view"})
        self.assertEqual(
            trajectory["steps"][0]["image_path"],
            "images/traj_000000/0_get_env_image_top_view_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][1]["image_path"],
            "images/traj_000000/1_get_env_image_room_view_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][4]["args"],
            {"agent_id": "agent_0", "view": "agentview_center"},
        )
        self.assertEqual(
            trajectory["steps"][6]["args"],
            {"agent_id": "agent_0", "view": "agentview_right"},
        )
        self.assertEqual(
            trajectory["steps"][11]["args"],
            {"agent_id": "agent_0", "view": "wrist"},
        )
        self.assertEqual(
            trajectory["steps"][4]["image_path"],
            "images/traj_000000/4_get_agent_image_agentview_center_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][11]["image_path"],
            "images/traj_000000/11_get_agent_image_wrist_agent_0.png",
        )
        self.assertEqual(
            trajectory["steps"][1]["reasoning"],
            "I need an initial room-view image before the task begins.",
        )
        self.assertEqual(
            trajectory["steps"][4]["reasoning"],
            (
                "I need a center agent-view image from my perspective before I "
                "execute navigate_to_fixture."
            ),
        )
        self.assertEqual(
            trajectory["steps"][11]["reasoning"],
            (
                "I need a wrist image from my perspective before I execute "
                "pick_up_object."
            ),
        )

    def test_post_process_trajectory_v1_keeps_rewritten_steps_schema_valid(self):
        trajectory = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v1",
        )
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

    def test_post_process_trajectory_v2_keeps_rewritten_steps_schema_valid(self):
        trajectory = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v2",
        )
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

    def test_post_process_trajectory_v1_is_idempotent(self):
        first_pass = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v1",
        )
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

    def test_post_process_trajectory_v2_is_idempotent(self):
        first_pass = post_process_trajectory(
            make_sample_trajectory(),
            image_tool_version="v2",
        )
        second_pass = post_process_trajectory(first_pass, image_tool_version="v2")

        self.assertEqual(second_pass["steps"], first_pass["steps"])
        self.assertEqual(
            second_pass["validation"]["signature"],
            first_pass["validation"]["signature"],
        )
        self.assertEqual(
            second_pass["validation"]["error_type"],
            POST_PROCESS_VALIDATION_ERROR_TYPE,
        )

    def test_post_process_trajectory_does_not_insert_agents_when_missing(self):
        trajectory = make_sample_trajectory()
        trajectory.pop("agents")

        processed = post_process_trajectory(trajectory)

        self.assertNotIn("agents", processed)

    def test_resolve_output_dataset_path_targets_w_images_copy(self):
        dataset_path = Path("/tmp/data/raw/prepare_coffee/summary.json")

        self.assertEqual(
            resolve_output_dataset_path(dataset_path),
            Path("/tmp/data/w_images/prepare_coffee/summary.json"),
        )

    def test_resolve_output_dataset_path_preserves_request_layout(self):
        dataset_path = Path(
            "/tmp/data/raw/requests/20260316T022801Z/prepare_coffee/summary.json"
        )

        self.assertEqual(
            resolve_output_dataset_path(dataset_path),
            Path(
                "/tmp/data/w_images/requests/20260316T022801Z/prepare_coffee/summary.json"
            ),
        )

    def test_post_process_dataset_writes_summary_copy_without_mutating_source(self):
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
            dataset_path = (
                Path(tmpdir) / "data" / "raw" / "prepare_coffee" / "summary.json"
            )
            trajectory_path = dataset_path.parent / "trajectories" / "traj_000000.json"
            prompt_path = dataset_path.parent / "prompts" / "traj_000000.md"
            error_summary_path = dataset_path.parent / "summary_errors.json"
            output_dataset_path = resolve_output_dataset_path(dataset_path)
            output_trajectory_path = (
                output_dataset_path.parent / "trajectories" / "traj_000000.json"
            )
            output_prompt_path = (
                output_dataset_path.parent / "prompts" / "traj_000000.md"
            )
            output_error_summary_path = (
                output_dataset_path.parent / "summary_errors.json"
            )
            output_images_dir = output_dataset_path.parent / "images"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(make_sample_trajectory(), indent=2),
                encoding="utf-8",
            )
            prompt_path.write_text("prompt copy me", encoding="utf-8")
            error_summary_path.write_text(
                json.dumps({"total_errors": 1}, indent=2),
                encoding="utf-8",
            )

            processed_count = post_process_dataset(
                dataset_path,
                disable_progress=True,
                image_tool_version="v1",
            )

            self.assertEqual(processed_count, 1)
            source_trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            updated_trajectory = json.loads(
                output_trajectory_path.read_text(encoding="utf-8")
            )
            self.assertEqual(source_trajectory["steps"][0]["tool"], "communicate")
            self.assertEqual(updated_trajectory["steps"][0]["tool"], "get_image")
            self.assertEqual(
                updated_trajectory["steps"][0]["image_path"],
                "images/traj_000000/0_top_view_agent_0.png",
            )
            self.assertEqual(
                json.loads(output_dataset_path.read_text(encoding="utf-8"))[
                    "trajectory_files"
                ],
                summary_payload["trajectory_files"],
            )
            self.assertEqual(
                json.loads(dataset_path.read_text(encoding="utf-8"))[
                    "trajectory_files"
                ],
                summary_payload["trajectory_files"],
            )
            self.assertEqual(
                output_prompt_path.read_text(encoding="utf-8"), "prompt copy me"
            )
            self.assertEqual(
                json.loads(output_error_summary_path.read_text(encoding="utf-8")),
                {"total_errors": 1},
            )
            self.assertTrue(output_images_dir.is_dir())

    def test_parse_args_requires_explicit_image_tool_version(self):
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "summary.json"])

    def test_parse_args_accepts_v2_image_tool_version(self):
        args = parse_args(
            [
                "--dataset",
                "summary.json",
                "--image-tool-version",
                "v2",
            ]
        )

        self.assertEqual(args.dataset, Path("summary.json"))
        self.assertEqual(args.image_tool_version, "v2")


if __name__ == "__main__":
    unittest.main()
