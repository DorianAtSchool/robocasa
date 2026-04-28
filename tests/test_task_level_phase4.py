from __future__ import annotations

import os
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from data_generation.task_level.pipeline.phase4 import (
    Phase4TaskResult,
    _load_phase3_results,
    _run_sweep,
)
from data_generation.task_level.runtime.render_env import normalize_mujoco_render_env


class RenderEnvTests(unittest.TestCase):
    def test_linux_defaults_to_egl_when_pyopengl_platform_requests_it(self):
        env = {"PYOPENGL_PLATFORM": "egl"}

        normalize_mujoco_render_env(env, platform="linux")

        self.assertEqual(env["MUJOCO_GL"], "egl")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "egl")

    def test_linux_reconciles_inconsistent_osmesa_and_egl(self):
        env = {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "egl",
        }

        normalize_mujoco_render_env(env, platform="linux")

        self.assertEqual(env["MUJOCO_GL"], "osmesa")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "osmesa")


class Phase4SweepTests(unittest.TestCase):
    def test_load_phase3_results_matches_lowercase_task_selectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            phase3_dir = output_dir / "phase3"
            phase3_dir.mkdir(parents=True)
            results_path = phase3_dir / "results.json"
            results_path.write_text(
                json.dumps(
                    [
                        {"task_name": "ArrangeTea", "output_dir": "arrangetea"},
                        {"task_name": "SetupBowls", "output_dir": "setupbowls"},
                    ]
                ),
                encoding="utf-8",
            )

            selected = _load_phase3_results(
                output_dir,
                task_names=["arrangetea"],
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["task_name"], "ArrangeTea")

    def test_run_sweep_passes_normalized_render_env_to_child_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            successful_tasks = [
                Phase4TaskResult(
                    task_name="PrepareCoffee",
                    task_dir_name="preparecoffee",
                    input_summary_path="in.json",
                    output_summary_path="out.json",
                    stdout_log_path="stdout.log",
                    stderr_log_path="stderr.log",
                    exit_code=0,
                    completed=True,
                )
            ]
            captured_env: dict[str, str] = {}

            def _fake_run(*args, **kwargs):
                nonlocal captured_env
                captured_env = dict(kwargs["env"])
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.dict(
                os.environ,
                {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "egl"},
                clear=False,
            ):
                with mock.patch(
                    "data_generation.task_level.pipeline.phase4.subprocess.run",
                    side_effect=_fake_run,
                ):
                    _run_sweep(
                        successful_tasks,
                        output_dir=output_dir,
                        workers=1,
                        videos=False,
                        dry_run=False,
                    )

        self.assertEqual(captured_env["MUJOCO_GL"], "osmesa")
        self.assertEqual(captured_env["PYOPENGL_PLATFORM"], "osmesa")


if __name__ == "__main__":
    unittest.main()
