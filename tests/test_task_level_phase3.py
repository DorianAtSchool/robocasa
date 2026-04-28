from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from data_generation.task_level.pipeline.phase3 import (
    Phase3TaskResult,
    _run_one_task,
    run_phase3,
)


class Phase3Tests(unittest.TestCase):
    def _write_spec(self, directory: Path, filename: str, task_name: str) -> Path:
        spec_path = directory / filename
        spec_path.write_text(
            json.dumps({"composite_task": task_name}),
            encoding="utf-8",
        )
        return spec_path

    def test_run_one_task_contains_unexpected_subprocess_exceptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            spec_dir = output_dir / "phase1" / "specs"
            spec_dir.mkdir(parents=True)
            spec_path = self._write_spec(
                spec_dir,
                "toastheatableingredients.json",
                "ToastHeatableIngredients",
            )

            with mock.patch(
                "data_generation.task_level.pipeline.phase3.subprocess.run",
                side_effect=OSError("boom"),
            ):
                result = _run_one_task(
                    spec_path,
                    output_dir=output_dir,
                    num_runs=1,
                    model=None,
                    project=None,
                    location="global",
                    max_retries=1,
                    generation_timeout_sec=300,
                    spec_dir=spec_dir,
                    dry_run=False,
                )

        self.assertEqual(result.task_name, "ToastHeatableIngredients")
        self.assertFalse(result.completed)
        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.pending_run_indices, [0])
        self.assertIn("OSError: boom", result.error or "")

    def test_run_phase3_emits_heartbeat_and_persists_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            spec_dir = output_dir / "phase1" / "specs"
            spec_dir.mkdir(parents=True)
            fast_spec = self._write_spec(spec_dir, "fasttask.json", "FastTask")
            slow_spec = self._write_spec(spec_dir, "slowtask.json", "SlowTask")
            heartbeat_events: list[tuple[int, int, list[tuple[str, float]], int]] = []

            def _fake_result(spec_path: Path, **_: object) -> Phase3TaskResult:
                task_name = "FastTask" if spec_path == fast_spec else "SlowTask"
                time.sleep(0.01 if spec_path == fast_spec else 0.06)
                return Phase3TaskResult(
                    task_name=task_name,
                    spec_path=str(spec_path),
                    output_dir=str(output_dir / "phase3" / "raw" / spec_path.stem),
                    summary_path=str(output_dir / "phase3" / "raw" / spec_path.stem / "summary.json"),
                    error_summary_path=str(
                        output_dir / "phase3" / "raw" / spec_path.stem / "summary_errors.json"
                    ),
                    stdout_log_path=str(output_dir / "phase3" / "logs" / f"{spec_path.stem}.stdout.log"),
                    stderr_log_path=str(output_dir / "phase3" / "logs" / f"{spec_path.stem}.stderr.log"),
                    exit_code=0,
                    completed=True,
                    num_runs=1,
                    num_trajectories=1,
                    completed_run_indices=[0],
                    failed_run_indices=[],
                    pending_run_indices=[],
                    error_counts_by_type=[],
                    distinct_errors=[],
                    error=None,
                )

            with mock.patch(
                "data_generation.task_level.pipeline.phase3._run_one_task",
                side_effect=_fake_result,
            ):
                results = run_phase3(
                    output_dir=output_dir,
                    spec_paths=[fast_spec, slow_spec],
                    workers=2,
                    heartbeat_callback=lambda completed, total, pending_status, pending_count: heartbeat_events.append(
                        (completed, total, pending_status, pending_count)
                    ),
                    heartbeat_interval_sec=0.02,
                    generation_timeout_sec=300,
                )

            summary_path = output_dir / "phase3" / "summary.json"
            results_path = output_dir / "phase3" / "results.json"
            self.assertEqual(len(results), 2)
            self.assertTrue(summary_path.exists())
            self.assertTrue(results_path.exists())
            self.assertTrue(heartbeat_events)
            self.assertTrue(
                any(
                    any(task_name == "SlowTask" for task_name, _ in pending_status)
                    for _, _, pending_status, _ in heartbeat_events
                )
            )


if __name__ == "__main__":
    unittest.main()
