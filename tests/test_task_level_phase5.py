from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_generation.task_level.pipeline.phase5 import run_phase5


class Phase5ReportTests(unittest.TestCase):
    def test_phase5_reports_sweep_failures_from_sweep_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            phase4_dir = output_dir / "phase4"
            sweeps_dir = phase4_dir / "sweeps"
            sweeps_dir.mkdir(parents=True)

            (phase4_dir / "results.json").write_text(
                json.dumps(
                    {
                        "pre_image_results": [
                            {"task_name": "TaskA", "completed": True},
                        ],
                        "sweep": {"completed": True, "exit_code": 0},
                    }
                ),
                encoding="utf-8",
            )
            (phase4_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "selected_phase3_tasks": 1,
                        "pre_image_completed": 1,
                        "pre_image_failed": 0,
                        "sweep_completed": True,
                    }
                ),
                encoding="utf-8",
            )
            (sweeps_dir / "sweep_summary.json").write_text(
                json.dumps(
                    {
                        "total": 1,
                        "succeeded": 0,
                        "failed": 1,
                        "results": [
                            {
                                "status": "error",
                                "task_dir": "taska",
                                "error": "load_initial_state requires an explicit site for fixture 'fridge_right_group'.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = run_phase5(output_dir=output_dir)

        self.assertEqual(report["phase4"]["sweep_summary"]["failed"], 1)
        self.assertEqual(report["phase4"]["sweep_summary"]["succeeded"], 0)
        self.assertTrue(
            any(
                "Phase 4 sweep produced failing trajectories" in recommendation
                for recommendation in report["recommendations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
