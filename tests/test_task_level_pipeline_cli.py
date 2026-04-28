from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_generation.task_level.pipeline import cli


class PipelineCliPhaseSelectionTests(unittest.TestCase):
    def test_parse_args_phase1_sim_normalization_defaults_off(self):
        args = cli.parse_args(["--phase", "1"])

        self.assertFalse(args.phase1_sim_normalization)

    def test_parse_args_phase1_sim_normalization_flag_enables_it(self):
        args = cli.parse_args(["--phase", "1", "--phase1-sim-normalization"])

        self.assertTrue(args.phase1_sim_normalization)

    def test_parse_args_normalizes_phase_subset_in_pipeline_order(self):
        args = cli.parse_args(["--phase", "4", "1", "3"])

        self.assertEqual(args.phase, ("1", "3", "4"))

    def test_parse_args_expands_all_to_full_pipeline(self):
        args = cli.parse_args(["--phase", "all"])

        self.assertEqual(args.phase, cli.PIPELINE_PHASE_ORDER)

    def test_parse_args_rejects_all_mixed_with_explicit_phases(self):
        with self.assertRaises(SystemExit):
            cli.parse_args(["--phase", "all", "4"])

    def test_main_runs_requested_subset_once_in_pipeline_order(self):
        phase_calls: list[str] = []

        def _record_phase(phase_name: str):
            def _runner(**_: object):
                phase_calls.append(phase_name)
                if phase_name == "0a":
                    return []
                if phase_name == "0b":
                    return []
                return None

            return _runner

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_dir = Path(temp_dir)
            with mock.patch(
                "data_generation.task_level.pipeline.cli._load_best_available_candidates",
                return_value=[],
            ):
                with mock.patch(
                    "data_generation.task_level.pipeline.cli._run_phase1",
                    side_effect=_record_phase("1"),
                ):
                    with mock.patch(
                        "data_generation.task_level.pipeline.cli._run_phase3",
                        side_effect=_record_phase("3"),
                    ):
                        with mock.patch(
                            "data_generation.task_level.pipeline.cli._run_phase4",
                            side_effect=_record_phase("4"),
                        ):
                            cli.main(
                                [
                                    "--phase",
                                    "4",
                                    "1",
                                    "3",
                                    "--resume",
                                    str(resume_dir),
                                ]
                            )

        self.assertEqual(phase_calls, ["1", "3", "4"])

    def test_load_best_available_candidates_falls_back_to_phase0a(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            phase0a_dir = run_dir / "phase0a"
            phase0a_dir.mkdir(parents=True)
            (phase0a_dir / "candidates.json").write_text(
                json.dumps(
                    [
                        {
                            "task_name": "TaskA",
                            "module_path": "example.task_a",
                            "file_path": "/tmp/task_a.py",
                            "activity": "prep",
                            "batch": "batch1",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            candidates = cli._load_best_available_candidates(
                run_dir,
                batch="batch1",
                task_names=None,
            )

        self.assertEqual([candidate.task_name for candidate in candidates], ["TaskA"])

    def test_load_best_available_candidates_rebuilds_from_source_when_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            rebuilt_candidate = mock.Mock()
            rebuilt_candidate.task_name = "ArrangeTea"
            rebuilt_candidate.batch = "batch1"

            with mock.patch(
                "data_generation.task_level.pipeline.phase0a.analyze_all_tasks",
                return_value=([rebuilt_candidate], []),
            ):
                candidates = cli._load_best_available_candidates(
                    run_dir,
                    batch="batch1",
                    task_names=["arrangetea"],
                )

        self.assertEqual([candidate.task_name for candidate in candidates], ["ArrangeTea"])

    def test_apply_selection_filters_accepts_slug_task_names(self):
        candidate = mock.Mock()
        candidate.task_name = "ArrangeTea"
        candidate.batch = "batch1"

        selected = cli._apply_selection_filters(
            [candidate],
            batch="batch1",
            task_names=["arrangetea"],
        )

        self.assertEqual(selected, [candidate])

    def test_load_phase2_passed_spec_paths_accepts_slug_task_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            phase2_dir = run_dir / "phase2"
            phase2_dir.mkdir(parents=True)
            spec_path = run_dir / "phase1" / "specs" / "arrangetea.json"
            spec_path.parent.mkdir(parents=True)
            spec_path.write_text("{}", encoding="utf-8")
            (phase2_dir / "validation_results.json").write_text(
                json.dumps(
                    [
                        {
                            "task_name": "ArrangeTea",
                            "spec_path": str(spec_path),
                            "passed": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            spec_paths = cli._load_phase2_passed_spec_paths(
                run_dir,
                task_names=["arrangetea"],
            )

        self.assertEqual(spec_paths, [spec_path])

    def test_load_best_available_phase3_specs_falls_back_to_phase1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            specs_dir = run_dir / "phase1" / "specs"
            specs_dir.mkdir(parents=True)
            spec_path = specs_dir / "task_a.json"
            spec_path.write_text("{}", encoding="utf-8")

            spec_paths = cli._load_best_available_phase3_spec_paths(
                run_dir,
                task_names=None,
            )

        self.assertEqual(spec_paths, [spec_path])

    def test_run_phase3_falls_back_to_phase1_specs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            specs_dir = run_dir / "phase1" / "specs"
            specs_dir.mkdir(parents=True)
            spec_path = specs_dir / "task_a.json"
            spec_path.write_text("{}", encoding="utf-8")

            with mock.patch(
                "data_generation.task_level.pipeline.phase3.run_phase3",
                return_value=[],
            ) as mocked_phase3_impl:
                cli._run_phase3(
                    run_dir=run_dir,
                    args=cli.parse_args(["--phase", "3", "--resume", str(run_dir)]),
                    state=mock.Mock(),
                )

        mocked_phase3_impl.assert_called_once()
        _, kwargs = mocked_phase3_impl.call_args
        self.assertEqual(kwargs["spec_paths"], [spec_path])
        self.assertTrue(callable(kwargs["heartbeat_callback"]))

    def test_run_phase1_passes_sim_normalization_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            args = cli.parse_args(
                ["--phase", "1", "--resume", str(run_dir), "--phase1-sim-normalization"]
            )

            with mock.patch(
                "data_generation.task_level.pipeline.phase1.run_phase1",
                return_value=[],
            ) as mocked_phase1_impl:
                cli._run_phase1(
                    run_dir=run_dir,
                    args=args,
                    state=mock.Mock(),
                    candidates=[],
                )

        mocked_phase1_impl.assert_not_called()

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            args = cli.parse_args(
                ["--phase", "1", "--resume", str(run_dir), "--phase1-sim-normalization"]
            )

            candidate = mock.Mock()
            candidate.task_name = "TaskA"
            candidate.batch = "batch1"

            with mock.patch(
                "data_generation.task_level.pipeline.phase1.run_phase1",
                return_value=[],
            ) as mocked_phase1_impl:
                cli._run_phase1(
                    run_dir=run_dir,
                    args=args,
                    state=mock.Mock(),
                    candidates=[candidate],
                )

        mocked_phase1_impl.assert_called_once()
        _, kwargs = mocked_phase1_impl.call_args
        self.assertTrue(kwargs["sim_normalization"])


if __name__ == "__main__":
    unittest.main()
