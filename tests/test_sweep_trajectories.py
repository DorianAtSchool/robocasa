"""Tests for trajectory sweep worker orchestration and wrapper parsing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT_PATH = REPO_ROOT / "scripts" / "sweep_trajectories.py"
SWEEP_WRAPPER_PATH = REPO_ROOT / "scripts" / "generate_and_insert_images.sh"

_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "sweep_trajectories_script",
    SWEEP_SCRIPT_PATH,
)
assert _SWEEP_SPEC is not None
assert _SWEEP_SPEC.loader is not None
sweep_trajectories_script = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(sweep_trajectories_script)


class FakeProgressDisplay:
    """Capture sweep progress updates without constructing a real Rich console."""

    def __init__(self) -> None:
        self.total_runs: int | None = None
        self.worker_count: int | None = None
        self.overall_updates: list[int] = []
        self.worker_assignments: list[tuple[int, str, int, int]] = []
        self.worker_run_completions: list[tuple[int, str, int, int, int]] = []
        self.worker_finalizations: list[tuple[int, int, int]] = []
        self.log_lines: list[str] = []
        self.closed = False

    def assign_worker(
        self,
        worker_slot: int,
        *,
        task_name: str,
        traj_idx: int,
        total_runs: int,
    ) -> None:
        self.worker_assignments.append((worker_slot, task_name, traj_idx, total_runs))

    def record_run_completion(
        self,
        worker_slot: int,
        *,
        result: dict[str, object],
    ) -> None:
        self.overall_updates.append(1)
        self.worker_run_completions.append(
            (
                worker_slot,
                str(result["status"]),
                int(result["layout"]),
                int(result["style"]),
                int(result["seed"]),
            )
        )

    def finalize_worker(
        self,
        worker_slot: int,
        *,
        error_count: int,
        missing_runs: int,
    ) -> None:
        if missing_runs > 0:
            self.overall_updates.append(missing_runs)
        self.worker_finalizations.append((worker_slot, error_count, missing_runs))

    def write_log_line(self, log_line: str) -> None:
        self.log_lines.append(log_line)
        print(log_line, file=sys.stderr)

    def close(self) -> None:
        self.closed = True


class SweepTrajectoryWorkerTests(unittest.TestCase):
    """Validate that parallel sweep execution stays deterministic."""

    def test_rich_progress_display_uses_non_expanding_layout(self) -> None:
        fake_progress = mock.Mock()
        fake_progress.add_task.return_value = 101

        with mock.patch.object(
            sweep_trajectories_script,
            "Console",
            return_value=mock.sentinel.console,
        ) as console_cls:
            with mock.patch.object(
                sweep_trajectories_script,
                "TextColumn",
                side_effect=[
                    mock.sentinel.description_column,
                    mock.sentinel.status_column,
                ],
            ):
                with mock.patch.object(
                    sweep_trajectories_script,
                    "BarColumn",
                    return_value=mock.sentinel.bar_column,
                ):
                    with mock.patch.object(
                        sweep_trajectories_script,
                        "TaskProgressColumn",
                        return_value=mock.sentinel.task_progress_column,
                    ):
                        with mock.patch.object(
                            sweep_trajectories_script,
                            "MofNCompleteColumn",
                            return_value=mock.sentinel.mofn_column,
                        ):
                            with mock.patch.object(
                                sweep_trajectories_script,
                                "StaticQueuedTimeElapsedColumn",
                                return_value=mock.sentinel.elapsed_column,
                            ):
                                with mock.patch.object(
                                    sweep_trajectories_script,
                                    "SweepOverallEtaColumn",
                                    return_value=mock.sentinel.eta_column,
                                ) as eta_column_cls:
                                    with mock.patch.object(
                                        sweep_trajectories_script,
                                        "RichProgress",
                                        return_value=fake_progress,
                                    ) as rich_progress:
                                        display = sweep_trajectories_script.SweepRichProgressDisplay(
                                            total_runs=4,
                                            worker_count=2,
                                        )

        console_cls.assert_called_once_with(stderr=True)
        eta_column_cls.assert_called_once_with()
        self.assertEqual(
            rich_progress.call_args.kwargs["console"], mock.sentinel.console
        )
        self.assertFalse(rich_progress.call_args.kwargs["expand"])
        fake_progress.start.assert_called_once()
        self.assertEqual(fake_progress.add_task.call_count, 1)
        self.assertIn("runs", fake_progress.add_task.call_args_list[0].args[0])
        self.assertTrue(fake_progress.add_task.call_args_list[0].kwargs["show_eta"])
        display.close()
        fake_progress.stop.assert_called_once()

    def test_execute_sweep_preserves_summary_order_with_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output"
            entries = [
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000000.json",
                    "traj_idx": 0,
                },
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000001.json",
                    "traj_idx": 1,
                },
            ]
            combos = ((11, 34, 42), (56, 42, 99))
            observed_output_dirs: list[Path] = []

            def fake_run_one(**kwargs):
                traj_path = Path(kwargs["traj_file"])
                if traj_path.stem.endswith("000000"):
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)
                observed_output_dirs.append(Path(kwargs["output_dir"]))
                return {
                    "status": "ok",
                    "task": "PrepareCoffee",
                    "steps_succeeded": 5,
                    "steps_total": 6,
                    "images_rendered": 7,
                }

            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(
                    sweep_trajectories_script,
                    "run_one",
                    side_effect=fake_run_one,
                ):
                    results = sweep_trajectories_script.execute_sweep(
                        entries,
                        combos=combos,
                        output_root=output_root,
                        workers=2,
                        robots=2,
                        placement="grid",
                        cell_size=0.05,
                        robot_spawn="trajectory",
                        skip_videos=True,
                        executor_factory=ThreadPoolExecutor,
                    )

        self.assertEqual(
            [
                (result["traj_idx"], result["layout"], result["style"], result["seed"])
                for result in results
            ],
            [
                (0, 11, 34, 42),
                (0, 56, 42, 99),
                (1, 11, 34, 42),
                (1, 56, 42, 99),
            ],
        )
        self.assertEqual(
            sorted(
                path.relative_to(output_root).as_posix()
                for path in observed_output_dirs
            ),
            [
                "prepare_coffee/traj_000000/L11_S34_sd42",
                "prepare_coffee/traj_000000/L56_S42_sd99",
                "prepare_coffee/traj_000001/L11_S34_sd42",
                "prepare_coffee/traj_000001/L56_S42_sd99",
            ],
        )

    def test_execute_sweep_quiet_mode_updates_progress_and_hides_success_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output"
            entries = [
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000000.json",
                    "traj_idx": 0,
                },
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000001.json",
                    "traj_idx": 1,
                },
            ]
            progress_display = FakeProgressDisplay()

            def fake_progress_factory(
                *,
                total_runs: int,
                worker_count: int,
            ) -> FakeProgressDisplay:
                progress_display.total_runs = total_runs
                progress_display.worker_count = worker_count
                return progress_display

            def fake_run_one(**kwargs):
                traj_path = Path(kwargs["traj_file"])
                print(f"stdout from {traj_path.name}")
                print(
                    "[robosuite INFO] Loading controller configuration", file=sys.stderr
                )
                if traj_path.stem.endswith("000001"):
                    print("WARNING: retained diagnostic", file=sys.stderr)
                    raise RuntimeError("worker boom")
                return {
                    "status": "ok",
                    "task": "PrepareCoffee",
                    "steps_succeeded": 5,
                    "steps_total": 6,
                    "images_rendered": 7,
                }

            with contextlib.redirect_stdout(io.StringIO()) as stdout_buffer:
                with contextlib.redirect_stderr(io.StringIO()) as stderr_buffer:
                    with mock.patch.object(
                        sweep_trajectories_script,
                        "run_one",
                        side_effect=fake_run_one,
                    ):
                        results = sweep_trajectories_script.execute_sweep(
                            entries,
                            combos=((11, 34, 42),),
                            output_root=output_root,
                            workers=1,
                            robots=2,
                            placement="grid",
                            cell_size=0.05,
                            robot_spawn="trajectory",
                            skip_videos=True,
                            show_progress=True,
                            log_run_completions=False,
                            suppress_run_stdout=True,
                            progress_factory=fake_progress_factory,
                        )

        self.assertEqual(
            [result["status"] for result in results],
            ["ok", "error"],
        )
        self.assertEqual(progress_display.total_runs, 2)
        self.assertEqual(progress_display.worker_count, 1)
        self.assertEqual(progress_display.overall_updates, [1, 1])
        self.assertEqual(
            progress_display.worker_assignments,
            [
                (0, "prepare_coffee", 0, 1),
                (0, "prepare_coffee", 1, 1),
            ],
        )
        self.assertEqual(
            progress_display.worker_finalizations,
            [(0, 0, 0), (0, 1, 0)],
        )
        self.assertTrue(progress_display.closed)
        self.assertEqual(stdout_buffer.getvalue(), "")
        self.assertNotIn("[robosuite INFO]", stderr_buffer.getvalue())
        self.assertIn("WARNING: retained diagnostic", stderr_buffer.getvalue())
        self.assertIn("ERROR: worker boom", stderr_buffer.getvalue())
        self.assertNotIn(
            "prepare_coffee/traj_000000",
            stderr_buffer.getvalue(),
        )

    def test_execute_sweep_parallel_progress_reuses_worker_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output"
            entries = [
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000000.json",
                    "traj_idx": 0,
                },
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000001.json",
                    "traj_idx": 1,
                },
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": temp_path / "traj_000002.json",
                    "traj_idx": 2,
                },
            ]
            progress_display = FakeProgressDisplay()

            def fake_progress_factory(
                *,
                total_runs: int,
                worker_count: int,
            ) -> FakeProgressDisplay:
                progress_display.total_runs = total_runs
                progress_display.worker_count = worker_count
                return progress_display

            def fake_run_one(**kwargs):
                traj_path = Path(kwargs["traj_file"])
                if traj_path.stem.endswith("000000"):
                    time.sleep(0.05)
                else:
                    time.sleep(0.01)
                return {
                    "status": "ok",
                    "task": "PrepareCoffee",
                    "steps_succeeded": 5,
                    "steps_total": 6,
                    "images_rendered": 7,
                }

            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    with mock.patch.object(
                        sweep_trajectories_script,
                        "run_one",
                        side_effect=fake_run_one,
                    ):
                        sweep_trajectories_script.execute_sweep(
                            entries,
                            combos=((11, 34, 42), (56, 42, 99)),
                            output_root=output_root,
                            workers=2,
                            robots=2,
                            placement="grid",
                            cell_size=0.05,
                            robot_spawn="trajectory",
                            skip_videos=True,
                            executor_factory=ThreadPoolExecutor,
                            progress_factory=fake_progress_factory,
                        )

        self.assertEqual(progress_display.total_runs, 6)
        self.assertEqual(progress_display.worker_count, 2)
        self.assertEqual(
            progress_display.worker_assignments,
            [
                (0, "prepare_coffee", 0, 2),
                (1, "prepare_coffee", 1, 2),
                (1, "prepare_coffee", 2, 2),
            ],
        )
        self.assertEqual(sum(progress_display.overall_updates), 6)
        self.assertEqual(
            [
                worker_slot
                for worker_slot, _, _ in progress_display.worker_finalizations
            ],
            [1, 1, 0],
        )


class SweepTrajectoryCliTests(unittest.TestCase):
    """Validate user-facing CLI output for quiet and non-quiet runs."""

    def test_main_quiet_mode_still_prints_final_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            summary_path = output_dir / "sweep_summary.json"
            fake_results = [
                {
                    "status": "ok",
                    "task_dir": "prepare_coffee",
                    "traj_idx": 0,
                    "layout": 11,
                    "style": 34,
                    "seed": 42,
                }
            ]
            observed_verbose_setting: list[str] = []

            def fake_execute_sweep(*args, **kwargs):
                observed_verbose_setting.append(os.environ["ROBOCASA_SWEEP_VERBOSE"])
                return fake_results

            with contextlib.redirect_stdout(io.StringIO()) as stdout_buffer:
                with contextlib.redirect_stderr(io.StringIO()) as stderr_buffer:
                    with mock.patch.object(
                        sweep_trajectories_script,
                        "discover_trajectories",
                        return_value=[
                            {
                                "task_dir_name": "prepare_coffee",
                                "traj_file": input_dir / "traj_000000.json",
                                "traj_idx": 0,
                            }
                        ],
                    ):
                        with mock.patch.object(
                            sweep_trajectories_script,
                            "execute_sweep",
                            side_effect=fake_execute_sweep,
                        ):
                            with mock.patch.object(
                                sys,
                                "argv",
                                [
                                    "sweep_trajectories.py",
                                    "--input-dir",
                                    str(input_dir),
                                    "--output-dir",
                                    str(output_dir),
                                    "--quiet",
                                ],
                            ):
                                sweep_trajectories_script.main()

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(stderr_buffer.getvalue(), "")
            self.assertEqual(
                stdout_buffer.getvalue(),
                f"\nDone: 1/1 succeeded\nSummary: {summary_path}\n",
            )
            self.assertEqual(observed_verbose_setting, ["0"])
            self.assertEqual(summary_payload["succeeded"], 1)

    def test_main_non_quiet_enables_simulator_verbose_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            observed_verbose_setting: list[str] = []

            def fake_execute_sweep(*args, **kwargs):
                observed_verbose_setting.append(os.environ["ROBOCASA_SWEEP_VERBOSE"])
                return [
                    {
                        "status": "ok",
                        "task_dir": "prepare_coffee",
                        "traj_idx": 0,
                        "layout": 11,
                        "style": 34,
                        "seed": 42,
                    }
                ]

            with mock.patch.object(
                sweep_trajectories_script,
                "discover_trajectories",
                return_value=[
                    {
                        "task_dir_name": "prepare_coffee",
                        "traj_file": input_dir / "traj_000000.json",
                        "traj_idx": 0,
                    }
                ],
            ):
                with mock.patch.object(
                    sweep_trajectories_script,
                    "execute_sweep",
                    side_effect=fake_execute_sweep,
                ):
                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "sweep_trajectories.py",
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

        self.assertEqual(observed_verbose_setting, ["1"])


class SweepTaskLevelWrapperTests(unittest.TestCase):
    """Validate wrapper-owned argument parsing for task-level sweep runs."""

    def _run_wrapper_with_stub_python(
        self,
        *,
        wrapper_args: list[str],
    ) -> tuple[subprocess.CompletedProcess[str], list[str], Path, Path]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_root = temp_path / "task_level_data"
            run_timestamp = "20260401T000000Z"
            input_dir = data_root / "pre_image" / run_timestamp
            output_dir = data_root / "image" / run_timestamp
            input_dir.mkdir(parents=True)

            stub_bin_dir = temp_path / "bin"
            stub_bin_dir.mkdir()
            captured_args_path = temp_path / "captured_args.txt"
            stub_python_path = stub_bin_dir / "python"
            stub_python_path.write_text(
                "#!/usr/bin/env bash\n"
                'printf \'%s\\n\' "$@" > "$CAPTURED_ARGS_PATH"\n',
                encoding="utf-8",
            )
            stub_python_path.chmod(0o755)

            env = os.environ.copy()
            env["CAPTURED_ARGS_PATH"] = str(captured_args_path)
            env["PATH"] = f"{stub_bin_dir}:{env['PATH']}"
            env["ROBOCASA_TASK_LEVEL_DATA_ROOT"] = str(data_root)

            completed = subprocess.run(
                ["bash", str(SWEEP_WRAPPER_PATH), *wrapper_args],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            captured_args = captured_args_path.read_text(encoding="utf-8").splitlines()

        return completed, captured_args, input_dir, output_dir

    def test_wrapper_defaults_to_quiet_python_cli(self) -> None:
        completed, captured_args, input_dir, output_dir = (
            self._run_wrapper_with_stub_python(
                wrapper_args=[
                    "20260401T000000Z",
                    "--workers",
                    "3",
                    "--layouts",
                    "11",
                    "--styles",
                    "34",
                ]
            )
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(
            captured_args,
            [
                "scripts/sweep_trajectories.py",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "3",
                "--quiet",
                "--layouts",
                "11",
                "--styles",
                "34",
            ],
        )

    def test_wrapper_verbose_mode_skips_quiet_python_cli_flag(self) -> None:
        completed, captured_args, input_dir, output_dir = (
            self._run_wrapper_with_stub_python(
                wrapper_args=[
                    "20260401T000000Z",
                    "--workers",
                    "3",
                    "--verbose",
                    "--layouts",
                    "11",
                ]
            )
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(
            captured_args,
            [
                "scripts/sweep_trajectories.py",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "3",
                "--layouts",
                "11",
            ],
        )


if __name__ == "__main__":
    unittest.main()
