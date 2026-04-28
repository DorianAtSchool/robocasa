"""Tests for trajectory sweep worker orchestration and wrapper parsing."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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

    def setUp(self) -> None:
        sweep_trajectories_script.clear_executor_cache()

    def tearDown(self) -> None:
        sweep_trajectories_script.clear_executor_cache()

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
                                        "SweepAverageTrajectoryTimeColumn",
                                        return_value=mock.sentinel.avg_column,
                                    ) as avg_column_cls:
                                        with mock.patch.object(
                                            sweep_trajectories_script,
                                            "SweepTrajectoryRateColumn",
                                            return_value=mock.sentinel.rate_column,
                                        ) as rate_column_cls:
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
        eta_column_cls.assert_called_once()
        self.assertEqual(len(eta_column_cls.call_args.args), 1)
        self.assertTrue(callable(eta_column_cls.call_args.args[0]))
        avg_column_cls.assert_called_once()
        self.assertTrue(callable(avg_column_cls.call_args.args[0]))
        rate_column_cls.assert_called_once()
        self.assertTrue(callable(rate_column_cls.call_args.args[0]))
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

    def test_rich_progress_display_estimates_eta_from_average_trajectory_time(
        self,
    ) -> None:
        fake_progress = mock.Mock()
        fake_progress.add_task.return_value = 101

        with mock.patch.object(
            sweep_trajectories_script,
            "Text",
            mock.Mock(),
        ):
            with mock.patch.object(
                sweep_trajectories_script,
                "Console",
                return_value=mock.sentinel.console,
            ):
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
                                        "RichProgress",
                                        return_value=fake_progress,
                                    ):
                                        display = sweep_trajectories_script.SweepRichProgressDisplay(
                                            total_runs=6,
                                            worker_count=2,
                                        )

        with mock.patch.object(
            sweep_trajectories_script.time,
            "monotonic",
            side_effect=[0.0, 0.0, 10.0, 12.0, 12.0],
        ):
            display.assign_worker(
                0,
                task_name="prepare_coffee",
                traj_idx=0,
                total_runs=2,
            )
            display.assign_worker(
                1,
                task_name="prepare_coffee",
                traj_idx=1,
                total_runs=2,
            )
            display.finalize_worker(
                0,
                error_count=0,
                missing_runs=0,
            )
            self.assertEqual(display._estimate_remaining_seconds(), 10.0)
            self.assertEqual(display._average_trajectory_seconds(), 10.0)
            self.assertAlmostEqual(display._trajectories_per_minute(), 5.0)
            display.close()

    def test_execute_sweep_returns_empty_without_creating_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = sweep_trajectories_script.execute_sweep(
                [],
                combos=((11, 34, 42),),
                output_root=Path(temp_dir),
                workers=112,
                robots=2,
                placement="grid",
                cell_size=0.05,
            )

        self.assertEqual(results, [])

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

    def test_get_gpu_allocation_supports_round_robin_and_custom_counts(self) -> None:
        self.assertEqual(
            sweep_trajectories_script.get_gpu_allocation(5, [0, 1]),
            [0, 1, 0, 1, 0],
        )
        self.assertEqual(
            sweep_trajectories_script.get_gpu_allocation(5, [0, 1], [3, 2]),
            [0, 0, 0, 1, 1],
        )
        self.assertEqual(
            sweep_trajectories_script.get_gpu_allocation(4, [0, 1], [3, 3]),
            [0, 0, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "same length as --gpu-ids"):
            sweep_trajectories_script.get_gpu_allocation(2, [0, 1], [2])

    def test_run_trajectory_entry_scopes_gpu_env_and_gl_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            traj_file = temp_path / "traj_000000.json"
            traj_file.write_text(
                json.dumps({"composite_task": "PrepareCoffee"}),
                encoding="utf-8",
            )
            entry = {
                "task_dir_name": "prepare_coffee",
                "traj_file": traj_file,
                "traj_idx": 0,
            }
            observed_calls: list[
                tuple[str | None, str | None, str | None, str | None, str]
            ] = []
            env_keys = (
                "CUDA_VISIBLE_DEVICES",
                "GPUS",
                "MUJOCO_EGL_DEVICE_ID",
                "MUJOCO_GL",
            )
            original_env = {key: os.environ.get(key) for key in env_keys}

            def fake_run_one(**kwargs):
                observed_calls.append(
                    (
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                        os.environ.get("GPUS"),
                        os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                        os.environ.get("MUJOCO_GL"),
                        str(kwargs["gl_backend"]),
                    )
                )
                return {
                    "status": "ok",
                    "task": "PrepareCoffee",
                    "steps_succeeded": 5,
                    "steps_total": 6,
                    "images_rendered": 7,
                }

            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "prior-visible"
                os.environ["GPUS"] = "prior-gpus"
                os.environ["MUJOCO_EGL_DEVICE_ID"] = "prior-egl"
                with mock.patch.object(
                    sweep_trajectories_script,
                    "run_one",
                    side_effect=fake_run_one,
                ):
                    result = sweep_trajectories_script.run_trajectory_entry(
                        entry,
                        entry_index=0,
                        total_runs=1,
                        combos=((11, 34, 42),),
                        output_root=temp_path / "output",
                        robots=2,
                        placement="grid",
                        cell_size=0.05,
                        robot_spawn="trajectory",
                        skip_videos=True,
                        gpu_id=7,
                        gl_backend="egl",
                    )
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(
            observed_calls,
            [("7", "7", "7", "egl", "egl")],
        )
        self.assertEqual(
            os.environ.get("CUDA_VISIBLE_DEVICES"), original_env["CUDA_VISIBLE_DEVICES"]
        )
        self.assertEqual(os.environ.get("GPUS"), original_env["GPUS"])
        self.assertEqual(
            os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            original_env["MUJOCO_EGL_DEVICE_ID"],
        )
        self.assertEqual(os.environ.get("MUJOCO_GL"), original_env["MUJOCO_GL"])

    def test_run_trajectory_entry_clears_stale_egl_env_for_osmesa(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            traj_file = temp_path / "traj_000000.json"
            traj_file.write_text(
                json.dumps({"composite_task": "PrepareCoffee"}),
                encoding="utf-8",
            )
            entry = {
                "task_dir_name": "prepare_coffee",
                "traj_file": traj_file,
                "traj_idx": 0,
            }
            observed_calls: list[tuple[str | None, str | None, str]] = []
            env_keys = ("MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL")
            original_env = {key: os.environ.get(key) for key in env_keys}

            def fake_run_one(**kwargs):
                observed_calls.append(
                    (
                        os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                        os.environ.get("MUJOCO_GL"),
                        str(kwargs["gl_backend"]),
                    )
                )
                return {
                    "status": "ok",
                    "task": "PrepareCoffee",
                    "steps_succeeded": 5,
                    "steps_total": 6,
                    "images_rendered": 7,
                }

            try:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = "prior-egl"
                os.environ["MUJOCO_GL"] = "egl"
                with mock.patch.object(
                    sweep_trajectories_script,
                    "run_one",
                    side_effect=fake_run_one,
                ):
                    result = sweep_trajectories_script.run_trajectory_entry(
                        entry,
                        entry_index=0,
                        total_runs=1,
                        combos=((11, 34, 42),),
                        output_root=temp_path / "output",
                        robots=2,
                        placement="grid",
                        cell_size=0.05,
                        robot_spawn="trajectory",
                        skip_videos=True,
                        gpu_id=None,
                        gl_backend="osmesa",
                    )
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(observed_calls, [(None, "osmesa", "osmesa")])
        self.assertEqual(
            os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            original_env["MUJOCO_EGL_DEVICE_ID"],
        )
        self.assertEqual(os.environ.get("MUJOCO_GL"), original_env["MUJOCO_GL"])
        self.assertEqual([run["status"] for run in result["results"]], ["ok"])

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

    def test_execute_sweep_forwards_max_tasks_per_child_to_process_pool(self) -> None:
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
            combos = ((11, 34, 42),)
            observed_executor_kwargs: dict[str, object] = {}

            class FakeProcessPoolExecutor:
                """Capture ProcessPoolExecutor kwargs without spawning workers."""

                def __init__(
                    self,
                    max_workers: int,
                    mp_context: object | None = None,
                    max_tasks_per_child: int | None = None,
                ) -> None:
                    observed_executor_kwargs["max_workers"] = max_workers
                    observed_executor_kwargs["mp_context"] = mp_context
                    observed_executor_kwargs[
                        "max_tasks_per_child"
                    ] = max_tasks_per_child

                def __enter__(self) -> "FakeProcessPoolExecutor":
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

                def submit(self, fn, *args, **kwargs) -> Future:
                    future: Future = Future()
                    future.set_result(fn(*args, **kwargs))
                    return future

            def fake_run_trajectory_entry(
                entry: dict[str, object],
                *,
                entry_index: int,
                combos: tuple[tuple[int, int, int], ...],
                **kwargs,
            ) -> dict[str, object]:
                del entry_index, kwargs
                return {
                    "entry_index": 0,
                    "diagnostic_lines": [],
                    "log_lines": [],
                    "results": [
                        {
                            "status": "ok",
                            "task_dir": entry["task_dir_name"],
                            "traj_idx": entry["traj_idx"],
                            "layout": layout,
                            "style": style,
                            "seed": seed,
                        }
                        for layout, style, seed in combos
                    ],
                }

            with mock.patch.object(
                sweep_trajectories_script,
                "ProcessPoolExecutor",
                FakeProcessPoolExecutor,
            ):
                with mock.patch.object(
                    sweep_trajectories_script.multiprocessing,
                    "get_context",
                    return_value=mock.sentinel.spawn_context,
                ):
                    with mock.patch.object(
                        sweep_trajectories_script,
                        "run_trajectory_entry",
                        side_effect=fake_run_trajectory_entry,
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.execute_sweep(
                                    entries,
                                    combos=combos,
                                    output_root=output_root,
                                    workers=2,
                                    robots=2,
                                    placement="grid",
                                    cell_size=0.05,
                                    robot_spawn="trajectory",
                                    skip_videos=True,
                                    max_tasks_per_child=1,
                                )

        self.assertEqual(observed_executor_kwargs["max_workers"], 2)
        self.assertIs(
            observed_executor_kwargs["mp_context"],
            mock.sentinel.spawn_context,
        )
        self.assertEqual(observed_executor_kwargs["max_tasks_per_child"], 1)

    def test_execute_sweep_interrupt_terminates_active_executor(self) -> None:
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
            observed_executor: dict[str, object] = {}

            class FakeWorkerProcess:
                """Expose process lifecycle hooks used by interrupt cleanup."""

                def __init__(self) -> None:
                    self.alive = True
                    self.terminate_calls = 0
                    self.kill_calls = 0
                    self.join_timeouts: list[float | None] = []

                def is_alive(self) -> bool:
                    return self.alive

                def terminate(self) -> None:
                    self.terminate_calls += 1
                    self.alive = False

                def join(self, timeout: float | None = None) -> None:
                    self.join_timeouts.append(timeout)

                def kill(self) -> None:
                    self.kill_calls += 1
                    self.alive = False

            class FakeInterruptingExecutor:
                """Capture immediate shutdown requests during cancellation."""

                def __init__(self, max_workers: int) -> None:
                    del max_workers
                    self.shutdown_calls: list[tuple[bool, bool]] = []
                    self._processes = {
                        0: FakeWorkerProcess(),
                        1: FakeWorkerProcess(),
                    }
                    observed_executor["executor"] = self

                def __enter__(self) -> "FakeInterruptingExecutor":
                    return self

                def __exit__(self, exc_type, exc, tb) -> bool:
                    return False

                def submit(self, fn, *args, **kwargs) -> Future:
                    del fn, args, kwargs
                    future: Future = Future()
                    return future

                def shutdown(
                    self,
                    wait: bool = True,
                    cancel_futures: bool = False,
                ) -> None:
                    self.shutdown_calls.append((wait, cancel_futures))

            with mock.patch.object(
                sweep_trajectories_script,
                "wait",
                side_effect=KeyboardInterrupt,
            ):
                with mock.patch.object(
                    sweep_trajectories_script,
                    "clear_executor_cache",
                ) as clear_executor_cache:
                    with self.assertRaises(KeyboardInterrupt):
                        sweep_trajectories_script.execute_sweep(
                            entries,
                            combos=((11, 34, 42),),
                            output_root=output_root,
                            workers=2,
                            robots=2,
                            placement="grid",
                            cell_size=0.05,
                            robot_spawn="trajectory",
                            skip_videos=True,
                            executor_factory=FakeInterruptingExecutor,
                        )

            fake_executor = observed_executor["executor"]
            assert isinstance(fake_executor, FakeInterruptingExecutor)
            self.assertEqual(fake_executor.shutdown_calls, [(False, True)])
            for process in fake_executor._processes.values():
                self.assertEqual(process.terminate_calls, 1)
                self.assertEqual(process.kill_calls, 0)
                self.assertEqual(process.join_timeouts, [0.2])
            clear_executor_cache.assert_called_once_with()

    def test_cached_executor_reuses_matching_config_and_replaces_old_one(self) -> None:
        created_executors = []

        class FakeCachedExecutor:
            """Tracks worker-local executor reuse without constructing the sim."""

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.closed = False
                created_executors.append(self)

            def close(self) -> None:
                self.closed = True

        first = sweep_trajectories_script._get_or_create_cached_executor(
            task_name="PrepareCoffee",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            placement="grid",
            cell_size=0.05,
            robot_spawn="trajectory",
            gl_backend="egl",
            executor_factory=FakeCachedExecutor,
        )
        second = sweep_trajectories_script._get_or_create_cached_executor(
            task_name="PrepareCoffee",
            robots=2,
            layout=11,
            style=34,
            seed=42,
            placement="grid",
            cell_size=0.05,
            robot_spawn="trajectory",
            gl_backend="egl",
            executor_factory=FakeCachedExecutor,
        )
        third = sweep_trajectories_script._get_or_create_cached_executor(
            task_name="PrepareCoffee",
            robots=2,
            layout=11,
            style=34,
            seed=99,
            placement="grid",
            cell_size=0.05,
            robot_spawn="trajectory",
            gl_backend="egl",
            executor_factory=FakeCachedExecutor,
        )
        fourth = sweep_trajectories_script._get_or_create_cached_executor(
            task_name="PrepareCoffee",
            robots=2,
            layout=11,
            style=34,
            seed=99,
            placement="grid",
            cell_size=0.05,
            robot_spawn="trajectory",
            gl_backend="egl",
            render_width=256,
            render_height=256,
            executor_factory=FakeCachedExecutor,
        )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertIsNot(third, fourth)
        self.assertEqual(len(created_executors), 3)
        self.assertTrue(first.closed)
        self.assertTrue(third.closed)
        self.assertFalse(fourth.closed)

    def test_run_one_restores_cached_executor_before_execution(self) -> None:
        class FakeCachedExecutor:
            """Minimal cached executor used to verify pre-run restore behavior."""

            def __init__(self) -> None:
                self.restore_calls = 0

            def restore_baseline_state(self) -> None:
                self.restore_calls += 1

        fake_executor = FakeCachedExecutor()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            traj_file = temp_path / "traj_000000.json"
            traj_file.write_text(
                json.dumps({"composite_task": "PrepareCoffee"}),
                encoding="utf-8",
            )
            output_dir = temp_path / "output"

            with mock.patch.object(
                sweep_trajectories_script,
                "_get_or_create_cached_executor",
                return_value=fake_executor,
            ):
                with mock.patch(
                    "robocasa.utils.trajectory_adapter.execute_trajectory",
                    return_value={
                        "steps": [
                            {"tool": "navigate_to_fixture", "success": True},
                            {"tool": "get_image", "success": True},
                        ]
                    },
                ):
                    result = sweep_trajectories_script.run_one(
                        traj_file=traj_file,
                        output_dir=output_dir,
                        layout=11,
                        style=34,
                        seed=42,
                        robots=2,
                        placement="grid",
                        cell_size=0.05,
                        robot_spawn="trajectory",
                        skip_videos=True,
                        gl_backend="egl",
                    )

        self.assertEqual(fake_executor.restore_calls, 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["steps_succeeded"], 1)
        self.assertEqual(result["steps_total"], 1)
        self.assertEqual(result["images_rendered"], 1)


class SweepTrajectoryShardTests(unittest.TestCase):
    """Validate deterministic trajectory sharding behavior."""

    def test_select_trajectory_shard_uses_round_robin_order(self) -> None:
        entries = [
            {
                "task_dir_name": "task_a",
                "traj_file": Path(f"traj_{idx}.json"),
                "traj_idx": idx,
            }
            for idx in range(7)
        ]

        shard_entries = sweep_trajectories_script.select_trajectory_shard(
            entries,
            num_shards=3,
            shard_index=1,
        )

        self.assertEqual(
            [entry["traj_idx"] for entry in shard_entries],
            [1, 4],
        )

    def test_select_trajectory_shard_rejects_out_of_range_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "--shard-index"):
            sweep_trajectories_script.select_trajectory_shard(
                [],
                num_shards=2,
                shard_index=2,
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

    def test_main_writes_custom_summary_path_for_sharded_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            custom_summary_path = temp_path / "summaries" / "shard_1.json"
            input_dir.mkdir()
            observed_kwargs: dict[str, object] = {}

            def fake_execute_sweep(*args, **kwargs):
                observed_kwargs.update(kwargs)
                return [
                    {
                        "status": "ok",
                        "task_dir": "prepare_coffee",
                        "traj_idx": 1,
                        "layout": 11,
                        "style": 34,
                        "seed": 42,
                    }
                ]

            discovered_entries = [
                {
                    "task_dir_name": "prepare_coffee",
                    "traj_file": input_dir / f"traj_{idx:06d}.json",
                    "traj_idx": idx,
                }
                for idx in range(3)
            ]

            with mock.patch.object(
                sweep_trajectories_script,
                "discover_trajectories",
                return_value=discovered_entries,
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
                            "--summary-path",
                            str(custom_summary_path),
                            "--num-shards",
                            "2",
                            "--shard-index",
                            "1",
                            "--quiet",
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

            summary_payload = json.loads(custom_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["trajectories"], 1)
            self.assertEqual(summary_payload["num_shards"], 2)
            self.assertEqual(summary_payload["shard_index"], 1)
            self.assertEqual(observed_kwargs["workers"], 1)

    def test_main_allows_empty_shard_and_writes_zero_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            execute_calls = 0

            def fake_execute_sweep(*args, **kwargs):
                nonlocal execute_calls
                execute_calls += 1
                return []

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
                            "--num-shards",
                            "4",
                            "--shard-index",
                            "3",
                            "--quiet",
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

            summary_payload = json.loads(
                (output_dir / "sweep_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(execute_calls, 1)
            self.assertEqual(summary_payload["trajectories"], 0)
            self.assertEqual(summary_payload["total"], 0)
            self.assertEqual(summary_payload["results"], [])

    def test_main_gpu_ids_default_to_egl_and_forward_allocation_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            observed_kwargs: dict[str, object] = {}

            def fake_execute_sweep(*args, **kwargs):
                observed_kwargs.update(kwargs)
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
                            "--workers",
                            "4",
                            "--gpu-ids",
                            "0",
                            "1",
                            "--procs-per-gpu",
                            "2",
                            "2",
                            "--quiet",
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

        self.assertEqual(observed_kwargs["gpu_ids"], [0, 1])
        self.assertEqual(observed_kwargs["procs_per_gpu"], [2, 2])
        self.assertEqual(observed_kwargs["gl_backend"], "egl")

    def test_main_forwards_max_tasks_per_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            observed_kwargs: dict[str, object] = {}

            def fake_execute_sweep(*args, **kwargs):
                observed_kwargs.update(kwargs)
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
                            "--workers",
                            "4",
                            "--max-tasks-per-child",
                            "1",
                            "--quiet",
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

        self.assertEqual(observed_kwargs["max_tasks_per_child"], 1)

    def test_main_forwards_render_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            observed_kwargs: dict[str, object] = {}

            def fake_execute_sweep(*args, **kwargs):
                observed_kwargs.update(kwargs)
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
                            "--render-width",
                            "256",
                            "--render-height",
                            "192",
                            "--quiet",
                        ],
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                sweep_trajectories_script.main()

        self.assertEqual(observed_kwargs["render_width"], 256)
        self.assertEqual(observed_kwargs["render_height"], 192)

    def test_main_signal_handler_cancels_active_sweep_executor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            registered_handlers: dict[int, object] = {}
            installed_handlers: dict[int, object] = {}
            signal_calls: list[tuple[int, object]] = []

            class FakeSweepExecutor:
                """Expose shutdown calls triggered by the installed signal handler."""

                def __init__(self) -> None:
                    self.shutdown_calls: list[tuple[bool, bool]] = []

                def shutdown(
                    self,
                    wait: bool = True,
                    cancel_futures: bool = False,
                ) -> None:
                    self.shutdown_calls.append((wait, cancel_futures))

            fake_executor = FakeSweepExecutor()
            previous_handlers = {
                sweep_trajectories_script.signal.SIGINT: mock.sentinel.prev_sigint,
                sweep_trajectories_script.signal.SIGTERM: mock.sentinel.prev_sigterm,
            }

            def fake_signal(signum: int, handler: object) -> object:
                signal_calls.append((signum, handler))
                registered_handlers[signum] = handler
                if callable(handler):
                    installed_handlers[signum] = handler
                return mock.sentinel.signal_result

            def fake_execute_sweep(*args, **kwargs):
                del args
                cancellation_controller = kwargs["cancellation_controller"]
                cancellation_controller.attach_executor(fake_executor)
                sigterm_handler = registered_handlers[
                    sweep_trajectories_script.signal.SIGTERM
                ]
                assert callable(sigterm_handler)
                sigterm_handler(sweep_trajectories_script.signal.SIGTERM, None)
                raise AssertionError("unreachable after signal handler")

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
                    sweep_trajectories_script.signal,
                    "getsignal",
                    side_effect=lambda signum: previous_handlers[signum],
                ):
                    with mock.patch.object(
                        sweep_trajectories_script.signal,
                        "signal",
                        side_effect=fake_signal,
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
                                with self.assertRaises(KeyboardInterrupt):
                                    sweep_trajectories_script.main()

        self.assertEqual(fake_executor.shutdown_calls, [(False, True)])
        self.assertEqual(
            signal_calls,
            [
                (
                    sweep_trajectories_script.signal.SIGINT,
                    installed_handlers[sweep_trajectories_script.signal.SIGINT],
                ),
                (
                    sweep_trajectories_script.signal.SIGTERM,
                    installed_handlers[sweep_trajectories_script.signal.SIGTERM],
                ),
                (
                    sweep_trajectories_script.signal.SIGINT,
                    mock.sentinel.prev_sigint,
                ),
                (
                    sweep_trajectories_script.signal.SIGTERM,
                    mock.sentinel.prev_sigterm,
                ),
            ],
        )


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
        (
            completed,
            captured_args,
            input_dir,
            output_dir,
        ) = self._run_wrapper_with_stub_python(
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
        (
            completed,
            captured_args,
            input_dir,
            output_dir,
        ) = self._run_wrapper_with_stub_python(
            wrapper_args=[
                "20260401T000000Z",
                "--workers",
                "3",
                "--verbose",
                "--layouts",
                "11",
            ]
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

    def test_wrapper_forwards_multi_gpu_sweep_args(self) -> None:
        (
            completed,
            captured_args,
            input_dir,
            output_dir,
        ) = self._run_wrapper_with_stub_python(
            wrapper_args=[
                "20260401T000000Z",
                "--workers",
                "4",
                "--gpu-ids",
                "0",
                "1",
                "--procs-per-gpu",
                "2",
                "2",
                "--gl-backend",
                "egl",
            ]
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
                "4",
                "--quiet",
                "--gpu-ids",
                "0",
                "1",
                "--procs-per-gpu",
                "2",
                "2",
                "--gl-backend",
                "egl",
            ],
        )


if __name__ == "__main__":
    unittest.main()
