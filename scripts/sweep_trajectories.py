#!/usr/bin/env python3
"""Sweep all trajectories in a dataset directory through the sim tool executor.

Discovers all traj_*.json files under <input_dir>/<task_name>/trajectories/,
executes each one across all (layout, style, seed) combinations, and writes
outputs: adapted trajectory, execution metadata, and images rendered by
get_image tool calls to their specified paths.

Usage:
    # Single combo (default):
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output

    # Sweep across multiple layouts, styles, and seeds:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 56 \
        --styles 34 42 \
        --seeds 42 99

    # Limit to specific task dirs:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --tasks hot_dog_setup prepare_coffee

    # Limit to specific trajectory indices:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --indices 0 1 2

    # Dry run — show what would be executed without running anything:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 56 --styles 34 42 --seeds 42 99 \
        --dry-run

    # Balanced ~1k trajectory sweep from 24 base trajectories:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output_1k \
        --workers 4 \
        --layouts 11 42 56 \
        --styles 34 42 \
        --seeds 1 2 3 4 5 6 7

    # Sweep and push the flattened dataset to Hugging Face Hub:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 42 56 \
        --styles 34 42 \
        --seeds 1 2 3 4 5 6 7 \
        --push-to-hub DorianAtSchool/robocasa-trajectories-single
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from functools import partial
import io
import itertools
import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Make repo-root imports work when this file is executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.task_level.generation.raw import progress as raw_progress

BarColumn = raw_progress.BarColumn
Console = raw_progress.Console
MofNCompleteColumn = raw_progress.MofNCompleteColumn
OVERALL_PROGRESS_COLOR = raw_progress.OVERALL_PROGRESS_COLOR
PROGRESS_BAR_WIDTH = raw_progress.PROGRESS_BAR_WIDTH
ProgressColumn = raw_progress.ProgressColumn
RichProgress = raw_progress.RichProgress
RichTaskProgressAdapter = raw_progress.RichTaskProgressAdapter
StaticQueuedTimeElapsedColumn = raw_progress.StaticQueuedTimeElapsedColumn
TaskProgressColumn = raw_progress.TaskProgressColumn
Text = raw_progress.Text
TextColumn = raw_progress.TextColumn

try:
    from rich.progress import TimeRemainingColumn
except ImportError:  # pragma: no cover
    TimeRemainingColumn = None

CLI_EPILOG = textwrap.dedent("""\
    Examples:
      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output

      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output \\
        --workers 4 \\
        --tasks hot_dog_setup prepare_coffee \\
        --layouts 11 42 56 \\
        --styles 34 42 \\
        --seeds 1 2 3 4 5 6 7

      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output_1k \\
        --layouts 11 42 56 \\
        --styles 34 42 \\
        --seeds 1 2 3 4 5 6 7 \\
        --push-to-hub DorianAtSchool/robocasa-trajectories-single
    """)

QUIET_DIAGNOSTIC_PATTERN = re.compile(
    r"\b(warn(?:ing)?|error|exception|traceback|critical|fatal)\b",
    re.IGNORECASE,
)


class SweepOverallEtaColumn(ProgressColumn):
    """Shows ETA only for the overall runs row."""

    def __init__(self) -> None:
        """Initializes the shared ETA column for sweep progress."""

        if TimeRemainingColumn is None or Text is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._delegate = TimeRemainingColumn()

    def render(self, task: Any) -> Any:
        """Renders ETA for the overall row and blanks for worker rows."""

        if not getattr(task, "fields", {}).get("show_eta", False):
            return Text("")
        return self._delegate.render(task)


class SweepRichProgressDisplay:
    """Owns the Rich progress layout used by sweep execution."""

    def __init__(self, *, total_runs: int, worker_count: int) -> None:
        """Initializes the shared overall row used by sweep execution."""

        if RichProgress is None or Console is None:
            raise RuntimeError("rich progress support is unavailable")
        # The scheduler still tracks worker slots internally, but the terminal
        # now renders only the overall runs bar.
        _ = worker_count

        self.console = Console(stderr=True)
        self._progress = RichProgress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            StaticQueuedTimeElapsedColumn(),
            SweepOverallEtaColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
            expand=False,
        )
        self._progress.start()
        overall_task_id = self._progress.add_task(
            f"[{OVERALL_PROGRESS_COLOR}]runs[/{OVERALL_PROGRESS_COLOR}]",
            total=total_runs,
            status="running",
            show_eta=True,
        )
        self.overall_progress = RichTaskProgressAdapter(
            self._progress,
            overall_task_id,
            total_runs,
            status="running",
        )

    def assign_worker(
        self,
        worker_slot: int,
        *,
        task_name: str,
        traj_idx: int,
        total_runs: int,
    ) -> None:
        """Keeps compatibility with the sweep scheduler's worker-slot hooks."""

        del worker_slot, task_name, traj_idx, total_runs

    def record_run_completion(
        self,
        worker_slot: int,
        *,
        result: dict[str, Any],
    ) -> None:
        """Advances the overall bar by one completed run."""

        del worker_slot, result
        self.overall_progress.update(1)

    def finalize_worker(
        self,
        worker_slot: int,
        *,
        error_count: int,
        missing_runs: int,
    ) -> None:
        """Backfills the overall bar if a worker exits without progress events."""

        del worker_slot, error_count
        if missing_runs > 0:
            self.overall_progress.update(missing_runs)

    def write_log_line(self, log_line: str) -> None:
        """Prints one log line without corrupting the active Rich display."""

        self.console.print(log_line, markup=False, highlight=False)

    def close(self) -> None:
        """Stops the Rich progress display."""

        self._progress.stop()


def _create_sweep_progress_display(
    *,
    total_runs: int,
    worker_count: int,
    progress_factory: Callable[..., Any] | None = None,
) -> Any | None:
    """Builds the interactive sweep progress display when possible."""

    if total_runs <= 0 or worker_count <= 0:
        return None
    if progress_factory is not None:
        return progress_factory(total_runs=total_runs, worker_count=worker_count)
    if not os.isatty(2) or RichProgress is None or Console is None:
        return None
    return SweepRichProgressDisplay(total_runs=total_runs, worker_count=worker_count)


def _record_local_progress_event(
    progress_display: Any,
    slot_completed_runs: dict[int, int],
    worker_slot: int,
    event: dict[str, Any],
) -> None:
    """Applies one run-completed event directly in the main process."""

    _apply_progress_event(
        progress_display,
        slot_completed_runs,
        {"worker_slot": worker_slot, **event},
    )


def _enqueue_progress_event(
    progress_queue: Any,
    worker_slot: int,
    event: dict[str, Any],
) -> None:
    """Queues one run-completed event from a background worker."""

    progress_queue.put({"worker_slot": worker_slot, **event})


def _apply_progress_event(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    event: dict[str, Any],
) -> None:
    """Applies one queued run-completed event to the active display."""

    if progress_display is None:
        return
    worker_slot = int(event["worker_slot"])
    slot_completed_runs[worker_slot] = slot_completed_runs.get(worker_slot, 0) + 1
    progress_display.record_run_completion(
        worker_slot,
        result={
            "status": event["status"],
            "layout": event["layout"],
            "style": event["style"],
            "seed": event["seed"],
        },
    )


def _drain_progress_events(
    progress_queue: Any | None,
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
) -> None:
    """Consumes every pending background progress event."""

    if progress_queue is None or progress_display is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        _apply_progress_event(progress_display, slot_completed_runs, event)


def _create_progress_event_queue(
    *,
    executor_factory: Callable[..., Any],
) -> tuple[Any, Any | None]:
    """Builds the queue used to stream run completions back to the main process."""

    if executor_factory is ThreadPoolExecutor:
        return queue.Queue(), None
    progress_manager = multiprocessing.Manager()
    return progress_manager.Queue(), progress_manager


def _assign_progress_worker(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    *,
    worker_slot: int,
    entry: dict[str, Any],
    combo_count: int,
) -> None:
    """Resets one worker row for a newly scheduled trajectory entry."""

    if progress_display is None:
        return
    slot_completed_runs[worker_slot] = 0
    progress_display.assign_worker(
        worker_slot,
        task_name=str(entry["task_dir_name"]),
        traj_idx=int(entry["traj_idx"]),
        total_runs=combo_count,
    )


def _finalize_progress_worker(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    *,
    worker_slot: int,
    results: list[dict[str, Any]],
) -> None:
    """Completes one worker row once its trajectory entry finishes."""

    if progress_display is None:
        return
    completed_runs = slot_completed_runs.get(worker_slot, 0)
    total_runs = len(results)
    error_count = sum(1 for result in results if result["status"] == "error")
    progress_display.finalize_worker(
        worker_slot,
        error_count=error_count,
        missing_runs=max(total_runs - completed_runs, 0),
    )
    slot_completed_runs[worker_slot] = 0


def _emit_sweep_log_line(
    log_line: str,
    *,
    log_writer: Callable[[str], None] | None,
    use_stderr: bool,
) -> None:
    """Write one log line without corrupting an active progress display."""

    if log_writer is not None:
        log_writer(log_line)
        return
    output_stream = sys.stderr if use_stderr else sys.stdout
    print(log_line, file=output_stream, flush=True)


def _quiet_diagnostic_lines(stderr_text: str) -> list[str]:
    """Keep only warning and error stderr lines for quiet sweep runs."""

    diagnostics: list[str] = []
    for raw_line in stderr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if QUIET_DIAGNOSTIC_PATTERN.search(line):
            diagnostics.append(line)
    return diagnostics


@contextlib.contextmanager
def _quiet_run_output_context(*, suppress_output: bool) -> Any:
    """Suppress normal run chatter while keeping warning/error diagnostics."""

    if not suppress_output:
        yield None
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    robosuite_logger = None
    original_level = None

    try:
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        robosuite_logger = ROBOSUITE_DEFAULT_LOGGER
        original_level = robosuite_logger.level
    except Exception:
        robosuite_logger = None
        original_level = None

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
        stderr_buffer
    ):
        if robosuite_logger is not None:
            robosuite_logger.setLevel(logging.WARNING)
        try:
            yield stderr_buffer
        finally:
            if robosuite_logger is not None and original_level is not None:
                robosuite_logger.setLevel(original_level)


@contextlib.contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Any:
    """Apply environment overrides for the duration of one context."""

    previous_values = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _emit_trajectory_log_lines(
    trajectory_result: dict[str, Any],
    *,
    log_writer: Callable[[str], None] | None,
    log_run_completions: bool,
) -> None:
    """Print the selected run logs from one finished trajectory entry."""

    for diagnostic_line in trajectory_result.get("diagnostic_lines", []):
        _emit_sweep_log_line(
            diagnostic_line,
            log_writer=log_writer,
            use_stderr=True,
        )

    for result, log_line in zip(
        trajectory_result["results"],
        trajectory_result["log_lines"],
    ):
        use_stderr = result["status"] == "error"
        if log_run_completions or use_stderr:
            _emit_sweep_log_line(
                log_line,
                log_writer=log_writer,
                use_stderr=use_stderr,
            )


def discover_trajectories(
    input_dir: Path,
    task_filter: list[str] | None = None,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Find all trajectory JSONs under input_dir/<task>/trajectories/."""
    entries = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        traj_dir = task_dir / "trajectories"
        if not traj_dir.is_dir():
            continue
        for traj_file in sorted(traj_dir.glob("traj_*.json")):
            traj_idx = int(traj_file.stem.split("_")[-1])
            if indices is not None and traj_idx not in indices:
                continue
            entries.append(
                {
                    "task_dir_name": task_dir.name,
                    "traj_file": traj_file,
                    "traj_idx": traj_idx,
                }
            )
    return entries


def _resolve_run_output_dir(
    output_root: Path,
    *,
    task_name: str,
    traj_idx: int,
    combo_count: int,
    layout: int,
    style: int,
    seed: int,
) -> tuple[Path, str]:
    """Resolve the output directory and display label for one run."""

    if combo_count == 1:
        return output_root / task_name / f"traj_{traj_idx:06d}", ""

    combo_dir = f"L{layout}_S{style}_sd{seed}"
    combo_label = _resolve_combo_label(
        combo_count=combo_count,
        layout=layout,
        style=style,
        seed=seed,
    )
    return output_root / task_name / f"traj_{traj_idx:06d}" / combo_dir, combo_label


def _resolve_combo_label(
    *,
    combo_count: int,
    layout: int,
    style: int,
    seed: int,
) -> str:
    """Build the display label for one scene combo."""

    if combo_count == 1:
        return ""
    return f" L{layout}/S{style}/sd{seed}"


def _format_run_log_line(
    *,
    run_num: int,
    total_runs: int,
    task_name: str,
    traj_idx: int,
    combo_label: str,
    result: dict[str, Any],
) -> str:
    """Build the CLI log line for one completed sweep run."""

    label = f"[{run_num}/{total_runs}] {task_name}/traj_{traj_idx:06d}{combo_label}"
    elapsed_seconds = float(result["elapsed_s"])
    if result["status"] == "ok":
        return (
            f"{label} "
            f"{result['steps_succeeded']}/{result['steps_total']} steps ok, "
            f"{result['images_rendered']} images ({elapsed_seconds:.1f}s)"
        )
    return f"{label} ERROR: {result['error']} ({elapsed_seconds:.1f}s)"


def run_one(
    traj_file: Path,
    output_dir: Path,
    layout: int,
    style: int,
    seed: int,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    gl_backend: str = "osmesa",
) -> dict:
    """Execute a single trajectory and return summary info."""
    from robocasa.utils.sim_tool_executor import SimToolExecutor
    from robocasa.utils.trajectory_adapter import execute_trajectory

    with open(traj_file) as f:
        trajectory = json.load(f)

    task_name = trajectory.get("composite_task", "Kitchen")

    executor = SimToolExecutor(
        task_name=task_name,
        robots=robots,
        layout=layout,
        style=style,
        seed=seed,
        placement=placement,
        cell_size=cell_size,
        robot_spawn=robot_spawn,
        gl_backend=gl_backend,
    )

    try:
        # Copy original trajectory JSON to output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traj_file, output_dir / "original_trajectory.json")

        metadata = execute_trajectory(
            executor=executor,
            trajectory=trajectory,
            output_dir=str(output_dir),
            skip_videos=skip_videos,
        )

        # Count successes (skip get_image steps which always succeed)
        steps = metadata.get("steps", [])
        action_steps = [s for s in steps if s.get("tool") != "get_image"]
        n_success = sum(1 for s in action_steps if s.get("success"))
        n_total = len(action_steps)
        n_images = sum(1 for s in steps if s.get("tool") == "get_image")

        return {
            "status": "ok",
            "task": task_name,
            "steps_succeeded": n_success,
            "steps_total": n_total,
            "images_rendered": n_images,
        }
    finally:
        executor.close()


def run_trajectory_entry(
    entry: dict[str, Any],
    *,
    entry_index: int,
    total_runs: int,
    combos: tuple[tuple[int, int, int], ...],
    output_root: Path,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    suppress_stdout: bool = False,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    gpu_id: int | None = None,
    gl_backend: str = "osmesa",
) -> dict[str, Any]:
    """Execute one discovered trajectory across every requested scene combo."""

    gpu_environment = contextlib.nullcontext()
    if gpu_id is not None:
        # Constrain each worker invocation to one GPU before constructing any
        # simulator state so concurrent entries can be spread across GPUs.
        gpu_environment = _temporary_environment(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "GPUS": str(gpu_id),
                "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
            }
        )

    with gpu_environment:
        task_name = str(entry["task_dir_name"])
        traj_idx = int(entry["traj_idx"])
        traj_file = Path(entry["traj_file"])
        combo_count = len(combos)
        results: list[dict[str, Any]] = []
        diagnostic_lines: list[str] = []
        log_lines: list[str] = []

        for combo_offset, (layout, style, seed) in enumerate(combos):
            run_num = entry_index * combo_count + combo_offset + 1
            traj_output_dir, combo_label = _resolve_run_output_dir(
                output_root,
                task_name=task_name,
                traj_idx=traj_idx,
                combo_count=combo_count,
                layout=layout,
                style=style,
                seed=seed,
            )

            started_at = time.time()
            quiet_stderr_buffer = None
            try:
                with _quiet_run_output_context(
                    suppress_output=suppress_stdout
                ) as quiet_stderr_buffer:
                    result = run_one(
                        traj_file=traj_file,
                        output_dir=traj_output_dir,
                        layout=layout,
                        style=style,
                        seed=seed,
                        robots=robots,
                        placement=placement,
                        cell_size=cell_size,
                        robot_spawn=robot_spawn,
                        skip_videos=skip_videos,
                        gl_backend=gl_backend,
                    )
                elapsed_seconds = time.time() - started_at
                result["elapsed_s"] = round(elapsed_seconds, 1)
            except Exception as exc:  # pragma: no cover - exercised via callers
                elapsed_seconds = time.time() - started_at
                result = {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_s": round(elapsed_seconds, 1),
                }
            if quiet_stderr_buffer is not None:
                diagnostic_lines.extend(
                    _quiet_diagnostic_lines(quiet_stderr_buffer.getvalue())
                )

            result["task_dir"] = task_name
            result["traj_idx"] = traj_idx
            result["traj_file"] = str(traj_file)
            result["layout"] = layout
            result["style"] = style
            result["seed"] = seed
            results.append(result)
            if progress_reporter is not None:
                progress_reporter(
                    {
                        "status": result["status"],
                        "layout": layout,
                        "style": style,
                        "seed": seed,
                    }
                )
            log_lines.append(
                _format_run_log_line(
                    run_num=run_num,
                    total_runs=total_runs,
                    task_name=task_name,
                    traj_idx=traj_idx,
                    combo_label=combo_label,
                    result=result,
                )
            )

    return {
        "diagnostic_lines": diagnostic_lines,
        "entry_index": entry_index,
        "results": results,
        "log_lines": log_lines,
    }


def _build_trajectory_crash_results(
    entry: dict[str, Any],
    *,
    entry_index: int,
    total_runs: int,
    combos: tuple[tuple[int, int, int], ...],
    error_message: str,
    traceback_text: str,
) -> dict[str, Any]:
    """Build error records when a worker fails before returning any run data."""

    task_name = str(entry["task_dir_name"])
    traj_idx = int(entry["traj_idx"])
    traj_file = Path(entry["traj_file"])
    combo_count = len(combos)
    results: list[dict[str, Any]] = []
    log_lines: list[str] = []

    for combo_offset, (layout, style, seed) in enumerate(combos):
        run_num = entry_index * combo_count + combo_offset + 1
        combo_label = _resolve_combo_label(
            combo_count=combo_count,
            layout=layout,
            style=style,
            seed=seed,
        )
        result = {
            "status": "error",
            "error": error_message,
            "traceback": traceback_text,
            "elapsed_s": 0.0,
            "task_dir": task_name,
            "traj_idx": traj_idx,
            "traj_file": str(traj_file),
            "layout": layout,
            "style": style,
            "seed": seed,
        }
        results.append(result)
        log_lines.append(
            _format_run_log_line(
                run_num=run_num,
                total_runs=total_runs,
                task_name=task_name,
                traj_idx=traj_idx,
                combo_label=combo_label,
                result=result,
            )
        )

    return {
        "entry_index": entry_index,
        "results": results,
        "log_lines": log_lines,
    }


def get_gpu_allocation(
    num_workers: int,
    gpu_ids: list[int] | None,
    procs_per_gpu: list[int] | None = None,
) -> list[int] | None:
    """Resolve one GPU assignment per concurrent worker slot."""

    if gpu_ids is None:
        if procs_per_gpu is not None:
            raise ValueError("--procs-per-gpu requires --gpu-ids.")
        return None
    if not gpu_ids:
        raise ValueError("--gpu-ids must include at least one GPU id.")
    if num_workers <= 0:
        return []

    if procs_per_gpu is None:
        return [gpu_ids[i % len(gpu_ids)] for i in range(num_workers)]

    if len(procs_per_gpu) != len(gpu_ids):
        raise ValueError(
            "--procs-per-gpu must have the same length as --gpu-ids."
        )

    adjusted_counts = list(procs_per_gpu)
    total_allocated = sum(adjusted_counts)
    if total_allocated < num_workers:
        raise ValueError(
            f"Sum of --procs-per-gpu ({total_allocated}) must be at least the "
            f"number of concurrent workers ({num_workers})."
        )

    while total_allocated > num_workers:
        gpu_index = max(range(len(adjusted_counts)), key=adjusted_counts.__getitem__)
        adjusted_counts[gpu_index] -= 1
        total_allocated -= 1

    gpu_allocation: list[int] = []
    for gpu_id, worker_count in zip(gpu_ids, adjusted_counts):
        gpu_allocation.extend([gpu_id] * worker_count)
    return gpu_allocation


def execute_sweep(
    entries: list[dict[str, Any]],
    *,
    combos: tuple[tuple[int, int, int], ...],
    output_root: Path,
    workers: int,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    executor_factory: Callable[..., Any] | None = None,
    show_progress: bool = True,
    log_run_completions: bool = True,
    suppress_run_stdout: bool = False,
    progress_factory: Callable[..., Any] | None = None,
    gpu_ids: list[int] | None = None,
    procs_per_gpu: list[int] | None = None,
    gl_backend: str = "osmesa",
) -> list[dict[str, Any]]:
    """Execute the discovered trajectories and preserve summary ordering."""

    total_runs = len(entries) * len(combos)
    ordered_results: list[list[dict[str, Any]] | None] = [None] * len(entries)
    combo_count = len(combos)
    max_workers = min(workers, len(entries)) if entries else 0
    progress_worker_count = max_workers
    gpu_allocation = get_gpu_allocation(max_workers, gpu_ids, procs_per_gpu)
    progress_display = None
    slot_completed_runs: dict[int, int] = {}
    sweep_executor_factory = executor_factory or ProcessPoolExecutor

    try:
        if show_progress and total_runs > 0:
            progress_display = _create_sweep_progress_display(
                total_runs=total_runs,
                worker_count=progress_worker_count,
                progress_factory=progress_factory,
            )
        log_writer = (
            progress_display.write_log_line if progress_display is not None else None
        )

        if workers == 1 or len(entries) == 1:
            worker_slot = 0
            for entry_index, entry in enumerate(entries):
                _assign_progress_worker(
                    progress_display,
                    slot_completed_runs,
                    worker_slot=worker_slot,
                    entry=entry,
                    combo_count=combo_count,
                )
                trajectory_result = run_trajectory_entry(
                    entry,
                    entry_index=entry_index,
                    total_runs=total_runs,
                    combos=combos,
                    output_root=output_root,
                    robots=robots,
                    placement=placement,
                    cell_size=cell_size,
                    robot_spawn=robot_spawn,
                    skip_videos=skip_videos,
                    suppress_stdout=suppress_run_stdout,
                    gpu_id=gpu_allocation[worker_slot] if gpu_allocation else None,
                    gl_backend=gl_backend,
                    progress_reporter=(
                        partial(
                            _record_local_progress_event,
                            progress_display,
                            slot_completed_runs,
                            worker_slot,
                        )
                        if progress_display is not None
                        else None
                    ),
                )
                ordered_results[entry_index] = trajectory_result["results"]
                _finalize_progress_worker(
                    progress_display,
                    slot_completed_runs,
                    worker_slot=worker_slot,
                    results=trajectory_result["results"],
                )
                _emit_trajectory_log_lines(
                    trajectory_result,
                    log_writer=log_writer,
                    log_run_completions=log_run_completions,
                )
        else:
            with contextlib.ExitStack() as exit_stack:
                progress_queue = None
                progress_manager = None
                if progress_display is not None:
                    progress_queue, progress_manager = _create_progress_event_queue(
                        executor_factory=sweep_executor_factory
                    )
                    if progress_manager is not None:
                        exit_stack.enter_context(progress_manager)
                with sweep_executor_factory(max_workers=max_workers) as executor:
                    pending_entries = iter(enumerate(entries))
                    future_to_context: dict[Any, tuple[int, int, dict[str, Any]]] = {}

                    def submit_entry(worker_slot: int) -> bool:
                        """Schedules the next entry on the requested worker slot."""

                        try:
                            entry_index, entry = next(pending_entries)
                        except StopIteration:
                            return False
                        _assign_progress_worker(
                            progress_display,
                            slot_completed_runs,
                            worker_slot=worker_slot,
                            entry=entry,
                            combo_count=combo_count,
                        )
                        progress_reporter = None
                        if progress_queue is not None:
                            progress_reporter = partial(
                                _enqueue_progress_event,
                                progress_queue,
                                worker_slot,
                            )
                        future = executor.submit(
                            run_trajectory_entry,
                            entry,
                            entry_index=entry_index,
                            total_runs=total_runs,
                            combos=combos,
                            output_root=output_root,
                            robots=robots,
                            placement=placement,
                            cell_size=cell_size,
                            robot_spawn=robot_spawn,
                            skip_videos=skip_videos,
                            suppress_stdout=suppress_run_stdout,
                            gpu_id=(
                                gpu_allocation[worker_slot]
                                if gpu_allocation is not None
                                else None
                            ),
                            gl_backend=gl_backend,
                            progress_reporter=progress_reporter,
                        )
                        future_to_context[future] = (entry_index, worker_slot, entry)
                        return True

                    for worker_slot in range(max_workers):
                        if not submit_entry(worker_slot):
                            break

                    while future_to_context:
                        _drain_progress_events(
                            progress_queue,
                            progress_display,
                            slot_completed_runs,
                        )
                        done, _ = wait(
                            tuple(future_to_context),
                            timeout=0.1,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            continue
                        _drain_progress_events(
                            progress_queue,
                            progress_display,
                            slot_completed_runs,
                        )
                        for future in done:
                            entry_index, worker_slot, entry = future_to_context.pop(
                                future
                            )
                            try:
                                trajectory_result = future.result()
                            except (
                                Exception
                            ) as exc:  # pragma: no cover - defensive path
                                trajectory_result = _build_trajectory_crash_results(
                                    entry,
                                    entry_index=entry_index,
                                    total_runs=total_runs,
                                    combos=combos,
                                    error_message=str(exc),
                                    traceback_text=traceback.format_exc(),
                                )

                            _drain_progress_events(
                                progress_queue,
                                progress_display,
                                slot_completed_runs,
                            )
                            ordered_results[entry_index] = trajectory_result["results"]
                            _finalize_progress_worker(
                                progress_display,
                                slot_completed_runs,
                                worker_slot=worker_slot,
                                results=trajectory_result["results"],
                            )
                            _emit_trajectory_log_lines(
                                trajectory_result,
                                log_writer=log_writer,
                                log_run_completions=log_run_completions,
                            )
                            submit_entry(worker_slot)
                    _drain_progress_events(
                        progress_queue,
                        progress_display,
                        slot_completed_runs,
                    )
    finally:
        if progress_display is not None:
            progress_display.close()

    flattened_results: list[dict[str, Any]] = []
    for entry_results in ordered_results:
        if entry_results is None:
            continue
        flattened_results.extend(entry_results)
    return flattened_results


def _load_sweep_summary(output_root: Path) -> dict:
    summary_path = output_root / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No sweep_summary.json in {output_root}")
    with open(summary_path) as f:
        return json.load(f)


def _iter_completed_runs(output_root: Path):
    summary = _load_sweep_summary(output_root)
    combo_count = summary.get("scene_combos", 1)

    for run_result in summary["results"]:
        if run_result.get("status") != "ok":
            continue

        task_dir = run_result["task_dir"]
        traj_idx = run_result["traj_idx"]
        layout = run_result["layout"]
        style = run_result["style"]
        seed = run_result["seed"]

        if combo_count == 1:
            run_dir = output_root / task_dir / f"traj_{traj_idx:06d}"
        else:
            run_dir = (
                output_root
                / task_dir
                / f"traj_{traj_idx:06d}"
                / f"L{layout}_S{style}_sd{seed}"
            )

        meta_path = run_dir / "trajectory_execution_metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        episode_id = f"{task_dir}/traj_{traj_idx:06d}/L{layout}_S{style}_sd{seed}"
        task_name = metadata.get("composite_task") or metadata.get("task") or task_dir
        rel_run_dir = run_dir.relative_to(output_root)

        yield {
            "episode_id": episode_id,
            "task": task_name,
            "task_dir": task_dir,
            "traj_idx": traj_idx,
            "layout": layout,
            "style": style,
            "seed": seed,
            "run_dir": run_dir,
            "run_dir_rel": str(rel_run_dir),
            "metadata": metadata,
            "adapted_trajectory_path": str(rel_run_dir / "adapted_trajectory.json"),
            "original_trajectory_path": str(rel_run_dir / "original_trajectory.json"),
            "execution_metadata_path": str(
                rel_run_dir / "trajectory_execution_metadata.json"
            ),
        }


def iter_sweep_metadata_paths(output_root: Path):
    """Yield repo-relative JSON artifact paths that should accompany the dataset."""
    seen = {Path("sweep_summary.json")}
    yield output_root / "sweep_summary.json", "sweep_summary.json"

    for run in _iter_completed_runs(output_root):
        for key in (
            "adapted_trajectory_path",
            "original_trajectory_path",
            "execution_metadata_path",
        ):
            rel_path = Path(run[key])
            if rel_path in seen:
                continue
            abs_path = output_root / rel_path
            if not abs_path.exists():
                continue
            seen.add(rel_path)
            yield abs_path, str(rel_path)


def upload_sweep_metadata_files(repo_id: str, output_root: Path) -> None:
    """Upload referenced episode JSON files alongside the parquet dataset."""
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        allow_patterns=[
            "sweep_summary.json",
            "**/adapted_trajectory.json",
            "**/original_trajectory.json",
            "**/trajectory_execution_metadata.json",
        ],
        commit_message="Upload sweep metadata files",
    )


def sweep_output_to_dataset(output_root: Path) -> "datasets.Dataset":
    """Convert sweep output directory into a single flat step-level dataset.

    Each row is one tool step. Large episode-level JSON blobs are kept as
    separate metadata files in the dataset repo and referenced by path
    columns so the HF table stays flat and viewable.
    """
    from datasets import Dataset, Features, Value, Image as HFImage

    IMAGE_COLUMNS = [
        "room_view",
        "top_view",
        "map",
        "agentview_center",
        "agentview_left",
        "agentview_right",
        "wrist",
    ]

    VIEW_TOKENS = IMAGE_COLUMNS  # filename tokens match column names
    rows: list[dict] = []
    for run in _iter_completed_runs(output_root):
        metadata = run["metadata"]
        num_steps = len(metadata.get("steps", []))

        for step in metadata.get("steps", []):
            step_idx = step.get("step_index", 0)
            tool = step.get("tool", "")
            robot_idx = step.get("robot_idx", 0)
            args = step.get("args", {})
            success = step.get("success", False)

            images = {col: None for col in IMAGE_COLUMNS}
            image_paths = step.get("image_paths") or []
            for img_path_str in image_paths:
                img_path = Path(img_path_str)
                if not img_path.exists() and img_path.suffix == ".png":
                    img_path = img_path.with_suffix(".jpg")
                if not img_path.exists():
                    continue
                fname = img_path.stem
                for view_token in VIEW_TOKENS:
                    if f"_{view_token}_" in f"_{fname}_":
                        images[view_token] = str(img_path)
                        break

            args_clean = {k: v for k, v in args.items() if k != "image_paths"}

            rows.append(
                {
                    "episode_id": run["episode_id"],
                    "task": run["task"],
                    "task_dir": run["task_dir"],
                    "layout": run["layout"],
                    "style": run["style"],
                    "seed": run["seed"],
                    "num_steps": num_steps,
                    "run_dir": run["run_dir_rel"],
                    "adapted_trajectory_path": run["adapted_trajectory_path"],
                    "original_trajectory_path": run["original_trajectory_path"],
                    "execution_metadata_path": run["execution_metadata_path"],
                    "step_index": step_idx,
                    "tool_name": tool,
                    "tool_args": json.dumps(args_clean, separators=(",", ":")),
                    "robot_idx": robot_idx,
                    "success": success,
                    **images,
                }
            )

    features = Features(
        {
            "episode_id": Value("string"),
            "task": Value("string"),
            "task_dir": Value("string"),
            "layout": Value("int32"),
            "style": Value("int32"),
            "seed": Value("int32"),
            "num_steps": Value("int32"),
            "run_dir": Value("string"),
            "adapted_trajectory_path": Value("string"),
            "original_trajectory_path": Value("string"),
            "execution_metadata_path": Value("string"),
            "step_index": Value("int32"),
            "tool_name": Value("string"),
            "tool_args": Value("string"),
            "robot_idx": Value("int32"),
            "success": Value("bool"),
            **{col: HFImage() for col in IMAGE_COLUMNS},
        }
    )

    ds = Dataset.from_list(rows, features=features)
    print(
        f"Built dataset: {len(ds)} step rows across {len(set(ds['episode_id']))} episodes"
    )
    return ds


def build_dataset_card(repo_id: str, ds: "datasets.Dataset") -> str:
    """Build a readable HuggingFace dataset card."""
    tasks = sorted(set(ds["task"]))
    episode_ids = ds["episode_id"]
    episodes = len(set(episode_ids))
    avg_steps = len(ds) / max(episodes, 1)
    task_list = ", ".join(tasks) if tasks else "Unknown"
    return textwrap.dedent(f"""\
        ---
        pretty_name: RoboCasa Trajectories Single
        configs:
        - config_name: default
          data_files:
          - split: train
            path: data/train-*
        ---

        # RoboCasa Trajectories Single

        This dataset contains flat RoboCasa step rows with separate episode JSON files.

        ## Structure

        Each row is one tool step.

        Episode-level JSON is not duplicated into parquet. Instead, each row
        carries repo-relative references:

        - `adapted_trajectory_path`
        - `original_trajectory_path`
        - `execution_metadata_path`

        Image columns stay inline and viewable in the dataset table:

        - `room_view`
        - `top_view`
        - `map`
        - `agentview_center`
        - `agentview_left`
        - `agentview_right`
        - `wrist`

        ## Summary

        - rows: {len(ds)}
        - episodes: {episodes}
        - tasks: {task_list}
        - average steps per episode: {avg_steps:.1f}

        ## Load

        ```python
        from datasets import load_dataset

        ds = load_dataset("{repo_id}", split="train")
        ```

        ## Notes

        - Camera renders are stored as JPEG. Maps remain PNG.
        - MP4 videos are not included in the dataset.
        - Debug `initial` / `pre_initial_state` camera frames are not part of the dataset.
        - Episode JSON files are available in the repo files at the paths referenced by `*_path` columns.
        """)


def upload_dataset_card(repo_id: str, ds: "datasets.Dataset") -> None:
    """Overwrite the auto-generated Hub README with a readable dataset card."""
    from io import BytesIO

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=BytesIO(build_dataset_card(repo_id, ds).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep trajectories through the sim executor and optionally publish the dataset.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root dir")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific task dir names",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific traj indices",
    )
    parser.add_argument(
        "--layouts",
        type=int,
        nargs="+",
        default=[11],
        help="Kitchen layout ids (default: 11)",
    )
    parser.add_argument(
        "--styles",
        type=int,
        nargs="+",
        default=[34],
        help="Kitchen style ids (default: 34)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Environment seeds (default: 42)",
    )
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--placement", choices=["grid", "continuous"], default="grid")
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum parallel trajectory workers (default: 1).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional GPU IDs to assign across concurrent workers. When set, "
            "workers are distributed across these GPUs."
        ),
    )
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional worker counts per GPU. Must match --gpu-ids in length and "
            "sum to at least the number of concurrent workers."
        ),
    )
    parser.add_argument(
        "--gl-backend",
        choices=["osmesa", "egl"],
        default=None,
        help=(
            "OpenGL backend. Defaults to 'egl' when --gpu-ids is set, "
            "otherwise 'osmesa'."
        ),
    )
    parser.add_argument(
        "--robot-spawn",
        choices=["sim", "trajectory"],
        default="trajectory",
        help=(
            "Robot initial placement source. 'sim': all robots at "
            "init_robot_base_ref. 'trajectory' (default): each robot at its trajectory location."
        ),
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Record per-camera MP4 videos for each run",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Push dataset to HuggingFace Hub (e.g. 'username/robocasa-trajectories')",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without executing"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show the run progress bar but suppress normal informational output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if args.workers <= 0:
        parser.error("--workers must be greater than 0.")

    # Child workers inherit this process environment, so set the shared
    # simulator debug gate before launching any sweep work.
    os.environ["ROBOCASA_SWEEP_VERBOSE"] = "0" if args.quiet else "1"

    entries = discover_trajectories(
        input_dir, task_filter=args.tasks, indices=args.indices
    )
    if not entries:
        print("No trajectories found.", file=sys.stderr, flush=True)
        sys.exit(1)

    combos = list(itertools.product(args.layouts, args.styles, args.seeds))
    total_runs = len(entries) * len(combos)
    concurrent_workers = min(args.workers, len(entries))
    resolved_gl_backend = args.gl_backend or ("egl" if args.gpu_ids else "osmesa")
    if args.gpu_ids is not None and resolved_gl_backend != "egl":
        parser.error("--gpu-ids requires --gl-backend egl.")
    try:
        gpu_allocation = get_gpu_allocation(
            concurrent_workers,
            args.gpu_ids,
            args.procs_per_gpu,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            f"Found {len(entries)} trajectories x {len(combos)} scene combos = {total_runs} runs"
        )
        if args.workers > 1:
            print(f"Using {concurrent_workers} trajectory workers")
        if gpu_allocation is not None:
            print(f"Using GL backend: {resolved_gl_backend}")
            print(f"GPU allocation by worker slot: {gpu_allocation}")
        if len(combos) > 1:
            print(f"  layouts: {args.layouts}")
            print(f"  styles:  {args.styles}")
            print(f"  seeds:   {args.seeds}")
        print()

    if args.dry_run:
        if not args.quiet:
            for entry in entries:
                for layout, style, seed in combos:
                    task_name = entry["task_dir_name"]
                    traj_idx = entry["traj_idx"]
                    out, combo_label = _resolve_run_output_dir(
                        output_root,
                        task_name=task_name,
                        traj_idx=traj_idx,
                        combo_count=len(combos),
                        layout=layout,
                        style=style,
                        seed=seed,
                    )
                    if combo_label:
                        print(
                            f"  {task_name}/traj_{traj_idx:06d}{combo_label} -> {out}"
                        )
                    else:
                        print(f"  {task_name}/traj_{traj_idx:06d} -> {out}")
            print(f"\n{total_runs} runs (dry run, nothing executed)")
        return

    results = execute_sweep(
        entries,
        combos=tuple(combos),
        output_root=output_root,
        workers=args.workers,
        robots=args.robots,
        placement=args.placement,
        cell_size=args.cell_size,
        robot_spawn=args.robot_spawn,
        skip_videos=not args.videos,
        log_run_completions=not args.quiet,
        show_progress=True,
        suppress_run_stdout=args.quiet,
        gpu_ids=args.gpu_ids,
        procs_per_gpu=args.procs_per_gpu,
        gl_backend=resolved_gl_backend,
    )

    # Write sweep summary
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dir": str(input_dir),
        "layouts": args.layouts,
        "styles": args.styles,
        "seeds": args.seeds,
        "scene_combos": len(combos),
        "trajectories": len(entries),
        "total": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "gl_backend": resolved_gl_backend,
        "gpu_ids": args.gpu_ids,
        "procs_per_gpu": args.procs_per_gpu,
        "results": results,
    }
    with open(output_root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Done: {summary['succeeded']}/{summary['total']} succeeded")
    print(f"Summary: {output_root / 'sweep_summary.json'}")

    if summary["failed"] > 0:
        print(f"\nFailed runs:")
        for r in results:
            if r["status"] == "error":
                print(
                    f"  {r['task_dir']}/traj_{r['traj_idx']:06d} L{r['layout']}/S{r['style']}/sd{r['seed']}: {r['error']}"
                )

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        if not args.quiet:
            print(f"\nConverting sweep output to HuggingFace dataset...")
        ds = sweep_output_to_dataset(output_root)
        if not args.quiet:
            print(f"Pushing to {args.push_to_hub}...")
        ds.push_to_hub(args.push_to_hub)
        if not args.quiet:
            print("Uploading sweep metadata files...")
        upload_sweep_metadata_files(args.push_to_hub, output_root)
        if not args.quiet:
            print("Uploading dataset card...")
        upload_dataset_card(args.push_to_hub, ds)
        if not args.quiet:
            print(
                f"Done! Dataset pushed to https://huggingface.co/datasets/{args.push_to_hub}"
            )


if __name__ == "__main__":
    main()
