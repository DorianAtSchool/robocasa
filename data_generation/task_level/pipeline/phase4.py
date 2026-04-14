"""Phase 4: pre-image post-processing and layout sweep orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Phase4TaskResult:
    """One task's pre-image post-processing result."""

    task_name: str
    task_dir_name: str
    input_summary_path: str
    output_summary_path: str
    stdout_log_path: str
    stderr_log_path: str
    exit_code: int
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Phase4SweepResult:
    """One aggregate sweep execution result."""

    input_dir: str
    output_dir: str
    summary_path: str
    stdout_log_path: str
    stderr_log_path: str
    exit_code: int
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_phase3_results(
    output_dir: Path,
    *,
    task_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    results_path = output_dir / "phase3" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Phase 3 results not found at {results_path}. Run Phase 3 first."
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("phase3/results.json was not a JSON list")
    wanted = set(task_names or [])
    selected: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        task_name = entry.get("task_name")
        if wanted and task_name not in wanted:
            continue
        selected.append(entry)
    return selected


def _run_pre_image_for_task(
    phase3_result: dict[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
) -> Phase4TaskResult:
    task_name = str(phase3_result["task_name"])
    task_dir_name = Path(str(phase3_result["output_dir"])).name
    phase4_task_dir = output_dir / "phase4" / "pre_image" / task_dir_name
    output_summary_path = phase4_task_dir / "summary.json"
    log_dir = output_dir / "phase4" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_dir / f"{task_dir_name}.image.stdout.log"
    stderr_log_path = log_dir / f"{task_dir_name}.image.stderr.log"
    input_summary_path = Path(str(phase3_result["summary_path"]))

    command = [
        sys.executable,
        "-m",
        "data_generation.task_level.generation.image.cli",
        "--dataset",
        str(input_summary_path),
        "--output-dataset",
        str(output_summary_path),
    ]
    if dry_run:
        return Phase4TaskResult(
            task_name=task_name,
            task_dir_name=task_dir_name,
            input_summary_path=str(input_summary_path),
            output_summary_path=str(output_summary_path),
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            exit_code=0,
            completed=False,
            error="DRY RUN: " + " ".join(command),
        )

    completed_process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    stdout_log_path.write_text(completed_process.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(completed_process.stderr or "", encoding="utf-8")
    completed = completed_process.returncode == 0 and output_summary_path.exists()
    return Phase4TaskResult(
        task_name=task_name,
        task_dir_name=task_dir_name,
        input_summary_path=str(input_summary_path),
        output_summary_path=str(output_summary_path),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        exit_code=int(completed_process.returncode),
        completed=completed,
        error=(
            None
            if completed
            else (
                (completed_process.stderr or "").strip()
                or (completed_process.stdout or "").strip()
                or f"image post-processing exited with code {completed_process.returncode}"
            )
        ),
    )


def _run_sweep(
    successful_tasks: list[Phase4TaskResult],
    *,
    output_dir: Path,
    workers: int,
    videos: bool,
    dry_run: bool,
) -> Phase4SweepResult:
    input_dir = output_dir / "phase4" / "pre_image"
    sweep_output_dir = output_dir / "phase4" / "sweeps"
    summary_path = sweep_output_dir / "sweep_summary.json"
    log_dir = output_dir / "phase4" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_dir / "sweep.stdout.log"
    stderr_log_path = log_dir / "sweep.stderr.log"

    command = [
        sys.executable,
        "scripts/sweep_trajectories.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(sweep_output_dir),
        "--workers",
        str(max(1, workers)),
    ]
    if videos:
        command.append("--videos")
    if successful_tasks:
        command.extend(["--tasks", *[task.task_dir_name for task in successful_tasks]])
    if dry_run:
        return Phase4SweepResult(
            input_dir=str(input_dir),
            output_dir=str(sweep_output_dir),
            summary_path=str(summary_path),
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            exit_code=0,
            completed=False,
            error="DRY RUN: " + " ".join(command),
        )

    child_env = os.environ.copy()
    current_gl = child_env.get("MUJOCO_GL", "").strip().lower()
    if sys.platform == "darwin" and current_gl in {"", "osmesa", "egl"}:
        child_env["MUJOCO_GL"] = "cgl"

    completed_process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        env=child_env,
    )
    stdout_log_path.write_text(completed_process.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(completed_process.stderr or "", encoding="utf-8")
    completed = completed_process.returncode == 0 and summary_path.exists()
    return Phase4SweepResult(
        input_dir=str(input_dir),
        output_dir=str(sweep_output_dir),
        summary_path=str(summary_path),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        exit_code=int(completed_process.returncode),
        completed=completed,
        error=(
            None
            if completed
            else (
                (completed_process.stderr or "").strip()
                or (completed_process.stdout or "").strip()
                or f"sweep exited with code {completed_process.returncode}"
            )
        ),
    )


def run_phase4(
    *,
    output_dir: Path,
    task_names: list[str] | None = None,
    workers: int = 4,
    videos: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run image post-processing and then sweep the successful tasks."""

    phase_dir = output_dir / "phase4"
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase3_results = _load_phase3_results(output_dir, task_names=task_names)
    selected_phase3_results = [
        result for result in phase3_results if result.get("completed") is True
    ]

    pre_image_results = [
        _run_pre_image_for_task(
            phase3_result,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        for phase3_result in selected_phase3_results
    ]
    successful_tasks = [result for result in pre_image_results if result.completed]
    sweep_result = _run_sweep(
        successful_tasks,
        output_dir=output_dir,
        workers=workers,
        videos=videos,
        dry_run=dry_run or not successful_tasks,
    )

    results_payload = {
        "pre_image_results": [result.to_dict() for result in pre_image_results],
        "sweep": sweep_result.to_dict(),
    }
    summary_payload = {
        "selected_phase3_tasks": len(selected_phase3_results),
        "pre_image_completed": sum(1 for result in pre_image_results if result.completed),
        "pre_image_failed": sum(1 for result in pre_image_results if not result.completed),
        "sweep_completed": sweep_result.completed,
        "videos": videos,
    }
    if not dry_run:
        (phase_dir / "results.json").write_text(
            json.dumps(results_payload, indent=2),
            encoding="utf-8",
        )
        (phase_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )
    return results_payload
