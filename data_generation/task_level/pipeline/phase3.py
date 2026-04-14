"""Phase 3: trajectory generation for Phase 2-passing TaskSpecs.

This phase is intentionally thin. It does not reimplement raw generation.
Instead it shells out to the existing raw CLI in a fresh subprocess with the
run's spec directory pinned via `ROBOCASA_TASK_SPEC_DIR`. That keeps the task
registry aligned with the generated specs and avoids import-order issues.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Phase3TaskResult:
    """Persisted outcome for one task-level trajectory-generation run."""

    task_name: str
    spec_path: str
    output_dir: str
    summary_path: str
    error_summary_path: str
    stdout_log_path: str
    stderr_log_path: str
    exit_code: int
    completed: bool
    num_runs: int
    num_trajectories: int
    completed_run_indices: list[int]
    failed_run_indices: list[int]
    pending_run_indices: list[int]
    error_counts_by_type: list[dict[str, Any]]
    distinct_errors: list[dict[str, Any]]
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


def _task_output_dir(output_dir: Path, task_name: str) -> Path:
    return output_dir / "phase3" / "raw" / task_name


def _task_log_dir(output_dir: Path) -> Path:
    return output_dir / "phase3" / "logs"


def _summarize_phase3_results(results: list[Phase3TaskResult]) -> dict[str, Any]:
    total_errors = 0
    error_type_counts: dict[str, int] = {}
    completed = 0
    incomplete = 0
    for result in results:
        if result.completed:
            completed += 1
        else:
            incomplete += 1
        for error_count in result.error_counts_by_type:
            error_type = error_count.get("error_type")
            count = error_count.get("count")
            if not isinstance(error_type, str) or not isinstance(count, int):
                continue
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + count
            total_errors += count

    return {
        "total": len(results),
        "completed": completed,
        "incomplete": incomplete,
        "num_trajectories": sum(result.num_trajectories for result in results),
        "total_errors": total_errors,
        "error_counts_by_type": [
            {"error_type": error_type, "count": error_type_counts[error_type]}
            for error_type in sorted(error_type_counts)
        ],
    }


def _run_one_task(
    spec_path: Path,
    *,
    output_dir: Path,
    num_runs: int,
    model: str | None,
    project: str | None,
    location: str,
    max_retries: int,
    generation_timeout_sec: int | None,
    spec_dir: Path,
    dry_run: bool,
) -> Phase3TaskResult:
    spec_payload = _load_json_object(spec_path)
    task_name = str((spec_payload or {}).get("composite_task") or spec_path.stem)
    task_dir_name = spec_path.stem
    task_output_dir = _task_output_dir(output_dir, task_dir_name)
    summary_path = task_output_dir / "summary.json"
    error_summary_path = task_output_dir / "summary_errors.json"
    log_dir = _task_log_dir(output_dir)
    stdout_log_path = log_dir / f"{task_dir_name}.stdout.log"
    stderr_log_path = log_dir / f"{task_dir_name}.stderr.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "data_generation.task_level.generation.raw.cli",
        "--tasks",
        task_name,
        "--num-runs",
        str(num_runs),
        "--max-workers",
        "1",
        "--max-retries",
        str(max_retries),
        "--enable-validation",
        "--thinking-level",
        "low",
    ]
    if model:
        command.extend(["--model", model])
    if project:
        command.extend(["--project", project])
    if location:
        command.extend(["--location", location])
    if generation_timeout_sec is not None:
        command.extend(["--generation-timeout-sec", str(generation_timeout_sec)])
    if summary_path.exists():
        command.extend(["--resume", str(task_output_dir)])
    else:
        command.extend(["--summary-path", str(summary_path)])

    if dry_run:
        return Phase3TaskResult(
            task_name=task_name,
            spec_path=str(spec_path),
            output_dir=str(task_output_dir),
            summary_path=str(summary_path),
            error_summary_path=str(error_summary_path),
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
            exit_code=0,
            completed=False,
            num_runs=num_runs,
            num_trajectories=0,
            completed_run_indices=[],
            failed_run_indices=[],
            pending_run_indices=list(range(num_runs)),
            error_counts_by_type=[],
            distinct_errors=[],
            error="DRY RUN: " + " ".join(command),
        )

    env = dict(os.environ)
    env["ROBOCASA_TASK_SPEC_DIR"] = str(spec_dir.resolve())
    completed_process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
    )
    stdout_log_path.write_text(completed_process.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(completed_process.stderr or "", encoding="utf-8")

    summary_payload = _load_json_object(summary_path) or {}
    error_payload = _load_json_object(error_summary_path) or {}
    return Phase3TaskResult(
        task_name=task_name,
        spec_path=str(spec_path),
        output_dir=str(task_output_dir),
        summary_path=str(summary_path),
        error_summary_path=str(error_summary_path),
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        exit_code=int(completed_process.returncode),
        completed=bool(summary_payload.get("is_complete", False)),
        num_runs=int(summary_payload.get("num_runs", 1) or 1),
        num_trajectories=int(summary_payload.get("num_trajectories", 0) or 0),
        completed_run_indices=list(summary_payload.get("completed_run_indices", []) or []),
        failed_run_indices=list(summary_payload.get("failed_run_indices", []) or []),
        pending_run_indices=list(summary_payload.get("pending_run_indices", []) or []),
        error_counts_by_type=list(error_payload.get("error_counts_by_type", []) or []),
        distinct_errors=list(error_payload.get("distinct_errors", []) or []),
        error=(
            None
            if completed_process.returncode == 0
            else (
                (completed_process.stderr or "").strip()
                or (completed_process.stdout or "").strip()
                or f"raw generation exited with code {completed_process.returncode}"
            )
        ),
    )


def run_phase3(
    *,
    output_dir: Path,
    spec_paths: list[Path],
    model: str | None = None,
    num_runs: int = 1,
    workers: int = 4,
    max_retries: int = 5,
    project: str | None = None,
    location: str = "global",
    dry_run: bool = False,
    progress_callback: Any = None,
    generation_timeout_sec: int | None = 300,
) -> list[Phase3TaskResult]:
    """Generate raw trajectories for the given TaskSpec files."""

    phase_dir = output_dir / "phase3"
    phase_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = output_dir / "phase1" / "specs"

    if workers <= 1 or len(spec_paths) <= 1:
        results: list[Phase3TaskResult] = []
        for index, spec_path in enumerate(spec_paths, start=1):
            result = _run_one_task(
                spec_path,
                output_dir=output_dir,
                num_runs=num_runs,
                model=model,
                project=project,
                location=location,
                max_retries=max_retries,
                generation_timeout_sec=generation_timeout_sec,
                spec_dir=spec_dir,
                dry_run=dry_run,
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(index, len(spec_paths), result.task_name)
    else:
        ordered_results: list[Phase3TaskResult | None] = [None] * len(spec_paths)
        completed = 0
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    _run_one_task,
                    spec_path,
                    output_dir=output_dir,
                    num_runs=num_runs,
                    model=model,
                    project=project,
                    location=location,
                    max_retries=max_retries,
                    generation_timeout_sec=generation_timeout_sec,
                    spec_dir=spec_dir,
                    dry_run=dry_run,
                ): index
                for index, spec_path in enumerate(spec_paths)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered_results[index] = future.result()
                with lock:
                    completed += 1
                    current = completed
                if progress_callback is not None:
                    progress_callback(current, len(spec_paths), ordered_results[index].task_name)
        results = [result for result in ordered_results if result is not None]

    if not dry_run:
        results_path = phase_dir / "results.json"
        summary_path = phase_dir / "summary.json"
        results_path.write_text(
            json.dumps([result.to_dict() for result in results], indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(_summarize_phase3_results(results), indent=2),
            encoding="utf-8",
        )
    return results
