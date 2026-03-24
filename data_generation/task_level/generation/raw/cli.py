"""Parse CLI arguments and write task-level generation outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from data_generation.task_level.generation.raw.config import (
    ALL_COMPOSITE_TASKS_OPTION,
    DEFAULT_COMPOSITE_TASK,
    GENERATION_ERROR_EXIT_CODE,
    GOOGLE_CLOUD_BATCH_GCS_PREFIX_ENV_VAR,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_MESSAGE,
    RuntimeConfig,
    THINKING_LEVEL_CHOICES,
    _validate_runtime_config,
)
from data_generation.task_level.generation.raw.orchestrator import generate_trajectories
from data_generation.task_level.generation.raw.outputs import (
    _print_written_output_summary,
    _resolve_output_paths,
    _write_generation_outputs,
    _write_request_outputs,
    build_error_summary_output_payload,
    load_generation_output_payload,
    merge_generation_output_payloads,
    resolve_cost_output_path,
    resolve_dataset_output_path,
    resolve_error_output_path,
    resolve_request_output_path,
    resolve_request_task_output_path,
    validate_resume_payload,
)
from data_generation.task_level.generation.raw.runtime_support import (
    _exception_summary,
    _resolve_task_definitions_or_raise,
)
from data_generation.task_level.runtime.client import (
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_SDK,
    TrajectoryGenerationError,
    load_dotenv_file,
)
from data_generation.task_level.tasks import supported_task_names


def run_cli(argv: list[str] | None = None) -> int:
    """Runs the CLI and converts expected runtime failures into exit codes."""

    try:
        return main(argv)
    except TrajectoryGenerationError as exc:
        print(_exception_summary(exc), file=sys.stderr, flush=True)
        return GENERATION_ERROR_EXIT_CODE
    except KeyboardInterrupt as exc:
        print(str(exc) or INTERRUPTED_MESSAGE, file=sys.stderr, flush=True)
        os._exit(INTERRUPTED_EXIT_CODE)


def _resume_directory_summary_payload(resume_path: Path) -> dict[str, Any] | None:
    """Loads the existing summary payload from one resume directory when present."""

    summary_path = resume_path / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _validate_resume_directory_mode(runtime_config: RuntimeConfig) -> None:
    """Rejects resuming a request directory as a task directory or vice versa."""

    if runtime_config.resume_path is None:
        return
    summary_payload = _resume_directory_summary_payload(runtime_config.resume_path)
    if summary_payload is None:
        return
    if (
        len(runtime_config.composite_tasks) == 1
        and "composite_tasks" in summary_payload
    ):
        raise TrajectoryGenerationError(
            "Resume directory contains a multi-task request summary, but the current "
            "command selected only one task."
        )
    if len(runtime_config.composite_tasks) > 1 and "composite_task" in summary_payload:
        raise TrajectoryGenerationError(
            "Resume directory contains a single-task summary, but the current "
            "command selected multiple tasks."
        )


def _task_resume_output_paths(
    runtime_config: RuntimeConfig,
    *,
    composite_task: str,
    request_summary_path: Path | None = None,
) -> tuple[RuntimeConfig, Any]:
    """Resolves one task runtime config and output path set for fresh or resumed runs."""

    if runtime_config.resume_path is not None:
        if request_summary_path is None:
            summary_path = runtime_config.resume_path / "summary.json"
        else:
            summary_path = resolve_request_task_output_path(
                request_summary_path,
                composite_task,
            )
    elif request_summary_path is None:
        summary_path = resolve_dataset_output_path(
            composite_task,
            model=runtime_config.model,
        )
    else:
        summary_path = resolve_request_task_output_path(
            request_summary_path, composite_task
        )

    task_runtime_config = runtime_config.for_task(
        composite_task,
        summary_path=summary_path,
        cost_output_path=(
            None
            if request_summary_path is not None
            else runtime_config.cost_output_path
        ),
        resume_path=(
            summary_path.parent if runtime_config.resume_path is not None else None
        ),
    )
    return task_runtime_config, _resolve_output_paths(task_runtime_config)


def _task_is_complete(payload: dict[str, Any]) -> bool:
    """Returns whether one task payload has any run indices left to execute."""

    return not payload.get("pending_run_indices", [])


def _truncate_error_summary_text(text: str, *, max_length: int = 240) -> str:
    """Collapses whitespace and truncates long error summaries for CLI output."""

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return f"{normalized_text[: max_length - 3].rstrip()}..."


def _print_incomplete_task_summary(
    payload: dict[str, Any],
    *,
    composite_task: str,
    error_summary_path: Path,
) -> None:
    """Prints a concise failure summary when one task finished incomplete."""

    failed_run_indices = payload.get("failed_run_indices", [])
    if not isinstance(failed_run_indices, list) or not failed_run_indices:
        return

    task_name = composite_task
    payload_composite_task = payload.get("composite_task")
    if isinstance(payload_composite_task, str) and payload_composite_task.strip():
        task_name = payload_composite_task

    num_runs = payload.get("num_runs")
    failed_run_count = len(failed_run_indices)
    total_runs_text = str(num_runs) if isinstance(num_runs, int) else "?"
    print(
        f"Generation incomplete for {task_name}: "
        f"{failed_run_count}/{total_runs_text} runs failed."
    )

    error_summary_payload = build_error_summary_output_payload(payload)
    distinct_errors = error_summary_payload.get("distinct_errors", [])
    if not isinstance(distinct_errors, list) or not distinct_errors:
        print(f"Inspect {error_summary_path} for the full error log.")
        return

    for distinct_error in distinct_errors[:3]:
        if not isinstance(distinct_error, dict):
            continue
        summary = distinct_error.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        error_count = distinct_error.get("count")
        if isinstance(error_count, int) and error_count > 1:
            prefix = f"Failure ({error_count}x): "
        else:
            prefix = "Failure: "
        print(f"{prefix}{_truncate_error_summary_text(summary)}")

    if len(distinct_errors) > 3:
        print(f"{len(distinct_errors) - 3} additional distinct errors omitted.")
    print(f"Inspect {error_summary_path} for the full error log.")


def _generate_or_resume_task_payload(
    task_runtime_config: RuntimeConfig,
    *,
    output_paths: Any,
) -> tuple[dict[str, Any], bool]:
    """Generates one task payload or resumes only the still-pending run indices."""

    if (
        task_runtime_config.resume_path is not None
        and output_paths.summary_path.exists()
    ):
        existing_payload = load_generation_output_payload(output_paths)
        validate_resume_payload(task_runtime_config, existing_payload)
        pending_run_indices = tuple(existing_payload.get("pending_run_indices", []))
        if not pending_run_indices:
            return existing_payload, False
        resumed_runtime_config = task_runtime_config.for_task(
            task_runtime_config.composite_task,
            summary_path=output_paths.summary_path,
            cost_output_path=None,
            resume_path=task_runtime_config.resume_path,
            run_indices=pending_run_indices,
        )
        new_payload = generate_trajectories(resumed_runtime_config)
        return (
            merge_generation_output_payloads(
                task_runtime_config,
                existing_payload=existing_payload,
                new_payload=new_payload,
            ),
            True,
        )
    return generate_trajectories(task_runtime_config), True


def parse_args(argv: list[str] | None = None) -> RuntimeConfig:
    """Parses CLI arguments into one normalized runtime configuration."""

    load_dotenv_file()
    supported_tasks = ", ".join(supported_task_names())
    parser = argparse.ArgumentParser(
        description="Generate multi-agent task-level trajectories with google-genai on Vertex AI.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[DEFAULT_COMPOSITE_TASK],
        dest="composite_tasks",
        help=(
            "Task names to generate. Use "
            f"`{ALL_COMPOSITE_TASKS_OPTION}` for every task. Available tasks: "
            f"{supported_tasks}. --num-runs applies to each selected task."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        dest="num_runs",
        help="Number of model generation runs to execute.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        dest="num_runs",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cost-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for the cost summary file. Defaults to "
            "`cost_summary.json` alongside the summary output."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume from an existing output directory in place. For a single task, "
            "pass the task output directory. For multiple tasks, pass the request directory."
        ),
    )
    parser.add_argument(
        "--layout",
        type=int,
        default=None,
        help="Optional kitchen layout id to persist on saved trajectories.",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=None,
        help="Optional kitchen style id to persist on saved trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional scene seed to persist on saved trajectories.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Vertex model name.",
    )
    parser.add_argument(
        "--sdk",
        type=str,
        choices=[DEFAULT_SDK],
        default=DEFAULT_SDK,
        help="Generation client to use.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project ID.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION),
        help="Vertex location.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Model sampling temperature.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=("base", "verbalized"),
        default="base",
        help="Trajectory sampling strategy.",
    )
    parser.add_argument(
        "--verbalized-k",
        type=int,
        default=1,
        dest="verbalized_k",
        help="Number of trajectories to request per run when --sampling verbalized is enabled.",
    )
    parser.add_argument(
        "--verbalized_k",
        type=int,
        dest="verbalized_k",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        choices=THINKING_LEVEL_CHOICES,
        help=(
            "Optional Gemini 3 thinking level. Supported values: "
            + ", ".join(THINKING_LEVEL_CHOICES)
            + "."
        ),
    )
    parser.add_argument(
        "--thinking_level",
        type=str,
        dest="thinking_level",
        choices=THINKING_LEVEL_CHOICES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel trajectory workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum generation attempts per trajectory.",
    )
    parser.add_argument(
        "--batch-processing",
        action="store_true",
        dest="batch_processing",
        help="Use Vertex batch processing instead of online requests.",
    )
    parser.add_argument(
        "--batch_processing",
        action="store_true",
        dest="batch_processing",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-gcs-prefix",
        type=str,
        default=os.environ.get(GOOGLE_CLOUD_BATCH_GCS_PREFIX_ENV_VAR),
        help=(
            "GCS prefix for Vertex batch staging and output, for example "
            "gs://bucket/path."
        ),
    )
    parser.add_argument(
        "--batch_gcs_prefix",
        type=str,
        dest="batch_gcs_prefix",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(disable_validation=True)
    parser.add_argument(
        "--enable-validation",
        action="store_false",
        dest="disable_validation",
        help="Reject trajectories that fail symbolic validation. Validation is disabled by default.",
    )
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        dest="disable_validation",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    parsed_tasks = tuple(args.composite_tasks)
    normalized_tasks = RuntimeConfig._normalize_composite_tasks(None, parsed_tasks)
    # Resolve the default single-task summary path from the normalized task list
    # so special selectors like `all` follow the same output-path behavior.
    default_summary_path = (
        resolve_dataset_output_path(normalized_tasks[0], model=args.model)
        if len(normalized_tasks) == 1
        else None
    )

    return RuntimeConfig(
        composite_task=None,
        num_runs=args.num_runs,
        model=args.model,
        sdk=args.sdk,
        project=args.project,
        location=args.location,
        temperature=args.temperature,
        sampling=args.sampling,
        verbalized_k=args.verbalized_k,
        thinking_level=args.thinking_level,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        summary_path=default_summary_path,
        cost_output_path=args.cost_output,
        resume_path=args.resume,
        disable_validation=args.disable_validation,
        batch_processing=args.batch_processing,
        batch_gcs_prefix=args.batch_gcs_prefix,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        composite_tasks=parsed_tasks,
    )


def main(argv: list[str] | None = None) -> int:
    """Executes task-level generation for one or many requested tasks."""

    runtime_config = parse_args(argv)
    # Validate once up front so both execution paths below share the same
    # normalized configuration contract.
    _validate_runtime_config(runtime_config)
    _resolve_task_definitions_or_raise(runtime_config.composite_tasks)
    _validate_resume_directory_mode(runtime_config)

    if len(runtime_config.composite_tasks) == 1:
        # The single-task path writes one standalone dataset tree directly.
        task_runtime_config, output_paths = _task_resume_output_paths(
            runtime_config,
            composite_task=runtime_config.composite_task,
        )
        payload, should_write_outputs = _generate_or_resume_task_payload(
            task_runtime_config,
            output_paths=output_paths,
        )
        if should_write_outputs:
            (
                written_trajectory_paths,
                written_prompt_paths,
                written_output_paths,
            ) = _write_generation_outputs(
                payload,
                output_paths=output_paths,
            )
            _print_written_output_summary(
                output_paths,
                written_trajectory_paths=written_trajectory_paths,
                written_prompt_paths=written_prompt_paths,
                written_output_paths=written_output_paths,
            )
        else:
            print(
                f"Resume found no pending runs for {runtime_config.composite_task}. "
                f"Reusing existing outputs at {output_paths.summary_path}"
            )
        if not _task_is_complete(payload):
            _print_incomplete_task_summary(
                payload,
                composite_task=runtime_config.composite_task,
                error_summary_path=output_paths.error_summary_path,
            )
        return 0 if _task_is_complete(payload) else GENERATION_ERROR_EXIT_CODE

    request_summary_path = (
        runtime_config.resume_path / "summary.json"
        if runtime_config.resume_path is not None
        else resolve_request_output_path(model=runtime_config.model)
    )
    task_run_entries: list[dict[str, Any]] = []
    request_is_complete = True
    # The multi-task path runs each task independently, then writes request-
    # level summaries that point back to those per-task outputs.
    for composite_task in runtime_config.composite_tasks:
        task_runtime_config, output_paths = _task_resume_output_paths(
            runtime_config,
            composite_task=composite_task,
            request_summary_path=request_summary_path,
        )
        payload, should_write_outputs = _generate_or_resume_task_payload(
            task_runtime_config,
            output_paths=output_paths,
        )
        task_run_entries.append(
            {
                "composite_task": composite_task,
                "payload": payload,
                "output_paths": output_paths,
            }
        )
        if should_write_outputs:
            (
                written_trajectory_paths,
                written_prompt_paths,
                written_output_paths,
            ) = _write_generation_outputs(
                payload,
                output_paths=output_paths,
            )
            _print_written_output_summary(
                output_paths,
                written_trajectory_paths=written_trajectory_paths,
                written_prompt_paths=written_prompt_paths,
                written_output_paths=written_output_paths,
            )
        else:
            print(
                f"Resume found no pending runs for {composite_task}. "
                f"Reusing existing outputs at {output_paths.summary_path}"
            )
        if not _task_is_complete(payload):
            _print_incomplete_task_summary(
                payload,
                composite_task=composite_task,
                error_summary_path=output_paths.error_summary_path,
            )
        request_is_complete = request_is_complete and _task_is_complete(payload)

    _write_request_outputs(
        runtime_config,
        task_run_entries,
        request_summary_path=request_summary_path,
    )
    print(f"Wrote combined request summary to {request_summary_path}")
    print(
        "Wrote combined cost summary to "
        f"{resolve_cost_output_path(request_summary_path, runtime_config.cost_output_path)}"
    )
    print(
        "Wrote combined error summary to "
        f"{resolve_error_output_path(request_summary_path)}"
    )
    return 0 if request_is_complete else GENERATION_ERROR_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(run_cli())
