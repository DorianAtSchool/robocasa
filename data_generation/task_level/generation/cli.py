"""Parse CLI arguments and write task-level generation outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from data_generation.task_level.generation.config import (
    DEFAULT_COMPOSITE_TASK,
    GENERATION_ERROR_EXIT_CODE,
    GOOGLE_CLOUD_BATCH_GCS_PREFIX_ENV_VAR,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_MESSAGE,
    RuntimeConfig,
    THINKING_LEVEL_CHOICES,
    _validate_runtime_config,
)
from data_generation.task_level.generation.orchestrator import generate_trajectories
from data_generation.task_level.generation.outputs import (
    _print_written_output_summary,
    _resolve_output_paths,
    _write_generation_outputs,
    _write_request_outputs,
    resolve_cost_output_path,
    resolve_dataset_output_path,
    resolve_error_output_path,
    resolve_request_output_path,
    resolve_request_task_output_path,
)
from data_generation.task_level.generation.runtime_support import _exception_summary, _resolve_task_definitions_or_raise
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
            "Task names to generate. Available tasks: "
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
            "Optional JSON path for the cost summary sidecar. Defaults to "
            "`cost_summary.json` alongside the summary output."
        ),
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
    default_summary_path = (
        resolve_dataset_output_path(parsed_tasks[0])
        if len(parsed_tasks) == 1
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
        disable_validation=args.disable_validation,
        batch_processing=args.batch_processing,
        batch_gcs_prefix=args.batch_gcs_prefix,
        composite_tasks=parsed_tasks,
    )

def main(argv: list[str] | None = None) -> int:
    """Executes task-level generation for one or many requested tasks."""

    runtime_config = parse_args(argv)
    _validate_runtime_config(runtime_config)
    _resolve_task_definitions_or_raise(runtime_config.composite_tasks)

    if len(runtime_config.composite_tasks) == 1:
        output_paths = _resolve_output_paths(runtime_config)
        payload = generate_trajectories(runtime_config)
        written_trajectory_paths, written_prompt_paths, written_output_paths = _write_generation_outputs(
            payload,
            output_paths=output_paths,
        )
        _print_written_output_summary(
            output_paths,
            written_trajectory_paths=written_trajectory_paths,
            written_prompt_paths=written_prompt_paths,
            written_output_paths=written_output_paths,
        )
        return 0

    request_summary_path = resolve_request_output_path()
    task_run_entries: list[dict[str, Any]] = []
    for composite_task in runtime_config.composite_tasks:
        task_summary_path = resolve_request_task_output_path(
            request_summary_path,
            composite_task,
        )
        task_runtime_config = runtime_config.for_task(
            composite_task,
            summary_path=task_summary_path,
            cost_output_path=None,
        )
        output_paths = _resolve_output_paths(task_runtime_config)
        payload = generate_trajectories(task_runtime_config)
        written_trajectory_paths, written_prompt_paths, written_output_paths = _write_generation_outputs(
            payload,
            output_paths=output_paths,
        )
        task_run_entries.append(
            {
                "composite_task": composite_task,
                "payload": payload,
                "output_paths": output_paths,
            }
        )
        _print_written_output_summary(
            output_paths,
            written_trajectory_paths=written_trajectory_paths,
            written_prompt_paths=written_prompt_paths,
            written_output_paths=written_output_paths,
        )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
