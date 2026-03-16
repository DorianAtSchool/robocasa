"""Run task-level trajectory generation through the Vertex AI batch API."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from data_generation.task_level.generation.raw import costs as _costs
from data_generation.task_level.generation.raw import errors as _errors
from data_generation.task_level.generation.raw import outputs as _outputs
from data_generation.task_level.generation.raw import progress as _progress
from data_generation.task_level.generation.raw import (
    runtime_support as _runtime_support,
)
from data_generation.task_level.generation.raw.config import (
    BATCH_DIRECTORY_NAME,
    BATCH_INTERRUPTED_MESSAGE,
    BATCH_POLL_INTERVAL_SECONDS,
    DATASET_RUN_TIMESTAMP_FORMAT,
    RuntimeConfig,
)
from data_generation.task_level.runtime.client import (
    BATCH_TRAFFIC_TYPE,
    TrajectoryGenerationError,
    build_generation_usage_metadata,
    build_raw_google_genai_client,
)
from data_generation.task_level.tasks import (
    ResponseFormatValidationError,
    TaskDefinition,
    TrajectoryValidationError,
)
from data_generation.task_level.tasks.base import TaskInstance
from data_generation.utils import camel_to_snake_case


@dataclass(frozen=True)
class BatchTrajectoryRequest:
    trajectory_index: int
    attempt_number: int
    variation_key: str
    prompt: str
    task_instance: TaskInstance


@dataclass(frozen=True)
class BatchRoundArtifacts:
    local_input_path: Path
    gcs_input_uri: str
    gcs_output_prefix: str


@dataclass(frozen=True)
class BatchRunContext:
    run_id: str
    local_staging_dir: Path
    gcs_run_prefix: str


class BatchGenerationService:
    def __init__(self, runtime_config: RuntimeConfig):
        self._client = build_raw_google_genai_client(
            project=runtime_config.project,
            location=runtime_config.location,
        )

    def create_job(
        self,
        *,
        model: str,
        input_uri: str,
        output_prefix: str,
        display_name: str,
    ) -> Any:
        return self._client.batches.create(
            model=model,
            src={
                "format": "jsonl",
                "gcs_uri": [input_uri],
            },
            config={
                "display_name": display_name,
                "dest": {
                    "format": "jsonl",
                    "gcs_uri": output_prefix,
                },
            },
        )

    def get_job(self, *, name: str) -> Any:
        return self._client.batches.get(name=name)

    def cancel_job(self, *, name: str) -> None:
        self._client.batches.cancel(name=name)


class GCSBatchStorage:
    def __init__(self, runtime_config: RuntimeConfig):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise TrajectoryGenerationError(
                "google-cloud-storage is not installed. Install it with "
                "`uv pip install google-cloud-storage`."
            ) from exc
        self._client = storage.Client(project=runtime_config.project)

    def upload_text(self, *, text: str, gcs_uri: str) -> None:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text, content_type="application/jsonl")

    def download_texts(self, *, gcs_prefix: str) -> list[tuple[str, str]]:
        bucket_name, blob_prefix = _parse_gcs_uri(gcs_prefix)
        blobs = sorted(
            self._client.list_blobs(bucket_name, prefix=blob_prefix),
            key=lambda blob: blob.name,
        )
        return [
            (f"gs://{bucket_name}/{blob.name}", blob.download_as_text())
            for blob in blobs
            if blob.name.endswith(".jsonl")
        ]


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise TrajectoryGenerationError(
            f"Invalid GCS URI '{gcs_uri}'. Expected a value starting with gs://."
        )
    bucket_and_path = gcs_uri[len("gs://") :]
    bucket_name, _, blob_name = bucket_and_path.partition("/")
    if not bucket_name:
        raise TrajectoryGenerationError(
            f"Invalid GCS URI '{gcs_uri}'. Bucket name is required."
        )
    return bucket_name, blob_name


def _join_gcs_uri(prefix: str, *parts: str) -> str:
    normalized_prefix = prefix.rstrip("/")
    normalized_parts = [part.strip("/") for part in parts if part]
    if not normalized_parts:
        return normalized_prefix
    return "/".join([normalized_prefix, *normalized_parts])


def _build_batch_service_from_runtime(
    runtime_config: RuntimeConfig,
) -> BatchGenerationService:
    return BatchGenerationService(runtime_config)


def _build_batch_storage_from_runtime(runtime_config: RuntimeConfig) -> GCSBatchStorage:
    return GCSBatchStorage(runtime_config)


def _build_batch_run_context(runtime_config: RuntimeConfig) -> BatchRunContext:
    run_id = datetime.now(timezone.utc).strftime(DATASET_RUN_TIMESTAMP_FORMAT)
    local_output_path = _outputs._resolve_summary_path(runtime_config)
    # Keep one run-scoped staging prefix so retries for the same request stay
    # grouped together locally and in GCS.
    local_staging_dir = local_output_path.parent / BATCH_DIRECTORY_NAME / run_id
    gcs_run_prefix = _join_gcs_uri(
        runtime_config.batch_gcs_prefix or "",
        camel_to_snake_case(runtime_config.composite_task),
        run_id,
    )
    return BatchRunContext(
        run_id=run_id,
        local_staging_dir=local_staging_dir,
        gcs_run_prefix=gcs_run_prefix,
    )


def _build_batch_round_artifacts(
    batch_run_context: BatchRunContext,
    *,
    round_number: int,
) -> BatchRoundArtifacts:
    round_dir_name = f"round-{round_number:02d}"
    local_round_dir = batch_run_context.local_staging_dir / round_dir_name
    local_input_path = local_round_dir / "input.jsonl"
    return BatchRoundArtifacts(
        local_input_path=local_input_path,
        gcs_input_uri=_join_gcs_uri(
            batch_run_context.gcs_run_prefix,
            round_dir_name,
            "input.jsonl",
        ),
        gcs_output_prefix=_join_gcs_uri(
            batch_run_context.gcs_run_prefix,
            round_dir_name,
            "output",
        ),
    )


def _batch_request_payload(
    *,
    prompt: str,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> dict[str, Any]:
    sampling_strategy = _runtime_support._sampling_strategy_for_runtime(runtime_config)
    generation_config = {
        "temperature": runtime_config.temperature,
        "responseMimeType": "application/json",
        "responseSchema": sampling_strategy.response_schema(
            task_definition=task_definition,
            runtime_config=runtime_config,
        ),
    }
    if runtime_config.thinking_level is not None:
        generation_config["thinkingConfig"] = {
            "thinkingLevel": runtime_config.thinking_level,
        }

    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": generation_config,
    }


def _build_batch_trajectory_request(
    *,
    trajectory_index: int,
    attempt_number: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    task_instance: TaskInstance,
) -> BatchTrajectoryRequest:
    sampling_strategy = _runtime_support._sampling_strategy_for_runtime(runtime_config)
    variation_key = _runtime_support.format_trajectory_variation_key(
        trajectory_index,
        attempt_number - 1,
    )
    return BatchTrajectoryRequest(
        trajectory_index=trajectory_index,
        attempt_number=attempt_number,
        variation_key=variation_key,
        prompt=sampling_strategy.build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            task_instance=task_instance,
            variation_key=variation_key,
        ),
        task_instance=task_instance,
    )


def _write_batch_input_jsonl(
    batch_requests: list[BatchTrajectoryRequest],
    *,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    local_input_path: Path,
) -> str:
    local_input_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                # Store the stable request identifier outside the prompt so
                # batch result matching does not depend on prompt wording.
                "variation_key": batch_request.variation_key,
                "request": _batch_request_payload(
                    prompt=batch_request.prompt,
                    runtime_config=runtime_config,
                    task_definition=task_definition,
                ),
            },
            sort_keys=True,
        )
        for batch_request in batch_requests
    ]
    payload = "\n".join(lines)
    if payload:
        payload = f"{payload}\n"
    local_input_path.write_text(payload, encoding="utf-8")
    return payload


def _batch_job_state_name(batch_job: Any) -> str | None:
    state = getattr(batch_job, "state", None)
    if state is None:
        return None
    return getattr(state, "value", None) or getattr(state, "name", None) or str(state)


def _is_terminal_batch_job_state(state_name: str | None) -> bool:
    return state_name in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
        "JOB_STATE_EXPIRED",
    }


def _is_ignorable_batch_cancel_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "already cancelled",
            "already canceled",
            "already done",
            "not found",
            "404",
            "409",
        )
    )


def _extract_batch_response_payload(response_payload: Any) -> Any:
    if isinstance(response_payload, (dict, str)):
        if isinstance(response_payload, str):
            return response_payload
        candidates = response_payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                content = first_candidate.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, list):
                        text_parts = [
                            part.get("text")
                            for part in parts
                            if isinstance(part, dict)
                            and isinstance(part.get("text"), str)
                        ]
                        if text_parts:
                            return "\n".join(text_parts)
        return response_payload
    raise ResponseFormatValidationError(
        f"Unsupported batch response payload type: {type(response_payload).__name__}"
    )


def _load_batch_output_rows(
    storage_client: GCSBatchStorage,
    *,
    gcs_output_prefix: str,
) -> list[dict[str, Any]]:
    # Vertex may shard results across multiple JSONL files, so flatten them
    # here before matching rows back to requests.
    rows: list[dict[str, Any]] = []
    for blob_uri, blob_text in storage_client.download_texts(
        gcs_prefix=gcs_output_prefix
    ):
        for line_number, line in enumerate(blob_text.splitlines(), start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                rows.append(json.loads(stripped_line))
            except json.JSONDecodeError as exc:
                raise TrajectoryGenerationError(
                    f"Batch output file {blob_uri} line {line_number} contained invalid JSON."
                ) from exc
    return rows


class RichBatchProgressDisplay:
    """Owns the interactive Rich layout for batch trajectory generation."""

    def __init__(self, runtime_config: RuntimeConfig):
        if _progress.RichProgress is None or _progress.Console is None:
            raise RuntimeError("rich progress support is unavailable")

        self.console = _progress.Console(stderr=True)
        self._progress = _progress.RichProgress(
            _progress.TextColumn("[bold]{task.description}[/bold]"),
            _progress.BarColumn(bar_width=_progress.PROGRESS_BAR_WIDTH),
            _progress.TaskProgressColumn(),
            _progress.MofNCompleteColumn(),
            _progress.StaticQueuedTimeElapsedColumn(),
            _progress.TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
            expand=False,
        )
        self._progress.start()
        self.overall_progress = _progress.RichTaskProgressAdapter(
            self._progress,
            self._progress.add_task(
                "[cyan]runs[/cyan]",
                total=runtime_config.num_runs,
                status="waiting for batch results",
            ),
            runtime_config.num_runs,
        )

    def close(self) -> None:
        self._progress.stop()


def _create_batch_progress_handles(
    runtime_config: RuntimeConfig,
    *,
    disable_progress: bool,
) -> _progress.ProgressHandles:
    # Batch mode only needs an overall completion bar, so keep the Rich layout compact.
    if (
        not disable_progress
        and _progress.RichProgress is not None
        and _progress.Console is not None
    ):
        progress_display = RichBatchProgressDisplay(runtime_config)
        return _progress.ProgressHandles(
            display=progress_display,
            overall_progress=progress_display.overall_progress,
            trajectory_progress_bars=[],
            log_writer=progress_display.console.print,
        )

    overall_progress = tqdm(
        total=runtime_config.num_runs,
        desc="Runs",
        position=0,
        disable=disable_progress,
        dynamic_ncols=True,
        colour=_progress.OVERALL_PROGRESS_COLOR,
        bar_format=_progress.TQDM_BAR_FORMAT,
    )
    return _progress.ProgressHandles(
        display=None,
        overall_progress=overall_progress,
        trajectory_progress_bars=[],
        log_writer=None,
    )


def _set_batch_progress_status(
    progress_handles: _progress.ProgressHandles,
    status: str,
    *,
    accumulated_cost_text: str | None = None,
) -> None:
    """Updates the shared batch progress status text when a progress bar is active."""

    _progress._update_overall_progress_status(
        progress_handles.overall_progress,
        status=status,
        accumulated_cost_text=accumulated_cost_text,
    )


def _batch_round_display_name(
    runtime_config: RuntimeConfig,
    *,
    batch_run_context: BatchRunContext,
    round_number: int,
) -> str:
    return (
        f"{camel_to_snake_case(runtime_config.composite_task)}-"
        f"{batch_run_context.run_id}-round-{round_number:02d}"
    )


def _wait_for_batch_job_completion(
    batch_service: BatchGenerationService,
    *,
    job_name: str,
) -> Any:
    while True:
        batch_job = batch_service.get_job(name=job_name)
        state_name = _batch_job_state_name(batch_job)
        if _is_terminal_batch_job_state(state_name):
            return batch_job
        time.sleep(BATCH_POLL_INTERVAL_SECONDS)


def _cancel_active_batch_jobs(
    batch_service: BatchGenerationService,
    *,
    active_job_names: set[str],
    enabled: bool,
    writer: Callable[[str], None] | None,
) -> None:
    for job_name in sorted(active_job_names):
        try:
            batch_service.cancel_job(name=job_name)
        except Exception as exc:
            if _is_ignorable_batch_cancel_error(exc):
                continue
            _progress._log_runtime_message(
                f"Unable to cancel batch job {job_name}: {_runtime_support._exception_summary(exc)}",
                enabled=enabled,
                writer=writer,
            )


def _batch_job_failure_message(batch_job: Any) -> str:
    state_name = _batch_job_state_name(batch_job) or "unknown"
    error = getattr(batch_job, "error", None)
    if error is None:
        return f"Batch job ended in state {state_name}."
    return f"Batch job ended in state {state_name}: {error}"


def _batch_round_completion_message(
    *,
    round_number: int,
    succeeded_count: int,
    retryable_count: int,
    failed_count: int,
    completed_count: int,
    total_count: int,
) -> str:
    return (
        f"Batch round {round_number} complete: "
        f"{succeeded_count} succeeded, "
        f"{retryable_count} retryable, "
        f"{failed_count} failed, "
        f"{completed_count}/{total_count} complete."
    )


def _raise_exhausted_batch_errors(
    runtime_config: RuntimeConfig,
    exhausted_errors: dict[int, Exception],
) -> None:
    if len(exhausted_errors) == 1:
        trajectory_index, exc = next(iter(exhausted_errors.items()))
        raise TrajectoryGenerationError(
            f"Unable to generate a valid trajectory for index {trajectory_index} after "
            f"{runtime_config.max_retries} attempts: {_runtime_support._exception_summary(exc)}"
        )

    error_summary = ", ".join(
        f"{trajectory_index}: {_runtime_support._exception_summary(exc)}"
        for trajectory_index, exc in sorted(exhausted_errors.items())
    )
    raise TrajectoryGenerationError(
        "Unable to generate valid trajectories after "
        f"{runtime_config.max_retries} attempts: {error_summary}"
    )


def generate_trajectories_batch(
    runtime_config: RuntimeConfig,
    *,
    task_definition: TaskDefinition,
    show_progress: bool,
) -> dict[str, Any]:
    sampling_strategy = _runtime_support._sampling_strategy_for_runtime(runtime_config)
    seen_signatures: set[str] = set()
    seen_signatures_lock = threading.Lock()
    error_events: list[dict[str, Any]] = []
    error_events_lock = threading.Lock()
    attempt_prompts: list[dict[str, Any]] = []
    disable_progress = not show_progress or not os.isatty(2)
    progress_handles = _create_batch_progress_handles(
        runtime_config,
        disable_progress=disable_progress,
    )
    batch_service = _build_batch_service_from_runtime(runtime_config)
    storage_client = _build_batch_storage_from_runtime(runtime_config)
    batch_run_context = _build_batch_run_context(runtime_config)
    projected_cost_estimate = _costs._build_preflight_cost_estimate_summary(
        runtime_config,
        task_definition,
    )
    accumulated_cost_tracker = _progress.AccumulatedCostTracker(
        total_trajectories=runtime_config.num_runs
        * sampling_strategy.trajectories_per_run(runtime_config),
    )
    active_job_names: set[str] = set()
    results: dict[int, list[dict[str, Any]]] = {}
    task_instances = {
        trajectory_index: task_definition.build_task_instance(trajectory_index)
        for trajectory_index in range(runtime_config.num_runs)
    }
    attempt_numbers = {
        trajectory_index: 1 for trajectory_index in range(runtime_config.num_runs)
    }
    pending_indices = list(range(runtime_config.num_runs))

    _progress._log_cost_summary(
        label="Initial projected cost",
        cost_estimate=projected_cost_estimate,
        runtime_config=runtime_config,
        enabled=show_progress,
        writer=progress_handles.log_writer,
    )
    _progress._log_runtime_message(
        "--max-workers is ignored in batch mode.",
        enabled=show_progress,
        writer=progress_handles.log_writer,
    )
    _set_batch_progress_status(
        progress_handles,
        "starting",
        accumulated_cost_text=accumulated_cost_tracker.status_text(),
    )

    try:
        # Batch mode advances in rounds so failed runs can be resubmitted
        # without rebuilding successful trajectories.
        for round_number in range(1, runtime_config.max_retries + 1):
            if not pending_indices:
                break
            _set_batch_progress_status(
                progress_handles,
                f"round {round_number}: preparing {len(pending_indices)}",
                accumulated_cost_text=accumulated_cost_tracker.status_text(),
            )

            batch_requests = [
                _build_batch_trajectory_request(
                    trajectory_index=trajectory_index,
                    attempt_number=attempt_numbers[trajectory_index],
                    runtime_config=runtime_config,
                    task_definition=task_definition,
                    task_instance=task_instances[trajectory_index],
                )
                for trajectory_index in sorted(pending_indices)
            ]
            attempt_prompts.extend(
                {
                    "run_id": _outputs.format_attempt_prompt_owner_id(
                        runtime_config,
                        run_index=batch_request.trajectory_index,
                    ),
                    "attempt_number": batch_request.attempt_number,
                    "prompt": batch_request.prompt,
                }
                for batch_request in batch_requests
            )
            round_artifacts = _build_batch_round_artifacts(
                batch_run_context,
                round_number=round_number,
            )
            batch_input_payload = _write_batch_input_jsonl(
                batch_requests,
                runtime_config=runtime_config,
                task_definition=task_definition,
                local_input_path=round_artifacts.local_input_path,
            )
            storage_client.upload_text(
                text=batch_input_payload,
                gcs_uri=round_artifacts.gcs_input_uri,
            )
            batch_job = batch_service.create_job(
                model=runtime_config.model,
                input_uri=round_artifacts.gcs_input_uri,
                output_prefix=round_artifacts.gcs_output_prefix,
                display_name=_batch_round_display_name(
                    runtime_config,
                    batch_run_context=batch_run_context,
                    round_number=round_number,
                ),
            )
            batch_job_name = getattr(batch_job, "name", None)
            if not isinstance(batch_job_name, str) or not batch_job_name:
                raise TrajectoryGenerationError(
                    "Batch job creation did not return a job name."
                )
            active_job_names.add(batch_job_name)
            _set_batch_progress_status(
                progress_handles,
                f"round {round_number}: running",
                accumulated_cost_text=accumulated_cost_tracker.status_text(),
            )
            _progress._log_runtime_message(
                "Submitted batch round "
                f"{round_number}: {batch_job_name} -> {round_artifacts.gcs_output_prefix}",
                enabled=show_progress,
                writer=progress_handles.log_writer,
            )

            batch_job = _wait_for_batch_job_completion(
                batch_service,
                job_name=batch_job_name,
            )
            active_job_names.discard(batch_job_name)
            batch_job_state = _batch_job_state_name(batch_job)
            if batch_job_state in {
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED",
            }:
                raise TrajectoryGenerationError(_batch_job_failure_message(batch_job))
            _set_batch_progress_status(
                progress_handles,
                f"round {round_number}: processing results",
                accumulated_cost_text=accumulated_cost_tracker.status_text(),
            )

            round_rows = _load_batch_output_rows(
                storage_client,
                gcs_output_prefix=round_artifacts.gcs_output_prefix,
            )
            rows_by_variation_key: dict[str, dict[str, Any]] = {}
            for row in round_rows:
                variation_key = row.get("variation_key")
                if not isinstance(variation_key, str) or not variation_key.strip():
                    continue
                rows_by_variation_key.setdefault(variation_key, row)

            next_pending_indices: list[int] = []
            exhausted_errors: dict[int, Exception] = {}
            succeeded_count = 0
            retryable_count = 0
            failed_count = 0

            # Match every returned row back to the request variation key so
            # retries remain stable even if Vertex reorders output files.
            for batch_request in batch_requests:
                row = rows_by_variation_key.get(batch_request.variation_key)
                if row is None:
                    row_error: Exception = ResponseFormatValidationError(
                        "Batch output was missing a response row for the request."
                    )
                else:
                    status = row.get("status")
                    if isinstance(status, str) and status.strip():
                        row_error = TrajectoryGenerationError(
                            f"Batch row failed: {status.strip()}"
                        )
                    else:
                        try:
                            response_payload = _extract_batch_response_payload(
                                row.get("response")
                            )
                            sampled_candidates = sampling_strategy.extract_candidates(
                                raw_response=response_payload,
                                task_definition=task_definition,
                                runtime_config=runtime_config,
                            )
                            usage = build_generation_usage_metadata(
                                (
                                    row.get("response", {}).get("usageMetadata")
                                    if isinstance(row.get("response"), dict)
                                    else None
                                ),
                                default_traffic_type=BATCH_TRAFFIC_TYPE,
                            )
                            shared_generation_usage = (
                                _runtime_support._build_shared_generation_usage(
                                    runtime_config=runtime_config,
                                    sampled_candidates=sampled_candidates,
                                    prompt=batch_request.prompt,
                                    raw_response=response_payload,
                                    usage=usage,
                                    attempt_number=batch_request.attempt_number,
                                )
                            )
                            accumulated_cost_text = (
                                accumulated_cost_tracker.add_observed_cost(
                                    shared_generation_usage.get("observed_cost_usd")
                                )
                            )
                            _set_batch_progress_status(
                                progress_handles,
                                f"round {round_number}: processing results",
                                accumulated_cost_text=accumulated_cost_text,
                            )
                            trajectory_records = _runtime_support._build_trajectory_records_from_sampled_candidates(
                                run_index=batch_request.trajectory_index,
                                runtime_config=runtime_config,
                                task_definition=task_definition,
                                task_instance=batch_request.task_instance,
                                sampled_candidates=sampled_candidates,
                                prompt=batch_request.prompt,
                                raw_response=response_payload,
                                usage=usage,
                                validator=task_definition.validator_factory(
                                    batch_request.task_instance
                                ),
                                seen_signatures=seen_signatures,
                                seen_signatures_lock=seen_signatures_lock,
                                attempt_number=batch_request.attempt_number,
                            )
                        except Exception as exc:
                            row_error = exc
                        else:
                            results[batch_request.trajectory_index] = trajectory_records
                            for trajectory_record in trajectory_records:
                                _errors._append_error_event(
                                    error_events,
                                    _errors._validation_error_event(
                                        trajectory_record["validation"],
                                        source="batch",
                                        trajectory_id=trajectory_record[
                                            "trajectory_id"
                                        ],
                                        trajectory_index=batch_request.trajectory_index,
                                        attempt_number=batch_request.attempt_number,
                                    ),
                                    error_events_lock=error_events_lock,
                                )
                            _progress._update_overall_progress_status(
                                progress_handles.overall_progress,
                                amount=1,
                                status=f"round {round_number}: processing results",
                                accumulated_cost_text=accumulated_cost_tracker.complete_trajectories(
                                    len(trajectory_records)
                                ),
                            )
                            succeeded_count += 1
                            for trajectory_record in trajectory_records:
                                _progress._log_runtime_message(
                                    _progress._trajectory_completion_log_message(
                                        runtime_config,
                                        trajectory_id=trajectory_record[
                                            "trajectory_id"
                                        ],
                                        generation_usage=trajectory_record[
                                            "generation_usage"
                                        ],
                                        validation=trajectory_record["validation"],
                                    ),
                                    enabled=show_progress,
                                    writer=progress_handles.log_writer,
                                )
                            continue

                _errors._append_error_event(
                    error_events,
                    _errors._exception_error_event(
                        row_error,
                        source="batch",
                        stage=(
                            "validation"
                            if isinstance(row_error, TrajectoryValidationError)
                            else "generation"
                        ),
                        trajectory_index=batch_request.trajectory_index,
                        attempt_number=batch_request.attempt_number,
                        retryable=batch_request.attempt_number
                        < runtime_config.max_retries,
                    ),
                    error_events_lock=error_events_lock,
                )

                if batch_request.attempt_number >= runtime_config.max_retries:
                    exhausted_errors[batch_request.trajectory_index] = row_error
                    failed_count += 1
                else:
                    attempt_numbers[batch_request.trajectory_index] += 1
                    next_pending_indices.append(batch_request.trajectory_index)
                    retryable_count += 1

            _progress._log_runtime_message(
                _batch_round_completion_message(
                    round_number=round_number,
                    succeeded_count=succeeded_count,
                    retryable_count=retryable_count,
                    failed_count=failed_count,
                    completed_count=len(results),
                    total_count=runtime_config.num_runs,
                ),
                enabled=show_progress,
                writer=progress_handles.log_writer,
            )
            if exhausted_errors:
                _raise_exhausted_batch_errors(runtime_config, exhausted_errors)
            pending_indices = sorted(next_pending_indices)
        _set_batch_progress_status(
            progress_handles,
            "complete",
            accumulated_cost_text=accumulated_cost_tracker.status_text(),
        )
    except KeyboardInterrupt:
        _cancel_active_batch_jobs(
            batch_service,
            active_job_names=active_job_names,
            enabled=show_progress,
            writer=progress_handles.log_writer,
        )
        raise KeyboardInterrupt(BATCH_INTERRUPTED_MESSAGE)
    finally:
        _progress._close_progress_handles(progress_handles)

    ordered_trajectories = [
        trajectory_record
        for index in sorted(results)
        for trajectory_record in results[index]
    ]
    return _outputs._build_generation_payload(
        runtime_config,
        ordered_trajectories,
        error_events=error_events,
        attempt_prompts=attempt_prompts,
    )
