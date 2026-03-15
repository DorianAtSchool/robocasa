"""Run task-level trajectory generation through direct on-demand requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from typing import TYPE_CHECKING, Any

import data_generation.task_level.trajectory_generation as trajectory_generation
from data_generation.task_level.runtime.client import (
    TrajectoryGenerationError,
    build_generation_client,
)
from data_generation.task_level.tasks import TaskDefinition

if TYPE_CHECKING:
    from data_generation.task_level.trajectory_generation import RuntimeConfig


def generate_single_trajectory(
    *,
    trajectory_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    client_factory: Any = None,
    overall_progress: Any | None = None,
    trajectory_progress: Any | None = None,
    seen_signatures: set[str] | None = None,
    seen_signatures_lock: threading.Lock | None = None,
    accumulated_cost_tracker: Any | None = None,
    error_events: list[dict[str, Any]] | None = None,
    error_events_lock: threading.Lock | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Generates one run and returns one or many saved trajectory records."""

    trajectory_records = generate_single_run(
        run_index=trajectory_index,
        runtime_config=runtime_config,
        task_definition=task_definition,
        client_factory=client_factory,
        overall_progress=overall_progress,
        trajectory_progress=trajectory_progress,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
        accumulated_cost_tracker=accumulated_cost_tracker,
        error_events=error_events,
        error_events_lock=error_events_lock,
    )
    if runtime_config.sampling == "verbalized" and runtime_config.verbalized_k > 1:
        return trajectory_records
    return trajectory_records[0]


def generate_single_run(
    *,
    run_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    client_factory: Any = None,
    overall_progress: Any | None = None,
    trajectory_progress: Any | None = None,
    seen_signatures: set[str] | None = None,
    seen_signatures_lock: threading.Lock | None = None,
    accumulated_cost_tracker: Any | None = None,
    error_events: list[dict[str, Any]] | None = None,
    error_events_lock: threading.Lock | None = None,
) -> list[dict[str, Any]]:
    client = (
        client_factory()
        if client_factory is not None
        else build_generation_client(
            sdk=runtime_config.sdk,
            project=runtime_config.project,
            location=runtime_config.location,
        )
    )
    validator = task_definition.validator_factory()
    sampling_strategy = trajectory_generation._sampling_strategy_for_runtime(runtime_config)
    last_error: Exception | None = None
    trajectory_started = False
    retry_feedback: str | None = None

    for attempt_index in range(runtime_config.max_retries):
        # Variation keys give retries a stable way to ask for distinct traces.
        variation_key = trajectory_generation.format_trajectory_variation_key(
            run_index,
            attempt_index,
        )
        prompt = sampling_strategy.build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            variation_key=variation_key,
            retry_feedback=retry_feedback,
        )
        tool_call_count: int | None = None
        retry_feedback_candidate: dict[str, Any] | None = None
        if trajectory_progress is not None:
            # Start the elapsed timer only when this worker begins generation.
            if not trajectory_started:
                trajectory_progress.start()
                trajectory_started = True
            trajectory_progress.set_postfix_str(
                trajectory_generation._trajectory_generation_status(
                    runtime_config,
                    attempt_number=attempt_index + 1,
                )
            )
        try:
            raw_response = client.generate(
                model=runtime_config.model,
                prompt=prompt,
                response_schema=sampling_strategy.response_schema(
                    task_definition=task_definition,
                    runtime_config=runtime_config,
                ),
                temperature=runtime_config.temperature,
                thinking_level=runtime_config.thinking_level,
            )
            response_payload, usage = trajectory_generation._unwrap_generation_response(
                raw_response
            )
            sampled_candidates = sampling_strategy.extract_candidates(
                raw_response=response_payload,
                task_definition=task_definition,
                runtime_config=runtime_config,
            )
            if len(sampled_candidates) == 1:
                # Reuse the single invalid candidate as a tiny negative example on retries.
                retry_feedback_candidate = sampled_candidates[0].candidate
            tool_call_count = sum(
                trajectory_generation._tool_call_count(sampled_candidate.candidate) or 0
                for sampled_candidate in sampled_candidates
            ) or None
            shared_generation_usage = trajectory_generation._build_shared_generation_usage(
                runtime_config=runtime_config,
                sampled_candidates=sampled_candidates,
                prompt=prompt,
                raw_response=response_payload,
                usage=usage,
                attempt_number=attempt_index + 1,
            )
            accumulated_cost_text = (
                accumulated_cost_tracker.add_observed_cost(
                    shared_generation_usage.get("observed_cost_usd")
                )
                if accumulated_cost_tracker is not None
                else None
            )
            trajectory_generation._update_overall_progress_status(
                overall_progress,
                status="running",
                accumulated_cost_text=accumulated_cost_text,
            )
            trajectory_records = trajectory_generation._build_trajectory_records_from_sampled_candidates(
                run_index=run_index,
                runtime_config=runtime_config,
                task_definition=task_definition,
                sampled_candidates=sampled_candidates,
                prompt=prompt,
                raw_response=response_payload,
                usage=usage,
                validator=validator,
                seen_signatures=seen_signatures,
                seen_signatures_lock=seen_signatures_lock,
                attempt_number=attempt_index + 1,
            )
            for trajectory_record in trajectory_records:
                validation = trajectory_record["validation"]
                trajectory_generation._append_error_event(
                    error_events,
                    trajectory_generation._validation_error_event(
                        validation,
                        source="on_demand",
                        trajectory_id=trajectory_record["trajectory_id"],
                        trajectory_index=run_index,
                        attempt_number=attempt_index + 1,
                    ),
                    error_events_lock=error_events_lock,
                )

            completed_cost_text = (
                accumulated_cost_tracker.complete_trajectories(len(trajectory_records))
                if accumulated_cost_tracker is not None
                else accumulated_cost_text
            )
            trajectory_generation._update_overall_progress_status(
                overall_progress,
                amount=1,
                status="running",
                accumulated_cost_text=completed_cost_text,
            )
            total_observed_cost_usd = trajectory_generation._observed_cost_total(
                [
                    trajectory_record["generation_usage"]
                    for trajectory_record in trajectory_records
                ]
            )
            average_observed_cost_usd = (
                total_observed_cost_usd / len(trajectory_records)
                if total_observed_cost_usd is not None
                else None
            )
            successful_trajectory_count = sum(
                1
                for trajectory_record in trajectory_records
                if trajectory_record["validation"]["is_valid"]
            )
            invalid_validation = next(
                (
                    trajectory_record["validation"]
                    for trajectory_record in trajectory_records
                    if not trajectory_record["validation"]["is_valid"]
                ),
                None,
            )
            trajectory_generation._update_completed_trajectory_progress(
                trajectory_progress,
                attempt_number=attempt_index + 1,
                max_retries=runtime_config.max_retries,
                trajectory_count=len(trajectory_records),
                successful_trajectory_count=successful_trajectory_count,
                is_valid=all(
                    trajectory_record["validation"]["is_valid"]
                    for trajectory_record in trajectory_records
                ),
                total_observed_cost_usd=total_observed_cost_usd,
                average_observed_cost_usd=average_observed_cost_usd,
                tool_call_count=tool_call_count,
                validation=invalid_validation,
            )
            return trajectory_records
        except Exception as exc:
            last_error = exc
            if (
                not runtime_config.disable_validation
                and isinstance(exc, trajectory_generation.TrajectoryValidationError)
            ):
                retry_feedback = trajectory_generation._build_retry_feedback_text(
                    exc,
                    candidate=retry_feedback_candidate,
                )
            trajectory_generation._append_error_event(
                error_events,
                trajectory_generation._exception_error_event(
                    exc,
                    source="on_demand",
                    stage=(
                        "validation"
                        if isinstance(exc, trajectory_generation.TrajectoryValidationError)
                        else "generation"
                    ),
                    trajectory_index=run_index,
                    attempt_number=attempt_index + 1,
                    retryable=(
                        attempt_index + 1 < runtime_config.max_retries
                        and not trajectory_generation._is_non_retryable_generation_error(exc)
                    ),
                ),
                error_events_lock=error_events_lock,
            )
            if trajectory_generation._is_non_retryable_generation_error(exc):
                raise TrajectoryGenerationError(
                    "Trajectory generation failed with a non-retryable error: "
                    f"{trajectory_generation._exception_summary(exc)}"
                ) from exc
            if trajectory_progress is not None:
                trajectory_progress.set_postfix_str(
                    trajectory_generation._trajectory_retry_status(
                        runtime_config,
                        attempt_number=attempt_index + 1,
                        tool_call_count=tool_call_count,
                    )
                )

    raise TrajectoryGenerationError(
        f"Unable to generate a valid trajectory run for index {run_index} after "
        f"{runtime_config.max_retries} attempts: "
        f"{trajectory_generation._exception_summary(last_error) if last_error is not None else 'Unknown error'}"
    )


def generate_trajectories_on_demand(
    runtime_config: RuntimeConfig,
    *,
    task_definition: TaskDefinition,
    client_factory: Any = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    seen_signatures: set[str] = set()
    seen_signatures_lock = threading.Lock()
    projected_cost_estimate = trajectory_generation._build_preflight_cost_estimate_summary(
        runtime_config,
        task_definition,
    )
    accumulated_cost_tracker = trajectory_generation.AccumulatedCostTracker(
        total_trajectories=trajectory_generation._expected_saved_trajectory_count(
            runtime_config
        ),
    )
    error_events: list[dict[str, Any]] = []
    error_events_lock = threading.Lock()

    disable_progress = not show_progress or not os.isatty(2)
    progress_handles = trajectory_generation._create_progress_handles(
        runtime_config,
        disable_progress=disable_progress,
    )
    trajectory_generation._log_cost_summary(
        label="Initial projected cost",
        cost_estimate=projected_cost_estimate,
        runtime_config=runtime_config,
        enabled=show_progress,
        writer=progress_handles.log_writer,
    )
    trajectory_generation._update_overall_progress_status(
        progress_handles.overall_progress,
        status="running",
        accumulated_cost_text=accumulated_cost_tracker.status_text(),
    )

    # Collect by index first so the final JSON stays deterministic under concurrency.
    results: dict[int, list[dict[str, Any]]] = {}
    executor = ThreadPoolExecutor(
        max_workers=min(runtime_config.max_workers, runtime_config.num_runs)
    )
    futures: dict[Any, int] = {}
    wait_for_shutdown = True
    try:
        futures = {
            executor.submit(
                generate_single_run,
                run_index=index,
                runtime_config=runtime_config,
                task_definition=task_definition,
                client_factory=client_factory,
                overall_progress=progress_handles.overall_progress,
                trajectory_progress=progress_handles.trajectory_progress_bars[index],
                seen_signatures=seen_signatures,
                seen_signatures_lock=seen_signatures_lock,
                accumulated_cost_tracker=accumulated_cost_tracker,
                error_events=error_events,
                error_events_lock=error_events_lock,
            ): index
            for index in range(runtime_config.num_runs)
        }

        for future in as_completed(futures):
            run_index = futures[future]
            trajectory_records = future.result()
            results[run_index] = trajectory_records
    except KeyboardInterrupt:
        wait_for_shutdown = False
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if wait_for_shutdown:
            executor.shutdown(wait=True, cancel_futures=False)
        trajectory_generation._close_progress_handles(progress_handles)

    ordered_trajectories = [
        trajectory_record
        for index in sorted(results)
        for trajectory_record in results[index]
    ]
    return trajectory_generation._build_generation_payload(
        runtime_config,
        ordered_trajectories,
        error_events=error_events,
    )
