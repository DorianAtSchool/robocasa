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
) -> dict[str, Any]:
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
    last_error: Exception | None = None

    for attempt_index in range(runtime_config.max_retries):
        # Variation keys give retries a stable way to ask for distinct traces.
        variation_key = f"traj-{trajectory_index:03d}-attempt-{attempt_index:02d}"
        prompt = task_definition.build_prompt(variation_key)
        tool_call_count: int | None = None
        if trajectory_progress is not None:
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
                response_schema=task_definition.response_schema,
                temperature=runtime_config.temperature,
            )
            response_payload, usage = trajectory_generation._unwrap_generation_response(
                raw_response
            )
            candidate = trajectory_generation.extract_json_candidate(response_payload)
            tool_call_count = trajectory_generation._tool_call_count(candidate)
            trajectory_record = trajectory_generation._build_trajectory_record_from_candidate(
                trajectory_index=trajectory_index,
                runtime_config=runtime_config,
                task_definition=task_definition,
                candidate=candidate,
                prompt=prompt,
                usage=usage,
                validator=validator,
                seen_signatures=seen_signatures,
                seen_signatures_lock=seen_signatures_lock,
                attempt_number=attempt_index + 1,
            )
            validation = trajectory_record["validation"]

            if trajectory_progress is not None:
                trajectory_progress.update(1)
                if not runtime_config.disable_validation:
                    trajectory_progress.set_postfix_str("valid")
            if overall_progress is not None:
                overall_progress.update(1)
            generation_usage = trajectory_record["generation_usage"]
            trajectory_generation._update_completed_trajectory_progress(
                trajectory_progress,
                attempt_number=attempt_index + 1,
                is_valid=validation["is_valid"],
                observed_cost_usd=generation_usage["observed_cost_usd"],
                tool_call_count=(
                    trajectory_generation._tool_call_count(trajectory_record)
                    or tool_call_count
                ),
                validation=validation,
            )
            return trajectory_record
        except Exception as exc:
            last_error = exc
            if trajectory_generation._is_non_retryable_generation_error(exc):
                raise TrajectoryGenerationError(
                    "Trajectory generation failed with a non-retryable error: "
                    f"{trajectory_generation._exception_summary(exc)}"
                ) from exc
            if trajectory_progress is not None:
                trajectory_progress.update(1)
                trajectory_progress.set_postfix_str(
                    trajectory_generation._trajectory_retry_status(
                        runtime_config,
                        attempt_number=attempt_index + 1,
                        tool_call_count=tool_call_count,
                    )
                )

    raise TrajectoryGenerationError(
        f"Unable to generate a valid trajectory for index {trajectory_index} after "
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

    disable_progress = not show_progress or not os.isatty(2)
    progress_handles = trajectory_generation._create_progress_handles(
        runtime_config,
        disable_progress=disable_progress,
    )

    projected_cost_estimate = trajectory_generation._build_preflight_cost_estimate_summary(
        runtime_config,
        task_definition,
    )
    trajectory_generation._log_cost_summary(
        label="Projected cost",
        cost_estimate=projected_cost_estimate,
        runtime_config=runtime_config,
        enabled=show_progress,
        writer=progress_handles.log_writer,
    )

    # Collect by index first so the final JSON stays deterministic under concurrency.
    results: dict[int, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(
        max_workers=min(runtime_config.max_workers, runtime_config.num_trajectories)
    )
    futures: dict[Any, int] = {}
    wait_for_shutdown = True
    try:
        futures = {
            executor.submit(
                generate_single_trajectory,
                trajectory_index=index,
                runtime_config=runtime_config,
                task_definition=task_definition,
                client_factory=client_factory,
                overall_progress=progress_handles.overall_progress,
                trajectory_progress=progress_handles.trajectory_progress_bars[index],
                seen_signatures=seen_signatures,
                seen_signatures_lock=seen_signatures_lock,
            ): index
            for index in range(runtime_config.num_trajectories)
        }

        for future in as_completed(futures):
            trajectory_index = futures[future]
            trajectory_record = future.result()
            results[trajectory_index] = trajectory_record
            generation_usage = trajectory_record["generation_usage"]
            trajectory_generation._log_runtime_message(
                trajectory_generation._trajectory_completion_log_message(
                    runtime_config,
                    trajectory_id=trajectory_record["trajectory_id"],
                    generation_usage=generation_usage,
                    validation=trajectory_record["validation"],
                ),
                enabled=show_progress,
                writer=progress_handles.log_writer,
            )
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

    ordered_trajectories = [results[index] for index in sorted(results)]
    return trajectory_generation._build_generation_payload(
        runtime_config,
        ordered_trajectories,
    )
