"""Run task-level trajectory generation through direct on-demand requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from typing import Any

from data_generation.task_level.generation import costs as _costs
from data_generation.task_level.generation import errors as _errors
from data_generation.task_level.generation import outputs as _outputs
from data_generation.task_level.generation import progress as _progress
from data_generation.task_level.generation import runtime_support as _runtime_support
from data_generation.task_level.generation.config import RuntimeConfig
from data_generation.task_level.runtime.client import (
    TrajectoryGenerationError,
    build_generation_client,
)
from data_generation.task_level.tasks import (
    DuplicateTrajectoryValidationError,
    TaskDefinition,
    TrajectoryValidationError,
)


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
    attempt_prompts: list[dict[str, Any]] | None = None,
    attempt_prompts_lock: threading.Lock | None = None,
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
        attempt_prompts=attempt_prompts,
        attempt_prompts_lock=attempt_prompts_lock,
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
    attempt_prompts: list[dict[str, Any]] | None = None,
    attempt_prompts_lock: threading.Lock | None = None,
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
    task_instance = task_definition.build_task_instance(run_index)
    validator = task_definition.validator_factory(task_instance)
    sampling_strategy = _runtime_support._sampling_strategy_for_runtime(runtime_config)
    last_error: Exception | None = None
    trajectory_started = False
    retry_feedback: str | None = None
    greedy_verbalized_validation = (
        not runtime_config.disable_validation
        and runtime_config.sampling == "verbalized"
        and runtime_config.verbalized_k > 1
    )
    run_completed = False
    accumulated_valid_results: list[tuple[Any, dict[str, Any], dict[str, Any], str]] = []
    accumulated_generation_usages: list[dict[str, Any]] = []
    reserved_run_signatures: set[str] = set()
    previous_invalid_summary: str | None = None

    for attempt_index in range(runtime_config.max_retries):
        # Variation keys give retries a stable way to ask for distinct traces.
        variation_key = _runtime_support.format_trajectory_variation_key(
            run_index,
            attempt_index,
        )
        prompt = sampling_strategy.build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            task_instance=task_instance,
            variation_key=variation_key,
            retry_feedback=retry_feedback,
        )
        attempt_prompt_entry = {
            "run_id": _outputs.format_attempt_prompt_owner_id(
                runtime_config,
                run_index=run_index,
            ),
            "attempt_number": attempt_index + 1,
            "prompt": prompt,
        }
        if attempt_prompts is not None:
            if attempt_prompts_lock is None:
                attempt_prompts.append(attempt_prompt_entry)
            else:
                with attempt_prompts_lock:
                    attempt_prompts.append(attempt_prompt_entry)
        tool_call_count: int | None = None
        retry_feedback_candidate: dict[str, Any] | None = None
        if trajectory_progress is not None:
            # Start the elapsed timer only when this worker begins generation.
            if not trajectory_started:
                trajectory_progress.start()
                trajectory_started = True
            trajectory_progress.set_postfix_str(
                _progress._trajectory_generation_status(
                    runtime_config,
                    attempt_number=attempt_index + 1,
                    previous_invalid_summary=previous_invalid_summary,
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
            response_payload, usage = _runtime_support._unwrap_generation_response(
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
                _runtime_support._tool_call_count(sampled_candidate.candidate) or 0
                for sampled_candidate in sampled_candidates
            ) or None
            shared_generation_usage = _runtime_support._build_shared_generation_usage(
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
            _progress._update_overall_progress_status(
                overall_progress,
                status="running",
                accumulated_cost_text=accumulated_cost_text,
            )
            if greedy_verbalized_validation:
                invalid_validations: list[dict[str, Any]] = []
                valid_results_this_attempt: list[
                    tuple[Any, dict[str, Any], dict[str, Any], str]
                ] = []
                needed_count = runtime_config.verbalized_k - len(accumulated_valid_results)

                for sampled_candidate in sampled_candidates:
                    validation, normalized_candidate = _runtime_support._validate_candidate(
                        sampled_candidate.candidate,
                        validator,
                        enforce_validation=False,
                    )
                    if validation["is_valid"] is not True:
                        invalid_validations.append(validation)
                        if retry_feedback_candidate is None:
                            retry_feedback_candidate = sampled_candidate.candidate
                        continue

                    try:
                        _runtime_support._maybe_reserve_signature(
                            validation,
                            disable_validation=False,
                            seen_signatures=seen_signatures,
                            seen_signatures_lock=seen_signatures_lock,
                        )
                    except DuplicateTrajectoryValidationError as exc:
                        invalid_validations.append(
                            _runtime_support._validation_error_payload(
                                exc,
                                candidate=normalized_candidate,
                            )
                        )
                        if retry_feedback_candidate is None:
                            retry_feedback_candidate = sampled_candidate.candidate
                        continue

                    reserved_run_signatures.add(validation["signature"])
                    valid_results_this_attempt.append(
                        (
                            sampled_candidate,
                            validation,
                            normalized_candidate,
                            prompt,
                        )
                    )
                    if len(valid_results_this_attempt) >= needed_count:
                        break

                for invalid_validation in invalid_validations:
                    _errors._append_error_event(
                        error_events,
                        _errors._validation_error_event_from_payload(
                            invalid_validation,
                            source="on_demand",
                            trajectory_index=run_index,
                            attempt_number=attempt_index + 1,
                            retryable=(
                                len(accumulated_valid_results) + len(valid_results_this_attempt)
                                < runtime_config.verbalized_k
                                and attempt_index + 1 < runtime_config.max_retries
                            ),
                        ),
                        error_events_lock=error_events_lock,
                    )

                if valid_results_this_attempt:
                    accumulated_generation_usages.append(shared_generation_usage)
                    accumulated_valid_results.extend(valid_results_this_attempt)

                if len(accumulated_valid_results) < runtime_config.verbalized_k:
                    last_error = _runtime_support._build_verbalized_insufficient_results_error(
                        required_count=runtime_config.verbalized_k,
                        collected_count=len(accumulated_valid_results),
                        invalid_validations=invalid_validations,
                    )
                    if trajectory_progress is not None:
                        invalid_summary = (
                            _runtime_support._validation_errors_retry_summary(
                                invalid_validations
                            )
                        )
                        previous_invalid_summary = invalid_summary
                        trajectory_progress.set_postfix_str(
                            _progress._trajectory_retry_status(
                                runtime_config,
                                attempt_number=attempt_index + 1,
                                tool_call_count=tool_call_count,
                                invalid_summary=invalid_summary,
                            )
                        )
                    if invalid_validations:
                        first_invalid = invalid_validations[0]
                        retry_feedback = _runtime_support._build_retry_feedback_text_from_validation(
                            first_invalid,
                            candidate=retry_feedback_candidate,
                        )
                    continue

                total_prompt_tokens = sum(
                    generation_usage["prompt_tokens"]
                    for generation_usage in accumulated_generation_usages
                )
                total_output_tokens = sum(
                    generation_usage["output_tokens"]
                    for generation_usage in accumulated_generation_usages
                )
                total_reasoning_tokens = sum(
                    _costs._reasoning_token_count(generation_usage)
                    for generation_usage in accumulated_generation_usages
                )
                total_observed_cost_usd = _costs._observed_cost_total(
                    accumulated_generation_usages
                )
                aggregate_generation_usage = dict(accumulated_generation_usages[0])
                aggregate_generation_usage["prompt_tokens"] = total_prompt_tokens
                aggregate_generation_usage["output_tokens"] = total_output_tokens
                aggregate_generation_usage["reasoning_tokens"] = total_reasoning_tokens
                aggregate_generation_usage["total_tokens"] = (
                    total_prompt_tokens + total_output_tokens + total_reasoning_tokens
                )
                aggregate_generation_usage["observed_cost_usd"] = total_observed_cost_usd
                aggregate_generation_usage["successful_attempt_number"] = attempt_index + 1
                aggregate_generation_usage["retry_costs_included"] = (
                    attempt_index + 1 > 1
                )
                split_generation_usages = (
                    _runtime_support._split_generation_usage_across_candidates(
                        aggregate_generation_usage,
                        candidate_count=len(accumulated_valid_results),
                    )
                )
                trajectory_records: list[dict[str, Any]] = []
                for candidate_index, (
                    (
                        sampled_candidate,
                        validation,
                        normalized_candidate,
                        candidate_prompt,
                    ),
                    generation_usage,
                ) in enumerate(
                    zip(accumulated_valid_results, split_generation_usages)
                ):
                    trajectory_id = _runtime_support.format_trajectory_id(
                        _runtime_support._global_trajectory_index(
                            runtime_config,
                            run_index=run_index,
                            candidate_index=candidate_index,
                        )
                    )
                    trajectory_record = task_definition.build_trajectory_record(
                        candidate=normalized_candidate,
                        validation=validation,
                        trajectory_id=trajectory_id,
                        generation_usage=generation_usage,
                        task_instance=task_instance,
                    )
                    if sampled_candidate.probability is not None:
                        trajectory_record["sampling_metadata"] = {
                            "strategy": runtime_config.sampling,
                            "probability": sampled_candidate.probability,
                            "candidate_index": candidate_index,
                            "run_index": run_index,
                        }
                    trajectory_record["prompt"] = candidate_prompt
                    trajectory_record["raw_output"] = sampled_candidate.raw_output
                    trajectory_records.append(trajectory_record)
            else:
                trajectory_records = _runtime_support._build_trajectory_records_from_sampled_candidates(
                    run_index=run_index,
                    runtime_config=runtime_config,
                    task_definition=task_definition,
                    task_instance=task_instance,
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
                _errors._append_error_event(
                    error_events,
                    _errors._validation_error_event(
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
            _progress._update_overall_progress_status(
                overall_progress,
                amount=1,
                status="running",
                accumulated_cost_text=completed_cost_text,
            )
            total_observed_cost_usd = _costs._observed_cost_total(
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
            _progress._update_completed_trajectory_progress(
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
            run_completed = True
            return trajectory_records
        except Exception as exc:
            last_error = exc
            if (
                not runtime_config.disable_validation
                and isinstance(exc, TrajectoryValidationError)
            ):
                retry_feedback = _runtime_support._build_retry_feedback_text(
                    exc,
                    candidate=retry_feedback_candidate,
                )
            _errors._append_error_event(
                error_events,
                _errors._exception_error_event(
                    exc,
                    source="on_demand",
                    stage=(
                        "validation"
                        if isinstance(exc, TrajectoryValidationError)
                        else "generation"
                    ),
                    trajectory_index=run_index,
                    attempt_number=attempt_index + 1,
                    retryable=(
                        attempt_index + 1 < runtime_config.max_retries
                        and not _runtime_support._is_non_retryable_generation_error(exc)
                    ),
                ),
                error_events_lock=error_events_lock,
            )
            if _runtime_support._is_non_retryable_generation_error(exc):
                raise TrajectoryGenerationError(
                    "Trajectory generation failed with a non-retryable error: "
                    f"{_runtime_support._exception_summary(exc)}"
                ) from exc
            if trajectory_progress is not None:
                invalid_summary = None
                if isinstance(exc, TrajectoryValidationError):
                    invalid_summary = _runtime_support._validation_error_retry_summary(
                        _runtime_support._validation_error_payload(exc)
                    )
                previous_invalid_summary = invalid_summary
                trajectory_progress.set_postfix_str(
                    _progress._trajectory_retry_status(
                        runtime_config,
                        attempt_number=attempt_index + 1,
                        tool_call_count=tool_call_count,
                        invalid_summary=invalid_summary,
                    )
                )

    if (
        not run_completed
        and reserved_run_signatures
        and seen_signatures is not None
        and seen_signatures_lock is not None
    ):
        with seen_signatures_lock:
            seen_signatures.difference_update(reserved_run_signatures)
    raise TrajectoryGenerationError(
        f"Unable to generate a valid trajectory run for index {run_index} after "
        f"{runtime_config.max_retries} attempts: "
        f"{_runtime_support._exception_summary(last_error) if last_error is not None else 'Unknown error'}"
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
    projected_cost_estimate = _costs._build_preflight_cost_estimate_summary(
        runtime_config,
        task_definition,
    )
    accumulated_cost_tracker = _progress.AccumulatedCostTracker(
        total_trajectories=_runtime_support._expected_saved_trajectory_count(
            runtime_config
        ),
    )
    error_events: list[dict[str, Any]] = []
    error_events_lock = threading.Lock()
    attempt_prompts: list[dict[str, Any]] = []
    attempt_prompts_lock = threading.Lock()

    disable_progress = not show_progress or not os.isatty(2)
    progress_handles = _progress._create_progress_handles(
        runtime_config,
        disable_progress=disable_progress,
    )
    _progress._log_cost_summary(
        label="Initial projected cost",
        cost_estimate=projected_cost_estimate,
        runtime_config=runtime_config,
        enabled=show_progress,
        writer=progress_handles.log_writer,
    )
    _progress._update_overall_progress_status(
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
                attempt_prompts=attempt_prompts,
                attempt_prompts_lock=attempt_prompts_lock,
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
