"""Provide shared runtime helpers for validation, sampling, and JSON parsing."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from data_generation.task_level.generation.raw.config import (
    RETRY_PROGRESS_ERROR_MESSAGE_MAX_LENGTH,
    RuntimeConfig,
    TRAJECTORY_ID_DIGITS,
)
from data_generation.task_level.runtime.client import (
    BATCH_TRAFFIC_TYPE,
    COST_DECIMAL_PLACES,
    GenerationResult,
    GenerationUsage,
    TrajectoryGenerationError,
    _generation_error_status_code,
    build_generation_usage,
)
from data_generation.task_level.sampling import (
    SampledTrajectoryCandidate,
    get_sampling_strategy,
)
from data_generation.task_level.tasks import (
    DuplicateTrajectoryValidationError,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    InsufficientValidUniqueTrajectoriesValidationError,
    ResponseFormatValidationError,
    TaskDefinition,
    TaskValidator,
    TrajectoryValidationError,
    get_task_definition,
    supported_task_names,
)
from data_generation.utils import round_cost, stable_json_sha256


def format_trajectory_id(trajectory_index: int) -> str:
    """Formats one persisted trajectory ID with enough padding for large runs."""

    return f"traj_{trajectory_index:0{TRAJECTORY_ID_DIGITS}d}"


def format_trajectory_variation_key(
    trajectory_index: int,
    attempt_index: int,
) -> str:
    """Formats the retry variation key used to diversify model attempts."""

    return (
        f"traj-{trajectory_index:0{TRAJECTORY_ID_DIGITS}d}-attempt-{attempt_index:02d}"
    )


def format_trajectory_progress_label(trajectory_index: int) -> str:
    """Formats the short progress-bar label for one generation run."""

    return (
        format_trajectory_id(trajectory_index).replace("traj", "run").replace("_", " ")
    )


def _retry_feedback_step_lines(
    candidate: dict[str, Any] | None,
    *,
    failing_step: int | None,
) -> list[str]:
    """Builds a tiny local counterexample around the failing step when possible."""

    if not isinstance(failing_step, int) or not isinstance(candidate, dict):
        return []
    steps = candidate.get("steps")
    if not isinstance(steps, list):
        return []

    step_by_index = {
        step["step"]: step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("step"), int)
    }
    lines: list[str] = []
    previous_step = step_by_index.get(failing_step - 1)
    if previous_step is not None:
        lines.append(
            f"- step {previous_step['step']}: {json.dumps(previous_step, sort_keys=True)}"
        )
    failing_step_payload = step_by_index.get(failing_step)
    if failing_step_payload is not None:
        lines.append(
            f"- step {failing_step_payload['step']}: {json.dumps(failing_step_payload, sort_keys=True)}"
        )
    return lines


def _build_retry_feedback_text(
    exc: TrajectoryValidationError,
    *,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Builds the compact retry block appended to the next generation prompt."""

    lines = [
        "Previous attempt failed validation.",
        "",
        "Failure summary:",
        f"- error_type: {exc.error_type}",
    ]
    if isinstance(exc.step, int):
        lines.append(f"- failing_step: {exc.step}")
    if str(exc).strip():
        lines.append(f"- message: {str(exc).strip()}")
    if exc.details:
        lines.append(f"- details: {json.dumps(exc.details, sort_keys=True)}")

    step_lines = _retry_feedback_step_lines(candidate, failing_step=exc.step)
    if step_lines:
        lines.extend(["", "Local bad example:", *step_lines])

    lines.extend(
        [
            "",
            "Repair instructions:",
            "- Regenerate the full trajectory from step 0.",
            "- Do not continue or patch the previous attempt.",
            "- Avoid the same validation failure and keep the whole trajectory symbolically consistent.",
        ]
    )
    return "\n".join(lines)


def _build_retry_feedback_text_from_validation(
    validation: dict[str, Any],
    *,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Builds retry feedback directly from a serialized validation payload."""

    error_type = _validation_error_type(validation) or "TrajectoryValidationError"
    step = validation.get("step")
    error_message = validation.get("error")
    error_details = validation.get("error_details")

    lines = [
        "Previous attempt failed validation.",
        "",
        "Failure summary:",
        f"- error_type: {error_type}",
    ]
    if isinstance(step, int):
        lines.append(f"- failing_step: {step}")
    if isinstance(error_message, str) and error_message.strip():
        lines.append(f"- message: {error_message.strip()}")
    if isinstance(error_details, dict) and error_details:
        lines.append(f"- details: {json.dumps(error_details, sort_keys=True)}")

    step_lines = _retry_feedback_step_lines(
        candidate, failing_step=step if isinstance(step, int) else None
    )
    if step_lines:
        lines.extend(["", "Local bad example:", *step_lines])

    lines.extend(
        [
            "",
            "Repair instructions:",
            "- Regenerate the full trajectory from step 0.",
            "- Do not continue or patch the previous attempt.",
            "- Avoid the same validation failure and keep the whole trajectory symbolically consistent.",
        ]
    )
    return "\n".join(lines)


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    if isinstance(exc, TrajectoryGenerationError):
        return True

    status_code = _generation_error_status_code(exc)
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return True

    text = str(exc)
    non_retryable_markers = (
        "400 INVALID_ARGUMENT",
        "403 PERMISSION_DENIED",
        "401 UNAUTHENTICATED",
        "429 RESOURCE_EXHAUSTED",
        "DefaultCredentialsError",
        "credentials were not found",
        "aiplatform.endpoints.predict",
        "API has not been used",
        "SERVICE_DISABLED",
    )
    return any(marker in text for marker in non_retryable_markers)


def _resolve_task_definition_or_raise(composite_task: str) -> TaskDefinition:
    # Keep CLI parsing and generation on the same task registry lookup path.
    task_definition = get_task_definition(composite_task)
    if task_definition is None:
        supported_tasks = ", ".join(supported_task_names())
        raise TrajectoryGenerationError(
            f"Unsupported task '{composite_task}'. "
            f"Available tasks: {supported_tasks}."
        )
    return task_definition


def _resolve_task_definitions_or_raise(
    composite_tasks: tuple[str, ...],
) -> tuple[TaskDefinition, ...]:
    """Resolves every selected task definition in CLI order."""

    return tuple(
        _resolve_task_definition_or_raise(composite_task)
        for composite_task in composite_tasks
    )


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return stable_json_sha256(candidate, default=str)


def _default_traffic_type_for_runtime(runtime_config: RuntimeConfig) -> str:
    if runtime_config.batch_processing:
        return BATCH_TRAFFIC_TYPE
    return "ON_DEMAND"


def _sampling_strategy_for_runtime(runtime_config: RuntimeConfig):
    """Returns the configured sampling strategy for the current runtime."""

    return get_sampling_strategy(getattr(runtime_config, "sampling", "base"))


def _trajectories_per_run(runtime_config: RuntimeConfig) -> int:
    """Returns how many saved trajectories one successful run should emit."""

    return _sampling_strategy_for_runtime(runtime_config).trajectories_per_run(
        runtime_config
    )


def _expected_saved_trajectory_count(runtime_config: RuntimeConfig) -> int:
    """Returns the expected saved trajectory count for one successful job."""

    return runtime_config.num_runs * _trajectories_per_run(runtime_config)


def _global_trajectory_index(
    runtime_config: RuntimeConfig,
    *,
    run_index: int,
    candidate_index: int,
) -> int:
    """Builds the flattened saved-trajectory index for one run candidate."""

    return (run_index * _trajectories_per_run(runtime_config)) + candidate_index


def _build_trajectory_record_from_candidate(
    *,
    trajectory_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    candidate: dict[str, Any],
    prompt: str,
    raw_output: Any,
    usage: GenerationUsage | None,
    validator: TaskValidator,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
    attempt_number: int,
) -> dict[str, Any]:
    # Reuse the shared sampled-candidate path so base and verbalized modes keep
    # one normalization and validation implementation.
    trajectory_records = _build_trajectory_records_from_sampled_candidates(
        run_index=trajectory_index,
        runtime_config=runtime_config,
        task_definition=task_definition,
        sampled_candidates=[
            SampledTrajectoryCandidate(
                candidate=candidate,
                raw_output=raw_output,
            )
        ],
        prompt=prompt,
        raw_response=raw_output,
        usage=usage,
        validator=validator,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
        attempt_number=attempt_number,
    )
    return trajectory_records[0]


def _exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _validation_error_type(validation: dict[str, Any]) -> str | None:
    error_type = validation.get("error_type")
    if isinstance(error_type, str) and error_type:
        return error_type
    return None


def _validation_error_base_type(validation: dict[str, Any]) -> str | None:
    """Returns the top-level validation family when one is available."""

    error_base_type = validation.get("error_base_type")
    if isinstance(error_base_type, str) and error_base_type:
        return error_base_type
    return None


def _validation_error_summary(validation: dict[str, Any]) -> str | None:
    error_type = _validation_error_type(validation)
    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message:
        if error_type is not None:
            return f"{error_type}: {error_message}"
        return error_message
    return error_type


def _validation_error_progress_summary(validation: dict[str, Any]) -> str | None:
    """Builds a compact validation summary for progress bars."""

    error_type = _validation_error_type(validation)
    step = validation.get("step")
    if error_type is not None:
        if isinstance(step, int):
            return f"{error_type} step={step}"
        return error_type

    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message:
        return error_message.split(":", 1)[0]
    return None


def _truncate_progress_text(text: str, *, max_length: int) -> str:
    """Collapses whitespace and truncates long progress-bar text with an ellipsis."""

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return f"{normalized_text[: max_length - 3].rstrip()}..."


def _validation_error_retry_summary(validation: dict[str, Any]) -> str | None:
    """Builds a retry status summary that includes the failing validation message."""

    progress_summary = _validation_error_progress_summary(validation)
    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message.strip():
        truncated_message = _truncate_progress_text(
            error_message,
            max_length=RETRY_PROGRESS_ERROR_MESSAGE_MAX_LENGTH,
        )
        if progress_summary is not None:
            return f"{progress_summary}: {truncated_message}"
        return truncated_message
    return progress_summary


def _validation_errors_retry_summary(validations: list[dict[str, Any]]) -> str | None:
    """Aggregates one attempt's invalid validations into a compact retry suffix."""

    distinct_summaries: list[str] = []
    for validation in validations:
        retry_summary = _validation_error_retry_summary(validation)
        if retry_summary is None or retry_summary in distinct_summaries:
            continue
        distinct_summaries.append(retry_summary)

    if not distinct_summaries:
        return None
    if len(distinct_summaries) == 1:
        return distinct_summaries[0]

    displayed_summaries = "; ".join(distinct_summaries[:2])
    if len(distinct_summaries) > 2:
        return f"{displayed_summaries}; +{len(distinct_summaries) - 2} more"
    return displayed_summaries


def _unwrap_generation_response(
    raw_response: Any,
) -> tuple[Any, GenerationUsage | None]:
    # Both SDK wrappers and test doubles feed through here, so normalize the
    # payload shape before sampling code looks at it.
    if isinstance(raw_response, GenerationResult):
        return raw_response.payload, raw_response.usage
    return raw_response, None


def _validate_candidate(
    candidate: dict[str, Any],
    validator: TaskValidator,
    *,
    enforce_validation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        validation = dict(validator.validate(candidate))
        normalized_candidate = validation.pop("normalized_candidate", candidate)
        return validation, normalized_candidate
    except TrajectoryValidationError as exc:
        if enforce_validation:
            raise
        # Preserve the invalid trace for inspection when validation is disabled.
        return (
            {
                "is_valid": False,
                "validation_disabled": True,
                "error_type": exc.error_type,
                "error_base_type": exc.error_base_type,
                "error": str(exc),
                "error_details": dict(exc.details) if exc.details else None,
                "step": exc.step if isinstance(exc.step, int) else None,
                "checks": [],
                "final_state": None,
                "signature": _candidate_signature(candidate),
            },
            candidate,
        )


def _validation_error_payload(
    exc: TrajectoryValidationError,
    *,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializes one validation exception into the shared validation shape."""

    payload = {
        "is_valid": False,
        "error_type": exc.error_type,
        "error_base_type": exc.error_base_type,
        "error": str(exc),
        "error_details": dict(exc.details) if exc.details else None,
        "step": exc.step if isinstance(exc.step, int) else None,
        "checks": [],
        "final_state": None,
    }
    if candidate is not None:
        payload["signature"] = _candidate_signature(candidate)
    return payload


def _build_verbalized_insufficient_results_error(
    *,
    required_count: int,
    collected_count: int,
    invalid_validations: list[dict[str, Any]],
) -> TrajectoryValidationError:
    """Builds one descriptive insufficiency error for verbalized sampling retries."""

    duplicate_count = 0
    invalid_count = 0
    for validation in invalid_validations:
        if _validation_error_type(validation) == "DuplicateTrajectoryValidationError":
            duplicate_count += 1
            continue
        invalid_count += 1

    missing_count = max(required_count - collected_count, 0)
    details = {
        "required_valid_unique_trajectories": required_count,
        "collected_valid_unique_trajectories": collected_count,
        "missing_valid_unique_trajectories": missing_count,
        "duplicate_candidate_count": duplicate_count,
        "invalid_candidate_count": invalid_count,
    }

    if duplicate_count and invalid_count:
        return InsufficientValidUniqueTrajectoriesMixedError(
            "Verbalized run did not produce enough valid unique trajectories because "
            "some candidates failed validation and others duplicated existing "
            "trajectories.",
            details=details,
        )
    if duplicate_count:
        return InsufficientValidUniqueTrajectoriesDuplicateError(
            "Verbalized run did not produce enough valid unique trajectories because "
            "some candidates duplicated existing trajectories.",
            details=details,
        )
    if invalid_count:
        return InsufficientValidUniqueTrajectoriesInvalidError(
            "Verbalized run did not produce enough valid unique trajectories because "
            "some candidates failed validation.",
            details=details,
        )
    return InsufficientValidUniqueTrajectoriesValidationError(
        "Verbalized run did not produce enough valid unique trajectories.",
        details=details,
    )


def _split_integer_total(total: int, parts: int) -> list[int]:
    """Splits one integer total across parts while preserving the sum."""

    base_value, remainder = divmod(total, parts)
    return [base_value + (1 if index < remainder else 0) for index in range(parts)]


def _split_float_total(total: float | None, parts: int) -> list[float | None]:
    """Splits one rounded float total across parts while preserving the sum."""

    if total is None:
        return [None] * parts
    if parts == 1:
        return [round_cost(total, decimal_places=COST_DECIMAL_PLACES)]

    split_values: list[float] = []
    remaining = total
    for index in range(parts):
        parts_left = parts - index
        if parts_left == 1:
            split_values.append(
                round_cost(remaining, decimal_places=COST_DECIMAL_PLACES) or 0.0
            )
            continue
        split_value = (
            round_cost(
                total / parts,
                decimal_places=COST_DECIMAL_PLACES,
            )
            or 0.0
        )
        split_values.append(split_value)
        remaining -= split_value
    return split_values


def _reasoning_token_count(generation_usage: dict[str, Any]) -> int:
    """Read reasoning tokens from usage payloads when splitting shared totals."""

    reasoning_tokens = generation_usage.get("reasoning_tokens")
    if not isinstance(reasoning_tokens, int):
        return 0
    return reasoning_tokens


def _split_generation_usage_across_candidates(
    generation_usage: dict[str, Any],
    *,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Splits one run's usage metadata across the saved candidates from that run."""

    if candidate_count == 1:
        return [dict(generation_usage)]

    prompt_splits = _split_integer_total(
        generation_usage["prompt_tokens"], candidate_count
    )
    output_splits = _split_integer_total(
        generation_usage["output_tokens"], candidate_count
    )
    reasoning_splits = _split_integer_total(
        _reasoning_token_count(generation_usage),
        candidate_count,
    )
    cost_splits = _split_float_total(
        generation_usage.get("observed_cost_usd"),
        candidate_count,
    )

    split_usages: list[dict[str, Any]] = []
    for index in range(candidate_count):
        split_usage = dict(generation_usage)
        split_usage["prompt_tokens"] = prompt_splits[index]
        split_usage["output_tokens"] = output_splits[index]
        split_usage["reasoning_tokens"] = reasoning_splits[index]
        split_usage["total_tokens"] = (
            split_usage["prompt_tokens"]
            + split_usage["output_tokens"]
            + split_usage["reasoning_tokens"]
        )
        split_usage["observed_cost_usd"] = cost_splits[index]
        split_usages.append(split_usage)
    return split_usages


def _reserve_signature_batch(
    validations: list[dict[str, Any]],
    *,
    disable_validation: bool,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
) -> None:
    """Atomically reserves all candidate signatures produced by one successful run."""

    if disable_validation or seen_signatures is None or seen_signatures_lock is None:
        return

    signatures = [validation["signature"] for validation in validations]
    if len(signatures) != len(set(signatures)):
        raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")

    with seen_signatures_lock:
        if any(signature in seen_signatures for signature in signatures):
            raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")
        seen_signatures.update(signatures)


def _build_trajectory_records_from_sampled_candidates(
    *,
    run_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    task_instance: Any,
    sampled_candidates: list[SampledTrajectoryCandidate],
    prompt: str,
    raw_response: Any,
    usage: GenerationUsage | None,
    validator: TaskValidator,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
    attempt_number: int,
) -> list[dict[str, Any]]:
    """Builds saved trajectory records for all candidates emitted by one run."""

    validations: list[dict[str, Any]] = []
    normalized_candidates: list[dict[str, Any]] = []
    for sampled_candidate in sampled_candidates:
        validation, normalized_candidate = _validate_candidate(
            sampled_candidate.candidate,
            validator,
            enforce_validation=not runtime_config.disable_validation,
        )
        validations.append(validation)
        normalized_candidates.append(normalized_candidate)

    _reserve_signature_batch(
        validations,
        disable_validation=runtime_config.disable_validation,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
    )

    # Split one attempt-level usage record across the candidates that will be
    # persisted so downstream outputs stay trajectory-centric.
    shared_usage = _build_shared_generation_usage(
        runtime_config=runtime_config,
        sampled_candidates=sampled_candidates,
        prompt=prompt,
        raw_response=raw_response,
        usage=usage,
        attempt_number=attempt_number,
    )
    split_generation_usages = _split_generation_usage_across_candidates(
        shared_usage,
        candidate_count=len(sampled_candidates),
    )

    trajectory_records: list[dict[str, Any]] = []
    for candidate_index, (
        sampled_candidate,
        validation,
        normalized_candidate,
        generation_usage,
    ) in enumerate(
        zip(
            sampled_candidates,
            validations,
            normalized_candidates,
            split_generation_usages,
        )
    ):
        trajectory_id = format_trajectory_id(
            _global_trajectory_index(
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
        trajectory_record["prompt"] = prompt
        trajectory_record["raw_output"] = sampled_candidate.raw_output
        trajectory_records.append(trajectory_record)
    return trajectory_records


def _build_shared_generation_usage(
    *,
    runtime_config: RuntimeConfig,
    sampled_candidates: list[SampledTrajectoryCandidate],
    prompt: str,
    raw_response: Any,
    usage: GenerationUsage | None,
    attempt_number: int,
) -> dict[str, Any]:
    """Builds one attempt-level usage payload before splitting across candidates."""

    usage_candidate: dict[str, Any]
    if len(sampled_candidates) == 1:
        usage_candidate = sampled_candidates[0].candidate
    elif isinstance(raw_response, dict):
        usage_candidate = raw_response
    else:
        usage_candidate = {
            "responses": [
                sampled_candidate.raw_output for sampled_candidate in sampled_candidates
            ]
        }
    return build_generation_usage(
        model=runtime_config.model,
        prompt=prompt,
        candidate=usage_candidate,
        usage=usage,
        attempt_number=attempt_number,
        default_traffic_type=_default_traffic_type_for_runtime(runtime_config),
    )


def _tool_call_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    return len(steps)


def _maybe_reserve_signature(
    validation: dict[str, Any],
    *,
    disable_validation: bool,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
) -> None:
    if disable_validation or seen_signatures is None or seen_signatures_lock is None:
        return

    signature = validation["signature"]
    # Reserve signatures before persisting outputs so concurrent workers do not
    # save the same normalized trajectory twice.
    # Enforce uniqueness only for validated trajectories we intend to keep.
    with seen_signatures_lock:
        if signature in seen_signatures:
            raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")
        seen_signatures.add(signature)


def extract_json_candidate(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ResponseFormatValidationError(
            f"Unsupported model response type: {type(raw_response).__name__}"
        )

    # Some SDK/model paths wrap the JSON object in fences or surrounding text.
    stripped = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if not json_match:
            raise ResponseFormatValidationError(
                "Model response did not contain JSON."
            ) from exc
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as inner_exc:
            raise ResponseFormatValidationError(
                "Model response contained invalid JSON."
            ) from inner_exc
