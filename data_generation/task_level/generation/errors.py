"""Serialize runtime, validation, and persisted payload errors."""

from __future__ import annotations

from typing import Any

from data_generation.task_level.generation.runtime_support import (
    _validation_error_base_type,
    _validation_error_type,
    format_trajectory_id,
)
from data_generation.task_level.tasks import TrajectoryValidationError


def _build_error_event(
    *,
    error_type: str,
    error_base_type: str | None,
    message: str | None,
    source: str,
    stage: str,
    trajectory_index: int | None = None,
    trajectory_id: str | None = None,
    attempt_number: int | None = None,
    retryable: bool | None = None,
    saved_in_output: bool = False,
) -> dict[str, Any]:
    """Serializes one observed generation error into a stable sidecar schema."""

    event: dict[str, Any] = {
        "error_type": error_type,
        "source": source,
        "stage": stage,
        "saved_in_output": saved_in_output,
    }
    if error_base_type and error_base_type != error_type:
        event["error_base_type"] = error_base_type
    if message:
        event["message"] = message
        event["summary"] = f"{error_type}: {message}"
    else:
        event["summary"] = error_type
    if trajectory_index is not None:
        event["trajectory_index"] = trajectory_index
    if trajectory_id is not None:
        event["trajectory_id"] = trajectory_id
    if attempt_number is not None:
        event["attempt_number"] = attempt_number
    if retryable is not None:
        event["retryable"] = retryable
    return event


def _validation_error_event_from_payload(
    validation: dict[str, Any],
    *,
    source: str,
    trajectory_index: int | None = None,
    attempt_number: int | None = None,
    retryable: bool | None = None,
) -> dict[str, Any] | None:
    """Builds one error event from an in-memory invalid validation payload."""

    if validation.get("is_valid") is not False:
        return None
    error_type = _validation_error_type(validation) or "TrajectoryValidationError"
    error_message = validation.get("error")
    return _build_error_event(
        error_type=error_type,
        error_base_type=_validation_error_base_type(validation),
        message=error_message if isinstance(error_message, str) else None,
        source=source,
        stage="validation",
        trajectory_index=trajectory_index,
        attempt_number=attempt_number,
        retryable=retryable,
    )


def _exception_error_event(
    exc: Exception,
    *,
    source: str,
    stage: str,
    trajectory_index: int | None = None,
    attempt_number: int | None = None,
    retryable: bool | None = None,
) -> dict[str, Any]:
    """Builds one error event from a raised exception."""

    trajectory_id = (
        format_trajectory_id(trajectory_index)
        if isinstance(trajectory_index, int)
        else None
    )
    return _build_error_event(
        error_type=(
            exc.error_type
            if isinstance(exc, TrajectoryValidationError)
            else type(exc).__name__
        ),
        error_base_type=(
            exc.error_base_type if isinstance(exc, TrajectoryValidationError) else None
        ),
        message=str(exc).strip() or None,
        source=source,
        stage=stage,
        trajectory_index=trajectory_index,
        trajectory_id=trajectory_id,
        attempt_number=attempt_number,
        retryable=retryable,
    )


def _validation_error_event(
    validation: dict[str, Any],
    *,
    source: str,
    trajectory_id: str,
    trajectory_index: int | None = None,
    attempt_number: int | None = None,
) -> dict[str, Any] | None:
    """Builds one error event from persisted invalid-trajectory validation data."""

    if validation.get("is_valid") is not False:
        return None
    error_type = _validation_error_type(validation) or "TrajectoryValidationError"
    error_message = validation.get("error")
    return _build_error_event(
        error_type=error_type,
        error_base_type=_validation_error_base_type(validation),
        message=error_message if isinstance(error_message, str) else None,
        source=source,
        stage="validation",
        trajectory_index=trajectory_index,
        trajectory_id=trajectory_id,
        attempt_number=attempt_number,
        retryable=False,
        saved_in_output=True,
    )


def _append_error_event(
    error_events: list[dict[str, Any]] | None,
    error_event: dict[str, Any] | None,
    *,
    error_events_lock: threading.Lock | None = None,
) -> None:
    """Appends one observed error event while preserving thread safety."""

    if error_events is None or error_event is None:
        return
    if error_events_lock is None:
        error_events.append(error_event)
        return
    with error_events_lock:
        error_events.append(error_event)


def _error_event_key(error_event: dict[str, Any]) -> tuple[Any, ...]:
    """Normalizes one error event for stable deduplication and sorting."""

    trajectory_identity = error_event.get("trajectory_id")
    if trajectory_identity in {None, ""}:
        trajectory_identity = error_event.get("trajectory_index", -1)
    return (
        trajectory_identity,
        error_event.get("attempt_number", -1),
        error_event.get("stage", ""),
        error_event.get("error_type", ""),
        error_event.get("message", ""),
        error_event.get("saved_in_output", False),
    )


def _collect_payload_error_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collects and deduplicates all observed errors represented in one payload."""

    deduped_events: dict[tuple[Any, ...], dict[str, Any]] = {}
    payload_error_events = payload.get("error_events", [])
    for error_event in payload_error_events:
        if isinstance(error_event, dict):
            deduped_events[_error_event_key(error_event)] = dict(error_event)

    # Older payloads may not include the explicit error event log.
    if not deduped_events:
        for trajectory in payload.get("trajectories", []):
            validation = trajectory.get("validation")
            if not isinstance(validation, dict):
                continue
            error_event = _validation_error_event(
                validation,
                source="saved_trajectory",
                trajectory_id=trajectory["trajectory_id"],
                attempt_number=trajectory.get("generation_usage", {}).get(
                    "successful_attempt_number",
                    1,
                ),
            )
            if error_event is not None:
                deduped_events[_error_event_key(error_event)] = error_event

    return [deduped_events[key] for key in sorted(deduped_events)]
