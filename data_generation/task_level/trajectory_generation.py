"""Coordinate end-to-end task-level trajectory generation and output writing."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any, Callable

from tqdm import tqdm

try:
    from rich.console import Console
    from rich.text import Text
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        ProgressColumn,
        Progress as RichProgress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:  # pragma: no cover
    Console = None
    BarColumn = None
    MofNCompleteColumn = None
    ProgressColumn = None
    RichProgress = None
    SpinnerColumn = None
    TaskProgressColumn = None
    TextColumn = None
    Text = None
    TimeElapsedColumn = None

from data_generation.task_level.runtime.client import (
    BATCH_TRAFFIC_TYPE,
    COST_DECIMAL_PLACES,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_SDK,
    GenerationResult,
    GenerationUsage,
    TrajectoryGenerationError,
    _generation_error_status_code,
    _resolve_pricing_tier,
    build_generation_usage,
    load_dotenv_file,
    reprice_generation_usage,
)
from data_generation.task_level.sampling import (
    SampledTrajectoryCandidate,
    get_sampling_strategy,
)
from data_generation.task_level.tasks import (
    DuplicateTrajectoryValidationError,
    ResponseFormatValidationError,
    TaskDefinition,
    TaskValidator,
    TrajectoryValidationError,
    get_task_definition,
    supported_task_names,
)
from data_generation.utils import (
    camel_to_snake_case,
    coerce_int,
    format_cost_usd,
    round_cost,
    stable_json_sha256,
    write_json_output,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_generation" / "task_level" / "data"
DEFAULT_COMPOSITE_TASK = supported_task_names()[0]
DATASET_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TRAJECTORY_DIRECTORY_NAME = "trajectories"
SUMMARY_OUTPUT_FILENAME = "summary.json"
COST_SUMMARY_OUTPUT_FILENAME = "cost_summary.json"
ERROR_SUMMARY_OUTPUT_FILENAME = "summary_errors.json"
TRAJECTORY_ID_DIGITS = 6
BATCH_DIRECTORY_NAME = "batch"
OVERALL_PROGRESS_COLOR = "cyan"
PROGRESS_BAR_WIDTH = 30
THINKING_LEVEL_CHOICES = ("minimal", "low", "medium", "high")
TQDM_BAR_FORMAT = f"{{l_bar}}{{bar:{PROGRESS_BAR_WIDTH}}}{{r_bar}}"
GENERATION_ERROR_EXIT_CODE = 1
BATCH_POLL_INTERVAL_SECONDS = 10
GOOGLE_CLOUD_BATCH_GCS_PREFIX_ENV_VAR = "GOOGLE_CLOUD_BATCH_GCS_PREFIX"
# Keep concurrent trajectory bars easy to tell apart in the terminal.
TRAJECTORY_PROGRESS_COLORS = ("green", "yellow", "blue", "magenta", "red", "cyan")
INTERRUPTED_EXIT_CODE = 130
INTERRUPTED_MESSAGE = (
    "Interrupted. Exiting immediately. Queued trajectories were cancelled; "
    "requests already in flight may still be billed."
)
BATCH_INTERRUPTED_MESSAGE = (
    "Interrupted. Active remote batch jobs were cancelled. "
    "No local outputs were written."
)


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

    return format_trajectory_id(trajectory_index).replace("traj", "run").replace("_", " ")


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


@dataclass(frozen=True)
class RuntimeConfig:
    composite_task: str
    num_runs: int
    model: str
    sdk: str
    project: str | None
    location: str
    temperature: float
    max_workers: int
    max_retries: int
    sampling: str = "base"
    verbalized_k: int = 1
    thinking_level: str | None = None
    summary_path: Path | None = None
    cost_output_path: Path | None = None
    disable_validation: bool = False
    batch_processing: bool = False
    batch_gcs_prefix: str | None = None


@dataclass(frozen=True)
class ProgressHandles:
    display: Any | None
    overall_progress: Any
    trajectory_progress_bars: list[Any]
    log_writer: Callable[[str], None] | None


@dataclass(frozen=True)
class OutputPaths:
    summary_path: Path
    trajectory_dir: Path
    prompt_dir: Path
    output_dir: Path
    cost_path: Path
    error_summary_path: Path


class AccumulatedCostTracker:
    """Tracks accumulated and projected cost text for CLI progress updates."""

    def __init__(
        self,
        *,
        total_trajectories: int,
    ) -> None:
        self._total_cost_usd = 0.0
        self._cost_available = True
        self._completed_trajectories = 0
        self._total_trajectories = total_trajectories
        self._lock = threading.Lock()

    def add_observed_cost(self, observed_cost_usd: float | None) -> str:
        """Adds one observed attempt cost and returns the new progress suffix."""

        with self._lock:
            if self._cost_available:
                if observed_cost_usd is None:
                    self._cost_available = False
                else:
                    self._total_cost_usd += observed_cost_usd
            return self._status_text_locked()

    def complete_trajectories(self, amount: int = 1) -> str:
        """Marks completed trajectories so projected totals use only remaining work."""

        with self._lock:
            self._completed_trajectories = min(
                self._completed_trajectories + amount,
                self._total_trajectories,
            )
            return self._status_text_locked()

    def status_text(self) -> str:
        """Formats the current accumulated cost for progress displays."""

        with self._lock:
            return self._status_text_locked()

    def _status_text_locked(self) -> str:
        unavailable_text = format_cost_usd(
            None,
            decimal_places=COST_DECIMAL_PLACES,
            unavailable="n/a",
        )
        if not self._cost_available:
            return f"accumulated={unavailable_text} projected={unavailable_text}"
        rounded_total_cost = round_cost(
            self._total_cost_usd,
            decimal_places=COST_DECIMAL_PLACES,
        )
        status_parts = [
            "accumulated="
            f"{format_cost_usd(rounded_total_cost, decimal_places=COST_DECIMAL_PLACES)}"
        ]
        projected_cost_text = self._projected_cost_text_locked()
        if projected_cost_text is not None:
            status_parts.append(projected_cost_text)
        return " ".join(status_parts)

    def _projected_cost_text_locked(self) -> str | None:
        if self._completed_trajectories <= 0:
            return "projected=NaN"
        projected_total = self._projected_total_cost_locked()
        if projected_total is None:
            return None
        return (
            "projected="
            f"{format_cost_usd(projected_total, decimal_places=COST_DECIMAL_PLACES)}"
        )

    def _projected_total_cost_locked(self) -> float | None:
        average_trajectory_cost_usd = self._average_trajectory_cost_locked()
        if average_trajectory_cost_usd is None:
            return None
        remaining_trajectories = max(
            self._total_trajectories - self._completed_trajectories,
            0,
        )
        return round_cost(
            self._total_cost_usd + (remaining_trajectories * average_trajectory_cost_usd),
            decimal_places=COST_DECIMAL_PLACES,
        )

    def _average_trajectory_cost_locked(self) -> float | None:
        if self._completed_trajectories > 0:
            return self._total_cost_usd / self._completed_trajectories
        return None


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


def _progress_run_count(runtime_config: Any) -> int:
    """Returns the progress-bar item count for runtime-like configs."""

    run_count = getattr(runtime_config, "num_runs", None)
    if isinstance(run_count, int):
        return run_count
    trajectory_count = getattr(runtime_config, "num_trajectories", None)
    if isinstance(trajectory_count, int):
        return trajectory_count
    raise AttributeError("runtime config must define num_runs or num_trajectories")


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


def _unwrap_generation_response(
    raw_response: Any,
) -> tuple[Any, GenerationUsage | None]:
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


def _split_integer_total(total: int, parts: int) -> list[int]:
    """Splits one integer total across parts while preserving the sum."""

    base_value, remainder = divmod(total, parts)
    return [
        base_value + (1 if index < remainder else 0)
        for index in range(parts)
    ]


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
        split_value = round_cost(
            total / parts,
            decimal_places=COST_DECIMAL_PLACES,
        ) or 0.0
        split_values.append(split_value)
        remaining -= split_value
    return split_values


def _split_generation_usage_across_candidates(
    generation_usage: dict[str, Any],
    *,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Splits one run's usage metadata across the saved candidates from that run."""

    if candidate_count == 1:
        return [dict(generation_usage)]

    prompt_splits = _split_integer_total(generation_usage["prompt_tokens"], candidate_count)
    output_splits = _split_integer_total(generation_usage["output_tokens"], candidate_count)
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
            "responses": [sampled_candidate.raw_output for sampled_candidate in sampled_candidates]
        }
    return build_generation_usage(
        model=runtime_config.model,
        prompt=prompt,
        candidate=usage_candidate,
        usage=usage,
        attempt_number=attempt_number,
        default_traffic_type=_default_traffic_type_for_runtime(runtime_config),
    )


def _successful_attempt_count(generation_usage: dict[str, Any]) -> int:
    successful_attempt_number = generation_usage.get("successful_attempt_number", 1)
    if not isinstance(successful_attempt_number, int) or successful_attempt_number < 1:
        return 1
    return successful_attempt_number


def _projected_attempt_count(runtime_config: RuntimeConfig) -> int:
    """Choose the attempt multiplier used for the single preflight projection."""

    if runtime_config.disable_validation:
        return 1
    return max(runtime_config.max_retries, 1)


def _attempt_counts_for_saved_trajectories(
    generation_usages: list[dict[str, Any]],
) -> list[int]:
    """Read the persisted successful attempt number for each saved trajectory."""

    return [
        _successful_attempt_count(generation_usage)
        for generation_usage in generation_usages
    ]


def _reasoning_token_count(generation_usage: dict[str, Any]) -> int:
    """Read reasoning tokens from persisted usage, defaulting old payloads to zero."""
    return coerce_int(generation_usage.get("reasoning_tokens")) or 0


def _billable_output_token_count(generation_usage: dict[str, Any]) -> int:
    """Reasoning tokens share the standard output-token billing tier."""
    return generation_usage["output_tokens"] + _reasoning_token_count(generation_usage)


def _scaled_token_totals(
    generation_usages: list[dict[str, Any]],
    attempt_counts: list[int],
) -> dict[str, int]:
    return {
        "prompt": sum(
            generation_usage["prompt_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(generation_usages, attempt_counts)
        ),
        "output": sum(
            generation_usage["output_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(generation_usages, attempt_counts)
        ),
        "reasoning": sum(
            _reasoning_token_count(generation_usage) * attempt_count
            for generation_usage, attempt_count in zip(generation_usages, attempt_counts)
        ),
        "total": sum(
            generation_usage["total_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(generation_usages, attempt_counts)
        ),
    }


def _scaled_total_cost(
    generation_usages: list[dict[str, Any]],
    attempt_counts: list[int],
) -> float | None:
    scaled_costs: list[float] = []
    for generation_usage, attempt_count in zip(generation_usages, attempt_counts):
        observed_cost_usd = generation_usage.get("observed_cost_usd")
        if observed_cost_usd is None:
            return None
        scaled_costs.append(observed_cost_usd * attempt_count)
    return sum(scaled_costs)


def _observed_cost_total(generation_usages: list[dict[str, Any]]) -> float | None:
    """Sums observed costs only when every usage payload includes one."""

    observed_costs = [
        generation_usage.get("observed_cost_usd")
        for generation_usage in generation_usages
    ]
    if any(observed_cost is None for observed_cost in observed_costs):
        return None
    return sum(observed_costs)


def _shared_pricing(
    generation_usages: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    if not generation_usages:
        return None

    resolved_pricings: list[dict[str, Any]] = []
    for generation_usage in generation_usages:
        pricing = generation_usage.get("pricing")
        if not isinstance(pricing, dict):
            if model is None:
                return None
            pricing_tier = _resolve_pricing_tier(
                model,
                generation_usage.get("traffic_type"),
            )
            if pricing_tier is None:
                return None
            pricing = {
                "model": pricing_tier.model,
                "input_usd_per_million_tokens": pricing_tier.input_usd_per_million_tokens,
                "output_usd_per_million_tokens": pricing_tier.output_usd_per_million_tokens,
            }
        resolved_pricings.append(pricing)

    first_pricing = resolved_pricings[0]
    if any(pricing != first_pricing for pricing in resolved_pricings[1:]):
        return None

    return dict(first_pricing)


def _best_case_cost_estimate_note(attempt_counts: list[int]) -> str:
    if any(attempt_count > 1 for attempt_count in attempt_counts):
        return (
            "Best case uses the observed successful attempt number for each "
            "saved trajectory and assumes earlier failed attempts had the "
            "same token profile as the successful attempt."
        )
    return "Best case assumes each trajectory succeeds on the first attempt."


def _append_sampling_cost_note(
    summary: dict[str, Any],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    """Adds sampling-specific cost notes when totals are shared across candidates."""

    if runtime_config.sampling != "verbalized":
        return summary
    summary["notes"].append(
        "For verbalized sampling, token and cost totals are counted once per model response and apportioned across the saved trajectories from that response."
    )
    return summary


def _build_cost_estimate_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    # Post-run estimates should honor how many attempts each saved trajectory took.
    best_case_attempt_counts = _attempt_counts_for_saved_trajectories(
        generation_usages
    )
    worst_case_attempt_counts = [
        runtime_config.max_retries for _ in generation_usages
    ]

    best_case_tokens = _scaled_token_totals(
        generation_usages,
        best_case_attempt_counts,
    )
    worst_case_tokens = _scaled_token_totals(
        generation_usages,
        worst_case_attempt_counts,
    )
    best_case_total_cost = _scaled_total_cost(
        generation_usages,
        best_case_attempt_counts,
    )
    worst_case_total_cost = _scaled_total_cost(
        generation_usages,
        worst_case_attempt_counts,
    )
    shared_pricing = _shared_pricing(
        generation_usages,
        model=runtime_config.model,
    )

    summary = {
        "best_case_total_usd": round_cost(
            best_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "best_case_per_trajectory_usd": round_cost(
            best_case_total_cost / max(len(generation_usages), 1)
            if best_case_total_cost is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_total_usd": round_cost(
            worst_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_per_trajectory_usd": round_cost(
            worst_case_total_cost / max(len(generation_usages), 1)
            if worst_case_total_cost is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "best_case_tokens": best_case_tokens,
        "worst_case_tokens": worst_case_tokens,
        "max_retries_assumed_for_worst_case": runtime_config.max_retries,
        "notes": [
            _best_case_cost_estimate_note(best_case_attempt_counts),
            "Worst case assumes every trajectory consumes the full retry budget and each attempt has the same token profile as the successful attempt.",
            "Token counts use Vertex usage metadata when available and otherwise fall back to a local character-based estimate.",
        ],
    }
    if shared_pricing is not None:
        summary["pricing"] = shared_pricing
    return _append_sampling_cost_note(summary, runtime_config)


def _build_cost_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
) -> dict[str, Any]:
    # Scale saved trajectories by their successful attempt number so totals include
    # the retry work required to produce each persisted output.
    trajectory_count = len(generation_usages)
    attempt_counts = _attempt_counts_for_saved_trajectories(generation_usages)
    scaled_tokens = _scaled_token_totals(
        generation_usages,
        attempt_counts,
    )
    total_prompt_tokens = scaled_tokens["prompt"]
    total_output_tokens = scaled_tokens["output"]
    total_reasoning_tokens = scaled_tokens["reasoning"]
    total_tokens = scaled_tokens["total"]
    shared_pricing = _shared_pricing(generation_usages)
    pricing_supported = all("pricing" in usage for usage in generation_usages)

    input_cost = None
    output_cost = None
    total_cost = None
    if pricing_supported:
        # Keep the cost fields aligned with the retry-inclusive token totals.
        input_cost = sum(
            (usage["prompt_tokens"] * attempt_count / 1_000_000)
            * usage["pricing"]["input_usd_per_million_tokens"]
            for usage, attempt_count in zip(generation_usages, attempt_counts)
        )
        output_cost = sum(
            (_billable_output_token_count(usage) * attempt_count / 1_000_000)
            * usage["pricing"]["output_usd_per_million_tokens"]
            for usage, attempt_count in zip(generation_usages, attempt_counts)
        )
        total_cost = input_cost + output_cost

    summary = {
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "reasoning_tokens": total_reasoning_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round_cost(
            input_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "output_cost_usd": round_cost(
            output_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "total_cost_usd": round_cost(
            total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "average_trajectory_cost_usd": round_cost(
            total_cost / trajectory_count
            if total_cost is not None and trajectory_count > 0
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "notes": [
            "Cost summary scales each saved trajectory by its successful attempt number.",
            "Retry-inclusive totals assume earlier failed attempts used the same token profile as the successful attempt.",
        ],
    }
    if shared_pricing is not None:
        summary["pricing"] = shared_pricing
    return summary


def _build_preflight_cost_estimate_summary(
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> dict[str, Any]:
    manual_estimate = _sampling_strategy_for_runtime(
        runtime_config
    ).preflight_token_estimate(
        task_definition=task_definition,
        runtime_config=runtime_config,
    )
    generation_usage = reprice_generation_usage(
        {
            "successful_attempt_number": _projected_attempt_count(runtime_config),
            "prompt_tokens": manual_estimate.prompt_tokens,
            "output_tokens": manual_estimate.output_tokens,
            "reasoning_tokens": manual_estimate.reasoning_tokens,
            "total_tokens": (
                manual_estimate.prompt_tokens
                + manual_estimate.output_tokens
                + manual_estimate.reasoning_tokens
            ),
            "usage_source": "manual_task_estimate",
            "traffic_type": _default_traffic_type_for_runtime(runtime_config),
            "observed_cost_usd": 0.0,
        },
        model=runtime_config.model,
        traffic_type=_default_traffic_type_for_runtime(runtime_config),
        round_observed_cost=False,
    )
    summary = _build_cost_estimate_summary_from_generation_usages(
        [generation_usage] * runtime_config.num_runs,
        runtime_config=runtime_config,
    )
    if _projected_attempt_count(runtime_config) > 1:
        summary["notes"][0] = (
            "Projected total uses the manual task token estimate maintained in the "
            "task definition and scales it by the configured retry budget."
        )
    else:
        summary["notes"][0] = (
            "Best case uses the manual task token estimate maintained in the "
            "task definition."
        )
    summary["notes"][2] = (
        "Token counts come from the manual task token estimate rather than "
        "observed API usage metadata."
    )
    return _append_sampling_cost_note(summary, runtime_config)


def resolve_cost_output_path(
    summary_path: Path,
    cost_output_path: Path | None = None,
) -> Path:
    if cost_output_path is not None:
        return cost_output_path
    return summary_path.with_name(COST_SUMMARY_OUTPUT_FILENAME)


def resolve_error_output_path(summary_path: Path) -> Path:
    """Resolves the default error-summary sidecar path for one run."""

    return summary_path.with_name(ERROR_SUMMARY_OUTPUT_FILENAME)


def resolve_dataset_output_path(
    composite_task: str,
    *,
    generated_at: datetime | None = None,
) -> Path:
    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    task_dir = camel_to_snake_case(composite_task)
    return (
        DEFAULT_OUTPUT_DIR
        / task_dir
        / timestamp.strftime(DATASET_RUN_TIMESTAMP_FORMAT)
        / SUMMARY_OUTPUT_FILENAME
    )


def _resolve_summary_path(runtime_config: RuntimeConfig) -> Path:
    """Returns one stable summary path for the current generation run."""

    if runtime_config.summary_path is not None:
        return runtime_config.summary_path
    return resolve_dataset_output_path(runtime_config.composite_task)


def resolve_trajectory_output_dir(output_path: Path) -> Path:
    return output_path.parent / TRAJECTORY_DIRECTORY_NAME


def resolve_prompt_output_dir(output_path: Path) -> Path:
    """Resolves the sibling prompt output directory for one dataset summary."""

    return output_path.parent / "prompts"


def resolve_raw_output_dir(output_path: Path) -> Path:
    """Resolves the sibling raw-output directory for one dataset summary."""

    return output_path.parent / "outputs"


def _trajectory_output_filename(trajectory_id: str) -> str:
    return f"{trajectory_id}.json"


def _prompt_output_filename(trajectory_id: str) -> str:
    """Formats one prompt sidecar filename to match its trajectory basename."""

    return f"{trajectory_id}.md"


def _raw_output_filename(trajectory_id: str) -> str:
    """Formats one raw-output sidecar filename using the trajectory basename."""

    return f"{trajectory_id}.txt"


def _payload_run_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    run_metadata = {
        "composite_task": payload["composite_task"],
        "sdk": payload["sdk"],
        "model": payload["model"],
        "num_runs": payload.get("num_runs", payload["num_trajectories"]),
        "num_trajectories": payload["num_trajectories"],
        "generated_at": payload["generated_at"],
    }
    if "model_config" in payload:
        run_metadata["model_config"] = payload["model_config"]
    return run_metadata


def _build_model_config_payload(runtime_config: RuntimeConfig) -> dict[str, Any]:
    """Serializes request-time model settings for dataset metadata."""

    sampling_payload = {
        "temperature": runtime_config.temperature,
        "strategy": runtime_config.sampling,
    }
    if runtime_config.sampling == "verbalized":
        sampling_payload["verbalized_k"] = runtime_config.verbalized_k

    return {
        "reasoning": {
            "thinking_level": runtime_config.thinking_level,
        },
        "sampling": sampling_payload,
    }


def _summary_trajectory_entry(trajectory: dict[str, Any]) -> dict[str, str]:
    trajectory_id = trajectory["trajectory_id"]
    return {
        "trajectory_id": trajectory_id,
        "path": (
            Path(TRAJECTORY_DIRECTORY_NAME) / _trajectory_output_filename(trajectory_id)
        ).as_posix(),
    }


def _summary_trajectory_stats(
    trajectories: list[dict[str, Any]],
) -> dict[str, int | float]:
    """Builds aggregate completion and validation stats for summary metadata."""

    completed_trajectories = len(trajectories)
    invalid_trajectories = 0
    successful_trajectories = 0

    for trajectory in trajectories:
        validation = trajectory.get("validation")

        # Older payloads used in tests may not include validation metadata.
        if not isinstance(validation, dict):
            successful_trajectories += 1
            continue

        if validation.get("is_valid") is False:
            invalid_trajectories += 1
            continue

        successful_trajectories += 1

    successful_trajectory_fraction = 0.0
    if completed_trajectories > 0:
        successful_trajectory_fraction = (
            successful_trajectories / completed_trajectories
        )

    return {
        "completed_trajectories": completed_trajectories,
        "invalid_trajectories": invalid_trajectories,
        "successful_trajectory_fraction": successful_trajectory_fraction,
    }


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

    return [
        deduped_events[key]
        for key in sorted(deduped_events)
    ]


def _trajectory_cost_entry(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "generation_usage": trajectory["generation_usage"],
    }


def build_summary_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = _payload_run_metadata(payload)
    if "cost_summary" in payload:
        summary_payload["cost_summary"] = payload["cost_summary"]
    summary_payload.update(_summary_trajectory_stats(payload["trajectories"]))
    summary_payload["trajectory_directory"] = TRAJECTORY_DIRECTORY_NAME
    summary_payload["trajectory_files"] = [
        _summary_trajectory_entry(trajectory)
        for trajectory in payload["trajectories"]
    ]
    return summary_payload


def write_trajectory_output_payloads(
    trajectories: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    written_paths: list[Path] = []
    for trajectory in trajectories:
        output_path = output_dir / _trajectory_output_filename(
            trajectory["trajectory_id"]
        )
        write_json_output(
            {
                key: value
                for key, value in trajectory.items()
                if key not in {"prompt", "raw_output"}
            },
            output_path,
        )
        written_paths.append(output_path)
    return written_paths


def write_prompt_output_payloads(
    trajectory_prompts: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    """Writes one prompt sidecar per trajectory using matching trajectory IDs."""

    written_paths: list[Path] = []
    for prompt_entry in trajectory_prompts:
        output_path = output_dir / _prompt_output_filename(
            prompt_entry["trajectory_id"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(prompt_entry["prompt"], encoding="utf-8")
        written_paths.append(output_path)
    return written_paths


def _raw_output_text(raw_output: Any) -> str:
    """Serializes the stored raw model output into a text sidecar."""

    if isinstance(raw_output, str):
        return raw_output
    return json.dumps(raw_output, indent=2, sort_keys=True)


def write_raw_output_payloads(
    trajectory_outputs: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Writes one raw model output sidecar per trajectory."""

    written_paths: list[Path] = []
    for output_entry in trajectory_outputs:
        output_path = output_dir / _raw_output_filename(output_entry["trajectory_id"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _raw_output_text(output_entry["raw_output"]),
            encoding="utf-8",
        )
        written_paths.append(output_path)
    return written_paths


def build_cost_output_payload(
    payload: dict[str, Any],
    *,
    trajectory_output_path: Path | None = None,
) -> dict[str, Any]:
    # Keep the sidecar small enough to inspect without loading full trajectories.
    cost_payload = {
        **_payload_run_metadata(payload),
        "trajectory_output_path": (
            str(trajectory_output_path) if trajectory_output_path is not None else None
        ),
        "trajectory_directory": (
            str(resolve_trajectory_output_dir(trajectory_output_path))
            if trajectory_output_path is not None
            else None
        ),
        "trajectory_costs": [
            _trajectory_cost_entry(trajectory)
            for trajectory in payload["trajectories"]
        ],
    }
    if "cost_summary" in payload:
        cost_payload["cost_summary"] = payload["cost_summary"]
    return cost_payload


def build_error_summary_output_payload(
    payload: dict[str, Any],
    *,
    trajectory_output_path: Path | None = None,
) -> dict[str, Any]:
    """Builds the sidecar payload that aggregates all observed run errors."""

    error_events = _collect_payload_error_events(payload)
    error_counts_by_type: dict[str, int] = {}
    distinct_error_counts: dict[tuple[str, str], int] = {}

    for error_event in error_events:
        error_type = error_event["error_type"]
        error_counts_by_type[error_type] = error_counts_by_type.get(error_type, 0) + 1
        error_message = error_event.get("message", "")
        distinct_key = (error_type, error_message)
        distinct_error_counts[distinct_key] = (
            distinct_error_counts.get(distinct_key, 0) + 1
        )

    error_payload = {
        **_payload_run_metadata(payload),
        "trajectory_output_path": (
            str(trajectory_output_path) if trajectory_output_path is not None else None
        ),
        "trajectory_directory": (
            str(resolve_trajectory_output_dir(trajectory_output_path))
            if trajectory_output_path is not None
            else None
        ),
        "total_errors": len(error_events),
        "possible_errors": sorted(error_counts_by_type),
        "error_counts_by_type": [
            {
                "error_type": error_type,
                "count": error_counts_by_type[error_type],
            }
            for error_type in sorted(error_counts_by_type)
        ],
        "distinct_errors": [
            {
                "error_type": error_type,
                "message": error_message,
                "summary": (
                    f"{error_type}: {error_message}"
                    if error_message
                    else error_type
                ),
                "count": distinct_error_counts[(error_type, error_message)],
            }
            for error_type, error_message in sorted(distinct_error_counts)
        ],
        "error_events": error_events,
    }
    return error_payload


def _log_runtime_message(
    message: str,
    *,
    enabled: bool,
    writer: Callable[[str], None] | None = None,
) -> None:
    if not enabled:
        return
    if writer is not None:
        writer(message)
        return
    tqdm.write(message)


def _progress_status_with_accumulated_cost(
    status: str,
    *,
    accumulated_cost_text: str | None,
) -> str:
    """Appends accumulated cost text to a progress status when available."""

    if not accumulated_cost_text:
        return status
    if not status:
        return accumulated_cost_text
    return f"{status} {accumulated_cost_text}"


def _update_overall_progress_status(
    overall_progress: Any | None,
    *,
    amount: int = 0,
    status: str = "",
    accumulated_cost_text: str | None = None,
) -> None:
    """Updates the shared overall progress bar and refreshes its status text."""

    if overall_progress is None:
        return
    if amount:
        overall_progress.update(amount)
    status_text = _progress_status_with_accumulated_cost(
        status,
        accumulated_cost_text=accumulated_cost_text,
    )
    if not status_text:
        return
    overall_progress.set_postfix_str(status_text)
    overall_progress.refresh()


def _trajectory_progress_color(index: int) -> str:
    return TRAJECTORY_PROGRESS_COLORS[index % len(TRAJECTORY_PROGRESS_COLORS)]


class StaticQueuedTimeElapsedColumn(ProgressColumn):
    """Shows a fixed zero timer for queued tasks and real elapsed time once started."""

    def __init__(self) -> None:
        """Initializes the shared queued-aware elapsed column."""

        if TimeElapsedColumn is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._delegate = TimeElapsedColumn()

    def render(self, task: Any) -> Any:
        """Renders zero elapsed time until the task leaves the queued state."""

        if getattr(task, "fields", {}).get("queued", False):
            return Text("0:00:00")
        return self._delegate.render(task)


class RichTaskProgressAdapter:
    """Adapts one Rich progress task to the shared progress-bar interface."""

    def __init__(
        self,
        progress: RichProgress,
        task_id: int,
        total: int,
        *,
        started: bool = True,
        status: str = "",
    ):
        """Caches task state so Rich resets preserve custom fields like status."""

        self._progress = progress
        self._task_id = task_id
        self._total = total
        self._completed = 0
        self._started = started
        self._status = status

    def start(self) -> None:
        """Starts elapsed-time tracking only when generation actually begins."""

        if self._started:
            return
        self._progress.reset(
            self._task_id,
            start=True,
            total=self._total,
            completed=0,
            queued=False,
            status=self._status,
        )
        self._started = True

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value
        self._completed = min(self._completed, value)
        self._progress.update(
            self._task_id,
            total=value,
            completed=self._completed,
        )

    def update(self, amount: int = 1) -> None:
        self.start()
        self._completed += amount
        self._progress.advance(self._task_id, amount)

    def set_postfix_str(self, text: str) -> None:
        self._status = text
        self._progress.update(self._task_id, status=text)

    def refresh(self) -> None:
        self._progress.refresh()

    def complete(self, total: int) -> None:
        """Marks the task complete at the requested total."""

        self.start()
        self._total = total
        self._progress.update(
            self._task_id,
            total=total,
            completed=total,
        )

    def close(self) -> None:
        return


class TqdmTaskProgressAdapter:
    """Adapts one tqdm progress bar to the shared progress-bar interface."""

    def __init__(self, progress_bar: tqdm, total: int):
        self._progress_bar = progress_bar
        self._total = total
        self._started = False

    def start(self) -> None:
        """Resets elapsed-time bookkeeping when the first real attempt starts."""

        if self._started:
            return
        current_time = time()
        self._progress_bar.start_t = current_time
        self._progress_bar.last_print_t = current_time
        self._started = True

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value
        self._progress_bar.total = value
        if self._progress_bar.n > value:
            self._progress_bar.n = value
        self._progress_bar.refresh()

    def update(self, amount: int = 1) -> None:
        self.start()
        self._progress_bar.update(amount)

    def set_postfix_str(self, text: str) -> None:
        self._progress_bar.set_postfix_str(text)

    def refresh(self) -> None:
        self._progress_bar.refresh()

    def complete(self, total: int) -> None:
        """Marks the progress bar complete at the requested total."""

        self.start()
        self._total = total
        self._progress_bar.total = total
        self._progress_bar.n = total
        self._progress_bar.refresh()

    def close(self) -> None:
        self._progress_bar.close()


class RichProgressDisplay:
    """Owns the interactive Rich progress layout used by the CLI."""

    def __init__(self, runtime_config: RuntimeConfig):
        if RichProgress is None or Console is None:
            raise RuntimeError("rich progress support is unavailable")

        self.console = Console(stderr=True)
        self._progress = RichProgress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            StaticQueuedTimeElapsedColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
            expand=False,
        )
        self._progress.start()
        progress_count = _progress_run_count(runtime_config)
        overall_task_id = self._progress.add_task(
            "[cyan]runs[/cyan]",
            total=progress_count,
            status="running",
        )
        self.overall_progress = RichTaskProgressAdapter(
            self._progress,
            overall_task_id,
            progress_count,
            status="running",
        )
        self.trajectory_progress_bars = [
            RichTaskProgressAdapter(
                self._progress,
                self._progress.add_task(
                    f"[{_trajectory_progress_color(index)}]{format_trajectory_progress_label(index)}[/{_trajectory_progress_color(index)}]",
                    total=_trajectories_per_run(runtime_config),
                    status="queued",
                    start=True,
                    queued=True,
                ),
                _trajectories_per_run(runtime_config),
                started=False,
                status="queued",
            )
            for index in range(progress_count)
        ]

    def close(self) -> None:
        self._progress.stop()


def _create_progress_handles(
    runtime_config: RuntimeConfig,
    *,
    disable_progress: bool,
) -> ProgressHandles:
    # Prefer rich in interactive terminals, but keep tqdm as a zero-dependency fallback.
    if not disable_progress and RichProgress is not None:
        progress_display = RichProgressDisplay(runtime_config)
        return ProgressHandles(
            display=progress_display,
            overall_progress=progress_display.overall_progress,
            trajectory_progress_bars=progress_display.trajectory_progress_bars,
            log_writer=progress_display.console.print,
        )

    progress_count = _progress_run_count(runtime_config)
    overall_progress = tqdm(
        total=progress_count,
        desc="Runs",
        bar_format=TQDM_BAR_FORMAT,
        position=0,
        disable=disable_progress,
        dynamic_ncols=True,
        colour=OVERALL_PROGRESS_COLOR,
    )
    trajectory_progress_bars = [
        TqdmTaskProgressAdapter(
            tqdm(
                total=_trajectories_per_run(runtime_config),
                desc=format_trajectory_progress_label(index),
                bar_format=TQDM_BAR_FORMAT,
                position=index + 1,
                leave=True,
                disable=disable_progress,
                dynamic_ncols=True,
                colour=_trajectory_progress_color(index),
            ),
            total=_trajectories_per_run(runtime_config),
        )
        for index in range(progress_count)
    ]
    for progress_bar in trajectory_progress_bars:
        progress_bar.set_postfix_str("queued")
    return ProgressHandles(
        display=None,
        overall_progress=overall_progress,
        trajectory_progress_bars=trajectory_progress_bars,
        log_writer=None,
    )


def _close_progress_handles(progress_handles: ProgressHandles) -> None:
    if progress_handles.display is not None:
        progress_handles.display.close()
        return
    progress_handles.overall_progress.close()
    for progress_bar in progress_handles.trajectory_progress_bars:
        progress_bar.close()


def _log_cost_summary(
    *,
    label: str,
    cost_estimate: dict[str, Any],
    runtime_config: RuntimeConfig,
    enabled: bool,
    writer: Callable[[str], None] | None,
) -> None:
    message = _cost_summary_message(
        label=label,
        cost_estimate=cost_estimate,
        runtime_config=runtime_config,
    )
    if message is None:
        return
    _log_runtime_message(
        message,
        enabled=enabled,
        writer=writer,
    )


def _cost_summary_message(
    *,
    label: str,
    cost_estimate: dict[str, Any],
    runtime_config: RuntimeConfig,
) -> str | None:
    best_case_cost = cost_estimate["best_case_total_usd"]
    if best_case_cost is None:
        return None
    return (
        f"{label}: "
        f"{format_cost_usd(best_case_cost, decimal_places=COST_DECIMAL_PLACES)}"
    )


def _tool_call_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    return len(steps)


def _update_completed_trajectory_progress(
    trajectory_progress: Any | None,
    *,
    attempt_number: int,
    max_retries: int,
    trajectory_count: int,
    successful_trajectory_count: int | None,
    is_valid: bool,
    total_observed_cost_usd: float | None,
    average_observed_cost_usd: float | None,
    tool_call_count: int | None = None,
    validation: dict[str, Any] | None = None,
) -> None:
    if trajectory_progress is None:
        return
    trajectory_progress.complete(trajectory_count)
    cost_text = f"done attempts={attempt_number}/{max_retries}"
    if total_observed_cost_usd is not None:
        cost_text = (
            f"{cost_text} total="
            f"{format_cost_usd(total_observed_cost_usd, decimal_places=COST_DECIMAL_PLACES)}"
        )
    if average_observed_cost_usd is not None:
        cost_text = (
            f"{cost_text} avg="
            f"{format_cost_usd(average_observed_cost_usd, decimal_places=COST_DECIMAL_PLACES)}"
        )
    if (
        successful_trajectory_count is not None
        and trajectory_count > 1
    ):
        cost_text = (
            f"{cost_text} success={successful_trajectory_count}/{trajectory_count}"
        )
    if tool_call_count is not None:
        if trajectory_count > 1:
            average_tool_calls = tool_call_count / trajectory_count
            average_tool_calls_text = (
                str(int(average_tool_calls))
                if average_tool_calls.is_integer()
                else f"{average_tool_calls:.1f}"
            )
            cost_text = f"{cost_text} avg_calls={average_tool_calls_text}"
        else:
            cost_text = f"{cost_text} calls={tool_call_count}"
    if not is_valid:
        cost_text = f"{cost_text} invalid"
        if validation is not None:
            validation_summary = _validation_error_progress_summary(validation)
            if validation_summary is not None:
                cost_text = f"{cost_text} {validation_summary}"
    trajectory_progress.set_postfix_str(cost_text)


def _trajectory_generation_status(
    runtime_config: RuntimeConfig,
    *,
    attempt_number: int,
) -> str:
    if runtime_config.disable_validation:
        return "generating"
    return f"attempt {attempt_number}/{runtime_config.max_retries} generating"


def _trajectory_retry_status(
    runtime_config: RuntimeConfig,
    *,
    attempt_number: int,
    tool_call_count: int | None = None,
) -> str:
    if runtime_config.disable_validation:
        retry_text = "retrying"
    else:
        retry_text = f"attempt {attempt_number}/{runtime_config.max_retries} retry"
    if tool_call_count is None:
        return retry_text
    return f"{retry_text} calls={tool_call_count}"


def _trajectory_completion_log_message(
    runtime_config: RuntimeConfig,
    *,
    trajectory_id: str,
    generation_usage: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    if runtime_config.disable_validation and not validation["is_valid"]:
        validation_summary = _validation_error_summary(validation)
        if validation_summary is not None:
            return f"Generated {trajectory_id} invalid {validation_summary}"
        return f"Generated {trajectory_id} invalid"
    if runtime_config.disable_validation:
        return f"Generated {trajectory_id}"
    return (
        f"Generated {trajectory_id} "
        f"(attempt {generation_usage['successful_attempt_number']}/"
        f"{runtime_config.max_retries})"
    )


def _sanitize_generation_usage_for_output(
    generation_usage: dict[str, Any],
    *,
    disable_validation: bool,
) -> dict[str, Any]:
    if not disable_validation:
        return generation_usage
    return {
        key: value
        for key, value in generation_usage.items()
        if key != "successful_attempt_number"
    }


def _sanitize_trajectory_for_output(
    trajectory: dict[str, Any],
    *,
    disable_validation: bool,
) -> dict[str, Any]:
    sanitized_trajectory = {
        key: value
        for key, value in trajectory.items()
        if key not in {"prompt", "raw_output"}
    }
    if not disable_validation:
        return sanitized_trajectory
    return {
        **sanitized_trajectory,
        "generation_usage": _sanitize_generation_usage_for_output(
            trajectory["generation_usage"],
            disable_validation=disable_validation,
        ),
    }


def _validate_runtime_config(runtime_config: RuntimeConfig) -> None:
    if runtime_config.num_runs <= 0:
        raise TrajectoryGenerationError("--num-runs must be greater than 0.")
    if runtime_config.max_workers <= 0:
        raise TrajectoryGenerationError("--max-workers must be greater than 0.")
    if runtime_config.max_retries <= 0:
        raise TrajectoryGenerationError("--max-retries must be greater than 0.")
    if runtime_config.sampling == "verbalized" and runtime_config.verbalized_k <= 0:
        raise TrajectoryGenerationError(
            "--verbalized-k must be greater than 0 when --sampling verbalized is enabled."
        )
    if runtime_config.sampling != "verbalized" and runtime_config.verbalized_k != 1:
        raise TrajectoryGenerationError(
            "--verbalized-k may only be used when --sampling verbalized is enabled."
        )
    if runtime_config.batch_processing and not runtime_config.batch_gcs_prefix:
        raise TrajectoryGenerationError(
            "--batch-gcs-prefix or GOOGLE_CLOUD_BATCH_GCS_PREFIX is required "
            "when --batch-processing is enabled."
        )


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
    # Enforce uniqueness only for validated trajectories we intend to keep.
    with seen_signatures_lock:
        if signature in seen_signatures:
            raise DuplicateTrajectoryValidationError(
                "Duplicate trajectory signature."
            )
        seen_signatures.add(signature)


def _build_generation_payload(
    runtime_config: RuntimeConfig,
    ordered_trajectories: list[dict[str, Any]],
    *,
    error_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generation_usages = [
        trajectory["generation_usage"] for trajectory in ordered_trajectories
    ]
    output_trajectories = [
        _sanitize_trajectory_for_output(
            trajectory,
            disable_validation=runtime_config.disable_validation,
        )
        for trajectory in ordered_trajectories
    ]
    trajectory_prompts = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "prompt": trajectory["prompt"],
        }
        for trajectory in ordered_trajectories
        if isinstance(trajectory.get("prompt"), str)
    ]
    trajectory_outputs = [
        {
            "trajectory_id": trajectory["trajectory_id"],
            "raw_output": trajectory["raw_output"],
        }
        for trajectory in ordered_trajectories
        if "raw_output" in trajectory
    ]
    cost_summary = _build_cost_summary_from_generation_usages(
        generation_usages
    )
    cost_summary = _append_sampling_cost_note(cost_summary, runtime_config)
    return {
        "composite_task": runtime_config.composite_task,
        "sdk": runtime_config.sdk,
        "model": runtime_config.model,
        "model_config": _build_model_config_payload(runtime_config),
        "num_runs": runtime_config.num_runs,
        "num_trajectories": len(output_trajectories),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_summary": cost_summary,
        "error_events": (
            [dict(error_event) for error_event in sorted(error_events, key=_error_event_key)]
            if error_events is not None
            else []
        ),
        "trajectory_prompts": trajectory_prompts,
        "trajectory_outputs": trajectory_outputs,
        "trajectories": output_trajectories,
    }


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
) -> dict[str, Any] | list[dict[str, Any]]:
    from data_generation.task_level.runtime.on_demand_generation import (
        generate_single_trajectory as generate_single_trajectory_on_demand,
    )

    return generate_single_trajectory_on_demand(
        trajectory_index=trajectory_index,
        runtime_config=runtime_config,
        task_definition=task_definition,
        client_factory=client_factory,
        overall_progress=overall_progress,
        trajectory_progress=trajectory_progress,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
    )


def generate_trajectories(
    runtime_config: RuntimeConfig,
    *,
    client_factory: Any = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    task_definition = _resolve_task_definition_or_raise(runtime_config.composite_task)
    _validate_runtime_config(runtime_config)
    if runtime_config.batch_processing:
        from data_generation.task_level.runtime.batch_generation import (
            generate_trajectories_batch,
        )

        return generate_trajectories_batch(
            runtime_config,
            task_definition=task_definition,
            show_progress=show_progress,
        )
    from data_generation.task_level.runtime.on_demand_generation import (
        generate_trajectories_on_demand,
    )

    return generate_trajectories_on_demand(
        runtime_config,
        task_definition=task_definition,
        client_factory=client_factory,
        show_progress=show_progress,
    )


def run_cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except TrajectoryGenerationError as exc:
        print(_exception_summary(exc), file=sys.stderr, flush=True)
        return GENERATION_ERROR_EXIT_CODE
    except KeyboardInterrupt as exc:
        print(str(exc) or INTERRUPTED_MESSAGE, file=sys.stderr, flush=True)
        os._exit(INTERRUPTED_EXIT_CODE)


def parse_args(argv: list[str] | None = None) -> RuntimeConfig:
    load_dotenv_file()
    supported_tasks = ", ".join(supported_task_names())
    parser = argparse.ArgumentParser(
        description="Generate multi-agent task-level trajectories with google-genai on Vertex AI."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_COMPOSITE_TASK,
        dest="composite_task",
        help=f"Task name. Available tasks: {supported_tasks}.",
    )
    parser.add_argument(
        "--composite-task",
        type=str,
        dest="composite_task",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--composite_task",
        type=str,
        dest="composite_task",
        help=argparse.SUPPRESS,
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

    return RuntimeConfig(
        composite_task=args.composite_task,
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
        summary_path=resolve_dataset_output_path(args.composite_task),
        cost_output_path=args.cost_output,
        disable_validation=args.disable_validation,
        batch_processing=args.batch_processing,
        batch_gcs_prefix=args.batch_gcs_prefix,
    )


def _resolve_output_paths(runtime_config: RuntimeConfig) -> OutputPaths:
    summary_path = _resolve_summary_path(runtime_config)
    cost_path = resolve_cost_output_path(
        summary_path,
        runtime_config.cost_output_path,
    )
    if cost_path.resolve() == summary_path.resolve():
        raise TrajectoryGenerationError(
            "--cost-output must differ from the generated summary output path."
        )

    return OutputPaths(
        summary_path=summary_path,
        trajectory_dir=resolve_trajectory_output_dir(summary_path),
        prompt_dir=resolve_prompt_output_dir(summary_path),
        output_dir=resolve_raw_output_dir(summary_path),
        cost_path=cost_path,
        error_summary_path=resolve_error_output_path(summary_path),
    )


def _write_generation_outputs(
    payload: dict[str, Any],
    *,
    output_paths: OutputPaths,
) -> tuple[list[Path], list[Path], list[Path]]:
    summary_payload = build_summary_output_payload(payload)
    cost_payload = build_cost_output_payload(
        payload,
        trajectory_output_path=output_paths.summary_path,
    )
    error_summary_payload = build_error_summary_output_payload(
        payload,
        trajectory_output_path=output_paths.summary_path,
    )
    written_trajectory_paths = write_trajectory_output_payloads(
        payload["trajectories"],
        output_paths.trajectory_dir,
    )
    written_prompt_paths = write_prompt_output_payloads(
        payload.get("trajectory_prompts", []),
        output_paths.prompt_dir,
    )
    written_output_paths = write_raw_output_payloads(
        payload.get("trajectory_outputs", []),
        output_paths.output_dir,
    )
    write_json_output(summary_payload, output_paths.summary_path)
    write_json_output(cost_payload, output_paths.cost_path)
    write_json_output(error_summary_payload, output_paths.error_summary_path)
    return written_trajectory_paths, written_prompt_paths, written_output_paths


def main(argv: list[str] | None = None) -> int:
    runtime_config = parse_args(argv)
    _resolve_task_definition_or_raise(runtime_config.composite_task)
    output_paths = _resolve_output_paths(runtime_config)
    payload = generate_trajectories(runtime_config)
    written_trajectory_paths, written_prompt_paths, written_output_paths = _write_generation_outputs(
        payload,
        output_paths=output_paths,
    )
    print(f"Wrote trajectory summary to {output_paths.summary_path}")
    print(
        f"Wrote {len(written_trajectory_paths)} trajectory files to "
        f"{output_paths.trajectory_dir}"
    )
    print(
        f"Wrote {len(written_prompt_paths)} prompt files to "
        f"{output_paths.prompt_dir}"
    )
    print(
        f"Wrote {len(written_output_paths)} raw output files to "
        f"{output_paths.output_dir}"
    )
    print(f"Wrote cost summary to {output_paths.cost_path}")
    print(f"Wrote error summary to {output_paths.error_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
