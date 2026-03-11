from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
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
    RichProgress = None
    SpinnerColumn = None
    TaskProgressColumn = None
    TextColumn = None
    TimeElapsedColumn = None

from data_generation.task_level.client import (
    COST_DECIMAL_PLACES,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_SDK,
    GenerationResult,
    GenerationUsage,
    TrajectoryGenerationError,
    VERTEX_AI_PRICING_URL,
    _resolve_pricing_tier,
    build_generation_usage,
    build_generation_client,
    load_dotenv_file,
)
from data_generation.task_level.tasks import (
    TaskDefinition,
    TaskValidator,
    TrajectoryValidationError,
    get_task_definition,
    supported_task_names,
)
from data_generation.utils import (
    camel_to_snake_case,
    format_cost_usd,
    round_cost,
    stable_json_sha256,
    write_json_output,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_generation" / "task_level" / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "trajectories.json"
DEFAULT_COMPOSITE_TASK = supported_task_names()[0]
DATASET_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TRAJECTORY_DIRECTORY_NAME = "trajectories"
OVERALL_PROGRESS_COLOR = "cyan"
# Keep concurrent trajectory bars easy to tell apart in the terminal.
TRAJECTORY_PROGRESS_COLORS = ("green", "yellow", "blue", "magenta", "red", "cyan")
INTERRUPTED_EXIT_CODE = 130
INTERRUPTED_MESSAGE = (
    "Interrupted. Exiting immediately. Queued trajectories were cancelled; "
    "requests already in flight may still be billed."
)


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    if isinstance(exc, TrajectoryGenerationError):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return True

    text = str(exc)
    non_retryable_markers = (
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
            f"Unsupported composite task '{composite_task}'. "
            f"Available tasks: {supported_tasks}."
        )
    return task_definition


@dataclass(frozen=True)
class RuntimeConfig:
    composite_task: str
    num_trajectories: int
    output_path: Path
    model: str
    sdk: str
    project: str | None
    location: str
    temperature: float
    max_workers: int
    max_retries: int
    cost_output_path: Path | None = None
    disable_validation: bool = False


@dataclass(frozen=True)
class ProgressHandles:
    display: RichProgressDisplay | None
    overall_progress: Any
    trajectory_progress_bars: list[Any]
    log_writer: Callable[[str], None] | None


@dataclass(frozen=True)
class OutputPaths:
    summary_path: Path
    trajectory_dir: Path
    cost_path: Path


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return stable_json_sha256(candidate, default=str)


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
) -> dict[str, Any]:
    try:
        return validator.validate(candidate)
    except TrajectoryValidationError as exc:
        if enforce_validation:
            raise
        # Preserve the invalid trace for inspection when validation is disabled.
        return {
            "is_valid": False,
            "validation_disabled": True,
            "error": str(exc),
            "checks": [],
            "final_state": None,
            "signature": _candidate_signature(candidate),
        }


def _successful_attempt_count(generation_usage: dict[str, Any]) -> int:
    successful_attempt_number = generation_usage.get("successful_attempt_number", 1)
    if not isinstance(successful_attempt_number, int) or successful_attempt_number < 1:
        return 1
    return successful_attempt_number


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


def _build_cost_estimate_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    # Post-run estimates should honor how many attempts each saved trajectory took.
    best_case_attempt_counts = [
        _successful_attempt_count(generation_usage)
        for generation_usage in generation_usages
    ]
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
    pricing_supported = (
        best_case_total_cost is not None and worst_case_total_cost is not None
    )

    summary = {
        "currency": "USD",
        "pricing_reference": VERTEX_AI_PRICING_URL,
        "pricing_supported": pricing_supported,
        "best_case_total_usd": round_cost(
            best_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "best_case_per_trajectory_usd": round_cost(
            best_case_total_cost / runtime_config.num_trajectories
            if best_case_total_cost is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_total_usd": round_cost(
            worst_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_per_trajectory_usd": round_cost(
            worst_case_total_cost / runtime_config.num_trajectories
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
    return summary


def _build_cost_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
) -> dict[str, Any]:
    # This is the post-run rollup of the trajectories we actually kept.
    total_prompt_tokens = sum(
        usage["prompt_tokens"] for usage in generation_usages
    )
    total_output_tokens = sum(
        usage["output_tokens"] for usage in generation_usages
    )
    total_tokens = sum(usage["total_tokens"] for usage in generation_usages)
    usage_sources = sorted({usage["usage_source"] for usage in generation_usages})
    shared_pricing = _shared_pricing(generation_usages)
    pricing_supported = all("pricing" in usage for usage in generation_usages)

    input_cost = None
    output_cost = None
    total_cost = None
    if pricing_supported:
        # Sum the saved input/output usage directly instead of retry projections.
        input_cost = sum(
            (usage["prompt_tokens"] / 1_000_000)
            * usage["pricing"]["input_usd_per_million_tokens"]
            for usage in generation_usages
        )
        output_cost = sum(
            (usage["output_tokens"] / 1_000_000)
            * usage["pricing"]["output_usd_per_million_tokens"]
            for usage in generation_usages
        )
        total_cost = input_cost + output_cost

    summary = {
        "currency": "USD",
        "pricing_reference": VERTEX_AI_PRICING_URL,
        "pricing_supported": pricing_supported,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
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
        "usage_sources": usage_sources,
        "all_trajectories_used_api_usage_metadata": all(
            usage["usage_source"] == "api_usage_metadata"
            for usage in generation_usages
        ),
        "notes": [
            "Cost summary sums the saved token counts from each completed trajectory.",
            "Retry attempts that did not produce a saved trajectory are not included.",
        ],
    }
    if shared_pricing is not None:
        summary["pricing"] = shared_pricing
    return summary


def _historical_cost_paths(runtime_config: RuntimeConfig) -> list[Path]:
    task_output_dir = DEFAULT_OUTPUT_DIR / camel_to_snake_case(
        runtime_config.composite_task
    )
    if not task_output_dir.exists():
        return []
    return sorted(task_output_dir.rglob("*_trajectories_costs.json"))


def _load_historical_generation_usages(
    runtime_config: RuntimeConfig,
) -> list[dict[str, Any]]:
    historical_usages: list[dict[str, Any]] = []
    for cost_path in _historical_cost_paths(runtime_config):
        try:
            payload = json.loads(cost_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("composite_task") != runtime_config.composite_task:
            continue
        if payload.get("model") != runtime_config.model:
            continue

        for trajectory_cost in payload.get("trajectory_costs", []):
            generation_usage = trajectory_cost.get("generation_usage")
            if not _is_complete_generation_usage(generation_usage):
                continue
            historical_usages.append(generation_usage)
    return historical_usages


def _is_complete_generation_usage(generation_usage: Any) -> bool:
    if not isinstance(generation_usage, dict):
        return False
    if generation_usage.get("observed_cost_usd") is None:
        return False
    return all(
        isinstance(generation_usage.get(token_field), int)
        for token_field in ("prompt_tokens", "output_tokens", "total_tokens")
    )


def _preferred_historical_generation_usages(
    historical_usages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    api_usage_metadata = [
        generation_usage
        for generation_usage in historical_usages
        if generation_usage.get("usage_source") == "api_usage_metadata"
    ]
    if api_usage_metadata:
        return api_usage_metadata
    return historical_usages


def _mean_generation_usage(
    historical_usages: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_count = len(historical_usages)
    preferred_usage_source = historical_usages[0].get("usage_source")
    mean_usage = {
        "successful_attempt_number": 1,
        "prompt_tokens": round(
            sum(usage["prompt_tokens"] for usage in historical_usages) / sample_count
        ),
        "output_tokens": round(
            sum(usage["output_tokens"] for usage in historical_usages) / sample_count
        ),
        "total_tokens": round(
            sum(usage["total_tokens"] for usage in historical_usages) / sample_count
        ),
        "usage_source": (
            "historical_api_usage_metadata_mean"
            if preferred_usage_source == "api_usage_metadata"
            else "historical_mean"
        ),
        "traffic_type": historical_usages[0].get("traffic_type", "ON_DEMAND"),
        "observed_cost_usd": (
            sum(usage["observed_cost_usd"] for usage in historical_usages)
            / sample_count
        ),
    }
    shared_pricing = _shared_pricing(historical_usages)
    if shared_pricing is not None:
        mean_usage["pricing"] = shared_pricing
    return mean_usage


def _build_historical_preflight_generation_usage(
    runtime_config: RuntimeConfig,
) -> tuple[dict[str, Any], int] | None:
    historical_usages = _load_historical_generation_usages(runtime_config)
    if not historical_usages:
        return None

    preferred_historical_usages = _preferred_historical_generation_usages(
        historical_usages
    )
    return (
        _mean_generation_usage(preferred_historical_usages),
        len(preferred_historical_usages),
    )


def _build_preflight_cost_estimate_summary(
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> dict[str, Any]:
    historical_profile = _build_historical_preflight_generation_usage(runtime_config)
    if historical_profile is not None:
        generation_usage, sample_count = historical_profile
        summary = _build_cost_estimate_summary_from_generation_usages(
            [generation_usage] * runtime_config.num_trajectories,
            runtime_config=runtime_config,
        )
        summary["notes"][0] = (
            "Best case uses the mean token profile from "
            f"{sample_count} historical saved trajectories matching this task/model."
        )
        return summary

    # Fall back to a task-owned reference trace when no local history is available.
    generation_usage = build_generation_usage(
        model=runtime_config.model,
        prompt=task_definition.build_prompt("preflight"),
        candidate=task_definition.preflight_reference_candidate,
        usage=None,
        attempt_number=1,
    )
    summary = _build_cost_estimate_summary_from_generation_usages(
        [generation_usage] * runtime_config.num_trajectories,
        runtime_config=runtime_config,
    )
    summary["notes"][0] = (
        "Best case falls back to the task reference candidate and a local "
        "character-based token heuristic because no historical usage profile "
        "was found."
    )
    return summary


def resolve_cost_output_path(
    output_path: Path,
    cost_output_path: Path | None = None,
) -> Path:
    if cost_output_path is not None:
        return cost_output_path
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{output_path.stem}_costs{suffix}")


def resolve_dataset_output_path(
    output_path: Path,
    composite_task: str,
    *,
    generated_at: datetime | None = None,
) -> Path:
    # Only rewrite the default path so explicit outputs stay predictable.
    if output_path != DEFAULT_OUTPUT_PATH:
        return output_path

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    task_dir = camel_to_snake_case(composite_task)
    return (
        DEFAULT_OUTPUT_DIR
        / task_dir
        / timestamp.strftime(DATASET_RUN_TIMESTAMP_FORMAT)
        / f"{task_dir}_trajectories.json"
    )


def resolve_trajectory_output_dir(output_path: Path) -> Path:
    return output_path.parent / TRAJECTORY_DIRECTORY_NAME


def _trajectory_output_filename(trajectory_id: str) -> str:
    return f"{trajectory_id}.json"


def _payload_run_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "composite_task": payload["composite_task"],
        "sdk": payload["sdk"],
        "model": payload["model"],
        "project": payload["project"],
        "location": payload["location"],
        "num_trajectories": payload["num_trajectories"],
        "generated_at": payload["generated_at"],
    }


def _summary_trajectory_entry(trajectory: dict[str, Any]) -> dict[str, str]:
    trajectory_id = trajectory["trajectory_id"]
    return {
        "trajectory_id": trajectory_id,
        "path": (
            Path(TRAJECTORY_DIRECTORY_NAME) / _trajectory_output_filename(trajectory_id)
        ).as_posix(),
    }


def _trajectory_cost_entry(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "generation_usage": trajectory["generation_usage"],
    }


def build_summary_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = _payload_run_metadata(payload)
    if "cost_summary" in payload:
        summary_payload["cost_summary"] = payload["cost_summary"]
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
        write_json_output(trajectory, output_path)
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


def _trajectory_progress_color(index: int) -> str:
    return TRAJECTORY_PROGRESS_COLORS[index % len(TRAJECTORY_PROGRESS_COLORS)]


class RichTaskProgressAdapter:
    def __init__(self, progress: RichProgress, task_id: int, total: int):
        self._progress = progress
        self._task_id = task_id
        self._total = total
        self._completed = 0

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
        self._completed += amount
        self._progress.advance(self._task_id, amount)

    def set_postfix_str(self, text: str) -> None:
        self._progress.update(self._task_id, status=text)

    def refresh(self) -> None:
        self._progress.refresh()

    def close(self) -> None:
        return


class RichProgressDisplay:
    def __init__(self, runtime_config: RuntimeConfig):
        if RichProgress is None or Console is None:
            raise RuntimeError("rich progress support is unavailable")

        self.console = Console(stderr=True)
        self._progress = RichProgress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
            expand=True,
        )
        self._progress.start()
        overall_task_id = self._progress.add_task(
            "[cyan]trajectories[/cyan]",
            total=runtime_config.num_trajectories,
            status="running",
        )
        self.overall_progress = RichTaskProgressAdapter(
            self._progress,
            overall_task_id,
            runtime_config.num_trajectories,
        )
        self.trajectory_progress_bars = [
            RichTaskProgressAdapter(
                self._progress,
                self._progress.add_task(
                    f"[{_trajectory_progress_color(index)}]traj {index:03d}[/{_trajectory_progress_color(index)}]",
                    total=runtime_config.max_retries,
                    status="queued",
                ),
                runtime_config.max_retries,
            )
            for index in range(runtime_config.num_trajectories)
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

    overall_progress = tqdm(
        total=runtime_config.num_trajectories,
        desc="Trajectories",
        position=0,
        disable=disable_progress,
        dynamic_ncols=True,
        colour=OVERALL_PROGRESS_COLOR,
    )
    trajectory_progress_bars = [
        tqdm(
            total=runtime_config.max_retries,
            desc=f"traj {index:03d}",
            position=index + 1,
            leave=True,
            disable=disable_progress,
            dynamic_ncols=True,
            colour=_trajectory_progress_color(index),
        )
        for index in range(runtime_config.num_trajectories)
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
    worst_case_cost = cost_estimate["worst_case_total_usd"]
    if runtime_config.disable_validation:
        if best_case_cost is None:
            return None
        return (
            f"{label}: "
            f"{format_cost_usd(best_case_cost, decimal_places=COST_DECIMAL_PLACES)}"
        )
    if best_case_cost is None or worst_case_cost is None:
        return None
    return (
        f"{label} "
        f"(best case 1 try / worst case {runtime_config.max_retries} tries): "
        f"{format_cost_usd(best_case_cost, decimal_places=COST_DECIMAL_PLACES)} / "
        f"{format_cost_usd(worst_case_cost, decimal_places=COST_DECIMAL_PLACES)}"
    )


def _update_completed_trajectory_progress(
    trajectory_progress: Any | None,
    *,
    attempt_number: int,
    is_valid: bool,
    observed_cost_usd: float | None,
) -> None:
    if trajectory_progress is None:
        return
    trajectory_progress.total = attempt_number
    trajectory_progress.refresh()
    cost_text = "done"
    if observed_cost_usd is not None:
        cost_text = (
            f"{cost_text} "
            f"{format_cost_usd(observed_cost_usd, decimal_places=COST_DECIMAL_PLACES)}"
        )
    if not is_valid:
        cost_text = f"{cost_text} invalid"
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
) -> str:
    if runtime_config.disable_validation:
        return "retrying"
    return f"attempt {attempt_number}/{runtime_config.max_retries} retry"


def _trajectory_completion_log_message(
    runtime_config: RuntimeConfig,
    *,
    trajectory_id: str,
    generation_usage: dict[str, Any],
) -> str:
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
    if not disable_validation:
        return trajectory
    return {
        **trajectory,
        "generation_usage": _sanitize_generation_usage_for_output(
            trajectory["generation_usage"],
            disable_validation=disable_validation,
        ),
    }


def _validate_runtime_config(runtime_config: RuntimeConfig) -> None:
    if runtime_config.num_trajectories <= 0:
        raise TrajectoryGenerationError("--n must be greater than 0.")
    if runtime_config.max_workers <= 0:
        raise TrajectoryGenerationError("--max-workers must be greater than 0.")
    if runtime_config.max_retries <= 0:
        raise TrajectoryGenerationError("--max-retries must be greater than 0.")


def _build_generation_client_from_runtime(runtime_config: RuntimeConfig) -> Any:
    return build_generation_client(
        sdk=runtime_config.sdk,
        project=runtime_config.project,
        location=runtime_config.location,
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
            raise TrajectoryValidationError("Duplicate trajectory signature.")
        seen_signatures.add(signature)


def _build_generation_payload(
    runtime_config: RuntimeConfig,
    ordered_trajectories: list[dict[str, Any]],
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
    return {
        "composite_task": runtime_config.composite_task,
        "sdk": runtime_config.sdk,
        "model": runtime_config.model,
        "project": runtime_config.project,
        "location": runtime_config.location,
        "num_trajectories": runtime_config.num_trajectories,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cost_summary": _build_cost_summary_from_generation_usages(
            generation_usages
        ),
        "trajectories": output_trajectories,
    }


def extract_json_candidate(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise TrajectoryValidationError(
            f"Unsupported model response type: {type(raw_response).__name__}"
        )

    # Some SDK/model paths wrap the JSON object in fences or surrounding text.
    stripped = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if not json_match:
            raise TrajectoryValidationError("Model response did not contain JSON.")
        return json.loads(json_match.group(1))


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
        else _build_generation_client_from_runtime(runtime_config)
    )
    validator = task_definition.validator_factory()
    last_error: Exception | None = None

    for attempt_index in range(runtime_config.max_retries):
        # Variation keys give retries a stable way to ask for distinct traces.
        variation_key = f"traj-{trajectory_index:03d}-attempt-{attempt_index:02d}"
        prompt = task_definition.build_prompt(variation_key)
        if trajectory_progress is not None:
            trajectory_progress.set_postfix_str(
                _trajectory_generation_status(
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
            response_payload, usage = _unwrap_generation_response(raw_response)
            candidate = extract_json_candidate(response_payload)
            validation = _validate_candidate(
                candidate,
                validator,
                enforce_validation=not runtime_config.disable_validation,
            )
            _maybe_reserve_signature(
                validation,
                disable_validation=runtime_config.disable_validation,
                seen_signatures=seen_signatures,
                seen_signatures_lock=seen_signatures_lock,
            )

            if trajectory_progress is not None:
                trajectory_progress.update(1)
                if not runtime_config.disable_validation:
                    trajectory_progress.set_postfix_str("valid")
            if overall_progress is not None:
                overall_progress.update(1)
            generation_usage = build_generation_usage(
                model=runtime_config.model,
                prompt=prompt,
                candidate=candidate,
                usage=usage,
                attempt_number=attempt_index + 1,
            )
            _update_completed_trajectory_progress(
                trajectory_progress,
                attempt_number=attempt_index + 1,
                is_valid=validation["is_valid"],
                observed_cost_usd=generation_usage["observed_cost_usd"],
            )
            return task_definition.build_trajectory_record(
                candidate=candidate,
                validation=validation,
                trajectory_id=f"traj_{trajectory_index:03d}",
                generation_usage=generation_usage,
            )
        except Exception as exc:
            last_error = exc
            if _is_non_retryable_generation_error(exc):
                raise TrajectoryGenerationError(
                    f"Trajectory generation failed with a non-retryable error: {exc}"
                ) from exc
            if trajectory_progress is not None:
                trajectory_progress.update(1)
                trajectory_progress.set_postfix_str(
                    _trajectory_retry_status(
                        runtime_config,
                        attempt_number=attempt_index + 1,
                    )
                )

    raise TrajectoryGenerationError(
        f"Unable to generate a valid trajectory for index {trajectory_index} after "
        f"{runtime_config.max_retries} attempts: {last_error}"
    )


def generate_trajectories(
    runtime_config: RuntimeConfig,
    *,
    client_factory: Any = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    task_definition = _resolve_task_definition_or_raise(runtime_config.composite_task)
    _validate_runtime_config(runtime_config)

    seen_signatures: set[str] = set()
    seen_signatures_lock = threading.Lock()

    disable_progress = not show_progress or not os.isatty(2)
    progress_handles = _create_progress_handles(
        runtime_config,
        disable_progress=disable_progress,
    )

    projected_cost_estimate = _build_preflight_cost_estimate_summary(
        runtime_config,
        task_definition,
    )
    _log_cost_summary(
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
            _log_runtime_message(
                _trajectory_completion_log_message(
                    runtime_config,
                    trajectory_id=trajectory_record["trajectory_id"],
                    generation_usage=generation_usage,
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
        _close_progress_handles(progress_handles)

    ordered_trajectories = [results[index] for index in sorted(results)]
    return _build_generation_payload(runtime_config, ordered_trajectories)


def run_cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except KeyboardInterrupt:
        print(INTERRUPTED_MESSAGE, file=sys.stderr, flush=True)
        os._exit(INTERRUPTED_EXIT_CODE)


def parse_args(argv: list[str] | None = None) -> RuntimeConfig:
    load_dotenv_file()
    supported_tasks = ", ".join(supported_task_names())
    parser = argparse.ArgumentParser(
        description="Generate multi-agent task-level trajectories with google-genai on Vertex AI."
    )
    parser.add_argument(
        "--composite-task",
        type=str,
        default=DEFAULT_COMPOSITE_TASK,
        help=f"Composite task name. Available tasks: {supported_tasks}.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--cost-output",
        type=Path,
        default=None,
        help="Optional JSON path for the cost summary sidecar. Defaults to <output>_costs.json.",
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
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel trajectory workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum generation attempts per trajectory.",
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
        num_trajectories=args.n,
        output_path=args.output,
        model=args.model,
        sdk=args.sdk,
        project=args.project,
        location=args.location,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        cost_output_path=args.cost_output,
        disable_validation=args.disable_validation,
    )


def _resolve_output_paths(runtime_config: RuntimeConfig) -> OutputPaths:
    summary_path = resolve_dataset_output_path(
        runtime_config.output_path,
        runtime_config.composite_task,
    )
    cost_path = resolve_cost_output_path(
        summary_path,
        runtime_config.cost_output_path,
    )
    if cost_path.resolve() == summary_path.resolve():
        raise TrajectoryGenerationError("--cost-output must differ from --output.")

    return OutputPaths(
        summary_path=summary_path,
        trajectory_dir=resolve_trajectory_output_dir(summary_path),
        cost_path=cost_path,
    )


def _write_generation_outputs(
    payload: dict[str, Any],
    *,
    output_paths: OutputPaths,
) -> list[Path]:
    summary_payload = build_summary_output_payload(payload)
    cost_payload = build_cost_output_payload(
        payload,
        trajectory_output_path=output_paths.summary_path,
    )
    written_trajectory_paths = write_trajectory_output_payloads(
        payload["trajectories"],
        output_paths.trajectory_dir,
    )
    write_json_output(summary_payload, output_paths.summary_path)
    write_json_output(cost_payload, output_paths.cost_path)
    return written_trajectory_paths


def main(argv: list[str] | None = None) -> int:
    runtime_config = parse_args(argv)
    _resolve_task_definition_or_raise(runtime_config.composite_task)
    output_paths = _resolve_output_paths(runtime_config)
    payload = generate_trajectories(runtime_config)
    written_trajectory_paths = _write_generation_outputs(
        payload,
        output_paths=output_paths,
    )
    print(f"Wrote trajectory summary to {output_paths.summary_path}")
    print(
        f"Wrote {len(written_trajectory_paths)} trajectory files to "
        f"{output_paths.trajectory_dir}"
    )
    print(f"Wrote cost summary to {output_paths.cost_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
