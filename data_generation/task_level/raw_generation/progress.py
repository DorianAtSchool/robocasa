"""Render runtime progress updates and accumulate observed cost text."""

from __future__ import annotations

import threading
from dataclasses import dataclass
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

from data_generation.task_level.raw_generation.config import RuntimeConfig
from data_generation.task_level.raw_generation.runtime_support import (
    _expected_saved_trajectory_count,
    _trajectories_per_run,
    _validation_error_progress_summary,
    _validation_error_summary,
    format_trajectory_progress_label,
)
from data_generation.task_level.runtime.client import COST_DECIMAL_PLACES
from data_generation.utils import format_cost_usd, round_cost

OVERALL_PROGRESS_COLOR = "cyan"
PROGRESS_BAR_WIDTH = 30
TQDM_BAR_FORMAT = f"{{l_bar}}{{bar:{PROGRESS_BAR_WIDTH}}}{{r_bar}}"
# Keep concurrent trajectory bars easy to tell apart in the terminal.
TRAJECTORY_PROGRESS_COLORS = ("green", "yellow", "blue", "magenta", "red", "cyan")


@dataclass(frozen=True)
class ProgressHandles:
    display: Any | None
    overall_progress: Any
    trajectory_progress_bars: list[Any]
    log_writer: Callable[[str], None] | None


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
            self._total_cost_usd
            + (remaining_trajectories * average_trajectory_cost_usd),
            decimal_places=COST_DECIMAL_PLACES,
        )

    def _average_trajectory_cost_locked(self) -> float | None:
        if self._completed_trajectories > 0:
            return self._total_cost_usd / self._completed_trajectories
        return None


def _progress_run_count(runtime_config: Any) -> int:
    """Returns the progress-bar item count for runtime-like configs."""

    run_count = getattr(runtime_config, "num_runs", None)
    if isinstance(run_count, int):
        return run_count
    trajectory_count = getattr(runtime_config, "num_trajectories", None)
    if isinstance(trajectory_count, int):
        return trajectory_count
    raise AttributeError("runtime config must define num_runs or num_trajectories")


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
    if successful_trajectory_count is not None and trajectory_count > 1:
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
    previous_invalid_summary: str | None = None,
) -> str:
    if runtime_config.disable_validation:
        status_text = "generating"
    else:
        status_text = (
            f"attempt {attempt_number}/{runtime_config.max_retries} generating"
        )
    if previous_invalid_summary:
        return f"{status_text} after invalid {previous_invalid_summary}"
    return status_text


def _trajectory_retry_status(
    runtime_config: RuntimeConfig,
    *,
    attempt_number: int,
    tool_call_count: int | None = None,
    invalid_summary: str | None = None,
) -> str:
    if runtime_config.disable_validation:
        retry_text = "retrying"
    else:
        retry_text = f"attempt {attempt_number}/{runtime_config.max_retries} retry"
    if tool_call_count is None:
        status_text = retry_text
    else:
        status_text = f"{retry_text} calls={tool_call_count}"
    if invalid_summary:
        return f"{status_text} invalid {invalid_summary}"
    return status_text


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
