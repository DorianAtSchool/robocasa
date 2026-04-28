"""Phase 0b: LLM-based transferability rating for 2-agent suitability.

Takes Phase 0a candidates and rates each one HIGH / MEDIUM / LOW for how
well it fits the 2-agent collaborative setting. Low-rated tasks are kept
in `rated.json` and `excluded_low.json` so the filtering decision is
auditable — only `filtered.json` is the input to downstream phases.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_generation.task_level.runtime.client import (
    DEFAULT_GENERATION_TIMEOUT_SEC,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_SDK,
    BaseGenerationClient,
    build_generation_client,
)

from .models import TaskAnalysis
from .prompts.transferability import (
    RATING_VALUES,
    TRANSFERABILITY_RESPONSE_SCHEMA,
    build_transferability_prompt,
)

# Ordering from least to most permissive. A threshold of MEDIUM keeps both
# MEDIUM and HIGH; a threshold of HIGH keeps only HIGH.
_RATING_RANK: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass
class TransferabilityRating:
    """Per-task output of the Phase 0b LLM rating call."""

    task_name: str
    module_path: str
    batch: str
    rating: str | None
    rationale: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _passes_threshold(rating: str | None, threshold: str) -> bool:
    """Return True when the rating meets or exceeds the threshold."""

    if rating is None:
        return False
    return _RATING_RANK.get(rating, -1) >= _RATING_RANK.get(threshold, 1)


def _load_task_source(candidate: TaskAnalysis) -> str:
    """Read the composite task's source file from disk."""

    return Path(candidate.file_path).read_text(encoding="utf-8")


def _parse_rating_payload(payload: Any) -> tuple[str | None, str]:
    """Parse a rating + rationale out of the model's JSON response.

    Returns (rating, rationale). If the payload cannot be parsed, the
    rating is None and the rationale contains the parse failure.
    """

    text: str | None = None
    if isinstance(payload, str):
        text = payload
    elif payload is not None:
        text = str(payload)

    if not text:
        return None, "empty model response"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from model: {exc.msg}"

    if not isinstance(data, dict):
        return None, "model response was not a JSON object"

    rating = data.get("rating")
    rationale = data.get("rationale") or ""

    if not isinstance(rating, str) or rating not in RATING_VALUES:
        return None, f"invalid rating value: {rating!r}"

    if not isinstance(rationale, str):
        rationale = str(rationale)

    return rating, rationale.strip()


def _rate_one_task(
    candidate: TaskAnalysis,
    *,
    client: BaseGenerationClient,
    model: str,
    temperature: float,
) -> TransferabilityRating:
    """Run one LLM rating call for a single candidate task."""

    try:
        source_code = _load_task_source(candidate)
    except OSError as exc:
        return TransferabilityRating(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            rating=None,
            rationale="",
            error=f"failed to read source file: {exc}",
        )

    prompt = build_transferability_prompt(
        task_name=candidate.task_name,
        source_code=source_code,
        metadata=candidate.to_dict(),
    )

    try:
        result = client.generate(
            model=model,
            prompt=prompt,
            response_schema=TRANSFERABILITY_RESPONSE_SCHEMA,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 — surface any client error as a rating failure
        return TransferabilityRating(
            task_name=candidate.task_name,
            module_path=candidate.module_path,
            batch=candidate.batch,
            rating=None,
            rationale="",
            error=f"{type(exc).__name__}: {exc}",
        )

    payload = getattr(result, "payload", None)
    rating, rationale = _parse_rating_payload(payload)

    return TransferabilityRating(
        task_name=candidate.task_name,
        module_path=candidate.module_path,
        batch=candidate.batch,
        rating=rating,
        rationale=rationale,
        error=None if rating is not None else "failed to parse rating",
    )


def rate_candidates(
    candidates: list[TaskAnalysis],
    *,
    client: BaseGenerationClient,
    model: str,
    temperature: float = 0.1,
    workers: int = 4,
    progress_callback: Any = None,
) -> list[TransferabilityRating]:
    """Rate a list of candidates in parallel and return the ratings.

    Preserves the input order in the returned list. `progress_callback` is
    invoked with (completed_count, total_count, task_name) after each task
    finishes so callers can drive their own progress display.
    """

    ratings: list[TransferabilityRating | None] = [None] * len(candidates)
    total = len(candidates)
    completed = 0
    lock = threading.Lock()

    if workers <= 1 or total <= 1:
        for index, candidate in enumerate(candidates):
            ratings[index] = _rate_one_task(
                candidate,
                client=client,
                model=model,
                temperature=temperature,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, candidate.task_name)
        return [r for r in ratings if r is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _rate_one_task,
                candidate,
                client=client,
                model=model,
                temperature=temperature,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ratings[index] = future.result()
            with lock:
                completed += 1
                current = completed
            if progress_callback is not None:
                progress_callback(current, total, candidates[index].task_name)

    return [r for r in ratings if r is not None]


def _merge_candidate_with_rating(
    candidate: TaskAnalysis,
    rating: TransferabilityRating,
) -> dict[str, Any]:
    """Build one rated.json entry that carries both the Phase 0a fields and
    the Phase 0b rating, so downstream phases only need one input file."""

    merged = candidate.to_dict()
    merged["rating"] = rating.rating
    merged["rating_rationale"] = rating.rationale
    if rating.error:
        merged["rating_error"] = rating.error
    return merged


def _summarize_ratings(ratings: list[TransferabilityRating]) -> dict[str, Any]:
    """Aggregate rating counts for the summary file."""

    counts: dict[str, int] = {value: 0 for value in RATING_VALUES}
    error_count = 0
    for rating in ratings:
        if rating.rating in counts:
            counts[rating.rating] += 1
        if rating.error:
            error_count += 1
    return {
        "counts_by_rating": counts,
        "errors": error_count,
        "total": len(ratings),
    }


def run_phase0b(
    candidates: list[TaskAnalysis],
    *,
    output_dir: Path,
    model: str | None = None,
    workers: int = 4,
    temperature: float = 0.1,
    rating_threshold: str = "MEDIUM",
    project: str | None = None,
    location: str = DEFAULT_LOCATION,
    sdk: str = DEFAULT_SDK,
    dry_run: bool = False,
    client: BaseGenerationClient | None = None,
    progress_callback: Any = None,
    generation_timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> tuple[list[TransferabilityRating], list[TaskAnalysis], list[TaskAnalysis]]:
    """Execute Phase 0b on Phase 0a candidates and persist the outputs.

    Returns (ratings, filtered_candidates, excluded_low_candidates). The
    filtered list is what downstream phases should consume; the excluded
    list is kept for auditability.
    """

    if rating_threshold not in _RATING_RANK:
        raise ValueError(
            f"Invalid rating_threshold {rating_threshold!r}; "
            f"expected one of {sorted(_RATING_RANK)}."
        )

    resolved_model = model or DEFAULT_MODEL

    if dry_run:
        # Emit empty ratings so callers can see the intended flow without
        # burning API credits.
        return [], [], []

    if client is None:
        client = build_generation_client(
            sdk=sdk,
            project=project,
            location=location,
            timeout_sec=generation_timeout_sec,
        )

    ratings = rate_candidates(
        candidates,
        client=client,
        model=resolved_model,
        temperature=temperature,
        workers=workers,
        progress_callback=progress_callback,
    )

    # Reindex candidates by task_name so we can zip ratings back to the
    # original Phase 0a analysis entries even when threads complete out of
    # order.
    candidate_by_name = {c.task_name: c for c in candidates}

    filtered: list[TaskAnalysis] = []
    excluded_low: list[TaskAnalysis] = []
    rated_entries: list[dict[str, Any]] = []

    for rating in ratings:
        candidate = candidate_by_name.get(rating.task_name)
        if candidate is None:
            continue
        rated_entries.append(_merge_candidate_with_rating(candidate, rating))
        if _passes_threshold(rating.rating, rating_threshold):
            filtered.append(candidate)
        else:
            excluded_low.append(candidate)

    phase_dir = output_dir / "phase0b"
    phase_dir.mkdir(parents=True, exist_ok=True)

    with (phase_dir / "rated.json").open("w", encoding="utf-8") as f:
        json.dump(rated_entries, f, indent=2)

    filtered_entries = [
        entry
        for entry in rated_entries
        if _passes_threshold(entry.get("rating"), rating_threshold)
    ]
    with (phase_dir / "filtered.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_entries, f, indent=2)

    excluded_entries = [
        entry
        for entry in rated_entries
        if not _passes_threshold(entry.get("rating"), rating_threshold)
    ]
    with (phase_dir / "excluded_low.json").open("w", encoding="utf-8") as f:
        json.dump(excluded_entries, f, indent=2)

    summary = _summarize_ratings(ratings)
    summary["rating_threshold"] = rating_threshold
    summary["model"] = resolved_model
    summary["filtered"] = len(filtered_entries)
    summary["excluded_low"] = len(excluded_entries)
    with (phase_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return ratings, filtered, excluded_low
