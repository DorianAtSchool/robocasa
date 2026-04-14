"""Phase 2.5: LLM semantic review for Phase 2-passing TaskSpecs."""

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

from .prompts.spec_review import (
    SPEC_REVIEW_RESPONSE_SCHEMA,
    build_spec_review_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SpecReviewResult:
    """Per-spec semantic review output from Phase 2.5."""

    task_name: str
    spec_path: str
    approved: bool | None
    review_summary: str
    issues: list[str]
    suggested_fixes: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_spec_payload(spec_path: Path) -> dict[str, Any]:
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("spec file did not contain a JSON object")
    return payload


def _source_path_from_module(module_path: str) -> Path:
    """Resolve a dotted Python module path inside this repo to a `.py` file."""

    normalized_module_path = ".".join(
        part for part in module_path.strip().split(".") if part
    )
    if not normalized_module_path:
        raise ValueError("source_python_module was empty")
    return REPO_ROOT.joinpath(*normalized_module_path.split(".")).with_suffix(".py")


def _load_source_code(spec_payload: dict[str, Any]) -> str:
    source_module = spec_payload.get("source_python_module")
    if not isinstance(source_module, str):
        raise ValueError("source_python_module must be a string")
    source_path = _source_path_from_module(source_module)
    return source_path.read_text(encoding="utf-8")


def _parse_review_payload(payload: Any) -> tuple[bool | None, str, list[str], list[str], str | None]:
    """Parse one structured review payload from the model response."""

    text: str | None = None
    if isinstance(payload, str):
        text = payload
    elif payload is not None:
        text = str(payload)

    if not text:
        return None, "", [], [], "empty model response"

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "", [], [], f"invalid JSON from model: {exc.msg}"

    if not isinstance(data, dict):
        return None, "", [], [], "model response was not a JSON object"

    approved = data.get("approved")
    review_summary = data.get("review_summary")
    raw_issues = data.get("issues")
    raw_suggested_fixes = data.get("suggested_fixes")

    if not isinstance(approved, bool):
        return None, "", [], [], f"invalid approved value: {approved!r}"
    if not isinstance(review_summary, str):
        review_summary = str(review_summary or "")
    if not isinstance(raw_issues, list):
        return None, "", [], [], "issues must be a JSON array"
    if not isinstance(raw_suggested_fixes, list):
        return None, "", [], [], "suggested_fixes must be a JSON array"

    issues = [
        " ".join(str(issue).strip().split())
        for issue in raw_issues
        if " ".join(str(issue).strip().split())
    ]
    suggested_fixes = [
        " ".join(str(fix).strip().split())
        for fix in raw_suggested_fixes
        if " ".join(str(fix).strip().split())
    ]
    normalized_summary = " ".join(review_summary.strip().split())
    return approved, normalized_summary, issues, suggested_fixes, None


def _review_one_spec(
    spec_path: Path,
    *,
    client: BaseGenerationClient,
    model: str,
    temperature: float,
) -> SpecReviewResult:
    """Run one LLM semantic review call for a single TaskSpec."""

    try:
        spec_payload = _load_spec_payload(spec_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return SpecReviewResult(
            task_name=spec_path.stem,
            spec_path=str(spec_path),
            approved=None,
            review_summary="",
            issues=[],
            suggested_fixes=[],
            error=f"failed to read spec file: {exc}",
        )

    task_name = str(spec_payload.get("composite_task") or spec_path.stem)
    try:
        source_code = _load_source_code(spec_payload)
    except (OSError, ValueError) as exc:
        return SpecReviewResult(
            task_name=task_name,
            spec_path=str(spec_path),
            approved=None,
            review_summary="",
            issues=[],
            suggested_fixes=[],
            error=f"failed to read source file: {exc}",
        )

    prompt = build_spec_review_prompt(
        task_name=task_name,
        source_python=source_code,
        spec_payload=spec_payload,
    )
    try:
        result = client.generate(
            model=model,
            prompt=prompt,
            response_schema=SPEC_REVIEW_RESPONSE_SCHEMA,
            temperature=temperature,
            thinking_level="LOW",
        )
    except Exception as exc:  # noqa: BLE001
        return SpecReviewResult(
            task_name=task_name,
            spec_path=str(spec_path),
            approved=None,
            review_summary="",
            issues=[],
            suggested_fixes=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    approved, review_summary, issues, suggested_fixes, parse_error = _parse_review_payload(
        getattr(result, "payload", None)
    )
    return SpecReviewResult(
        task_name=task_name,
        spec_path=str(spec_path),
        approved=approved,
        review_summary=review_summary,
        issues=issues,
        suggested_fixes=suggested_fixes,
        error=parse_error,
    )


def review_specs(
    spec_paths: list[Path],
    *,
    client: BaseGenerationClient,
    model: str,
    temperature: float = 0.0,
    workers: int = 4,
    progress_callback: Any = None,
) -> list[SpecReviewResult]:
    """Review TaskSpecs in parallel while preserving input order."""

    results: list[SpecReviewResult | None] = [None] * len(spec_paths)
    total = len(spec_paths)
    completed = 0
    lock = threading.Lock()

    if workers <= 1 or total <= 1:
        for index, spec_path in enumerate(spec_paths):
            results[index] = _review_one_spec(
                spec_path,
                client=client,
                model=model,
                temperature=temperature,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, results[index].task_name)
        return [result for result in results if result is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _review_one_spec,
                spec_path,
                client=client,
                model=model,
                temperature=temperature,
            ): index
            for index, spec_path in enumerate(spec_paths)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            with lock:
                completed += 1
                current = completed
            if progress_callback is not None:
                progress_callback(current, total, results[index].task_name)

    return [result for result in results if result is not None]


def _summarize_reviews(results: list[SpecReviewResult]) -> dict[str, Any]:
    approved = sum(1 for result in results if result.approved is True)
    rejected = sum(1 for result in results if result.approved is False)
    errors = sum(1 for result in results if result.error)
    return {
        "total": len(results),
        "approved": approved,
        "rejected": rejected,
        "errors": errors,
    }


def run_phase2_5(
    *,
    output_dir: Path,
    spec_paths: list[Path],
    model: str | None = None,
    workers: int = 4,
    temperature: float = 0.0,
    project: str | None = None,
    location: str = DEFAULT_LOCATION,
    sdk: str = DEFAULT_SDK,
    dry_run: bool = False,
    client: BaseGenerationClient | None = None,
    progress_callback: Any = None,
    generation_timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> list[SpecReviewResult]:
    """Execute Phase 2.5 and persist semantic review results."""

    resolved_model = model or DEFAULT_MODEL
    if dry_run:
        return []

    if client is None:
        client = build_generation_client(
            sdk=sdk,
            project=project,
            location=location,
            timeout_sec=generation_timeout_sec,
        )

    results = review_specs(
        spec_paths,
        client=client,
        model=resolved_model,
        temperature=temperature,
        workers=workers,
        progress_callback=progress_callback,
    )

    phase_dir = output_dir / "phase2_5"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "review_results.json").write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    (phase_dir / "summary.json").write_text(
        json.dumps(
            {
                **_summarize_reviews(results),
                "model": resolved_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results
