"""CLI entry point for the automated TaskSpec generation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TaskAnalysis
from .state import PipelineState

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "pipeline_runs"
)

VALID_PHASES = ("0a", "0b", "1", "2", "2.5", "3", "4", "5", "all")
VALID_BATCHES = ("batch1", "batch2", "batch3", "all")
VALID_RATING_THRESHOLDS = ("LOW", "MEDIUM", "HIGH")
DEFAULT_PHASE2_REPAIR_RETRIES = 2
DEFAULT_PHASE2_5_REPAIR_RETRIES = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated TaskSpec generation pipeline for 2-agent RoboCasa tasks.",
    )
    parser.add_argument(
        "--phase",
        choices=VALID_PHASES,
        default="all",
        help="Which pipeline phase to run (default: all).",
    )
    parser.add_argument(
        "--batch",
        choices=VALID_BATCHES,
        default="all",
        help=(
            "Task-type batch to process (default: all): "
            "batch1=countertop prep/setup, "
            "batch2=appliance/sink/heat control, "
            "batch3=storage/articulated fixtures."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to existing run directory to resume.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override task list (specific task names).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name for phases that need it.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for trajectory generation (default: 5).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of raw trajectories to generate per task in Phase 3 (default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for LLM calls (default: 4).",
    )
    parser.add_argument(
        "--rating-threshold",
        choices=VALID_RATING_THRESHOLDS,
        default="MEDIUM",
        help=(
            "Phase 0b minimum rating to keep for downstream phases "
            "(default: MEDIUM). Low-rated tasks are still recorded in "
            "rated.json and excluded_low.json."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Google Cloud project for Vertex AI calls.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Vertex AI location (default: global).",
    )
    parser.add_argument(
        "--generation-timeout-sec",
        type=int,
        default=300,
        help=(
            "Per-request wall-clock cap (seconds) for LLM SDK calls used by "
            "phases 0b/1/2.5/3. Set to 0 to disable. Default: 300."
        ),
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Record per-camera MP4 videos during Phase 4 sweep runs.",
    )
    parser.add_argument(
        "--phase2-repair-retries",
        type=int,
        default=DEFAULT_PHASE2_REPAIR_RETRIES,
        help=(
            "When running `--phase all`, retry Phase 2-failing specs with "
            "validation feedback this many times (default: 2)."
        ),
    )
    parser.add_argument(
        "--phase2-5-repair-retries",
        type=int,
        default=DEFAULT_PHASE2_5_REPAIR_RETRIES,
        help=(
            "When running `--phase all`, retry Phase 2.5-rejected specs with "
            "review feedback this many times (default: 2)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    return parser.parse_args(argv)


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    """Get or create the run directory."""
    if args.resume:
        if not args.resume.exists():
            print(f"Error: Resume directory does not exist: {args.resume}", file=sys.stderr)
            sys.exit(1)
        return args.resume

    base = args.output_dir or DEFAULT_OUTPUT_DIR
    batch_label = args.batch if args.batch != "all" else "all_batches"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{batch_label}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _apply_selection_filters(
    candidates: list[TaskAnalysis],
    *,
    batch: str,
    task_names: list[str] | None,
) -> list[TaskAnalysis]:
    """Narrow a candidate list by batch label and explicit task allowlist."""

    if batch != "all":
        candidates = [c for c in candidates if c.batch == batch]
    if task_names:
        task_set = set(task_names)
        candidates = [c for c in candidates if c.task_name in task_set]
    return candidates


def _load_phase0a_candidates(run_dir: Path) -> list[TaskAnalysis]:
    """Load Phase 0a output from an existing run directory."""

    candidates_path = run_dir / "phase0a" / "candidates.json"
    if not candidates_path.exists():
        print(
            f"Error: Phase 0a output not found at {candidates_path}. "
            f"Run `--phase 0a` first or use `--phase all`.",
            file=sys.stderr,
        )
        sys.exit(1)
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    return [TaskAnalysis.from_dict(entry) for entry in payload]


def _load_phase0b_filtered(run_dir: Path) -> list[TaskAnalysis]:
    """Load the filtered (>= threshold) candidate list from Phase 0b."""

    filtered_path = run_dir / "phase0b" / "filtered.json"
    if not filtered_path.exists():
        print(
            f"Error: Phase 0b output not found at {filtered_path}. "
            f"Run `--phase 0b` first or use `--phase all`.",
            file=sys.stderr,
        )
        sys.exit(1)
    payload = json.loads(filtered_path.read_text(encoding="utf-8"))
    # Entries are merged dicts; TaskAnalysis.from_dict ignores extra rating keys.
    return [TaskAnalysis.from_dict(entry) for entry in payload]


def _ensure_selected_candidates(
    *,
    run_dir: Path,
    batch: str,
    task_names: list[str] | None,
    selected_candidates: list[TaskAnalysis] | None,
) -> list[TaskAnalysis]:
    """Load the downstream candidate subset if it is not already in memory."""

    if selected_candidates is not None:
        return selected_candidates
    loaded = _load_phase0b_filtered(run_dir)
    return _apply_selection_filters(
        loaded,
        batch=batch,
        task_names=task_names,
    )


def _run_phase0a(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> list[TaskAnalysis]:
    """Run Phase 0a and print a summary. Returns the selected candidate list."""

    from .phase0a import run_phase0a

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Phase 0a: Static candidate filtering..."
    )
    state.mark_phase_started("0a")
    candidates, excluded = run_phase0a(
        output_dir=run_dir,
        dry_run=args.dry_run,
    )

    batch_counts: dict[str, int] = {}
    for c in candidates:
        batch_counts[c.batch] = batch_counts.get(c.batch, 0) + 1

    print(f"  Scanned {len(candidates) + len(excluded)} task files")
    print(
        f"  Candidates: {len(candidates)} "
        f"(batch1={batch_counts.get('batch1', 0)}, "
        f"batch2={batch_counts.get('batch2', 0)}, "
        f"batch3={batch_counts.get('batch3', 0)})"
    )
    print(f"  Excluded: {len(excluded)}")

    selected = _apply_selection_filters(
        candidates,
        batch=args.batch,
        task_names=args.tasks,
    )
    if args.batch != "all" or args.tasks:
        print(f"  Selected: {len(selected)} candidates after batch/task filters")

    if not args.dry_run:
        state.mark_phase_completed("0a")

    return selected


def _run_phase0b(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
    candidates: list[TaskAnalysis],
) -> list[TaskAnalysis]:
    """Run Phase 0b on the given candidates and print a summary."""

    from .phase0b import run_phase0b

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Phase 0b: LLM transferability rating "
        f"({len(candidates)} candidates, threshold={args.rating_threshold})..."
    )

    if not candidates:
        print("  No candidates to rate. Skipping.")
        return []

    state.mark_phase_started("0b")

    def _progress(completed: int, total: int, task_name: str) -> None:
        # Keep this terse — one line per completion so stdout stays scannable.
        print(f"  [{completed}/{total}] rated {task_name}")

    ratings, filtered, excluded_low = run_phase0b(
        candidates,
        output_dir=run_dir,
        model=args.model,
        workers=args.workers,
        rating_threshold=args.rating_threshold,
        project=args.project,
        location=args.location or "global",
        dry_run=args.dry_run,
        progress_callback=_progress,
        generation_timeout_sec=(
            None if args.generation_timeout_sec == 0 else args.generation_timeout_sec
        ),
    )

    if args.dry_run:
        return filtered

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    error_count = 0
    for rating in ratings:
        if rating.rating in counts:
            counts[rating.rating] += 1
        if rating.error:
            error_count += 1
    print(
        f"  Rated {len(ratings)}: "
        f"HIGH={counts['HIGH']}, MEDIUM={counts['MEDIUM']}, LOW={counts['LOW']}, "
        f"errors={error_count}"
    )
    print(
        f"  Filtered (>= {args.rating_threshold}): {len(filtered)} | "
        f"Excluded below threshold: {len(excluded_low)}"
    )

    state.mark_phase_completed("0b")
    return filtered


def _run_phase1(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
    candidates: list[TaskAnalysis],
    repair_feedback_by_task: dict[str, Any] | None = None,
    display_label: str | None = None,
    progress_verb: str = "generated",
) -> list[Any]:
    """Run Phase 1 (LLM spec generation) on the given candidates."""

    from .phase1 import run_phase1

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"{display_label or f'Phase 1: LLM spec generation ({len(candidates)} candidates)...'}"
    )

    if not candidates:
        print("  No candidates to generate specs for. Skipping.")
        return []

    state.mark_phase_started("1")

    def _progress(completed: int, total: int, task_name: str) -> None:
        print(f"  [{completed}/{total}] {progress_verb} {task_name}")

    results = run_phase1(
        candidates,
        output_dir=run_dir,
        model=args.model,
        workers=args.workers,
        project=args.project,
        location=args.location or "global",
        dry_run=args.dry_run,
        progress_callback=_progress,
        generation_timeout_sec=(
            None if args.generation_timeout_sec == 0 else args.generation_timeout_sec
        ),
        repair_feedback_by_task=repair_feedback_by_task,
    )

    if args.dry_run:
        return results

    successes = sum(1 for r in results if r.spec_payload is not None)
    failures = len(results) - successes
    print(f"  Generated specs: {successes} | Failures: {failures}")
    if failures:
        for r in results:
            if r.spec_payload is None:
                print(f"    - {r.task_name}: {r.error}")

    state.mark_phase_completed("1")
    return results


def _load_phase1_spec_paths(
    run_dir: Path,
    *,
    task_names: list[str] | None,
) -> list[Path]:
    """Load the list of generated spec paths from Phase 1, optionally filtered."""

    specs_dir = run_dir / "phase1" / "specs"
    if not specs_dir.exists():
        print(
            f"Error: Phase 1 specs directory not found at {specs_dir}. "
            f"Run `--phase 1` first or use `--phase all`.",
            file=sys.stderr,
        )
        sys.exit(1)
    spec_paths = sorted(specs_dir.glob("*.json"))
    if task_names:
        from .phase1 import _spec_filename

        wanted = {_spec_filename(name) for name in task_names}
        spec_paths = [p for p in spec_paths if p.name in wanted]
    return spec_paths


def _resolve_recorded_spec_path(run_dir: Path, recorded_path: str) -> Path:
    """Resolve one spec path recorded in a results JSON back to a real file."""

    spec_path = Path(recorded_path)
    if spec_path.exists():
        return spec_path
    run_relative = run_dir / recorded_path
    if run_relative.exists():
        return run_relative
    return spec_path


def _load_phase2_passed_spec_paths(
    run_dir: Path,
    *,
    task_names: list[str] | None,
) -> list[Path]:
    """Load the subset of Phase 1 specs that passed Phase 2 validation."""

    validation_path = run_dir / "phase2" / "validation_results.json"
    if not validation_path.exists():
        print(
            f"Error: Phase 2 validation output not found at {validation_path}. "
            f"Run `--phase 2` first or use `--phase all`.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print(
            f"Error: Phase 2 validation output at {validation_path} was not a JSON list.",
            file=sys.stderr,
        )
        sys.exit(1)

    wanted_tasks = set(task_names or [])
    spec_paths: list[Path] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if not entry.get("passed", False):
            continue
        task_name = str(entry.get("task_name") or "")
        if wanted_tasks and task_name not in wanted_tasks:
            continue
        recorded_path = entry.get("spec_path")
        if not isinstance(recorded_path, str):
            continue
        spec_paths.append(_resolve_recorded_spec_path(run_dir, recorded_path))
    return spec_paths


def _load_phase2_5_approved_spec_paths(
    run_dir: Path,
    *,
    task_names: list[str] | None,
) -> list[Path]:
    """Prefer Phase 2.5-approved specs when review output exists."""

    review_path = run_dir / "phase2_5" / "review_results.json"
    if not review_path.exists():
        return _load_phase2_passed_spec_paths(run_dir, task_names=task_names)

    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print(
            f"Error: Phase 2.5 review output at {review_path} was not a JSON list.",
            file=sys.stderr,
        )
        sys.exit(1)

    wanted_tasks = set(task_names or [])
    spec_paths: list[Path] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("approved") is not True:
            continue
        task_name = str(entry.get("task_name") or "")
        if wanted_tasks and task_name not in wanted_tasks:
            continue
        recorded_path = entry.get("spec_path")
        if not isinstance(recorded_path, str):
            continue
        spec_paths.append(_resolve_recorded_spec_path(run_dir, recorded_path))
    return spec_paths


def _load_phase1_spec_payload(
    run_dir: Path,
    *,
    task_name: str,
) -> dict[str, Any] | None:
    """Load the current Phase 1 spec JSON for one task if it exists."""

    from .phase1 import _spec_filename

    spec_path = run_dir / "phase1" / "specs" / _spec_filename(task_name)
    if not spec_path.exists():
        return None
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_phase2_feedback_lines(result: Any) -> tuple[str, ...]:
    """Convert one Phase 2 validation result into prompt-ready repair feedback."""

    lines: list[str] = [
        "The previous TaskSpec failed static validation. Fix every issue below.",
    ]
    if getattr(result, "schema_error", None):
        lines.append(f"Schema error: {result.schema_error}")
    for error in getattr(result, "unsupported_kinds", ()) or ():
        lines.append(f"Unsupported kind: {error}")
    for error in getattr(result, "goal_consistency_errors", ()) or ():
        lines.append(f"Goal consistency error: {error}")
    for error in getattr(result, "grounding_errors", ()) or ():
        lines.append(f"Grounding error: {error}")
    for error in getattr(result, "simulation_errors", ()) or ():
        lines.append(f"Simulator alignment error: {error}")
    for error in getattr(result, "referential_errors", ()) or ():
        lines.append(f"Referential error: {error}")
    if getattr(result, "dry_run_error", None):
        lines.append(f"Example trajectory dry-run error: {result.dry_run_error}")
        lines.append(
            "Revise the example_trajectory so every non-communication action is "
            "preceded by navigate_to_fixture for the exact fixture that action "
            "operates at, including reference_fixture_id targets."
        )
    lines.append(
        "Return a corrected complete TaskSpec JSON. Preserve valid content and "
        "make the minimal coherent changes needed to pass validation."
    )
    return tuple(lines)


def _build_phase2_5_feedback_lines(result: Any) -> tuple[str, ...]:
    """Convert one Phase 2.5 review result into prompt-ready repair feedback."""

    lines: list[str] = [
        "The previous TaskSpec was rejected in semantic review against the source task.",
    ]
    review_summary = getattr(result, "review_summary", "")
    if isinstance(review_summary, str) and review_summary:
        lines.append(f"Review summary: {review_summary}")
    for issue in getattr(result, "issues", ()) or ():
        lines.append(f"Semantic issue: {issue}")
    for fix in getattr(result, "suggested_fixes", ()) or ():
        lines.append(f"Suggested fix: {fix}")
    lines.append(
        "Revise the existing TaskSpec JSON so it matches the source task semantics "
        "while staying internally consistent and validator-safe."
    )
    lines.append(
        "Keep to the supported TaskSpec schema and goal/precondition/effect kinds; "
        "if a suggested fix mentions an unsupported construct, translate it into the "
        "closest supported representation instead of copying it literally."
    )
    return tuple(lines)


def _merge_phase2_results_with_generation_failures(
    *,
    run_dir: Path,
    generation_results: list[Any],
    validation_results: list[Any],
) -> list[Any]:
    """Keep failed repair generations visible as synthetic Phase 2 failures."""

    from .phase1 import _spec_filename
    from .phase2 import SpecValidationResult

    results_by_task = {
        getattr(result, "task_name", ""): result
        for result in validation_results
        if getattr(result, "task_name", None)
    }
    for result in generation_results:
        task_name = getattr(result, "task_name", None)
        if not isinstance(task_name, str) or getattr(result, "spec_payload", None) is not None:
            continue
        results_by_task[task_name] = SpecValidationResult(
            task_name=task_name,
            spec_path=str(run_dir / "phase1" / "specs" / _spec_filename(task_name)),
            passed=False,
            schema_error=f"repair generation failed: {getattr(result, 'error', 'unknown error')}",
        )
    return [results_by_task[task_name] for task_name in sorted(results_by_task)]


def _repair_phase2_failures(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
    candidates: list[TaskAnalysis],
    validation_results: list[Any],
) -> list[Any]:
    """Retry Phase 2-failing specs by regenerating them with validation feedback."""

    from .phase1 import SpecRepairContext

    candidate_by_name = {candidate.task_name: candidate for candidate in candidates}
    current_results = validation_results
    max_retries = max(0, args.phase2_repair_retries)

    for attempt in range(1, max_retries + 1):
        failed_results = [
            result
            for result in current_results
            if not getattr(result, "passed", False)
            and getattr(result, "task_name", None) in candidate_by_name
        ]
        if not failed_results:
            break

        repair_candidates = [
            candidate_by_name[result.task_name] for result in failed_results
        ]
        repair_feedback_by_task = {
            result.task_name: SpecRepairContext(
                previous_spec_payload=_load_phase1_spec_payload(
                    run_dir,
                    task_name=result.task_name,
                ),
                feedback_lines=_build_phase2_feedback_lines(result),
            )
            for result in failed_results
        }
        print(
            f"  Repairing {len(repair_candidates)} Phase 2 failures "
            f"(attempt {attempt}/{max_retries})..."
        )
        repair_results = _run_phase1(
            run_dir=run_dir,
            args=args,
            state=state,
            candidates=repair_candidates,
            repair_feedback_by_task=repair_feedback_by_task,
            display_label=(
                f"Phase 1 repair: revising {len(repair_candidates)} specs from "
                f"Phase 2 feedback..."
            ),
            progress_verb="revised",
        )
        current_results = _run_phase2(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        current_results = _merge_phase2_results_with_generation_failures(
            run_dir=run_dir,
            generation_results=repair_results,
            validation_results=current_results,
        )

    return current_results


def _repair_phase2_5_rejections(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
    candidates: list[TaskAnalysis],
    review_results: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Retry Phase 2.5-rejected specs with semantic review feedback."""

    from .phase1 import SpecRepairContext

    candidate_by_name = {candidate.task_name: candidate for candidate in candidates}
    current_review_results = review_results
    current_phase2_results: list[Any] = []
    max_retries = max(0, args.phase2_5_repair_retries)

    for attempt in range(1, max_retries + 1):
        rejected_results = [
            result
            for result in current_review_results
            if getattr(result, "approved", None) is False
            and getattr(result, "task_name", None) in candidate_by_name
        ]
        if not rejected_results:
            break

        repair_candidates = [
            candidate_by_name[result.task_name] for result in rejected_results
        ]
        repair_feedback_by_task = {
            result.task_name: SpecRepairContext(
                previous_spec_payload=_load_phase1_spec_payload(
                    run_dir,
                    task_name=result.task_name,
                ),
                feedback_lines=_build_phase2_5_feedback_lines(result),
            )
            for result in rejected_results
        }
        print(
            f"  Repairing {len(repair_candidates)} Phase 2.5 rejections "
            f"(attempt {attempt}/{max_retries})..."
        )
        repair_results = _run_phase1(
            run_dir=run_dir,
            args=args,
            state=state,
            candidates=repair_candidates,
            repair_feedback_by_task=repair_feedback_by_task,
            display_label=(
                f"Phase 1 repair: revising {len(repair_candidates)} specs from "
                f"Phase 2.5 review feedback..."
            ),
            progress_verb="revised",
        )
        current_phase2_results = _run_phase2(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        current_phase2_results = _merge_phase2_results_with_generation_failures(
            run_dir=run_dir,
            generation_results=repair_results,
            validation_results=current_phase2_results,
        )
        current_phase2_results = _repair_phase2_failures(
            run_dir=run_dir,
            args=args,
            state=state,
            candidates=candidates,
            validation_results=current_phase2_results,
        )
        current_review_results = _run_phase2_5(
            run_dir=run_dir,
            args=args,
            state=state,
        )

    return current_phase2_results, current_review_results


def _run_phase2(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> list[Any]:
    """Run Phase 2 (static spec validation) on the generated specs."""

    from .phase2 import run_phase2

    spec_paths = _load_phase1_spec_paths(run_dir, task_names=args.tasks)

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Phase 2: Static spec validation ({len(spec_paths)} specs)..."
    )

    if args.dry_run:
        return []

    if not spec_paths:
        print("  No specs to validate. Skipping.")
        return []

    state.mark_phase_started("2")
    results = run_phase2(output_dir=run_dir, spec_paths=spec_paths)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"  Validated {len(results)}: passed={passed}, failed={failed}")
    if failed:
        for r in results:
            if r.passed:
                continue
            print(f"    - {r.task_name}:")
            if r.schema_error:
                print(f"        schema: {r.schema_error}")
            for err in r.unsupported_kinds:
                print(f"        unsupported: {err}")
            for err in r.goal_consistency_errors:
                print(f"        goal: {err}")
            for err in r.grounding_errors:
                print(f"        grounding: {err}")
            for err in r.referential_errors:
                print(f"        referential: {err}")
            if r.dry_run_error:
                # Keep dry-run lines short — the file has the full text.
                print(f"        dry_run: {r.dry_run_error[:200]}")

    state.mark_phase_completed("2")
    return results


def _run_phase2_5(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> list[Any]:
    """Run Phase 2.5 (LLM semantic spec review) on Phase 2-passing specs."""

    from .phase2_5 import run_phase2_5

    spec_paths = _load_phase2_passed_spec_paths(run_dir, task_names=args.tasks)
    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Phase 2.5: LLM spec review ({len(spec_paths)} Phase 2-passing specs)..."
    )

    if not spec_paths:
        print("  No Phase 2-passing specs to review. Skipping.")
        return []

    state.mark_phase_started("2.5")

    def _progress(completed: int, total: int, task_name: str) -> None:
        print(f"  [{completed}/{total}] reviewed {task_name}")

    results = run_phase2_5(
        output_dir=run_dir,
        spec_paths=spec_paths,
        model=args.model,
        workers=args.workers,
        project=args.project,
        location=args.location or "global",
        dry_run=args.dry_run,
        progress_callback=_progress,
        generation_timeout_sec=(
            None if args.generation_timeout_sec == 0 else args.generation_timeout_sec
        ),
    )

    if args.dry_run:
        return

    approved = sum(1 for result in results if result.approved is True)
    rejected = sum(1 for result in results if result.approved is False)
    errors = sum(1 for result in results if result.error)
    print(
        f"  Reviewed {len(results)}: approved={approved}, "
        f"rejected={rejected}, errors={errors}"
    )
    for result in results:
        if result.error:
            print(f"    - {result.task_name}: error: {result.error}")
            continue
        if result.approved:
            continue
        print(f"    - {result.task_name}: {result.review_summary}")
        for issue in result.issues:
            print(f"        issue: {issue}")
        for fix in result.suggested_fixes[:3]:
            print(f"        fix: {fix}")

    state.mark_phase_completed("2.5")
    return results


def _run_phase3(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> None:
    """Run Phase 3 trajectory generation on Phase 2-passing specs."""

    from .phase3 import run_phase3

    spec_paths = _load_phase2_5_approved_spec_paths(run_dir, task_names=args.tasks)
    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        f"Phase 3: Trajectory generation ({len(spec_paths)} approved specs, "
        f"num_runs={args.num_runs})..."
    )

    if not spec_paths:
        print("  No Phase 2-passing specs to generate. Skipping.")
        return

    state.mark_phase_started("3")

    def _progress(completed: int, total: int, task_name: str) -> None:
        print(f"  [{completed}/{total}] generated trajectories for {task_name}")

    results = run_phase3(
        output_dir=run_dir,
        spec_paths=spec_paths,
        model=args.model,
        num_runs=args.num_runs,
        workers=args.workers,
        max_retries=args.max_retries,
        project=args.project,
        location=args.location or "global",
        dry_run=args.dry_run,
        progress_callback=_progress,
        generation_timeout_sec=(
            None if args.generation_timeout_sec == 0 else args.generation_timeout_sec
        ),
    )

    if args.dry_run:
        return

    completed = sum(1 for result in results if result.completed)
    incomplete = len(results) - completed
    print(f"  Generated {len(results)} task outputs: completed={completed}, incomplete={incomplete}")
    for result in results:
        if result.completed:
            continue
        print(f"    - {result.task_name}: {result.error or 'incomplete'}")
        for distinct_error in result.distinct_errors[:2]:
            summary = distinct_error.get("summary")
            if isinstance(summary, str) and summary:
                print(f"        error: {summary}")

    state.mark_phase_completed("3")


def _run_phase4(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> None:
    """Run Phase 4 pre-image post-processing and sweep."""

    from .phase4 import run_phase4

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        "Phase 4: Pre-image + sweep..."
    )
    state.mark_phase_started("4")
    try:
        results = run_phase4(
            output_dir=run_dir,
            task_names=args.tasks,
            workers=args.workers,
            videos=args.videos,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        return

    pre_image_results = results.get("pre_image_results", [])
    sweep_result = results.get("sweep", {})
    pre_image_completed = sum(
        1
        for result in pre_image_results
        if isinstance(result, dict) and result.get("completed") is True
    )
    pre_image_failed = len(pre_image_results) - pre_image_completed
    print(
        f"  Pre-image tasks: completed={pre_image_completed}, failed={pre_image_failed}"
    )
    if isinstance(sweep_result, dict):
        print(
            f"  Sweep: {'completed' if sweep_result.get('completed') else 'incomplete'}"
        )
        if sweep_result.get("completed") is not True and sweep_result.get("error"):
            print(f"    - {sweep_result['error']}")

    state.mark_phase_completed("4")


def _run_phase5(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    state: PipelineState,
) -> None:
    """Run Phase 5 error aggregation."""

    from .phase5 import run_phase5

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}"
        "Phase 5: Error aggregation..."
    )
    if args.dry_run:
        return
    state.mark_phase_started("5")
    report = run_phase5(output_dir=run_dir, task_names=args.tasks)
    for recommendation in report.get("recommendations", []):
        print(f"  - {recommendation}")
    state.mark_phase_completed("5")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = _resolve_run_dir(args)
    state = PipelineState(
        run_dir,
        config={
            "batch": args.batch,
            "model": args.model,
            "max_retries": args.max_retries,
            "num_runs": args.num_runs,
            "workers": args.workers,
            "rating_threshold": args.rating_threshold,
            "phase2_repair_retries": args.phase2_repair_retries,
            "phase2_5_repair_retries": args.phase2_5_repair_retries,
        },
    )

    phase = args.phase
    should_run_all = phase == "all"
    phase2_results: list[Any] = []
    phase2_5_results: list[Any] = []

    # Phase 0a: Static candidate filtering.
    selected_candidates: list[TaskAnalysis] | None = None
    if phase in ("0a", "all"):
        selected_candidates = _run_phase0a(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if phase == "0a":
            return

    # Phase 0b: LLM transferability rating.
    if phase in ("0b", "all"):
        if selected_candidates is None:
            loaded = _load_phase0a_candidates(run_dir)
            selected_candidates = _apply_selection_filters(
                loaded,
                batch=args.batch,
                task_names=args.tasks,
            )
        selected_candidates = _run_phase0b(
            run_dir=run_dir,
            args=args,
            state=state,
            candidates=selected_candidates,
        )
        if phase == "0b":
            return

    # Phase 1: LLM spec generation.
    if phase in ("1", "all"):
        if selected_candidates is None:
            loaded = _load_phase0b_filtered(run_dir)
            selected_candidates = _apply_selection_filters(
                loaded,
                batch=args.batch,
                task_names=args.tasks,
            )
        _run_phase1(
            run_dir=run_dir,
            args=args,
            state=state,
            candidates=selected_candidates,
        )
        if phase == "1":
            return

    # Phase 2: Static spec validation.
    if phase in ("2", "all"):
        selected_candidates = _ensure_selected_candidates(
            run_dir=run_dir,
            batch=args.batch,
            task_names=args.tasks,
            selected_candidates=selected_candidates,
        )
        phase2_results = _run_phase2(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if selected_candidates:
            phase2_results = _repair_phase2_failures(
                run_dir=run_dir,
                args=args,
                state=state,
                candidates=selected_candidates,
                validation_results=phase2_results,
            )
        if phase == "2":
            return

    if phase in ("2.5", "all"):
        selected_candidates = _ensure_selected_candidates(
            run_dir=run_dir,
            batch=args.batch,
            task_names=args.tasks,
            selected_candidates=selected_candidates,
        )
        phase2_5_results = _run_phase2_5(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if selected_candidates:
            repaired_phase2_results, phase2_5_results = _repair_phase2_5_rejections(
                run_dir=run_dir,
                args=args,
                state=state,
                candidates=selected_candidates,
                review_results=phase2_5_results,
            )
            if repaired_phase2_results:
                phase2_results = repaired_phase2_results
        if phase == "2.5":
            return

    if phase in ("3", "all"):
        _run_phase3(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if phase == "3":
            return

    if phase in ("4", "all"):
        _run_phase4(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if phase == "4":
            return

    if phase in ("5", "all"):
        _run_phase5(
            run_dir=run_dir,
            args=args,
            state=state,
        )
        if phase == "5":
            return


if __name__ == "__main__":
    main()
