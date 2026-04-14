"""Phase 5: aggregate errors and bottlenecks across pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _filter_task_entries(
    entries: list[dict[str, Any]],
    *,
    task_names: list[str] | None,
    key: str = "task_name",
) -> list[dict[str, Any]]:
    if not task_names:
        return [entry for entry in entries if isinstance(entry, dict)]
    wanted = set(task_names)
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get(key) in wanted
    ]


def _count_strings(values: list[str]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return [
        {"value": value, "count": counts[value]}
        for value in sorted(counts, key=lambda item: (-counts[item], item))
    ]


def _phase2_section(output_dir: Path, *, task_names: list[str] | None) -> dict[str, Any] | None:
    payload = _load_json(output_dir / "phase2" / "validation_results.json")
    if not isinstance(payload, list):
        return None
    entries = _filter_task_entries(payload, task_names=task_names)
    failed = [entry for entry in entries if entry.get("passed") is not True]
    failure_messages: list[str] = []
    for entry in failed:
        for field_name in (
            "unsupported_kinds",
            "goal_consistency_errors",
            "grounding_errors",
            "referential_errors",
        ):
            for message in entry.get(field_name, []) or []:
                if isinstance(message, str):
                    failure_messages.append(message)
        for field_name in ("schema_error", "dry_run_error"):
            message = entry.get(field_name)
            if isinstance(message, str) and message:
                failure_messages.append(message)
    return {
        "total": len(entries),
        "passed": sum(1 for entry in entries if entry.get("passed") is True),
        "failed": len(failed),
        "failed_tasks": [entry.get("task_name") for entry in failed if entry.get("task_name")],
        "distinct_failures": _count_strings(failure_messages),
    }


def _phase2_5_section(output_dir: Path, *, task_names: list[str] | None) -> dict[str, Any] | None:
    payload = _load_json(output_dir / "phase2_5" / "review_results.json")
    if not isinstance(payload, list):
        return None
    entries = _filter_task_entries(payload, task_names=task_names)
    rejected = [entry for entry in entries if entry.get("approved") is False]
    issues: list[str] = []
    for entry in rejected:
        for issue in entry.get("issues", []) or []:
            if isinstance(issue, str):
                issues.append(issue)
        error = entry.get("error")
        if isinstance(error, str) and error:
            issues.append(error)
    return {
        "total": len(entries),
        "approved": sum(1 for entry in entries if entry.get("approved") is True),
        "rejected": len(rejected),
        "errors": sum(1 for entry in entries if entry.get("error")),
        "rejected_tasks": [
            entry.get("task_name") for entry in rejected if entry.get("task_name")
        ],
        "distinct_issues": _count_strings(issues),
    }


def _phase3_section(output_dir: Path, *, task_names: list[str] | None) -> dict[str, Any] | None:
    results_payload = _load_json(output_dir / "phase3" / "results.json")
    summary_payload = _load_json(output_dir / "phase3" / "summary.json")
    if not isinstance(results_payload, list):
        return None
    entries = _filter_task_entries(results_payload, task_names=task_names)
    incomplete = [entry for entry in entries if entry.get("completed") is not True]
    error_counts: dict[str, int] = {}
    for entry in entries:
        for error_count in entry.get("error_counts_by_type", []) or []:
            error_type = error_count.get("error_type")
            count = error_count.get("count")
            if isinstance(error_type, str) and isinstance(count, int):
                error_counts[error_type] = error_counts.get(error_type, 0) + count
    section = {
        "total": len(entries),
        "completed": sum(1 for entry in entries if entry.get("completed") is True),
        "incomplete": len(incomplete),
        "incomplete_tasks": [
            entry.get("task_name") for entry in incomplete if entry.get("task_name")
        ],
        "error_counts_by_type": [
            {"error_type": error_type, "count": error_counts[error_type]}
            for error_type in sorted(error_counts)
        ],
    }
    if isinstance(summary_payload, dict):
        section["summary"] = summary_payload
    return section


def _phase4_section(output_dir: Path, *, task_names: list[str] | None) -> dict[str, Any] | None:
    results_payload = _load_json(output_dir / "phase4" / "results.json")
    summary_payload = _load_json(output_dir / "phase4" / "summary.json")
    if not isinstance(results_payload, dict):
        return None
    pre_image_results = results_payload.get("pre_image_results")
    if not isinstance(pre_image_results, list):
        pre_image_results = []
    pre_image_entries = _filter_task_entries(
        pre_image_results,
        task_names=task_names,
    )
    failed_pre_image = [
        entry for entry in pre_image_entries if entry.get("completed") is not True
    ]
    section = {
        "pre_image_total": len(pre_image_entries),
        "pre_image_completed": sum(
            1 for entry in pre_image_entries if entry.get("completed") is True
        ),
        "pre_image_failed": len(failed_pre_image),
        "pre_image_failed_tasks": [
            entry.get("task_name") for entry in failed_pre_image if entry.get("task_name")
        ],
        "sweep": results_payload.get("sweep"),
    }
    if isinstance(summary_payload, dict):
        section["summary"] = summary_payload
    return section


def _recommendations(report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    phase2 = report.get("phase2") or {}
    phase3 = report.get("phase3") or {}
    phase4 = report.get("phase4") or {}

    if isinstance(phase2, dict) and int(phase2.get("failed", 0) or 0) > 0:
        recommendations.append(
            "Phase 2 is still blocking some tasks. Fix static spec failures before scaling generation."
        )
    if isinstance(phase3, dict) and int(phase3.get("incomplete", 0) or 0) > 0:
        recommendations.append(
            "Phase 3 has incomplete tasks. Use the aggregated error types to prioritize raw-generation fixes."
        )
    sweep = phase4.get("sweep") if isinstance(phase4, dict) else None
    if isinstance(sweep, dict) and sweep.get("completed") is not True:
        recommendations.append(
            "Phase 4 sweep is incomplete. Fix simulator-grounding or runtime failures before publishing data."
        )
    if not recommendations:
        recommendations.append(
            "No blocking failures detected in the available phase outputs."
        )
    return recommendations


def run_phase5(
    *,
    output_dir: Path,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate the currently available pipeline outputs into one report."""

    phase_dir = output_dir / "phase5"
    phase_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "phase2": _phase2_section(output_dir, task_names=task_names),
        "phase2_5": _phase2_5_section(output_dir, task_names=task_names),
        "phase3": _phase3_section(output_dir, task_names=task_names),
        "phase4": _phase4_section(output_dir, task_names=task_names),
    }
    report["recommendations"] = _recommendations(report)
    (phase_dir / "error_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (phase_dir / "summary.json").write_text(
        json.dumps(
            {
                "has_phase2": report["phase2"] is not None,
                "has_phase2_5": report["phase2_5"] is not None,
                "has_phase3": report["phase3"] is not None,
                "has_phase4": report["phase4"] is not None,
                "recommendations": report["recommendations"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
