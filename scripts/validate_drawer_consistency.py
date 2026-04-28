#!/usr/bin/env python3
"""Validate drawer open/pick consistency from trajectory execution metadata.

Checks, per trajectory_execution_metadata.json:
1) Preserved object fixture (from load_initial_state placement_events) exists.
2) First open_sliding_part target_id matches first pick_up_object source_id.
3) Open target matches preserved object fixture.
4) (Optional) If open_sliding_part details include moved_supported_children,
   ensure target object was moved by a non-zero delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _first_step(metadata: dict[str, Any], tool: str, *, object_id: str | None = None) -> dict[str, Any] | None:
    for step in metadata.get("steps", []):
        if step.get("tool") != tool:
            continue
        if object_id is not None:
            if (step.get("args") or {}).get("object_id") != object_id:
                continue
        return step
    return None


def _first_placement_event(metadata: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for event in (metadata.get("load_initial_state", {}) or {}).get("placement_events", []):
        if event.get("object_id") == object_id:
            return event
    return None


def _norm3(vec: Any) -> float:
    if not isinstance(vec, (list, tuple)) or len(vec) < 3:
        return 0.0
    x, y, z = vec[:3]
    try:
        return float((float(x) ** 2 + float(y) ** 2 + float(z) ** 2) ** 0.5)
    except Exception:
        return 0.0


def validate_metadata(path: Path, object_id: str) -> tuple[bool, list[str]]:
    data = json.loads(path.read_text())
    issues: list[str] = []

    placement = _first_placement_event(data, object_id)
    if placement is None:
        issues.append(f"missing placement_events entry for {object_id!r}")
        return False, issues
    preserved_fixture = placement.get("target_fixture_id")
    if not isinstance(preserved_fixture, str):
        issues.append(f"placement_events target_fixture_id missing for {object_id!r}")
        return False, issues

    open_step = _first_step(data, "open_sliding_part")
    if open_step is None:
        issues.append("missing open_sliding_part step")
        return False, issues
    open_target = (open_step.get("args") or {}).get("target_id")
    if not isinstance(open_target, str):
        issues.append("open_sliding_part.args.target_id missing")

    pick_step = _first_step(data, "pick_up_object", object_id=object_id)
    if pick_step is None:
        issues.append(f"missing pick_up_object step for {object_id!r}")
        return False, issues
    pick_source = (pick_step.get("args") or {}).get("source_id")
    if not isinstance(pick_source, str):
        issues.append(f"pick_up_object.args.source_id missing for {object_id!r}")

    if isinstance(open_target, str) and isinstance(pick_source, str) and open_target != pick_source:
        issues.append(
            f"open target {open_target!r} != pick source {pick_source!r}"
        )
    if isinstance(open_target, str) and open_target != preserved_fixture:
        issues.append(
            f"open target {open_target!r} != preserved fixture {preserved_fixture!r}"
        )
    if isinstance(pick_source, str) and pick_source != preserved_fixture:
        issues.append(
            f"pick source {pick_source!r} != preserved fixture {preserved_fixture!r}"
        )

    # Optional motion diagnostics (available in newer executor metadata).
    moved_children = (open_step.get("details") or {}).get("moved_supported_children")
    if moved_children is not None:
        target_move = None
        for child in moved_children:
            if isinstance(child, dict) and child.get("object_id") == object_id:
                target_move = child
                break
        if target_move is None:
            issues.append(f"{object_id!r} missing from moved_supported_children diagnostics")
        else:
            if _norm3(target_move.get("delta")) <= 1e-5:
                issues.append(f"{object_id!r} movement delta is ~0 in moved_supported_children")

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Run root containing sweep outputs (searched recursively for trajectory_execution_metadata.json).",
    )
    parser.add_argument(
        "--object-id",
        type=str,
        default="straw",
        help="Object ID to validate against drawer open/pick consistency.",
    )
    args = parser.parse_args()

    files = sorted(args.root.rglob("trajectory_execution_metadata.json"))
    if not files:
        print(f"No trajectory_execution_metadata.json found under {args.root}")
        return 2

    failed = 0
    for path in files:
        ok, issues = validate_metadata(path, args.object_id)
        rel = path.relative_to(args.root)
        if ok:
            print(f"[PASS] {rel}")
        else:
            failed += 1
            print(f"[FAIL] {rel}")
            for issue in issues:
                print(f"  - {issue}")

    if failed:
        print(f"\nFailed {failed}/{len(files)} metadata files.")
        return 1
    print(f"\nPassed {len(files)}/{len(files)} metadata files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

