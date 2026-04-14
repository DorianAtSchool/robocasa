#!/usr/bin/env python3
"""Merge per-shard sweep summaries into one standard sweep summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _result_sort_key(result: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(result["task_dir"]),
        int(result["traj_idx"]),
        int(result["layout"]),
        int(result["style"]),
        int(result["seed"]),
    )


def merge_sweep_summaries(summary_paths: list[Path]) -> dict[str, Any]:
    """Combine multiple shard summaries into one deterministic summary payload."""

    if not summary_paths:
        raise ValueError("Pass at least one shard summary to merge.")

    summaries = [_load_summary(path) for path in summary_paths]
    first = summaries[0]
    shared_keys = (
        "input_dir",
        "layouts",
        "styles",
        "seeds",
        "scene_combos",
        "gl_backend",
        "gpu_ids",
        "max_tasks_per_child",
        "procs_per_gpu",
        "render_width",
        "render_height",
    )

    for path, summary in zip(summary_paths[1:], summaries[1:]):
        for key in shared_keys:
            if summary.get(key) != first.get(key):
                raise ValueError(
                    f"Cannot merge {path}: key {key!r} differs across shard summaries."
                )

    merged_results: list[dict[str, Any]] = []
    seen_result_keys: set[tuple[str, int, int, int, int]] = set()
    trajectories = 0
    total = 0
    succeeded = 0
    failed = 0
    shard_counts = {
        int(summary["num_shards"])
        for summary in summaries
        if summary.get("num_shards") is not None
    }

    for path, summary in zip(summary_paths, summaries):
        trajectories += int(summary.get("trajectories", 0))
        total += int(summary.get("total", 0))
        succeeded += int(summary.get("succeeded", 0))
        failed += int(summary.get("failed", 0))
        for result in summary.get("results", []):
            result_key = _result_sort_key(result)
            if result_key in seen_result_keys:
                raise ValueError(
                    f"Cannot merge {path}: duplicate result for {result_key!r}."
                )
            seen_result_keys.add(result_key)
            merged_results.append(result)

    if len(shard_counts) > 1:
        raise ValueError("Shard summaries disagree on num_shards.")

    merged = {key: first.get(key) for key in shared_keys}
    merged.update(
        {
            "trajectories": trajectories,
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "num_shards": shard_counts.pop() if shard_counts else 1,
            "shard_index": None,
            "results": sorted(merged_results, key=_result_sort_key),
            "merged_from_summaries": [str(path) for path in summary_paths],
        }
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-shard sweep_summary.json files into one summary."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the merged sweep summary JSON.",
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        help="Per-shard summary JSON paths to merge.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    summary_paths = [Path(path) for path in args.summary_paths]
    merged_summary = merge_sweep_summaries(summary_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_summary, f, indent=2)


if __name__ == "__main__":
    main()
