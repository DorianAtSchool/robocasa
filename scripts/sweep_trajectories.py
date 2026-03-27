#!/usr/bin/env python3
"""Sweep all trajectories in a dataset directory through the sim tool executor.

Discovers all traj_*.json files under <input_dir>/<task_name>/trajectories/,
executes each one across all (layout, style, seed) combinations, and writes
outputs: adapted trajectory, execution metadata, and images rendered by
get_image tool calls to their specified paths.

Usage:
    # Single combo (default):
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output

    # Sweep across multiple layouts, styles, and seeds:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 56 \
        --styles 34 42 \
        --seeds 42 99

    # Limit to a specific task:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --task hot_dog_setup

    # Limit to specific trajectory indices:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --indices 0 1 2

    # Dry run — show what would be executed without running anything:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 56 --styles 34 42 --seeds 42 99 \
        --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path


def discover_trajectories(
    input_dir: Path,
    task_filter: str | None = None,
    indices: list[int] | None = None,
) -> list[dict]:
    """Find all trajectory JSONs under input_dir/<task>/trajectories/."""
    entries = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_dir.name != task_filter:
            continue
        traj_dir = task_dir / "trajectories"
        if not traj_dir.is_dir():
            continue
        for traj_file in sorted(traj_dir.glob("traj_*.json")):
            traj_idx = int(traj_file.stem.split("_")[-1])
            if indices is not None and traj_idx not in indices:
                continue
            entries.append({
                "task_dir_name": task_dir.name,
                "traj_file": traj_file,
                "traj_idx": traj_idx,
            })
    return entries


def run_one(
    traj_file: Path,
    output_dir: Path,
    layout: int,
    style: int,
    seed: int,
    robots: int,
    placement: str,
    cell_size: float,
) -> dict:
    """Execute a single trajectory and return summary info."""
    from robocasa.utils.sim_tool_executor import SimToolExecutor
    from robocasa.utils.trajectory_adapter import execute_trajectory

    with open(traj_file) as f:
        trajectory = json.load(f)

    task_name = trajectory.get("composite_task", "Kitchen")

    executor = SimToolExecutor(
        task_name=task_name,
        robots=robots,
        layout=layout,
        style=style,
        seed=seed,
        placement=placement,
        cell_size=cell_size,
    )

    try:
        metadata = execute_trajectory(
            executor=executor,
            trajectory=trajectory,
            output_dir=str(output_dir),
            skip_videos=True,
        )

        # Count successes (skip get_image steps which always succeed)
        steps = metadata.get("steps", [])
        action_steps = [s for s in steps if s.get("tool") != "get_image"]
        n_success = sum(1 for s in action_steps if s.get("success"))
        n_total = len(action_steps)
        n_images = sum(1 for s in steps if s.get("tool") == "get_image")

        return {
            "status": "ok",
            "task": task_name,
            "steps_succeeded": n_success,
            "steps_total": n_total,
            "images_rendered": n_images,
        }
    finally:
        executor.close()


def main():
    parser = argparse.ArgumentParser(description="Sweep trajectories through sim executor")
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root dir")
    parser.add_argument("--task", type=str, default=None, help="Filter to a single task dir name")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Filter to specific traj indices")
    parser.add_argument("--layouts", type=int, nargs="+", default=[11], help="Kitchen layout ids (default: 11)")
    parser.add_argument("--styles", type=int, nargs="+", default=[34], help="Kitchen style ids (default: 34)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Environment seeds (default: 42)")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--placement", choices=["grid", "continuous"], default="grid")
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    entries = discover_trajectories(input_dir, task_filter=args.task, indices=args.indices)
    if not entries:
        print("No trajectories found.")
        sys.exit(1)

    combos = list(itertools.product(args.layouts, args.styles, args.seeds))
    total_runs = len(entries) * len(combos)

    print(f"Found {len(entries)} trajectories x {len(combos)} scene combos = {total_runs} runs")
    if len(combos) > 1:
        print(f"  layouts: {args.layouts}")
        print(f"  styles:  {args.styles}")
        print(f"  seeds:   {args.seeds}")
    print()

    if args.dry_run:
        for entry in entries:
            for layout, style, seed in combos:
                task_name = entry["task_dir_name"]
                traj_idx = entry["traj_idx"]
                combo_dir = f"L{layout}_S{style}_sd{seed}"
                out = output_root / task_name / f"traj_{traj_idx:06d}" / combo_dir
                print(f"  {task_name}/traj_{traj_idx:06d}/{combo_dir} -> {out}")
        print(f"\n{total_runs} runs (dry run, nothing executed)")
        return

    results = []
    run_num = 0
    for entry in entries:
        task_name = entry["task_dir_name"]
        traj_idx = entry["traj_idx"]
        traj_file = entry["traj_file"]

        for layout, style, seed in combos:
            run_num += 1

            # When only one combo, keep flat output structure for backwards compat
            if len(combos) == 1:
                traj_output_dir = output_root / task_name / f"traj_{traj_idx:06d}"
                combo_label = ""
            else:
                combo_dir = f"L{layout}_S{style}_sd{seed}"
                traj_output_dir = output_root / task_name / f"traj_{traj_idx:06d}" / combo_dir
                combo_label = f" L{layout}/S{style}/sd{seed}"

            label = f"[{run_num}/{total_runs}] {task_name}/traj_{traj_idx:06d}{combo_label}"
            print(f"{label} ... ", end="", flush=True)

            t0 = time.time()
            try:
                result = run_one(
                    traj_file=traj_file,
                    output_dir=traj_output_dir,
                    layout=layout,
                    style=style,
                    seed=seed,
                    robots=args.robots,
                    placement=args.placement,
                    cell_size=args.cell_size,
                )
                elapsed = time.time() - t0
                result["elapsed_s"] = round(elapsed, 1)
                print(
                    f"{result['steps_succeeded']}/{result['steps_total']} steps ok, "
                    f"{result['images_rendered']} images ({elapsed:.1f}s)"
                )
            except Exception as e:
                elapsed = time.time() - t0
                result = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "elapsed_s": round(elapsed, 1),
                }
                print(f"ERROR: {e} ({elapsed:.1f}s)")

            result["task_dir"] = task_name
            result["traj_idx"] = traj_idx
            result["traj_file"] = str(traj_file)
            result["layout"] = layout
            result["style"] = style
            result["seed"] = seed
            results.append(result)

    # Write sweep summary
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dir": str(input_dir),
        "layouts": args.layouts,
        "styles": args.styles,
        "seeds": args.seeds,
        "scene_combos": len(combos),
        "trajectories": len(entries),
        "total": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    with open(output_root / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Done: {summary['succeeded']}/{summary['total']} succeeded")
    print(f"Summary: {output_root / 'sweep_summary.json'}")

    if summary["failed"] > 0:
        print(f"\nFailed runs:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['task_dir']}/traj_{r['traj_idx']:06d} L{r['layout']}/S{r['style']}/sd{r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
