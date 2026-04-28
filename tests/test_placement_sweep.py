#!/usr/bin/env python
"""
Sweep test for robot placement across tasks × layouts × styles × seeds.

Runs each (task, demo_plan, layout, style, seed) combination with the chosen
placement strategy and logs per-step robot positions, invariant checks, and
optional video/frame output.

Usage:
  # Default: 2 tasks × 2 layouts, grid placement
  python tests/test_placement_sweep.py --placement grid --output tmp/sweep_grid

  # Specific tasks and layouts
  python tests/test_placement_sweep.py --tasks HotDogSetup,PrepareSandwichStation \
      --layouts 11,56 --styles 34,42 --seeds 42 --output tmp/sweep

  # Quick smoke test (1 combo)
  python tests/test_placement_sweep.py --tasks HotDogSetup --layouts 11 --styles 34 --output tmp/quick

Options:
  --placement    grid (default: grid)
  --output       Directory for results (required)
  --tasks        Comma-separated task names (default: HotDogSetup,PrepareSandwichStation)
  --layouts      Comma-separated layout ids (default: 11,56)
  --styles       Comma-separated style ids (default: 34,42)
  --seeds        Comma-separated seeds (default: 42)
  --robots       Number of robots (default: 2)
  --robot-radius Robot collision radius (default: 0.18)
  --no-video     Skip video/frame output (just run invariant checks)
  --verbose      Print robot positions after each step
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import imageio
import numpy as np

from robocasa.utils.placement import (
    MAX_FRONT_WORKING_LATERAL_OFFSET,
    get_front_alignment_metrics,
)
from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.sim_tool_executor import _is_approach_center


# ---------------------------------------------------------------------------
# Demo plan mapping: task_name -> demo_plan_name
# ---------------------------------------------------------------------------
_TASK_DEMO_PLANS = {
    "HotDogSetup": "cooperative_hotdog_setup",
    "PrepareSandwichStation": "sandwich_station",
}

_FRONT_TOOL_ARG_BY_NAME = {
    "navigate_to_fixture": "fixture_id",
    "open_hinged_part": "target_id",
    "close_hinged_part": "target_id",
    "open_sliding_part": "target_id",
    "close_sliding_part": "target_id",
    "press_button": "target_id",
    "press_lever": "target_id",
    "set_rotary_control": "target_id",
    "pick_up_object": "source_id",
}


def _get_front_check_fixture_id(tool: str | None, args: dict | None) -> str | None:
    """Return the fixture id to validate for front-alignment, if any."""
    if tool is None or args is None:
        return None
    arg_name = _FRONT_TOOL_ARG_BY_NAME.get(tool)
    if arg_name is None:
        return None
    fixture_id = args.get(arg_name)
    if isinstance(fixture_id, str):
        return fixture_id
    return None


def _check_front_alignment(executor, robot_idx: int, fixture_id: str, step_tag: str):
    """Return front-alignment violations for one robot / fixture pair."""
    runner = executor.runner
    fixture = runner._fixtures.get(fixture_id)
    if fixture is None or not _is_approach_center(fixture):
        return []

    pos = runner._get_robot_position(robot_idx)[:2]
    target_xy = runner._get_fixture_front_target_xy(fixture_id)
    metrics = get_front_alignment_metrics(fixture, pos, target_xy=target_xy)
    if metrics is None:
        return []

    violations = []
    if not bool(metrics["on_front_face"] and metrics["within_span"]):
        violations.append(
            f"[{step_tag}] robot{robot_idx} not on front face of {fixture_id} at "
            f"({pos[0]:.3f}, {pos[1]:.3f})"
        )
    if float(metrics["lateral_offset"]) > MAX_FRONT_WORKING_LATERAL_OFFSET:
        violations.append(
            f"[{step_tag}] robot{robot_idx} too far from front working line of {fixture_id}: "
            f"offset={float(metrics['lateral_offset']):.3f}m"
        )
    return violations


def _check_invariants(
    executor,
    step_tag,
    verbose=False,
    tool: str | None = None,
    args: dict | None = None,
    robot_idx: int | None = None,
):
    """Check placement invariants, return list of violations."""
    runner = executor.runner
    violations = []
    positions = {}

    for ridx in range(runner._num_robots):
        pos = runner._get_robot_position(ridx)
        positions[f"robot{ridx}"] = [round(float(pos[0]), 3),
                                      round(float(pos[1]), 3),
                                      round(float(pos[2]), 3)]

        pos2d = pos[:2]

        grid = runner._occupancy_grid
        if grid is not None and not grid.is_free_of_fixtures(pos2d):
            violations.append(
                f"[{step_tag}] robot{ridx} in fixture-occupied cell at "
                f"({pos2d[0]:.3f}, {pos2d[1]:.3f})"
            )

        # Grid bounds check
        if grid is not None:
            cell = grid._world_to_grid(pos2d)
            r, c = cell
            if r < 0 or r >= grid._rows or c < 0 or c >= grid._cols:
                violations.append(
                    f"[{step_tag}] robot{ridx} outside grid at "
                    f"({pos2d[0]:.3f}, {pos2d[1]:.3f}) -> cell ({r}, {c})"
                )

    # Robot separation
    if runner._num_robots >= 2:
        pos0 = runner._get_robot_position(0)[:2]
        pos1 = runner._get_robot_position(1)[:2]
        dist = float(np.linalg.norm(pos0 - pos1))
        if dist < 0.1:
            violations.append(
                f"[{step_tag}] robots overlapping: dist={dist:.3f}m"
            )

    if verbose:
        pos_str = ", ".join(f"r{k[-1]}=({v[0]:.2f},{v[1]:.2f})" for k, v in positions.items())
        status = "OK" if not violations else f"FAIL({len(violations)})"
        print(f"  {step_tag}: {pos_str} [{status}]")

    return violations, positions


def run_single_combo(
    task: str,
    layout: int,
    style: int,
    seed: int,
    placement: str,
    robots: int,
    robot_radius: float,
    output_dir: Path | None,
    save_video: bool,
    verbose: bool,
) -> dict:
    """Run a single task/layout/style/seed combo. Returns a result dict."""
    combo_tag = f"{task}_L{layout}_S{style}_s{seed}_{placement}"
    combo_dir = output_dir / combo_tag if output_dir else None

    result = {
        "task": task,
        "layout": layout,
        "style": style,
        "seed": seed,
        "placement": placement,
        "robots": robots,
        "steps": [],
        "violations": [],
        "success": False,
        "error": None,
    }

    demo_plan_name = _TASK_DEMO_PLANS.get(task)
    if demo_plan_name is None:
        result["error"] = f"No demo plan for task {task}"
        print(f"  SKIP {combo_tag}: {result['error']}")
        return result

    try:
        executor = SimToolExecutor(
            task_name=task,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            render_width=320 if save_video else 160,
            render_height=240 if save_video else 128,
            placement=placement,
            robot_radius=robot_radius,
        )
    except Exception as e:
        result["error"] = f"Init failed: {e}"
        print(f"  FAIL {combo_tag}: {result['error']}")
        return result

    writers = {}
    try:
        # Save placement map
        if combo_dir:
            combo_dir.mkdir(parents=True, exist_ok=True)
            executor.save_placement_map(str(combo_dir), prefix="initial")

        # Init video writers
        if save_video and combo_dir:
            initial_frames = executor.render()
            for cam, img in initial_frames.items():
                w = imageio.get_writer(combo_dir / f"{cam}.mp4", fps=2)
                w.append_data(img)
                writers[cam] = w
                # Save initial frame
                (combo_dir / "frames" / cam).mkdir(parents=True, exist_ok=True)
                imageio.imwrite(combo_dir / "frames" / cam / "initial.png", img)

        # Check initial invariants
        viols, positions = _check_invariants(executor, f"{combo_tag}_init", verbose)
        result["violations"].extend(viols)

        # Build and run plan
        plan = executor.build_demo_plan(demo_plan_name)
        plan = executor.ground_plan_template(plan)

        for step_idx, step in enumerate(plan):
            tool = step["tool"]
            ridx = step.get("robot_idx", 0)
            args = step.get("args", {})

            try:
                tool_result = executor.execute(tool, robot_idx=ridx, **args)
                step_success = tool_result.success
            except Exception as e:
                step_success = False
                result["violations"].append(
                    f"[{combo_tag}_step{step_idx}] {tool} raised: {e}"
                )

            # Check invariants after step
            step_tag = f"{combo_tag}_step{step_idx}_{tool}"
            viols, positions = _check_invariants(
                executor,
                step_tag,
                verbose,
                tool=tool,
                args=args,
                robot_idx=ridx,
            )
            result["violations"].extend(viols)

            step_info = {
                "step_index": step_idx,
                "tool": tool,
                "robot_idx": ridx,
                "args": args,
                "success": step_success,
                "robot_positions": positions,
            }
            result["steps"].append(step_info)

            # Record video frame
            if writers:
                frames = executor.render()
                for cam, img in frames.items():
                    if cam in writers:
                        writers[cam].append_data(img)
                    cam_dir = combo_dir / "frames" / cam
                    imageio.imwrite(cam_dir / f"step_{step_idx:03d}_{tool}.png", img)

        all_steps_ok = all(s["success"] for s in result["steps"])
        no_violations = len(result["violations"]) == 0
        result["success"] = all_steps_ok and no_violations

        status = "OK" if result["success"] else "FAIL"
        n_viols = len(result["violations"])
        print(f"  {status} {combo_tag} "
              f"({len(result['steps'])} steps, {n_viols} violations)")

        if result["violations"] and not verbose:
            for v in result["violations"][:5]:
                print(f"    - {v}")
            if n_viols > 5:
                print(f"    ... and {n_viols - 5} more")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR {combo_tag}: {result['error']}")
        if verbose:
            traceback.print_exc()

    finally:
        for w in writers.values():
            w.close()
        executor.close()

        # Save result JSON
        if combo_dir:
            with open(combo_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sweep test: tasks × layouts × styles × seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--placement", default="grid", choices=["grid"],
        help="Placement strategy (default: grid)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--tasks", type=str, default="HotDogSetup,PrepareSandwichStation",
        help="Comma-separated task names (default: HotDogSetup,PrepareSandwichStation)",
    )
    parser.add_argument(
        "--layouts", type=str, default="11,56",
        help="Comma-separated layout ids (default: 11,56)",
    )
    parser.add_argument(
        "--styles", type=str, default="34,42",
        help="Comma-separated style ids (default: 34,42)",
    )
    parser.add_argument(
        "--seeds", type=str, default="42",
        help="Comma-separated seeds (default: 42)",
    )
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video/frame output (faster, just check invariants)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print robot positions after each step",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    layouts = [int(x.strip()) for x in args.layouts.split(",")]
    styles = [int(x.strip()) for x in args.styles.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_video = not args.no_video

    total = len(tasks) * len(layouts) * len(styles) * len(seeds)
    print(f"Placement sweep: {len(tasks)} tasks × {len(layouts)} layouts × "
          f"{len(styles)} styles × {len(seeds)} seeds = {total} combos")
    print(f"  placement={args.placement}, robots={args.robots}, "
          f"robot_radius={args.robot_radius}")
    print(f"  output={output_dir}, video={'yes' if save_video else 'no'}")
    print()

    results = []
    combo_idx = 0

    for task in tasks:
        for layout in layouts:
            for style in styles:
                for seed in seeds:
                    combo_idx += 1
                    print(f"[{combo_idx}/{total}] {task} L{layout} S{style} s{seed}")

                    r = run_single_combo(
                        task=task,
                        layout=layout,
                        style=style,
                        seed=seed,
                        placement=args.placement,
                        robots=args.robots,
                        robot_radius=args.robot_radius,
                        output_dir=output_dir,
                        save_video=save_video,
                        verbose=args.verbose,
                    )
                    results.append(r)

    # Summary
    print()
    print("=" * 60)
    n_ok = sum(1 for r in results if r["success"])
    n_fail = sum(1 for r in results if not r["success"] and r["error"] is None)
    n_error = sum(1 for r in results if r["error"] is not None)
    print(f"RESULTS: {n_ok}/{total} passed, {n_fail} failed, {n_error} errors")

    if n_fail + n_error > 0:
        print()
        print("Failures:")
        for r in results:
            if not r["success"]:
                tag = f"{r['task']}_L{r['layout']}_S{r['style']}_s{r['seed']}"
                if r["error"]:
                    print(f"  ERROR {tag}: {r['error']}")
                else:
                    print(f"  FAIL  {tag}: {len(r['violations'])} violations")
                    for v in r["violations"][:3]:
                        print(f"        {v}")

    # Save summary
    summary = {
        "placement": args.placement,
        "robots": args.robots,
        "total": total,
        "passed": n_ok,
        "failed": n_fail,
        "errors": n_error,
        "results": results,
    }
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    sys.exit(0 if n_ok == total else 1)


if __name__ == "__main__":
    main()
