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

    # Limit to specific task dirs:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --tasks hot_dog_setup prepare_coffee

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

    # Balanced ~1k trajectory sweep from 24 base trajectories:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output_1k \
        --layouts 11 42 56 \
        --styles 34 42 \
        --seeds 1 2 3 4 5 6 7

    # Sweep and push the flattened dataset to Hugging Face Hub:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output \
        --layouts 11 42 56 \
        --styles 34 42 \
        --seeds 1 2 3 4 5 6 7 \
        --push-to-hub DorianAtSchool/robocasa-trajectories-single

    # Write one row per trajectory instead of one row per step:
    python scripts/sweep_trajectories.py \
        --input-dir data_generation/task_level/data/image/20260324T031125Z \
        --output-dir tmp/sweep_output_traj \
        --row-granularity trajectory
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path


CLI_EPILOG = textwrap.dedent(
    """\
    Examples:
      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output

      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output \\
        --tasks hot_dog_setup prepare_coffee \\
        --layouts 11 42 56 \\
        --styles 34 42 \\
        --seeds 1 2 3 4 5 6 7

      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output_1k \\
        --layouts 11 42 56 \\
        --styles 34 42 \\
        --seeds 1 2 3 4 5 6 7 \\
        --push-to-hub DorianAtSchool/robocasa-trajectories-single

      python scripts/sweep_trajectories.py \\
        --input-dir data_generation/task_level/data/image/20260324T031125Z \\
        --output-dir tmp/sweep_output_traj \\
        --row-granularity trajectory
    """
)


def discover_trajectories(
    input_dir: Path,
    task_filter: list[str] | None = None,
    indices: list[int] | None = None,
) -> list[dict]:
    """Find all trajectory JSONs under input_dir/<task>/trajectories/."""
    entries = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_dir.name not in task_filter:
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
    robot_spawn: str = "sim",
    skip_videos: bool = True,
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
        robot_spawn=robot_spawn,
    )

    try:
        # Copy original trajectory JSON to output dir
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traj_file, output_dir / "original_trajectory.json")

        metadata = execute_trajectory(
            executor=executor,
            trajectory=trajectory,
            output_dir=str(output_dir),
            skip_videos=skip_videos,
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


def _load_sweep_summary(output_root: Path) -> dict:
    summary_path = output_root / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No sweep_summary.json in {output_root}")
    with open(summary_path) as f:
        return json.load(f)


def _iter_completed_runs(output_root: Path):
    summary = _load_sweep_summary(output_root)
    combo_count = summary.get("scene_combos", 1)

    for run_result in summary["results"]:
        if run_result.get("status") != "ok":
            continue

        task_dir = run_result["task_dir"]
        traj_idx = run_result["traj_idx"]
        layout = run_result["layout"]
        style = run_result["style"]
        seed = run_result["seed"]

        if combo_count == 1:
            run_dir = output_root / task_dir / f"traj_{traj_idx:06d}"
        else:
            run_dir = output_root / task_dir / f"traj_{traj_idx:06d}" / f"L{layout}_S{style}_sd{seed}"

        meta_path = run_dir / "trajectory_execution_metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        episode_id = f"{task_dir}/traj_{traj_idx:06d}/L{layout}_S{style}_sd{seed}"
        task_name = metadata.get("composite_task") or metadata.get("task") or task_dir
        rel_run_dir = run_dir.relative_to(output_root)

        yield {
            "episode_id": episode_id,
            "task": task_name,
            "task_dir": task_dir,
            "traj_idx": traj_idx,
            "layout": layout,
            "style": style,
            "seed": seed,
            "run_dir": run_dir,
            "run_dir_rel": str(rel_run_dir),
            "metadata": metadata,
            "adapted_trajectory_path": str(rel_run_dir / "adapted_trajectory.json"),
            "original_trajectory_path": str(rel_run_dir / "original_trajectory.json"),
            "execution_metadata_path": str(rel_run_dir / "trajectory_execution_metadata.json"),
        }


def iter_sweep_sidecar_paths(output_root: Path):
    """Yield repo-relative JSON artifact paths that should accompany the dataset."""
    seen = {Path("sweep_summary.json")}
    yield output_root / "sweep_summary.json", "sweep_summary.json"

    for run in _iter_completed_runs(output_root):
        for key in (
            "adapted_trajectory_path",
            "original_trajectory_path",
            "execution_metadata_path",
        ):
            rel_path = Path(run[key])
            if rel_path in seen:
                continue
            abs_path = output_root / rel_path
            if not abs_path.exists():
                continue
            seen.add(rel_path)
            yield abs_path, str(rel_path)


def upload_sweep_sidecars(repo_id: str, output_root: Path) -> None:
    """Upload referenced episode JSON files alongside the parquet dataset."""
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        allow_patterns=[
            "sweep_summary.json",
            "**/adapted_trajectory.json",
            "**/original_trajectory.json",
            "**/trajectory_execution_metadata.json",
        ],
        commit_message="Upload sweep metadata sidecars",
    )


def _resolve_step_images(image_paths: list[str] | None, image_columns: list[str]) -> dict[str, str | None]:
    images = {col: None for col in image_columns}
    for img_path_str in image_paths or []:
        img_path = Path(img_path_str)
        if not img_path.exists() and img_path.suffix == ".png":
            img_path = img_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        fname = img_path.stem
        for view_token in image_columns:
            if f"_{view_token}_" in f"_{fname}_":
                images[view_token] = str(img_path)
                break
    return images


def _read_compact_json(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path) as f:
        return json.dumps(json.load(f), separators=(",", ":"))


def _build_step_level_dataset(output_root: Path) -> "datasets.Dataset":
    """Convert sweep output directory into a flat step-level dataset."""
    from datasets import Dataset, Features, Value, Image as HFImage

    IMAGE_COLUMNS = [
        "room_view", "top_view", "map",
        "agentview_center", "agentview_left", "agentview_right", "wrist",
    ]

    rows: list[dict] = []
    for run in _iter_completed_runs(output_root):
        metadata = run["metadata"]
        num_steps = len(metadata.get("steps", []))

        for step in metadata.get("steps", []):
            step_idx = step.get("step_index", 0)
            tool = step.get("tool", "")
            robot_idx = step.get("robot_idx", 0)
            args = step.get("args", {})
            success = step.get("success", False)
            images = _resolve_step_images(step.get("image_paths"), IMAGE_COLUMNS)

            args_clean = {k: v for k, v in args.items() if k != "image_paths"}

            rows.append({
                "episode_id": run["episode_id"],
                "task": run["task"],
                "task_dir": run["task_dir"],
                "layout": run["layout"],
                "style": run["style"],
                "seed": run["seed"],
                "num_steps": num_steps,
                "run_dir": run["run_dir_rel"],
                "adapted_trajectory_path": run["adapted_trajectory_path"],
                "original_trajectory_path": run["original_trajectory_path"],
                "execution_metadata_path": run["execution_metadata_path"],
                "step_index": step_idx,
                "tool_name": tool,
                "tool_args": json.dumps(args_clean, separators=(",", ":")),
                "robot_idx": robot_idx,
                "success": success,
                **images,
            })

    features = Features({
        "episode_id": Value("string"),
        "task": Value("string"),
        "task_dir": Value("string"),
        "layout": Value("int32"),
        "style": Value("int32"),
        "seed": Value("int32"),
        "num_steps": Value("int32"),
        "run_dir": Value("string"),
        "adapted_trajectory_path": Value("string"),
        "original_trajectory_path": Value("string"),
        "execution_metadata_path": Value("string"),
        "step_index": Value("int32"),
        "tool_name": Value("string"),
        "tool_args": Value("string"),
        "robot_idx": Value("int32"),
        "success": Value("bool"),
        **{col: HFImage() for col in IMAGE_COLUMNS},
    })

    ds = Dataset.from_list(rows, features=features)
    print(f"Built dataset: {len(ds)} step rows across {len(set(ds['episode_id']))} episodes")
    return ds


def _build_trajectory_level_dataset(output_root: Path) -> "datasets.Dataset":
    """Convert sweep output directory into a trajectory-level dataset."""
    from datasets import Dataset, Features, Sequence, Value, Image as HFImage

    IMAGE_COLUMNS = [
        "room_view", "top_view", "map",
        "agentview_center", "agentview_left", "agentview_right", "wrist",
    ]

    rows: list[dict] = []
    for run in _iter_completed_runs(output_root):
        metadata = run["metadata"]
        row = {
            "episode_id": run["episode_id"],
            "task": run["task"],
            "task_dir": run["task_dir"],
            "layout": run["layout"],
            "style": run["style"],
            "seed": run["seed"],
            "num_steps": len(metadata.get("steps", [])),
            "run_dir": run["run_dir_rel"],
            "adapted_trajectory": _read_compact_json(run["run_dir"] / "adapted_trajectory.json"),
            "original_trajectory": _read_compact_json(run["run_dir"] / "original_trajectory.json"),
            "execution_metadata": json.dumps(metadata, separators=(",", ":")),
            "step_index": [],
            "tool_name": [],
            "tool_args": [],
            "robot_idx": [],
            "success": [],
            **{col: [] for col in IMAGE_COLUMNS},
        }

        for step in metadata.get("steps", []):
            images = _resolve_step_images(step.get("image_paths"), IMAGE_COLUMNS)
            args_clean = {k: v for k, v in (step.get("args") or {}).items() if k != "image_paths"}

            row["step_index"].append(step.get("step_index", 0))
            row["tool_name"].append(step.get("tool", ""))
            row["tool_args"].append(json.dumps(args_clean, separators=(",", ":")))
            row["robot_idx"].append(step.get("robot_idx", 0))
            row["success"].append(step.get("success", False))
            for col in IMAGE_COLUMNS:
                row[col].append(images[col])

        rows.append(row)

    features = Features({
        "episode_id": Value("string"),
        "task": Value("string"),
        "task_dir": Value("string"),
        "layout": Value("int32"),
        "style": Value("int32"),
        "seed": Value("int32"),
        "num_steps": Value("int32"),
        "run_dir": Value("string"),
        "adapted_trajectory": Value("large_string"),
        "original_trajectory": Value("large_string"),
        "execution_metadata": Value("large_string"),
        "step_index": Sequence(Value("int32")),
        "tool_name": Sequence(Value("string")),
        "tool_args": Sequence(Value("string")),
        "robot_idx": Sequence(Value("int32")),
        "success": Sequence(Value("bool")),
        **{col: Sequence(HFImage()) for col in IMAGE_COLUMNS},
    })

    ds = Dataset.from_list(rows, features=features)
    print(f"Built dataset: {len(ds)} trajectory rows")
    return ds


def sweep_output_to_dataset(
    output_root: Path,
    *,
    row_granularity: str = "step",
) -> "datasets.Dataset":
    """Convert sweep output into a dataset with configurable row granularity."""
    if row_granularity == "step":
        return _build_step_level_dataset(output_root)
    if row_granularity == "trajectory":
        return _build_trajectory_level_dataset(output_root)
    raise ValueError(f"Unsupported row granularity: {row_granularity}")


def build_dataset_card(
    repo_id: str,
    ds: "datasets.Dataset",
    *,
    row_granularity: str = "step",
) -> str:
    """Build a readable HuggingFace dataset card."""
    tasks = sorted(set(ds["task"]))
    episode_ids = ds["episode_id"]
    episodes = len(set(episode_ids))
    if row_granularity == "step":
        avg_steps = len(ds) / max(episodes, 1)
        intro = "This dataset contains flat RoboCasa step rows with sidecar episode JSON."
        row_text = "Each row is one tool step."
        episode_json_text = textwrap.dedent(
            """\
            Episode-level JSON is not duplicated into parquet. Instead, each row
            carries repo-relative references:

            - `adapted_trajectory_path`
            - `original_trajectory_path`
            - `execution_metadata_path`
            """
        ).strip()
        notes_tail = "- Episode JSON sidecars are available in the repo files at the paths referenced by `*_path` columns."
    else:
        avg_steps = sum(ds["num_steps"]) / max(len(ds), 1)
        intro = "This dataset contains one row per RoboCasa trajectory / episode."
        row_text = "Each row is one trajectory / episode."
        episode_json_text = textwrap.dedent(
            """\
            Episode-level JSON is stored inline:

            - `adapted_trajectory`
            - `original_trajectory`
            - `execution_metadata`

            Step-level data is stored in aligned sequence columns:

            - `step_index`
            - `tool_name`
            - `tool_args`
            - `robot_idx`
            - `success`
            """
        ).strip()
        notes_tail = "- This layout is self-contained under `load_dataset()`, but nested sequence columns are less friendly for the HF table viewer."
    task_list = ", ".join(tasks) if tasks else "Unknown"
    return textwrap.dedent(
        f"""\
        ---
        pretty_name: RoboCasa Trajectories Single
        configs:
        - config_name: default
          data_files:
          - split: train
            path: data/train-*
        ---

        # RoboCasa Trajectories Single

        {intro}

        ## Structure

        {row_text}

        {episode_json_text}

        Image columns stay inline and viewable in the dataset table:

        - `room_view`
        - `top_view`
        - `map`
        - `agentview_center`
        - `agentview_left`
        - `agentview_right`
        - `wrist`

        ## Summary

        - rows: {len(ds)}
        - episodes: {episodes}
        - tasks: {task_list}
        - average steps per episode: {avg_steps:.1f}

        ## Load

        ```python
        from datasets import load_dataset

        ds = load_dataset("{repo_id}", split="train")
        ```

        ## Notes

        - Camera renders are stored as JPEG. Maps remain PNG.
        - MP4 videos are not included in the dataset.
        - Debug `initial` / `pre_initial_state` camera frames are not part of the dataset.
        - Row granularity: `{row_granularity}`.
        {notes_tail}
        """
    )


def upload_dataset_card(
    repo_id: str,
    ds: "datasets.Dataset",
    *,
    row_granularity: str = "step",
) -> None:
    """Overwrite the auto-generated Hub README with a readable dataset card."""
    from io import BytesIO

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=BytesIO(build_dataset_card(repo_id, ds, row_granularity=row_granularity).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep trajectories through the sim executor and optionally publish the dataset.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root dir")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Filter to specific task dir names")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Filter to specific traj indices")
    parser.add_argument("--layouts", type=int, nargs="+", default=[11], help="Kitchen layout ids (default: 11)")
    parser.add_argument("--styles", type=int, nargs="+", default=[34], help="Kitchen style ids (default: 34)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Environment seeds (default: 42)")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--placement", choices=["grid", "continuous"], default="grid")
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument(
        "--robot-spawn",
        choices=["sim", "trajectory"],
        default="trajectory",
        help=(
            "Robot initial placement source. 'sim': all robots at "
            "init_robot_base_ref. 'trajectory' (default): each robot at its trajectory location."
        ),
    )
    parser.add_argument("--videos", action="store_true", help="Record per-camera MP4 videos for each run")
    parser.add_argument(
        "--row-granularity",
        choices=["step", "trajectory"],
        default="step",
        help="Dataset row shape when exporting or pushing (default: step)",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None, metavar="REPO_ID",
        help="Push dataset to HuggingFace Hub (e.g. 'username/robocasa-trajectories')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    entries = discover_trajectories(input_dir, task_filter=args.tasks, indices=args.indices)
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
                    robot_spawn=args.robot_spawn,
                    skip_videos=not args.videos,
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

    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print(f"\nConverting sweep output to HuggingFace dataset...")
        ds = sweep_output_to_dataset(output_root, row_granularity=args.row_granularity)
        print(f"Pushing to {args.push_to_hub}...")
        ds.push_to_hub(args.push_to_hub)
        if args.row_granularity == "step":
            print("Uploading sweep metadata sidecars...")
            upload_sweep_sidecars(args.push_to_hub, output_root)
        print("Uploading dataset card...")
        upload_dataset_card(args.push_to_hub, ds, row_granularity=args.row_granularity)
        print(f"Done! Dataset pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
