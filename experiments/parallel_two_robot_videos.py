#!/usr/bin/env python3
"""
Run N two-robot simulations in parallel and save videos for each run.

Each simulation writes:
  - room_view.mp4
  - robot0_view.mp4
  - robot1_view.mp4
  - robot0_wrist.mp4
  - robot1_wrist.mp4

under:
  <output-root>/sim_000/
  <output-root>/sim_001/
  ...
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sims", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-root", type=str, default="videos")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Per-run seed = seed-start + run index",
    )
    parser.add_argument(
        "--show-walls",
        action="store_true",
        help="Keep enclosing walls opaque",
    )
    parser.add_argument(
        "--gl",
        type=str,
        default="osmesa",
        choices=["osmesa", "egl", "glfw"],
    )
    return parser.parse_args()


def build_cmd(script_path: Path, run_output_dir: Path, args, run_idx: int):
    cmd = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(run_output_dir),
        "--steps",
        str(args.steps),
        "--fps",
        str(args.fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--layout",
        str(args.layout),
        "--style",
        str(args.style),
        "--seed",
        str(args.seed_start + run_idx),
        "--gl",
        args.gl,
    ]
    if args.show_walls:
        cmd.append("--show-walls")
    return cmd


def run_one(script_path: Path, output_root: Path, args, run_idx: int):
    run_dir = output_root / f"sim_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(script_path, run_dir, args, run_idx)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return run_idx, run_dir, proc.returncode, proc.stdout


def main():
    args = parse_args()
    if args.num_sims <= 0:
        raise ValueError("--num-sims must be > 0")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be > 0")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save each batch into its own timestamped directory to avoid overwrites.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = output_root / f"batch_{timestamp}"
    suffix = 1
    while batch_root.exists():
        batch_root = output_root / f"batch_{timestamp}_{suffix}"
        suffix += 1
    batch_root.mkdir(parents=True, exist_ok=False)

    script_path = Path(__file__).with_name("two_robot_video_sample.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    num_workers = min(args.max_workers, args.num_sims)
    print(
        f"Launching {args.num_sims} simulations with {num_workers} workers. "
        f"Batch output root: {batch_root}"
    )

    failures = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(run_one, script_path, batch_root, args, run_idx)
            for run_idx in range(args.num_sims)
        ]
        for fut in as_completed(futures):
            run_idx, run_dir, code, out = fut.result()
            if code == 0:
                print(f"[OK] sim_{run_idx:03d} -> {run_dir}")
            else:
                failures.append((run_idx, run_dir, code, out))
                print(f"[FAIL] sim_{run_idx:03d} (exit {code}) -> {run_dir}")

    if failures:
        print("\nFailed runs:")
        for run_idx, run_dir, code, out in failures:
            print(f"- sim_{run_idx:03d} (exit {code}) in {run_dir}")
            print(out)
        raise SystemExit(1)

    print("\nAll simulations finished successfully.")
    print("Per simulation, videos are:")
    print("- room_view.mp4")
    print("- robot0_view.mp4")
    print("- robot1_view.mp4")
    print("- robot0_wrist.mp4")
    print("- robot1_wrist.mp4")
    print(f"Saved under: {batch_root}")


if __name__ == "__main__":
    main()
