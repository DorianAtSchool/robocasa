#!/usr/bin/env python3
"""
Render placement maps: grid-based and continuous side by side.

Usage:
    python experiments/visualize_grid.py
    python experiments/visualize_grid.py --layout 56 --style 42 --seed 42
    python experiments/visualize_grid.py --mode grid        # grid only
    python experiments/visualize_grid.py --mode continuous   # continuous only
    python experiments/visualize_grid.py --mode both         # side by side (default)
"""

import argparse
import os

import platform
if platform.system() != "Darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robocasa.utils.trajectory_runner import TrajectoryRunner
from robocasa.utils.placement_map import draw_grid_map, draw_continuous_map


def visualize(layout: int, style: int, seed: int, output: str, mode: str):
    runner = TrajectoryRunner(
        task_name="HotDogSetup", robots=2,
        layout=layout, style=style, seed=seed,
        render_width=160, render_height=128,
        placement="continuous",
    )

    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        draw_grid_map(ax1, runner)
        draw_continuous_map(ax2, runner)
        fig.suptitle(
            f"Placement Comparison — layout={layout}, style={style}, seed={seed}",
            fontsize=14,
        )
    elif mode == "grid":
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        draw_grid_map(ax, runner)
        fig.suptitle(f"Grid — layout={layout}, style={style}, seed={seed}", fontsize=14)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        draw_continuous_map(ax, runner)
        fig.suptitle(f"Continuous — layout={layout}, style={style}, seed={seed}", fontsize=14)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved to {output}")
    runner.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize placement strategies")
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["grid", "continuous", "both"], default="both")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"tmp/placement_{args.mode}_L{args.layout}_S{args.style}_s{args.seed}.png"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    visualize(args.layout, args.style, args.seed, args.output, args.mode)
