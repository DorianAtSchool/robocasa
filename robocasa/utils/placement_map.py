"""
2D placement map rendering for grid and continuous strategies.

Reusable drawing functions consumed by ``SimToolExecutor.save_placement_map``
and ``experiments/visualize_grid.py``.
"""

from __future__ import annotations

import matplotlib.patches as patches
import numpy as np

from robocasa.utils.placement import (
    ContinuousPlacement,
    get_fixture_aabb,
    get_front_working_side_clearance,
    get_face_order,
    is_ground_obstacle,
    is_within_face_working_band,
)
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type

_FRONT_ONLY_FIXTURE_TYPES = [
    FixtureType.FRIDGE,
    FixtureType.CABINET,
    FixtureType.CABINET_WITH_DOOR,
    FixtureType.CABINET_SINGLE_DOOR,
    FixtureType.CABINET_DOUBLE_DOOR,
    FixtureType.DRAWER,
    FixtureType.TOP_DRAWER,
    FixtureType.MICROWAVE,
    FixtureType.OVEN,
    FixtureType.DISHWASHER,
]


def _draw_fixtures(ax, fixtures, label_fontsize=4):
    """Draw fixture AABBs on the axes."""
    for name, fxtr in fixtures.items():
        aabb = get_fixture_aabb(fxtr)
        if aabb is None:
            continue
        fmin, fmax = aabb

        obstacle = is_ground_obstacle(name, fxtr)
        color = "red" if obstacle else "green"
        alpha = 0.35 if obstacle else 0.15
        rect = patches.Rectangle(
            (fmin[0], fmin[1]),
            fmax[0] - fmin[0], fmax[1] - fmin[1],
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

        short = name.split("_main_")[0] if "_main_" in name else name
        cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
        ax.text(cx, cy, short, fontsize=label_fontsize, ha="center", va="center",
                color="black", clip_on=True)


def _draw_robots(ax, runner):
    """Draw robot positions."""
    for ridx, (color, label) in enumerate([("blue", "R0"), ("orange", "R1")]):
        if ridx >= runner._num_robots:
            break
        pos = runner._get_robot_position(ridx)[:2]
        ax.plot(pos[0], pos[1], "o", color=color, markersize=8, zorder=5)
        ax.text(pos[0], pos[1] + 0.15, label, fontsize=7, ha="center",
                va="bottom", color=color, fontweight="bold", zorder=5)


def draw_grid_map(ax, runner):
    """Draw the occupancy grid on axes."""
    grid = runner._occupancy_grid

    for r in range(grid._rows):
        for c in range(grid._cols):
            xy = grid._grid_to_world(r, c)
            half = grid.cell_size / 2
            color = "#444444" if grid._grid[r, c] else "#f0f0f0"
            rect = patches.Rectangle(
                (xy[0] - half, xy[1] - half),
                grid.cell_size, grid.cell_size,
                linewidth=0.3, edgecolor="#cccccc", facecolor=color,
            )
            ax.add_patch(rect)

    _draw_fixtures(ax, runner._fixtures)
    _draw_robots(ax, runner)

    x_min = grid._origin[0]
    x_max = grid._origin[0] + grid._cols * grid.cell_size
    y_min = grid._origin[1]
    y_max = grid._origin[1] + grid._rows * grid.cell_size
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Grid — {grid._rows}x{grid._cols} @ {grid.cell_size}m | "
        f"occupied={int(grid._grid.sum())}/{grid._grid.size}"
    )


def draw_continuous_map(ax, runner):
    """Draw the continuous placement candidates on axes."""
    cp = runner._continuous or ContinuousPlacement(runner._fixtures)
    grid = runner._occupancy_grid  # for bounds only

    # Light background
    x_min = grid._origin[0]
    x_max = grid._origin[0] + grid._cols * grid.cell_size
    y_min = grid._origin[1]
    y_max = grid._origin[1] + grid._rows * grid.cell_size
    ax.add_patch(patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor="#f8f8f8", edgecolor="#ccc", linewidth=0.5,
    ))

    _draw_fixtures(ax, runner._fixtures)

    def _classify_candidate(cp, pos, fixture_id, fxtr, face_key):
        """Return (color, marker, size) for a candidate position."""
        if any(fixture_is_type(fxtr, ft) for ft in _FRONT_ONLY_FIXTURE_TYPES):
            aabb = get_fixture_aabb(fxtr)
            if aabb is not None:
                fmin, fmax = aabb
                front_target = None
                if hasattr(runner, "_get_fixture_front_target_xy"):
                    front_target = runner._get_fixture_front_target_xy(fixture_id)
                front_face = get_face_order(fxtr, front_target_xy=front_target)[0]
                side_clearance = get_front_working_side_clearance(front_face, fmin, fmax)
                if face_key != front_face:
                    return "#cc3333", "x", 3
                if not is_within_face_working_band(
                    front_face,
                    pos,
                    fmin,
                    fmax,
                    side_clearance=side_clearance,
                ):
                    return "#cc3333", "x", 3
        if not cp.is_standable(pos):
            return "#cc3333", "x", 3   # red x: out of bounds or collides
        if cp._collides_with_obstacles(pos, exclude_fixture=fxtr):
            return "#cc3333", "x", 3   # red x: collides with obstacle
        if cp._is_enclosed(pos, exclude_fixture=fxtr):
            return "#cc8833", "x", 3   # orange x: enclosed / tight gap
        return "#33aa33", ".", 5        # green dot: valid

    # Draw candidate positions for ALL fixtures of key types (not just one
    # per type) so the map shows the full picture.
    target_types = [FixtureType.COUNTER, FixtureType.FRIDGE, FixtureType.SINK,
                    FixtureType.STOVE, FixtureType.CABINET,
                    FixtureType.DINING_COUNTER]
    for fid, fxtr in runner._fixtures.items():
        is_target = any(fixture_is_type(fxtr, ft) for ft in target_types)
        is_dining = "dining" in fid.lower()
        if not (is_target or is_dining):
            continue
        grouped_candidates = cp._generate_face_candidates_grouped(fxtr)
        for face_key, candidates in grouped_candidates.items():
            for pos, yaw in candidates:
                color, marker, size = _classify_candidate(cp, pos, fid, fxtr, face_key)
                ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                        zorder=3, alpha=0.7)

    _draw_robots(ax, runner)

    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Continuous — {len(cp._obstacle_aabbs)} obstacle AABBs | "
        f"green=valid, red=collides, orange=enclosed"
    )
