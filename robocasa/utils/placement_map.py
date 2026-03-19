"""
2D placement map rendering for grid and continuous strategies.

Reusable drawing functions consumed by ``SimToolExecutor.save_placement_map``
and ``experiments/visualize_grid.py``.
"""

from __future__ import annotations

import matplotlib.patches as patches
import numpy as np
import textwrap

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


def _format_fixture_label(name: str, wrap_width: int = 18) -> str:
    """Render a readable fixture label from an internal fixture id."""
    label = " ".join(str(name).replace("_", " ").split())
    if len(label) <= wrap_width:
        return label
    return textwrap.fill(
        label,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _aabb_overlaps(box_a, box_b, margin: float = 0.0) -> bool:
    """Return whether two axis-aligned boxes overlap."""
    return not (
        box_a[2] + margin < box_b[0]
        or box_b[2] + margin < box_a[0]
        or box_a[3] + margin < box_b[1]
        or box_b[3] + margin < box_a[1]
    )


def _estimate_label_extent(label: str, label_fontsize: int) -> tuple[float, float]:
    """Approximate label size in world coordinates for collision avoidance."""
    lines = label.splitlines() or [label]
    font_scale = max(label_fontsize / 6.0, 0.75)
    width = max(len(line) for line in lines) * 0.028 * font_scale + 0.08
    height = len(lines) * 0.11 * font_scale + 0.04
    return width, height


def _label_candidate_positions(fmin, fmax):
    """Return candidate anchor points for a fixture label."""
    cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
    dx = max((fmax[0] - fmin[0]) * 0.5 + 0.12, 0.18)
    dy = max((fmax[1] - fmin[1]) * 0.5 + 0.12, 0.18)
    return [
        (cx, fmax[1] + dy),
        (fmax[0] + dx, cy),
        (fmin[0] - dx, cy),
        (cx, fmin[1] - dy),
        (fmax[0] + dx, fmax[1] + dy),
        (fmin[0] - dx, fmax[1] + dy),
        (fmax[0] + dx, fmin[1] - dy),
        (fmin[0] - dx, fmin[1] - dy),
        (cx, cy),
    ]


def _pick_label_position(
    fixture_id: str,
    fmin,
    fmax,
    label: str,
    placed_label_boxes,
    fixture_boxes,
    label_fontsize: int,
):
    """Pick a label anchor with minimal overlap against other labels and fixtures."""
    cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
    label_width, label_height = _estimate_label_extent(label, label_fontsize)
    best_position = (cx, cy)
    best_box = (
        cx - label_width / 2,
        cy - label_height / 2,
        cx + label_width / 2,
        cy + label_height / 2,
    )
    best_score = float("inf")

    for x, y in _label_candidate_positions(fmin, fmax):
        candidate_box = (
            x - label_width / 2,
            y - label_height / 2,
            x + label_width / 2,
            y + label_height / 2,
        )
        label_overlap_count = sum(
            _aabb_overlaps(candidate_box, other_box, margin=0.03)
            for other_box in placed_label_boxes
        )
        fixture_overlap_count = sum(
            other_fixture_id != fixture_id
            and _aabb_overlaps(candidate_box, other_box, margin=0.02)
            for other_fixture_id, other_box in fixture_boxes
        )
        distance_penalty = float(np.linalg.norm(np.array([x - cx, y - cy])))
        center_penalty = 0.2 if np.allclose([x, y], [cx, cy]) else 0.0
        score = (
            100.0 * label_overlap_count
            + 12.0 * fixture_overlap_count
            + distance_penalty
            + center_penalty
        )
        if score < best_score:
            best_score = score
            best_position = (x, y)
            best_box = candidate_box

    return best_position, best_box


def _draw_fixtures(ax, fixtures, label_fontsize=5):
    """Draw fixture AABBs on the axes and return placed label boxes."""
    fixture_entries = []
    for name, fxtr in fixtures.items():
        aabb = get_fixture_aabb(fxtr)
        if aabb is None:
            continue
        fixture_entries.append((name, fxtr, *aabb))

    fixture_boxes = [(name, (fmin[0], fmin[1], fmax[0], fmax[1])) for name, _, fmin, fmax in fixture_entries]
    placed_label_boxes = []

    for name, fxtr, fmin, fmax in sorted(
        fixture_entries,
        key=lambda entry: (
            -float((entry[3][0] - entry[2][0]) * (entry[3][1] - entry[2][1])),
            entry[0],
        ),
    ):

        obstacle = is_ground_obstacle(name, fxtr)
        color = "red" if obstacle else "green"
        alpha = 0.35 if obstacle else 0.15
        rect = patches.Rectangle(
            (fmin[0], fmin[1]),
            fmax[0] - fmin[0], fmax[1] - fmin[1],
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

        label = _format_fixture_label(name)
        cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
        (label_x, label_y), label_box = _pick_label_position(
            name,
            fmin,
            fmax,
            label,
            placed_label_boxes,
            fixture_boxes,
            label_fontsize,
        )
        if not np.allclose([label_x, label_y], [cx, cy]):
            ax.plot(
                [cx, label_x],
                [cy, label_y],
                color="#444444",
                linewidth=0.4,
                alpha=0.45,
                zorder=4,
            )
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=label_fontsize,
            ha="center",
            va="center",
            color="black",
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#777777",
                "linewidth": 0.35,
                "alpha": 0.85,
            },
            zorder=5,
        )
        placed_label_boxes.append(label_box)

    return placed_label_boxes


def _draw_robots(ax, runner, placed_label_boxes=None):
    """Draw robot positions with readable agent labels."""
    if placed_label_boxes is None:
        placed_label_boxes = []

    colors = ["blue", "orange", "purple", "brown"]
    label_offsets = [
        np.array([0.18, 0.18]),
        np.array([0.18, -0.18]),
        np.array([-0.18, 0.18]),
        np.array([-0.18, -0.18]),
    ]
    for ridx, color in enumerate(colors):
        if ridx >= runner._num_robots:
            break
        pos = runner._get_robot_position(ridx)[:2]
        ax.plot(pos[0], pos[1], "o", color=color, markersize=8, zorder=5)
        label = f"agent {ridx}"
        label_width, label_height = _estimate_label_extent(label, label_fontsize=7)
        best_position = pos + label_offsets[ridx % len(label_offsets)]
        best_box = (
            best_position[0] - label_width / 2,
            best_position[1] - label_height / 2,
            best_position[0] + label_width / 2,
            best_position[1] + label_height / 2,
        )
        best_score = float("inf")
        for offset in label_offsets:
            candidate = pos + offset
            candidate_box = (
                candidate[0] - label_width / 2,
                candidate[1] - label_height / 2,
                candidate[0] + label_width / 2,
                candidate[1] + label_height / 2,
            )
            overlap_count = sum(
                _aabb_overlaps(candidate_box, other_box, margin=0.03)
                for other_box in placed_label_boxes
            )
            score = 100.0 * overlap_count + float(np.linalg.norm(offset))
            if score < best_score:
                best_score = score
                best_position = candidate
                best_box = candidate_box

        ax.text(
            best_position[0],
            best_position[1],
            label,
            fontsize=7,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.9,
            },
            zorder=6,
        )
        placed_label_boxes.append(best_box)


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

    placed_label_boxes = _draw_fixtures(ax, runner._fixtures)
    _draw_robots(ax, runner, placed_label_boxes=placed_label_boxes)

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

    placed_label_boxes = _draw_fixtures(ax, runner._fixtures)

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

    _draw_robots(ax, runner, placed_label_boxes=placed_label_boxes)

    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Continuous — {len(cp._obstacle_aabbs)} obstacle AABBs | "
        f"green=valid, red=collides, orange=enclosed"
    )
