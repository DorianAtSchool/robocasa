"""
2D placement map rendering for grid and continuous strategies.

Reusable drawing functions consumed by ``SimToolExecutor.save_placement_map``
and ``experiments/visualize_grid.py``.

Object rendering – current approach: **colored labels** (option 1).
Each object gets a colored text label placed at its position, using the
same overlap-avoidance logic as fixture labels.
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


def _format_fixture_label(name: str, clean: bool = True, wrap_width: int = 18) -> str:
    """Render a readable fixture label from an internal fixture id.

    When *clean* is True, strips layout-internal suffixes (``_group``,
    ``_main``) and collapses repeated segments while **keeping** positional
    tags (``_left``, ``_right``, ``_center``) for spatial context.

    When *clean* is False, returns the full raw fixture id (only wrapped).

    Examples (clean=True)::

        coffee_machine_left_group  → coffee_machine_left
        counter_1_left_group       → counter_1_left
        cab_1_main_group           → cab_1
        dining_dining_group        → dining
        fridge_right_group         → fridge_right
    """
    import re
    label = str(name).strip()
    if clean:
        # Strip trailing _group (layout-internal grouping suffix)
        label = re.sub(r"_group$", "", label)
        # Strip _main (generic grouping tag, not spatially meaningful)
        label = re.sub(r"_main$", "", label)
        # Collapse repeated segments (e.g. dining_dining → dining)
        parts = label.split("_")
        deduped = [parts[0]]
        for p in parts[1:]:
            if p != deduped[-1]:
                deduped.append(p)
        label = "_".join(deduped)
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


def _aabb_overlap_area(box_a, box_b, margin: float = 0.0) -> float:
    """Return the overlap area between two AABBs (with optional margin)."""
    ox = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + margin)
    oy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + margin)
    return ox * oy


def _estimate_label_extent(label: str, label_fontsize: int) -> tuple[float, float]:
    """Approximate label size in world coordinates for collision avoidance."""
    lines = label.splitlines() or [label]
    font_scale = max(label_fontsize / 6.0, 0.75)
    width = max(len(line) for line in lines) * 0.028 * font_scale + 0.08
    height = len(lines) * 0.11 * font_scale + 0.04
    return width, height


def _label_candidate_positions(fmin, fmax):
    """Return candidate anchor points for a fixture label (two rings + center)."""
    cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
    candidates = [(cx, cy)]
    for scale in (1.0, 2.0):
        dx = max((fmax[0] - fmin[0]) * 0.5 + 0.12 * scale, 0.18 * scale)
        dy = max((fmax[1] - fmin[1]) * 0.5 + 0.12 * scale, 0.18 * scale)
        candidates.extend([
            (cx, fmax[1] + dy),
            (fmax[0] + dx, cy),
            (fmin[0] - dx, cy),
            (cx, fmin[1] - dy),
            (fmax[0] + dx, fmax[1] + dy),
            (fmin[0] - dx, fmax[1] + dy),
            (fmax[0] + dx, fmin[1] - dy),
            (fmin[0] - dx, fmin[1] - dy),
        ])
    return candidates


def _pick_label_position(
    fmin,
    fmax,
    label: str,
    placed_label_boxes,
    label_fontsize: int,
):
    """Pick a label anchor with minimal overlap against other labels."""
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
        label_overlap_area = sum(
            _aabb_overlap_area(candidate_box, other_box, margin=0.03)
            for other_box in placed_label_boxes
        )
        distance_penalty = float(np.linalg.norm(np.array([x - cx, y - cy])))
        score = (
            500.0 * label_overlap_area
            + distance_penalty
        )
        if score < best_score:
            best_score = score
            best_position = (x, y)
            best_box = candidate_box

    return best_position, best_box


def _should_skip_fixture_label(name: str) -> bool:
    """Return True for fixture labels that should not be rendered."""
    name_lower = str(name).lower()
    if "floor" in name_lower or "wall" in name_lower or name_lower.startswith("stack_"):
        return True

    # Skip counter stack helper labels while keeping regular counter labels.
    if "counter" in name_lower and "stack" in name_lower:
        return True

    return False


def _draw_fixtures(ax, fixtures, label_fontsize=7, clean_labels=True):
    """Draw fixture AABBs on the axes and return placed label boxes."""
    fixture_entries = []
    for name, fxtr in fixtures.items():
        aabb = get_fixture_aabb(fxtr)
        if aabb is None:
            continue
        fixture_entries.append((name, fxtr, *aabb))

    # Draw all fixture rectangles first.
    for name, fxtr, fmin, fmax in fixture_entries:
        obstacle = is_ground_obstacle(name, fxtr)
        color = "red" if obstacle else "green"
        alpha = 0.35 if obstacle else 0.15
        rect = patches.Rectangle(
            (fmin[0], fmin[1]),
            fmax[0] - fmin[0], fmax[1] - fmin[1],
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

    # Filter to only fixtures that need labels, then place labels.
    label_entries = [e for e in fixture_entries if not _should_skip_fixture_label(e[0])]
    placed_label_boxes = []

    for name, fxtr, fmin, fmax in sorted(
        label_entries,
        key=lambda entry: (
            -float((entry[3][0] - entry[2][0]) * (entry[3][1] - entry[2][1])),
            entry[0],
        ),
    ):

        label = _format_fixture_label(name, clean=clean_labels)
        cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
        (label_x, label_y), label_box = _pick_label_position(
            fmin,
            fmax,
            label,
            placed_label_boxes,
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


def _get_object_positions(runner):
    """Return list of (display_name, x, y) for all task objects in the scene.

    Objects live in ``env.objects`` / ``env.obj_body_id``; their world
    positions come from the MuJoCo simulation state.  Display names use the
    object type (e.g. "mug") from ep_meta instead of the raw env key
    (e.g. "obj").
    """
    entries = []
    env = runner.env
    if not hasattr(env, "objects") or not env.objects:
        return entries

    # Build env_key → human-readable type from ep_meta object_cfgs
    obj_type_map = {}
    if hasattr(env, "get_ep_meta"):
        for cfg in env.get_ep_meta().get("object_cfgs", []):
            name = cfg.get("name", "")
            cat = (cfg.get("info") or {}).get("cat", "")
            if name and cat:
                obj_type_map[name] = cat

    for obj_name in env.objects:
        body_id = env.obj_body_id.get(obj_name)
        if body_id is None:
            continue
        pos = env.sim.data.body_xpos[body_id]
        display_name = obj_type_map.get(obj_name, obj_name)
        entries.append((display_name, float(pos[0]), float(pos[1])))
    return entries


_OBJECT_COLORS = [
    "#dd55ff", "#ff8800", "#00aadd", "#dd2255", "#44bb44", "#8866cc",
]


def _draw_objects(ax, runner, placed_label_boxes=None):
    """Draw colored labels at each object's position, avoiding collisions.

    Labels are placed using the same overlap-avoidance logic as fixture
    labels so they don't collide with fixtures, robots, or each other.
    """
    if placed_label_boxes is None:
        placed_label_boxes = []

    entries = _get_object_positions(runner)
    if not entries:
        return

    label_fontsize = 7
    for idx, (obj_name, x, y) in enumerate(entries):
        color = _OBJECT_COLORS[idx % len(_OBJECT_COLORS)]
        label = str(obj_name)
        # Draw marker dot at object position
        ax.plot(x, y, "D", color=color, markersize=5, zorder=7)
        fmin = np.array([x, y])
        fmax = np.array([x, y])
        (label_x, label_y), label_box = _pick_label_position(
            fmin, fmax, label, placed_label_boxes, label_fontsize,
        )
        ax.text(
            label_x, label_y, label,
            fontsize=label_fontsize, ha="center", va="center",
            color=color, fontweight="bold", zorder=8,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.88,
            },
        )
        placed_label_boxes.append(label_box)


def draw_grid_map(ax, runner, clean_labels=True):
    """Draw the occupancy grid on axes.

    *clean_labels*: when True (default), fixture labels are shortened
    (strip ``_group``, ``_main``, dedupe).  Set False for raw fixture ids.
    """
    grid = runner._occupancy_grid
    occupied_count = 0
    enclosed_count = 0
    free_count = 0

    for r in range(grid._rows):
        for c in range(grid._cols):
            xy = grid._grid_to_world(r, c)
            half = grid.cell_size / 2
            if grid._grid[r, c]:
                color = "#444444"  # hard occupied
                occupied_count += 1
            else:
                # Free in occupancy, but still not standable (tight corner pocket).
                if not grid.is_standable(xy):
                    color = "#d9923b"  # enclosed / corner-unplaceable
                    enclosed_count += 1
                else:
                    color = "#f0f0f0"  # standable free
                    free_count += 1
            rect = patches.Rectangle(
                (xy[0] - half, xy[1] - half),
                grid.cell_size, grid.cell_size,
                linewidth=0.3, edgecolor="#cccccc", facecolor=color,
            )
            ax.add_patch(rect)

    placed_label_boxes = _draw_fixtures(ax, runner._fixtures, clean_labels=clean_labels)
    _draw_robots(ax, runner, placed_label_boxes=placed_label_boxes)
    _draw_objects(ax, runner, placed_label_boxes=placed_label_boxes)

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
        f"occ={occupied_count}, enclosed={enclosed_count}, free={free_count}"
    )


def draw_continuous_map(ax, runner, clean_labels=True):
    """Draw the continuous placement candidates on axes.

    *clean_labels*: when True (default), fixture labels are shortened.
    Set False for raw fixture ids.
    """
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

    placed_label_boxes = _draw_fixtures(ax, runner._fixtures, clean_labels=clean_labels)

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
    _draw_objects(ax, runner, placed_label_boxes=placed_label_boxes)

    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Continuous — {len(cp._obstacle_aabbs)} obstacle AABBs | "
        f"green=valid, red=collides, orange=enclosed"
    )
