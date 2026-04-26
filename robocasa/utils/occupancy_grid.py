"""
2D occupancy grid for robot base placement.

Ground-level fixtures and walls mark cells as occupied.  Placement candidates
are generated at a fixed standoff distance from each fixture face (same
approach as ContinuousPlacement), then validated against the grid.

Multi-robot collision is handled by cell exclusion and physical distance
checks.
"""

from __future__ import annotations

import numpy as np

from robocasa.models.fixtures.fixture import Fixture
from robocasa.utils.placement import (
    _FRONT_WORKING_LATERAL_LIMITS,
    front_lateral_offset,
    get_face_center,
    get_face_order,
    get_face_target_point,
    get_front_working_side_clearance,
    get_fixture_aabb,
    is_within_face_working_band,
    prepend_face_target_candidate,
)

# Fixture name substrings that should NOT be rasterized as ground obstacles.
_SKIP_NAME_PATTERNS = ("floor",)

# Minimum height (z) of the fixture's lowest ext_site point for it to be
# considered "above ground" and therefore not a ground obstacle.
_ABOVE_GROUND_Z_THRESHOLD = 0.60


def _is_ground_obstacle(
    name: str,
    fxtr: Fixture,
    *,
    include_corner_cabinets: bool = True,
) -> bool:
    """Return True if *fxtr* is a ground-level physical obstacle."""
    name_lower = name.lower()
    for pat in _SKIP_NAME_PATTERNS:
        if pat in name_lower:
            return False
    if getattr(fxtr, "is_corner_cab", False) and not include_corner_cabinets:
        return False
    try:
        ext = fxtr.get_ext_sites(all_points=True, relative=False)
        min_z = min(float(np.asarray(p, dtype=float)[2]) for p in ext)
        if min_z > _ABOVE_GROUND_Z_THRESHOLD:
            return False
    except Exception:
        pass
    return True


class OccupancyGrid:
    """Grid-based placement planner for robot bases in a kitchen layout."""

    # Minimum physical separation between robots (meters).
    _MIN_ROBOT_SEPARATION = 0.40

    def __init__(
        self,
        fixtures: dict[str, Fixture],
        cell_size: float = 0.05,
        align_to_wall: bool = True,
        standoff: float = 0.40,
        sample_spacing: float = 0.08,
        include_corner_cabinets: bool = True,
    ):
        self.cell_size = cell_size
        self._fixtures = fixtures
        self._standoff = standoff
        self._sample_spacing = sample_spacing
        self._include_corner_cabinets = include_corner_cabinets
        self._obstacle_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        self._transient_obstacle_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Compute world AABB from ALL fixtures, rasterize ground obstacles.
        all_points: list[np.ndarray] = []
        obstacle_names: list[str] = []

        for name, fxtr in fixtures.items():
            try:
                ext = fxtr.get_ext_sites(all_points=True, relative=False)
                for p in ext:
                    all_points.append(np.asarray(p, dtype=float)[:2])
            except Exception:
                if hasattr(fxtr, "pos") and fxtr.pos is not None:
                    all_points.append(np.asarray(fxtr.pos, dtype=float)[:2])

            if _is_ground_obstacle(
                name,
                fxtr,
                include_corner_cabinets=include_corner_cabinets,
            ):
                obstacle_names.append(name)

        if not all_points:
            self._origin = np.array([0.0, 0.0])
            self._grid = np.zeros((1, 1), dtype=bool)
            self._rows = 1
            self._cols = 1
            self._fixture_grid = self._grid.copy()
            self._base_grid = self._grid.copy()
            self._base_fixture_grid = self._fixture_grid.copy()
            self._base_obstacle_aabbs = []
            return

        pts = np.array(all_points)
        margin = 1.0
        self._origin = pts.min(axis=0) - margin
        top_right = pts.max(axis=0) + margin

        if align_to_wall:
            import math
            self._origin[1] = -math.ceil(-self._origin[1] / cell_size) * cell_size
            self._origin[0] = -math.ceil(-self._origin[0] / cell_size) * cell_size

        extent = top_right - self._origin
        self._cols = max(1, int(np.ceil(extent[0] / cell_size)))
        self._rows = max(1, int(np.ceil(extent[1] / cell_size)))

        self._grid = np.zeros((self._rows, self._cols), dtype=bool)

        for name in obstacle_names:
            aabb = get_fixture_aabb(fixtures[name])
            if aabb is not None:
                self._obstacle_aabbs.append(aabb)
            self._rasterize_fixture(name, fixtures[name])

        # Save fixture-only occupancy (before flood-fill).  Front-face
        # candidates for interactive fixtures use this grid so that narrow
        # gaps between fixtures (e.g. fridge/counter) are not sealed off.
        self._fixture_grid = self._grid.copy()

        # Flood-fill from room interior to find reachable free cells.
        # Any free cell NOT reached is an enclosed pocket → mark occupied.
        self._seal_unreachable_cells(fixtures)
        self._base_grid = self._grid.copy()
        self._base_fixture_grid = self._fixture_grid.copy()
        self._base_obstacle_aabbs = list(self._obstacle_aabbs)

    # ------------------------------------------------------------------
    # Reachability
    # ------------------------------------------------------------------

    def _seal_unreachable_cells(self, fixtures: dict[str, Fixture]):
        """Mark free cells unreachable from the room interior as occupied.

        Uses BFS flood-fill from the room center.  Enclosed pockets
        (corners between cabinets, gaps behind fixtures) become occupied,
        preventing the robot from being placed there.

        Wall fixtures are dilated by 1 cell before the flood-fill so that
        single-cell gaps between wall segments are closed, preventing the
        fill from leaking outside the kitchen.  Only walls (names starting
        with ``wall_``) are dilated — internal fixtures like counters and
        cabinets are left untouched so narrow passages between furniture
        remain navigable.
        """
        from collections import deque

        # Find seed: room center from average of fixture positions
        positions = []
        for fxtr in fixtures.values():
            if hasattr(fxtr, "pos") and fxtr.pos is not None:
                positions.append(np.asarray(fxtr.pos, dtype=float)[:2])
        if not positions:
            return

        center = np.mean(positions, axis=0)
        seed_r, seed_c = self._world_to_grid(center)

        # If center cell is occupied, spiral outward to find a free cell
        if self._grid[seed_r, seed_c]:
            found = False
            for radius in range(1, max(self._rows, self._cols)):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        r, c = seed_r + dr, seed_c + dc
                        if 0 <= r < self._rows and 0 <= c < self._cols:
                            if not self._grid[r, c]:
                                seed_r, seed_c = r, c
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                return  # all cells occupied

        # Build a wall-only grid, then dilate it by 1 cell to close gaps
        # between wall segments.  This prevents the flood-fill from leaking
        # outside the kitchen through narrow gaps in the perimeter.
        wall_grid = np.zeros_like(self._grid)
        for name, fxtr in fixtures.items():
            if name.lower().startswith("wall_"):
                self._rasterize_onto(wall_grid, fxtr)
        # Dilate wall cells by 1 in each cardinal direction
        wall_dilated = wall_grid.copy()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(wall_grid, shift=(dr, dc), axis=(0, 1))
            if dr == -1:
                shifted[-1, :] = False
            elif dr == 1:
                shifted[0, :] = False
            if dc == -1:
                shifted[:, -1] = False
            elif dc == 1:
                shifted[:, 0] = False
            wall_dilated |= shifted
        # Combine: original obstacle grid + dilated wall cells
        flood_barrier = self._grid | wall_dilated

        # BFS flood-fill from seed using the barrier grid.
        reachable = np.zeros_like(self._grid, dtype=bool)
        if flood_barrier[seed_r, seed_c]:
            # Seed landed on a barrier cell — find nearest free cell
            found_seed = False
            for radius in range(1, max(self._rows, self._cols)):
                for sdr in range(-radius, radius + 1):
                    for sdc in range(-radius, radius + 1):
                        sr, sc = seed_r + sdr, seed_c + sdc
                        if 0 <= sr < self._rows and 0 <= sc < self._cols:
                            if not flood_barrier[sr, sc]:
                                seed_r, seed_c = sr, sc
                                found_seed = True
                                break
                    if found_seed:
                        break
                if found_seed:
                    break
            if not found_seed:
                return  # all cells blocked

        queue = deque([(seed_r, seed_c)])
        reachable[seed_r, seed_c] = True

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self._rows and 0 <= nc < self._cols:
                    if not reachable[nr, nc] and not flood_barrier[nr, nc]:
                        reachable[nr, nc] = True
                        queue.append((nr, nc))

        # Mark unreachable free cells as occupied (using original grid, not
        # the barrier — dilation was only used to prevent flood-fill leaks).
        unreachable_free = ~self._grid & ~reachable
        self._grid |= unreachable_free

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _world_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        """Convert world XY to (row, col) grid indices, clamped to bounds."""
        rel = (np.asarray(xy, dtype=float) - self._origin) / self.cell_size
        col = int(np.clip(int(np.floor(rel[0])), 0, self._cols - 1))
        row = int(np.clip(int(np.floor(rel[1])), 0, self._rows - 1))
        return (row, col)

    def _is_in_bounds(self, xy: np.ndarray) -> bool:
        """Return True if the world position falls within the grid extent."""
        rel = (np.asarray(xy, dtype=float) - self._origin) / self.cell_size
        col = int(np.floor(rel[0]))
        row = int(np.floor(rel[1]))
        return 0 <= col < self._cols and 0 <= row < self._rows

    def _grid_to_world(self, row: int, col: int) -> np.ndarray:
        """Return the world-frame center of grid cell (row, col)."""
        x = self._origin[0] + (col + 0.5) * self.cell_size
        y = self._origin[1] + (row + 0.5) * self.cell_size
        return np.array([x, y])

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _rasterize_world_aabb_onto(
        self,
        target: np.ndarray,
        min_xy: np.ndarray,
        max_xy: np.ndarray,
        *,
        padding_cells: int = 0,
    ) -> None:
        r_min, c_min = self._world_to_grid(min_xy)
        r_max, c_max = self._world_to_grid(max_xy)
        r_min = max(0, r_min - padding_cells)
        c_min = max(0, c_min - padding_cells)
        r_max = min(self._rows - 1, r_max + padding_cells)
        c_max = min(self._cols - 1, c_max + padding_cells)

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if 0 <= r < self._rows and 0 <= c < self._cols:
                    target[r, c] = True

    def _rasterize_onto(
        self,
        target: np.ndarray,
        fxtr: Fixture,
        *,
        padding_cells: int = 0,
    ):
        """Mark cells in *target* that overlap the fixture's 2D footprint."""
        try:
            ext = fxtr.get_ext_sites(all_points=True, relative=False)
            pts_3d = [np.asarray(p, dtype=float) for p in ext]
        except Exception:
            if hasattr(fxtr, "pos") and fxtr.pos is not None:
                r, c = self._world_to_grid(np.asarray(fxtr.pos[:2]))
                if 0 <= r < self._rows and 0 <= c < self._cols:
                    target[r, c] = True
            return

        pts_2d = np.array([p[:2] for p in pts_3d])
        min_xy = pts_2d.min(axis=0)
        max_xy = pts_2d.max(axis=0)
        self._rasterize_world_aabb_onto(
            target,
            min_xy,
            max_xy,
            padding_cells=padding_cells,
        )

    def _fixture_has_corner_walls(self, fixture_name: str, fxtr: Fixture) -> bool:
        fixture_aabb = get_fixture_aabb(fxtr)
        if fixture_aabb is None:
            return False
        near_pos_x = False
        near_neg_x = False
        near_pos_y = False
        near_neg_y = False
        proximity = self.cell_size * 1.05
        for other_name, other_fixture in self._fixtures.items():
            if other_name == fixture_name or not other_name.lower().startswith("wall_"):
                continue
            wall_aabb = get_fixture_aabb(other_fixture)
            if wall_aabb is None:
                continue
            wall_min, wall_max = wall_aabb
            fixture_min, fixture_max = fixture_aabb
            overlaps_x = min(float(fixture_max[0]), float(wall_max[0])) >= max(
                float(fixture_min[0]),
                float(wall_min[0]),
            ) - 1e-6
            overlaps_y = min(float(fixture_max[1]), float(wall_max[1])) >= max(
                float(fixture_min[1]),
                float(wall_min[1]),
            ) - 1e-6
            if overlaps_y:
                if 0.0 <= float(wall_min[0] - fixture_max[0]) <= proximity:
                    near_pos_x = True
                if 0.0 <= float(fixture_min[0] - wall_max[0]) <= proximity:
                    near_neg_x = True
            if overlaps_x:
                if 0.0 <= float(wall_min[1] - fixture_max[1]) <= proximity:
                    near_pos_y = True
                if 0.0 <= float(fixture_min[1] - wall_max[1]) <= proximity:
                    near_neg_y = True
        return (near_pos_x or near_neg_x) and (near_pos_y or near_neg_y)

    def _rasterize_fixture(self, fixture_name: str, fxtr: Fixture):
        """Mark grid cells overlapping the fixture's 2D footprint as occupied."""
        padding_cells = 1 if self._fixture_has_corner_walls(fixture_name, fxtr) else 0
        self._rasterize_onto(
            self._grid,
            fxtr,
            padding_cells=padding_cells,
        )

    def _rebuild_transient_occupancy(self) -> None:
        self._grid = self._base_grid.copy()
        self._fixture_grid = self._base_fixture_grid.copy()
        self._obstacle_aabbs = list(self._base_obstacle_aabbs)
        for aabb_min, aabb_max in self._transient_obstacle_aabbs.values():
            self._obstacle_aabbs.append((aabb_min, aabb_max))
            self._rasterize_world_aabb_onto(self._grid, aabb_min, aabb_max)
        self._seal_unreachable_cells(self._fixtures)

    def mark_world_aabb_occupied(
        self,
        aabb: tuple[np.ndarray, np.ndarray],
        obstacle_id: str | None = None,
    ) -> str:
        """Add a transient occupied AABB and refresh flood-filled reachability."""
        normalized_id = obstacle_id or f"transient_{len(self._transient_obstacle_aabbs)}"
        aabb_min = np.asarray(aabb[0], dtype=float)[:2]
        aabb_max = np.asarray(aabb[1], dtype=float)[:2]
        self._transient_obstacle_aabbs[normalized_id] = (aabb_min, aabb_max)
        self._rebuild_transient_occupancy()
        return normalized_id

    def clear_world_aabb_occupied(self, obstacle_id: str) -> None:
        """Remove one transient occupied AABB and refresh reachability."""
        if obstacle_id not in self._transient_obstacle_aabbs:
            return
        self._transient_obstacle_aabbs.pop(obstacle_id, None)
        self._rebuild_transient_occupancy()

    def _get_fixture_cells(self, fxtr: Fixture) -> list[tuple[int, int]]:
        """Return grid cells overlapped by a fixture footprint."""
        aabb = get_fixture_aabb(fxtr)
        if aabb is None:
            if hasattr(fxtr, "pos") and fxtr.pos is not None:
                return [self._world_to_grid(np.asarray(fxtr.pos[:2], dtype=float))]
            return []
        fmin, fmax = aabb
        r_min, c_min = self._world_to_grid(fmin)
        r_max, c_max = self._world_to_grid(fmax)
        cells: list[tuple[int, int]] = []
        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                if 0 <= row < self._rows and 0 <= col < self._cols:
                    cells.append((row, col))
        return cells

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_free(self, xy: np.ndarray) -> bool:
        """Return True if the cell at world position xy is unoccupied."""
        if not self._is_in_bounds(xy):
            return False
        r, c = self._world_to_grid(xy)
        return not self._grid[r, c]

    def is_free_of_fixtures(self, xy: np.ndarray) -> bool:
        """Return True if the cell is not occupied by any fixture (ignores flood-fill)."""
        if not self._is_in_bounds(xy):
            return False
        r, c = self._world_to_grid(xy)
        return not self._fixture_grid[r, c]

    def _is_enclosed(
        self,
        pos: np.ndarray,
        exclude_fixture: Fixture | None = None,
    ) -> bool:
        """Return True if *pos* sits in a tight pocket between obstacles.

        A position is considered unstandable when either:
        - It is tightly boxed in by 3+ blocking directions, or
        - It lies in an L-corner (one blocked x direction and one blocked y
          direction), which tends to produce cabinet / wall clipping.
        """
        exclude_aabb = None
        if exclude_fixture is not None:
            exclude_aabb = get_fixture_aabb(exclude_fixture)

        # Corner blocking distance for standability checks. A tighter value
        # misses practical cabinet/counter corners; this slightly larger value
        # better matches mobile-base clearance needs.
        threshold = 0.55
        blocked = [False, False, False, False]  # +x, -x, +y, -y
        pos_xy = np.asarray(pos, dtype=float)[:2]
        for aabb_min, aabb_max in self._obstacle_aabbs:
            if exclude_aabb is not None:
                if (
                    np.allclose(aabb_min, exclude_aabb[0], atol=0.01)
                    and np.allclose(aabb_max, exclude_aabb[1], atol=0.01)
                ):
                    continue
            if (
                aabb_min[0] - pos_xy[0] <= threshold
                and aabb_min[0] >= pos_xy[0]
                and pos_xy[1] >= aabb_min[1]
                and pos_xy[1] <= aabb_max[1]
            ):
                blocked[0] = True
            if (
                pos_xy[0] - aabb_max[0] <= threshold
                and aabb_max[0] <= pos_xy[0]
                and pos_xy[1] >= aabb_min[1]
                and pos_xy[1] <= aabb_max[1]
            ):
                blocked[1] = True
            if (
                aabb_min[1] - pos_xy[1] <= threshold
                and aabb_min[1] >= pos_xy[1]
                and pos_xy[0] >= aabb_min[0]
                and pos_xy[0] <= aabb_max[0]
            ):
                blocked[2] = True
            if (
                pos_xy[1] - aabb_max[1] <= threshold
                and aabb_max[1] <= pos_xy[1]
                and pos_xy[0] >= aabb_min[0]
                and pos_xy[0] <= aabb_max[0]
            ):
                blocked[3] = True
        blocked_x = blocked[0] or blocked[1]
        blocked_y = blocked[2] or blocked[3]
        corner_pocket = blocked_x and blocked_y
        return sum(blocked) >= 3 or corner_pocket

    def is_standable(self, xy: np.ndarray) -> bool:
        """Return True if a robot can stand at world position *xy*."""
        if not self.is_free(xy):
            return False
        return not self._is_enclosed(xy)

    def occupy(self, xy: np.ndarray):
        """Mark the cell at world position xy as occupied."""
        r, c = self._world_to_grid(xy)
        if 0 <= r < self._rows and 0 <= c < self._cols:
            self._grid[r, c] = True

    def release(self, xy: np.ndarray):
        """Mark the cell at world position xy as free."""
        r, c = self._world_to_grid(xy)
        if 0 <= r < self._rows and 0 <= c < self._cols:
            self._grid[r, c] = False

    # ------------------------------------------------------------------
    # Face-based candidate generation
    # ------------------------------------------------------------------

    def _get_face_order(
        self,
        fixture: Fixture,
        front_target_xy: np.ndarray | None = None,
    ) -> list[str]:
        """Return face keys ordered front -> sides -> back based on fixture.rot."""
        face_order = get_face_order(fixture, front_target_xy=front_target_xy)
        fixture_name = str(getattr(fixture, "name", "") or "").lower()
        if "counter_corner" in fixture_name and len(face_order) > 1:
            # Corner counter fronts can point into the dining/wall side of the
            # L-corner. Prefer the local -Y face, rotated into world space.
            angle = float(getattr(fixture, "rot", 0.0) or 0.0)
            kitchen_vec = np.array([np.sin(angle), -np.cos(angle)], dtype=float)
            face_vectors = {
                "neg_x": np.array([-1.0, 0.0], dtype=float),
                "pos_x": np.array([1.0, 0.0], dtype=float),
                "neg_y": np.array([0.0, -1.0], dtype=float),
                "pos_y": np.array([0.0, 1.0], dtype=float),
            }
            kitchen_face = max(
                face_vectors,
                key=lambda face_key: float(np.dot(face_vectors[face_key], kitchen_vec)),
            )
            if kitchen_face in face_order:
                return [
                    kitchen_face,
                    *(face_key for face_key in face_order if face_key != kitchen_face),
                ]
        return face_order

    def _sample_face(
        self,
        face_key: str,
        fmin: np.ndarray,
        fmax: np.ndarray,
        standoff: float | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Sample (position, yaw) candidates along one AABB face at standoff."""
        if standoff is None:
            standoff = self._standoff
        spacing = self._sample_spacing

        # face_key -> (perp_axis, sign, par_axis, par_min, par_max, edge_val, yaw)
        face_defs = {
            "neg_y": (1, -1, 0, fmin[0], fmax[0], fmin[1], np.pi / 2),
            "pos_y": (1, +1, 0, fmin[0], fmax[0], fmax[1], -np.pi / 2),
            "neg_x": (0, -1, 1, fmin[1], fmax[1], fmin[0], 0.0),
            "pos_x": (0, +1, 1, fmin[1], fmax[1], fmax[0], np.pi),
        }

        perp_axis, sign, par_axis, par_min, par_max, edge_val, yaw = face_defs[face_key]
        perp_val = edge_val + sign * standoff
        face_len = par_max - par_min
        candidates: list[tuple[np.ndarray, float]] = []

        if face_len <= 0:
            pos = np.zeros(2)
            pos[perp_axis] = perp_val
            pos[par_axis] = (par_min + par_max) / 2
            candidates.append((pos, yaw))
        else:
            n_samples = max(2, int(np.ceil(face_len / spacing)) + 1)
            for i in range(n_samples):
                t = i / max(n_samples - 1, 1)
                pos = np.zeros(2)
                pos[perp_axis] = perp_val
                pos[par_axis] = par_min + t * face_len
                candidates.append((pos, yaw))

        return candidates

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def find_placement(
        self,
        fixture: Fixture,
        robot_cells: list[tuple[int, int]] | None = None,
        ref_object_pos: np.ndarray | None = None,
        robot_positions: list[np.ndarray] | None = None,
        require_front: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Find a valid position near *fixture* at standoff distance.

        Args:
            require_front: If True (interactive fixtures like fridge/cabinet),
                only consider front-face candidates, keeping them close to the
                projected front working line. If ``ref_object_pos`` is
                provided in this mode, it is interpreted as the desired front
                working target, not as a contained-object bias. If False
                (surfaces like counters), consider ALL faces and use
                ``ref_object_pos`` to bias toward the referenced object when
                provided.

        Yaw is always axis-aligned (perpendicular to the fixture face).
        """
        if robot_cells is None:
            robot_cells = []
        robot_cell_set = set(robot_cells)
        if robot_positions is None:
            robot_positions = []
        robot_positions = [np.asarray(rp, dtype=float)[:2] for rp in robot_positions]

        aabb = get_fixture_aabb(fixture)
        if aabb is None:
            return None
        fmin, fmax = aabb
        fixture_center = np.asarray(fixture.pos[:2], dtype=float)

        def _filter_valid(candidates):
            """Filter candidates to reachable, collision-free positions."""
            valid: list[tuple[np.ndarray, float]] = []
            for pos, yaw in candidates:
                # Safety: never place inside the target fixture's AABB
                if (pos[0] >= fmin[0] and pos[0] <= fmax[0] and
                        pos[1] >= fmin[1] and pos[1] <= fmax[1]):
                    continue
                if not self._is_in_bounds(pos):
                    continue
                r, c = self._world_to_grid(pos)
                if self._grid[r, c]:
                    continue
                if self._is_enclosed(pos, exclude_fixture=fixture):
                    continue
                if (r, c) in robot_cell_set:
                    continue
                too_close = False
                for rp in robot_positions:
                    if float(np.linalg.norm(pos - rp)) < self._MIN_ROBOT_SEPARATION:
                        too_close = True
                        break
                if too_close:
                    continue
                valid.append((pos, yaw))
            return valid

        def _pick_best(valid, target_point: np.ndarray):
            target_2d = np.asarray(target_point, dtype=float)[:2]
            valid.sort(key=lambda x: float(np.linalg.norm(x[0] - target_2d)))
            return (valid[0][0], valid[0][1])

        def _has_clear_working_line(start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
            """Reject placements whose straight path to the work line crosses walls."""
            start = np.asarray(start_xy, dtype=float)[:2]
            end = np.asarray(end_xy, dtype=float)[:2]
            delta = end - start
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-9:
                return True
            sample_count = max(2, int(np.ceil(distance / max(self.cell_size * 0.5, 1e-3))))
            for step_idx in range(1, sample_count):
                alpha = step_idx / sample_count
                probe = start + alpha * delta
                if not self._is_in_bounds(probe):
                    return False
                row, col = self._world_to_grid(probe)
                if self._fixture_grid[row, col]:
                    return False
            return True

        if require_front:
            # Interactive fixture: stay on the front face, remain reachable,
            # and keep a clear working line to the fixture's front band.
            face_order = self._get_face_order(fixture, front_target_xy=ref_object_pos)
            front_face = face_order[0]
            side_clearance = get_front_working_side_clearance(front_face, fmin, fmax)
            is_corner_counter = "counter_corner" in str(
                getattr(fixture, "name", "") or ""
            ).lower()
            if is_corner_counter:
                side_clearance = 0.0
            for standoff in [self._standoff, 0.30, 0.20, 0.15, 0.10]:
                front_target = get_face_target_point(
                    front_face,
                    fmin,
                    fmax,
                    target_xy=ref_object_pos,
                    standoff=standoff,
                    side_clearance=side_clearance,
                )
                candidates = self._sample_face(front_face, fmin, fmax, standoff=standoff)
                candidates = prepend_face_target_candidate(
                    candidates,
                    front_face,
                    fmin,
                    fmax,
                    target_xy=ref_object_pos,
                    standoff=standoff,
                    side_clearance=side_clearance,
                )
                valid = _filter_valid(candidates)
                if valid:
                    valid = [
                        cand for cand in valid
                        if is_within_face_working_band(
                            front_face,
                            cand[0],
                            fmin,
                            fmax,
                            side_clearance=side_clearance,
                        )
                    ]
                if valid and not is_corner_counter:
                    valid = [
                        cand for cand in valid
                        if _has_clear_working_line(cand[0], front_target)
                    ]
                if valid and is_corner_counter:
                    return _pick_best(valid, front_target)
                if valid:
                    for lateral_limit in _FRONT_WORKING_LATERAL_LIMITS:
                        centered_valid = [
                            cand for cand in valid
                            if front_lateral_offset(front_face, cand[0], front_target) <= lateral_limit
                        ]
                        if centered_valid:
                            return _pick_best(centered_valid, front_target)
            return None
        else:
            # Surface: all faces, with a small outward fallback if the
            # nearest ring is blocked by neighboring fixtures or robots.
            standoffs = [
                self._standoff,
                self._standoff + self.cell_size,
                self._standoff + 2 * self.cell_size,
            ]
            target_point = fixture_center
            if ref_object_pos is not None:
                target_point = np.asarray(ref_object_pos, dtype=float)[:2]
            candidate_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]
            fixture_name = str(getattr(fixture, "name", "") or "").lower()
            if "counter_corner" in fixture_name:
                candidate_faces = self._get_face_order(
                    fixture,
                    front_target_xy=ref_object_pos,
                )[:1]
            for standoff in standoffs:
                all_candidates: list[tuple[np.ndarray, float]] = []
                for face_key in candidate_faces:
                    all_candidates.extend(
                        self._sample_face(face_key, fmin, fmax, standoff=standoff)
                    )
                valid = _filter_valid(all_candidates)
                if valid:
                    return _pick_best(valid, target_point)
            return None
