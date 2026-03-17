"""
Robot base placement strategies.

Two implementations with the same ``find_placement`` interface:

- **OccupancyGrid** (grid-based): rasterizes fixture footprints onto a 2D grid
  and picks free cells adjacent to the target fixture.  Fast, but suffers from
  quantization — a fixture that bleeds 1 cm into a cell blocks the entire cell.

- **ContinuousPlacement** (continuous / Option B): samples candidate positions
  along each face of the target fixture at a fixed standoff distance, then
  filters candidates that overlap any fixture AABB or are too close to another
  robot.  No grid quantization — positions are exact.

Both share helpers from ``_common`` (ground-obstacle filtering, fixture AABB
extraction).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from robocasa.models.fixtures.fixture import Fixture


# ======================================================================
# Shared helpers
# ======================================================================

# Fixture name substrings that should NOT be treated as ground obstacles.
# "floor" — walkable area, not an obstacle.
# NOTE: "backing" was previously skipped but this caused walls to be invisible
# to the placement system, allowing candidates behind walls.  Walls are now
# treated as obstacles; only "floor" is skipped.
_SKIP_NAME_PATTERNS = ("floor",)

# Minimum z of the fixture's lowest ext_site point to be "above ground."
_ABOVE_GROUND_Z_THRESHOLD = 0.60

# Robot radius used for circle-vs-AABB collision (continuous placement).
# The PandaOmron base collision box is 0.70 x 0.50 m, but fixture AABBs
# are also conservative (include seat overhang, door swing, etc.).
# Using the narrow half-extent (0.25) minus AABB conservatism margin.
_ROBOT_RADIUS = 0.18

# Default standoff from fixture face edge.
_FACE_STANDOFF = 0.40

# Minimum separation between two robots.
# Two robots side-by-side: 0.18 radius × 2 + 0.04 margin.
_MIN_ROBOT_SEPARATION = 0.40


def is_ground_obstacle(name: str, fxtr: Fixture) -> bool:
    """Return True if *fxtr* is a ground-level physical obstacle."""
    name_lower = name.lower()
    for pat in _SKIP_NAME_PATTERNS:
        if pat in name_lower:
            return False
    try:
        ext = fxtr.get_ext_sites(all_points=True, relative=False)
        min_z = min(float(np.asarray(p, dtype=float)[2]) for p in ext)
        if min_z > _ABOVE_GROUND_Z_THRESHOLD:
            return False
    except Exception:
        pass
    return True


def get_fixture_aabb(fxtr: Fixture) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (min_xy, max_xy) for the fixture's 2D AABB, or None."""
    try:
        ext = fxtr.get_ext_sites(all_points=True, relative=False)
        pts = np.array([np.asarray(p, dtype=float)[:2] for p in ext])
        return pts.min(axis=0), pts.max(axis=0)
    except Exception:
        return None


# ======================================================================
# Protocol — common interface for placement strategies
# ======================================================================

class PlacementStrategy(Protocol):
    """Minimal interface that both strategies expose."""

    def find_placement(
        self,
        fixture: Fixture,
        robot_positions: list[np.ndarray] | None = None,
        ref_object_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """Find a valid robot position near *fixture*.

        Args:
            fixture: Target fixture the robot needs to stand near.
            robot_positions: 2D world positions of other robots to avoid.
            ref_object_pos: Optional 2D position of a reference object;
                prefer the candidate closest to it.

        Returns:
            ``(pos_xy, yaw)`` or ``None`` if no valid position found.
        """
        ...

    def is_standable(self, xy: np.ndarray) -> bool:
        """Return True if a robot can physically stand at *xy*."""
        ...


# ======================================================================
# Option B: Continuous placement (no grid)
# ======================================================================

class ContinuousPlacement:
    """Placement by sampling along fixture faces with AABB collision checks.

    No grid discretization.  Candidate positions are generated at a fixed
    standoff distance along each face of the target fixture, sampled every
    ``sample_spacing`` meters.  Each candidate is checked for overlap against
    all ground-obstacle AABBs (using a circle-vs-AABB test with
    ``_ROBOT_RADIUS``) and against other robot positions.
    """

    def __init__(
        self,
        fixtures: dict[str, Fixture],
        standoff: float = _FACE_STANDOFF,
        sample_spacing: float = 0.08,
        robot_radius: float = _ROBOT_RADIUS,
        min_robot_separation: float = _MIN_ROBOT_SEPARATION,
    ):
        self._fixtures = fixtures
        self._standoff = standoff
        self._sample_spacing = sample_spacing
        self._robot_radius = robot_radius
        self._min_robot_separation = min_robot_separation

        # Pre-compute AABBs for all ground obstacles (done once at init).
        self._obstacle_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        all_pts: list[np.ndarray] = []
        for name, fxtr in fixtures.items():
            aabb = get_fixture_aabb(fxtr)
            if aabb is not None:
                all_pts.extend([aabb[0], aabb[1]])
                if is_ground_obstacle(name, fxtr):
                    self._obstacle_aabbs.append(aabb)

        # Compute room interior bounds from all fixture extents.
        # The robot must stay within the convex hull of fixtures (+ small margin)
        # to avoid being placed outside the room.
        if all_pts:
            pts_arr = np.array(all_pts)
            self._room_min = pts_arr.min(axis=0) - 0.1
            self._room_max = pts_arr.max(axis=0) + 0.1
        else:
            self._room_min = None
            self._room_max = None

    # ------------------------------------------------------------------
    # Collision checks
    # ------------------------------------------------------------------

    def _circle_overlaps_aabb(
        self,
        center: np.ndarray,
        radius: float,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
    ) -> bool:
        """Return True if a circle overlaps an axis-aligned bounding box."""
        # Closest point on the AABB to the circle center
        closest = np.clip(center, aabb_min, aabb_max)
        dist_sq = float(np.sum((center - closest) ** 2))
        return dist_sq < radius * radius

    def _collides_with_obstacles(
        self,
        pos: np.ndarray,
        exclude_fixture: Fixture | None = None,
    ) -> bool:
        """Return True if a robot at *pos* overlaps any ground obstacle."""
        exclude_aabb = None
        if exclude_fixture is not None:
            exclude_aabb = get_fixture_aabb(exclude_fixture)

        for aabb_min, aabb_max in self._obstacle_aabbs:
            # Skip the target fixture itself — robot stands next to it,
            # so the standoff handles clearance.
            if exclude_aabb is not None:
                if (np.allclose(aabb_min, exclude_aabb[0], atol=0.01) and
                        np.allclose(aabb_max, exclude_aabb[1], atol=0.01)):
                    continue
            if self._circle_overlaps_aabb(pos, self._robot_radius, aabb_min, aabb_max):
                return True
        return False

    def _is_enclosed(
        self,
        pos: np.ndarray,
        exclude_fixture: Fixture | None = None,
    ) -> bool:
        """Return True if *pos* is in a tight corner between obstacles.

        Casts four axis-aligned rays from *pos* and checks if obstacles
        block movement within a short distance.  A position is considered
        enclosed only when **3 or more** directions are blocked — this
        catches L-shaped corners (cabinet + wall + fridge) but NOT gaps
        between small fixtures like stools where only 2 directions (left
        and right) are blocked while front/back remain open.

        The target fixture is excluded because the robot is meant to stand
        in front of it — having the target fixture on one side is expected.
        """
        exclude_aabb = None
        if exclude_fixture is not None:
            exclude_aabb = get_fixture_aabb(exclude_fixture)

        threshold = self._robot_radius * 2
        blocked = [False, False, False, False]  # +x, -x, +y, -y
        for aabb_min, aabb_max in self._obstacle_aabbs:
            # Skip the target fixture
            if exclude_aabb is not None:
                if (np.allclose(aabb_min, exclude_aabb[0], atol=0.01) and
                        np.allclose(aabb_max, exclude_aabb[1], atol=0.01)):
                    continue
            # +x: obstacle to the right
            if (aabb_min[0] - pos[0] <= threshold and aabb_min[0] >= pos[0] and
                    pos[1] >= aabb_min[1] and pos[1] <= aabb_max[1]):
                blocked[0] = True
            # -x: obstacle to the left
            if (pos[0] - aabb_max[0] <= threshold and aabb_max[0] <= pos[0] and
                    pos[1] >= aabb_min[1] and pos[1] <= aabb_max[1]):
                blocked[1] = True
            # +y: obstacle above
            if (aabb_min[1] - pos[1] <= threshold and aabb_min[1] >= pos[1] and
                    pos[0] >= aabb_min[0] and pos[0] <= aabb_max[0]):
                blocked[2] = True
            # -y: obstacle below
            if (pos[1] - aabb_max[1] <= threshold and aabb_max[1] <= pos[1] and
                    pos[0] >= aabb_min[0] and pos[0] <= aabb_max[0]):
                blocked[3] = True
        # Enclosed only if 3+ directions are blocked (tight corner, not open gap)
        return sum(blocked) >= 3

    def _too_close_to_robots(
        self,
        pos: np.ndarray,
        robot_positions: list[np.ndarray],
    ) -> bool:
        """Return True if *pos* is within min separation of any robot."""
        for rp in robot_positions:
            if float(np.linalg.norm(pos - rp)) < self._min_robot_separation:
                return True
        return False

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _sample_face(
        self,
        perp_axis: int,
        sign: int,
        par_axis: int,
        par_min: float,
        par_max: float,
        edge_val: float,
        yaw: float,
    ) -> list[tuple[np.ndarray, float]]:
        """Sample (position, yaw) candidates along one AABB face."""
        perp_val = edge_val + sign * self._standoff
        face_len = par_max - par_min
        candidates: list[tuple[np.ndarray, float]] = []

        if face_len <= 0:
            pos = np.zeros(2)
            pos[perp_axis] = perp_val
            pos[par_axis] = (par_min + par_max) / 2
            candidates.append((pos, yaw))
        else:
            n_samples = max(2, int(np.ceil(face_len / self._sample_spacing)) + 1)
            for i in range(n_samples):
                t = i / max(n_samples - 1, 1)
                pos = np.zeros(2)
                pos[perp_axis] = perp_val
                pos[par_axis] = par_min + t * face_len
                candidates.append((pos, yaw))

        return candidates

    def _generate_face_candidates(
        self,
        fixture: Fixture,
    ) -> list[tuple[np.ndarray, float]]:
        """Generate (position, yaw) candidates along each face of the fixture.

        Returns a flat list of all candidates across all faces.
        """
        grouped = self._generate_face_candidates_grouped(fixture)
        flat: list[tuple[np.ndarray, float]] = []
        for cands in grouped.values():
            flat.extend(cands)
        return flat

    def _generate_face_candidates_grouped(
        self,
        fixture: Fixture,
    ) -> dict[str, list[tuple[np.ndarray, float]]]:
        """Generate candidates grouped by face key.

        Returns dict with keys ``neg_y``, ``pos_y``, ``neg_x``, ``pos_x``.
        """
        aabb = get_fixture_aabb(fixture)
        if aabb is None:
            return {}

        fmin, fmax = aabb
        return {
            # Bottom face (y_min): robot stands below, faces +y
            "neg_y": self._sample_face(1, -1, 0, fmin[0], fmax[0], fmin[1], np.pi / 2),
            # Top face (y_max): robot stands above, faces -y
            "pos_y": self._sample_face(1, +1, 0, fmin[0], fmax[0], fmax[1], -np.pi / 2),
            # Left face (x_min): robot stands left, faces +x
            "neg_x": self._sample_face(0, -1, 1, fmin[1], fmax[1], fmin[0], 0.0),
            # Right face (x_max): robot stands right, faces -x
            "pos_x": self._sample_face(0, +1, 1, fmin[1], fmax[1], fmax[0], np.pi),
        }

    def _get_face_order(self, fixture: Fixture) -> list[str]:
        """Return face keys ordered front → sides → back based on fixture.rot."""
        all_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]
        if not hasattr(fixture, "rot") or fixture.rot is None:
            return all_faces

        rot = float(fixture.rot)
        front_dir = np.array([-np.cos(rot), -np.sin(rot)])

        face_normals = {
            "neg_y": np.array([0.0, -1.0]),
            "pos_y": np.array([0.0, 1.0]),
            "neg_x": np.array([-1.0, 0.0]),
            "pos_x": np.array([1.0, 0.0]),
        }

        return sorted(all_faces,
                       key=lambda f: -float(np.dot(face_normals[f], front_dir)))

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def find_placement(
        self,
        fixture: Fixture,
        robot_positions: list[np.ndarray] | None = None,
        ref_object_pos: np.ndarray | None = None,
        require_front: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Find a collision-free position near *fixture*.

        Args:
            require_front: If True (interactive fixtures like fridge/cabinet),
                only consider front-face candidates (fall back to sides if
                none).  If False (surfaces like counters), consider ALL faces
                and pick the closest valid position.
        """
        if robot_positions is None:
            robot_positions = []
        robot_positions = [np.asarray(rp, dtype=float)[:2] for rp in robot_positions]

        face_groups = self._generate_face_candidates_grouped(fixture)
        if not face_groups:
            return None

        fixture_center = np.asarray(fixture.pos[:2], dtype=float)
        target_aabb = get_fixture_aabb(fixture)

        def _filter_valid(candidates):
            valid: list[tuple[np.ndarray, float]] = []
            for pos, yaw in candidates:
                # Safety: never place inside the target fixture's own AABB
                if target_aabb is not None:
                    if (pos[0] >= target_aabb[0][0] and pos[0] <= target_aabb[1][0] and
                            pos[1] >= target_aabb[0][1] and pos[1] <= target_aabb[1][1]):
                        continue
                if self._collides_with_obstacles(pos, exclude_fixture=fixture):
                    continue
                if not self._is_within_room(pos):
                    continue
                if self._is_enclosed(pos, exclude_fixture=fixture):
                    continue
                if self._too_close_to_robots(pos, robot_positions):
                    continue
                valid.append((pos, yaw))
            return valid

        def _pick_best(valid):
            if ref_object_pos is not None:
                ref_2d = np.asarray(ref_object_pos, dtype=float)[:2]
                valid.sort(key=lambda x: float(np.linalg.norm(x[0] - ref_2d)))
            else:
                valid.sort(key=lambda x: float(np.linalg.norm(x[0] - fixture_center)))
            return (valid[0][0], valid[0][1])

        if require_front:
            # Interactive fixture: try front face first, then fall back
            face_order = self._get_face_order(fixture)
            for face_key in face_order:
                valid = _filter_valid(face_groups.get(face_key, []))
                if valid:
                    return _pick_best(valid)
            return None
        else:
            # Surface: all faces, pick closest valid position
            all_candidates: list[tuple[np.ndarray, float]] = []
            for cands in face_groups.values():
                all_candidates.extend(cands)
            valid = _filter_valid(all_candidates)
            if not valid:
                return None
            return _pick_best(valid)

    def _is_within_room(self, pos: np.ndarray) -> bool:
        """Return True if *pos* is within the room bounds."""
        if self._room_min is None:
            return True
        p = np.asarray(pos, dtype=float)[:2]
        return not (np.any(p < self._room_min) or np.any(p > self._room_max))

    def is_inside_any_fixture(self, xy: np.ndarray) -> bool:
        """Return True if *xy* is inside any ground-obstacle AABB.

        Used as a post-placement safety check to guarantee the robot is
        never placed inside a fixture (fridge, cabinet, counter, etc.).
        """
        pos = np.asarray(xy, dtype=float)[:2]
        for aabb_min, aabb_max in self._obstacle_aabbs:
            if (pos[0] >= aabb_min[0] and pos[0] <= aabb_max[0] and
                    pos[1] >= aabb_min[1] and pos[1] <= aabb_max[1]):
                return True
        return False

    def is_standable(self, xy: np.ndarray) -> bool:
        """Return True if a robot can physically stand at *xy*.

        Checks both obstacle collision and room bounds.
        """
        if not self._is_within_room(xy):
            return False
        return not self._collides_with_obstacles(xy)
