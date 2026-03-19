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

# Front-working poses for enclosing fixtures should stay close to the center of
# the front face. Candidates are expanded gradually only along that face.
_FRONT_WORKING_LATERAL_LIMITS = (0.05, 0.12, 0.20, 0.24)
MAX_FRONT_WORKING_LATERAL_OFFSET = _FRONT_WORKING_LATERAL_LIMITS[-1]
_FRONT_WORKING_SIDE_CLEARANCE_RATIO = 0.25
_FRONT_WORKING_SIDE_CLEARANCE_MIN = 0.10
_FRONT_WORKING_SIDE_CLEARANCE_MAX = 0.18


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


_FACE_NORMALS = {
    "neg_y": np.array([0.0, -1.0]),
    "pos_y": np.array([0.0, 1.0]),
    "neg_x": np.array([-1.0, 0.0]),
    "pos_x": np.array([1.0, 0.0]),
}


def _get_rot_based_face_order(fixture: Fixture) -> list[str]:
    """Return face keys ordered front -> sides -> back based on fixture.rot."""
    all_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]
    if not hasattr(fixture, "rot") or fixture.rot is None:
        return all_faces

    rot = float(fixture.rot)
    front_dir = np.array([-np.cos(rot), -np.sin(rot)])
    return sorted(all_faces, key=lambda f: -float(np.dot(_FACE_NORMALS[f], front_dir)))


def infer_front_face_from_target(
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray,
) -> str:
    """Infer the front face from a world-space target on / near the active face."""
    target = np.asarray(target_xy, dtype=float)[:2]
    all_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]

    def _projected_distance(face_key: str) -> float:
        projected = get_face_target_point(
            face_key,
            fmin,
            fmax,
            target_xy=target,
            standoff=0.0,
            side_clearance=0.0,
        )
        return float(np.linalg.norm(projected - target))

    return min(all_faces, key=_projected_distance)


def get_face_order(
    fixture: Fixture,
    front_target_xy: np.ndarray | None = None,
) -> list[str]:
    """Return face keys ordered front -> sides -> back.

    If *front_target_xy* is provided, infer the front face from that target and
    only use fixture rotation to order the remaining faces.
    """
    base_order = _get_rot_based_face_order(fixture)
    if front_target_xy is None:
        return base_order

    aabb = get_fixture_aabb(fixture)
    if aabb is None:
        return base_order

    inferred_front = infer_front_face_from_target(aabb[0], aabb[1], front_target_xy)
    return [inferred_front, *[face for face in base_order if face != inferred_front]]


def get_face_center(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    standoff: float = _FACE_STANDOFF,
) -> np.ndarray:
    """Return the center point of a fixture face at the given standoff."""
    center = np.array([(fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2], dtype=float)
    if face_key == "neg_y":
        center[1] = fmin[1] - standoff
    elif face_key == "pos_y":
        center[1] = fmax[1] + standoff
    elif face_key == "neg_x":
        center[0] = fmin[0] - standoff
    elif face_key == "pos_x":
        center[0] = fmax[0] + standoff
    else:
        raise KeyError(f"Unknown face key: {face_key}")
    return center


def get_face_lateral_axis_and_span(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
) -> tuple[int, float, float]:
    """Return the lateral axis and min/max span for a face."""
    if face_key in {"neg_y", "pos_y"}:
        return 0, float(fmin[0]), float(fmax[0])
    if face_key in {"neg_x", "pos_x"}:
        return 1, float(fmin[1]), float(fmax[1])
    raise KeyError(f"Unknown face key: {face_key}")


def get_front_working_side_clearance(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
) -> float:
    """Return the side exclusion margin for a face's working band."""
    _, lateral_min, lateral_max = get_face_lateral_axis_and_span(face_key, fmin, fmax)
    span = max(0.0, lateral_max - lateral_min)
    if span <= 0.0:
        return 0.0
    clearance = float(np.clip(
        span * _FRONT_WORKING_SIDE_CLEARANCE_RATIO,
        _FRONT_WORKING_SIDE_CLEARANCE_MIN,
        _FRONT_WORKING_SIDE_CLEARANCE_MAX,
    ))
    return min(clearance, max(0.0, span / 2.0 - 1e-3))


def get_face_working_lateral_bounds(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    side_clearance: float | None = None,
) -> tuple[int, float, float]:
    """Return the safe lateral interval for a face's working band."""
    lateral_axis, lateral_min, lateral_max = get_face_lateral_axis_and_span(face_key, fmin, fmax)
    clearance = (
        get_front_working_side_clearance(face_key, fmin, fmax)
        if side_clearance is None else
        max(0.0, float(side_clearance))
    )
    if lateral_max - lateral_min <= 0.0:
        return lateral_axis, lateral_min, lateral_max

    safe_min = lateral_min + clearance
    safe_max = lateral_max - clearance
    if safe_min > safe_max:
        center = (lateral_min + lateral_max) / 2.0
        safe_min = center
        safe_max = center
    return lateral_axis, float(safe_min), float(safe_max)


def is_within_face_working_band(
    face_key: str,
    pos_xy: np.ndarray,
    fmin: np.ndarray,
    fmax: np.ndarray,
    side_clearance: float | None = None,
    margin: float = 1e-6,
) -> bool:
    """Return True if *pos_xy* lies within the face's safe working band."""
    lateral_axis, safe_min, safe_max = get_face_working_lateral_bounds(
        face_key,
        fmin,
        fmax,
        side_clearance=side_clearance,
    )
    pos = np.asarray(pos_xy, dtype=float)[:2]
    return bool(safe_min - margin <= pos[lateral_axis] <= safe_max + margin)


def get_face_target_point(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray | None = None,
    standoff: float = _FACE_STANDOFF,
    side_clearance: float | None = None,
) -> np.ndarray:
    """Project *target_xy* onto the face's working line at the given standoff."""
    face_target = get_face_center(face_key, fmin, fmax, standoff=standoff)
    lateral_axis, safe_min, safe_max = (
        get_face_working_lateral_bounds(face_key, fmin, fmax, side_clearance)
        if side_clearance is not None else
        get_face_lateral_axis_and_span(face_key, fmin, fmax)
    )
    if target_xy is None:
        face_target[lateral_axis] = float((safe_min + safe_max) / 2.0)
        return face_target

    target = np.asarray(target_xy, dtype=float)[:2]
    face_target[lateral_axis] = float(np.clip(target[lateral_axis], safe_min, safe_max))
    return face_target


def prepend_face_target_candidate(
    candidates: list[tuple[np.ndarray, float]],
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray | None = None,
    standoff: float = _FACE_STANDOFF,
    side_clearance: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Return candidates with the exact projected face-target candidate first."""
    target_pos = get_face_target_point(
        face_key,
        fmin,
        fmax,
        target_xy=target_xy,
        standoff=standoff,
        side_clearance=side_clearance,
    )
    if candidates:
        target_yaw = candidates[0][1]
        if any(np.allclose(pos, target_pos, atol=1e-6) for pos, _ in candidates):
            return candidates
    else:
        target_yaw = {
            "neg_y": np.pi / 2,
            "pos_y": -np.pi / 2,
            "neg_x": 0.0,
            "pos_x": np.pi,
        }[face_key]
    return [(target_pos, target_yaw), *candidates]


def prepend_face_center_candidate(
    candidates: list[tuple[np.ndarray, float]],
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    standoff: float = _FACE_STANDOFF,
) -> list[tuple[np.ndarray, float]]:
    """Return candidates with the exact face-center candidate placed first."""
    return prepend_face_target_candidate(
        candidates,
        face_key,
        fmin,
        fmax,
        target_xy=None,
        standoff=standoff,
    )


def get_front_alignment_metrics(
    fixture: Fixture,
    pos_xy: np.ndarray,
    span_margin: float = 0.05,
    target_xy: np.ndarray | None = None,
) -> dict[str, float | bool | str] | None:
    """Return front-face alignment metrics for *pos_xy* relative to *fixture*."""
    aabb = get_fixture_aabb(fixture)
    if aabb is None:
        return None

    fmin, fmax = aabb
    pos = np.asarray(pos_xy, dtype=float)[:2]
    front_face = get_face_order(fixture, front_target_xy=target_xy)[0]
    side_clearance = get_front_working_side_clearance(front_face, fmin, fmax)
    front_target = get_face_target_point(
        front_face,
        fmin,
        fmax,
        target_xy=target_xy,
        standoff=0.0,
        side_clearance=side_clearance,
    )

    lateral_axis, safe_min, safe_max = get_face_working_lateral_bounds(
        front_face,
        fmin,
        fmax,
        side_clearance=side_clearance,
    )
    centerline = float(front_target[lateral_axis])
    lateral_offset = abs(float(pos[lateral_axis] - centerline))
    within_span = bool(safe_min - span_margin <= pos[lateral_axis] <= safe_max + span_margin)

    if front_face in {"neg_y", "pos_y"}:
        front_gap = float(fmin[1] - pos[1]) if front_face == "neg_y" else float(pos[1] - fmax[1])
    else:
        front_gap = float(fmin[0] - pos[0]) if front_face == "neg_x" else float(pos[0] - fmax[0])

    return {
        "front_face": front_face,
        "lateral_offset": lateral_offset,
        "within_span": within_span,
        "on_front_face": front_gap > 0.0,
        "front_gap": front_gap,
    }


def front_lateral_offset(
    face_key: str,
    pos_xy: np.ndarray,
    front_target: np.ndarray,
) -> float:
    """Return lateral offset from *pos_xy* to a front-face working target."""
    lateral_axis = 0 if face_key in {"neg_y", "pos_y"} else 1
    pos = np.asarray(pos_xy, dtype=float)[:2]
    target = np.asarray(front_target, dtype=float)[:2]
    return abs(float(pos[lateral_axis] - target[lateral_axis]))


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
        standoff: float | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Sample (position, yaw) candidates along one AABB face."""
        if standoff is None:
            standoff = self._standoff
        perp_val = edge_val + sign * standoff
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

    # Face definitions: face_key -> (perp_axis, sign, par_axis, fmin/fmax indices for par, edge index, yaw)
    _FACE_DEFS = {
        "neg_y": (1, -1, 0, "fmin0", "fmax0", "fmin1", np.pi / 2),
        "pos_y": (1, +1, 0, "fmin0", "fmax0", "fmax1", -np.pi / 2),
        "neg_x": (0, -1, 1, "fmin1", "fmax1", "fmin0", 0.0),
        "pos_x": (0, +1, 1, "fmin1", "fmax1", "fmax0", np.pi),
    }

    def _sample_single_face(
        self,
        face_key: str,
        fmin: np.ndarray,
        fmax: np.ndarray,
        standoff: float | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Sample candidates for one face at given standoff."""
        vals = {"fmin0": fmin[0], "fmax0": fmax[0], "fmin1": fmin[1], "fmax1": fmax[1]}
        perp_axis, sign, par_axis, par_min_k, par_max_k, edge_k, yaw = self._FACE_DEFS[face_key]
        return self._sample_face(
            perp_axis, sign, par_axis,
            vals[par_min_k], vals[par_max_k], vals[edge_k], yaw,
            standoff=standoff,
        )

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
            face_key: self._sample_single_face(face_key, fmin, fmax)
            for face_key in ["neg_y", "pos_y", "neg_x", "pos_x"]
        }

    def _get_face_order(
        self,
        fixture: Fixture,
        front_target_xy: np.ndarray | None = None,
    ) -> list[str]:
        """Return face keys ordered front → sides → back based on fixture.rot."""
        return get_face_order(fixture, front_target_xy=front_target_xy)

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
                only consider front-face candidates, keeping them close to the
                fixture's projected front working line. If ``ref_object_pos``
                is provided in this mode, it is interpreted as the desired
                front working target, not as a contained-object bias. If False
                (surfaces like counters), consider ALL faces and use
                ``ref_object_pos`` to bias toward the referenced object when
                provided.
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

        def _pick_best(valid, target_point: np.ndarray):
            target_2d = np.asarray(target_point, dtype=float)[:2]
            valid.sort(key=lambda x: float(np.linalg.norm(x[0] - target_2d)))
            return (valid[0][0], valid[0][1])

        if require_front:
            # Interactive fixture: stay on the front face and keep as close to
            # the desired front working line as possible.
            aabb = get_fixture_aabb(fixture)
            face_order = self._get_face_order(fixture, front_target_xy=ref_object_pos)
            front_face = face_order[0]
            if aabb is not None:
                fmin, fmax = aabb
                side_clearance = get_front_working_side_clearance(front_face, fmin, fmax)
                for standoff in [self._standoff, 0.30, 0.20, 0.15, 0.10]:
                    front_target = get_face_target_point(
                        front_face,
                        fmin,
                        fmax,
                        target_xy=ref_object_pos,
                        standoff=standoff,
                        side_clearance=side_clearance,
                    )
                    candidates = self._sample_single_face(front_face, fmin, fmax, standoff=standoff)
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
            # Surface: all faces, pick closest valid position
            all_candidates: list[tuple[np.ndarray, float]] = []
            for cands in face_groups.values():
                all_candidates.extend(cands)
            valid = _filter_valid(all_candidates)
            if not valid:
                return None
            target_point = fixture_center
            if ref_object_pos is not None:
                target_point = np.asarray(ref_object_pos, dtype=float)[:2]
            return _pick_best(valid, target_point)

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
