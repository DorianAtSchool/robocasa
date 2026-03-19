"""Unit tests for ContinuousPlacement."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from robocasa.utils.placement import ContinuousPlacement


def _make_mock_fixture(
    pos: tuple[float, float, float],
    size: tuple[float, float, float] = (0.5, 0.5, 0.9),
    rot: float = 0.0,
    name: str = "fixture",
) -> MagicMock:
    """Create a mock fixture with a simple axis-aligned footprint."""
    fxtr = MagicMock()
    fxtr.pos = np.array(pos, dtype=float)
    fxtr.rot = rot
    fxtr.name = name

    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = pos
    corners = [
        np.array([cx - hx, cy - hy, cz - hz]),
        np.array([cx + hx, cy - hy, cz - hz]),
        np.array([cx - hx, cy + hy, cz - hz]),
        np.array([cx - hx, cy - hy, cz + hz]),
        np.array([cx - hx, cy + hy, cz + hz]),
        np.array([cx + hx, cy + hy, cz + hz]),
        np.array([cx + hx, cy + hy, cz - hz]),
        np.array([cx + hx, cy - hy, cz + hz]),
    ]
    fxtr.get_ext_sites = MagicMock(return_value=corners)
    return fxtr


class TestContinuousPlacementUnit(unittest.TestCase):
    def test_front_required_ignores_ref_object_pos_bias(self):
        """Front-required placement should honor an explicit front working target."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        placement = ContinuousPlacement({"target": target}, sample_spacing=0.08)

        result_left = placement.find_placement(
            target,
            ref_object_pos=np.array([-0.8, 0.6]),
            require_front=True,
        )
        result_right = placement.find_placement(
            target,
            ref_object_pos=np.array([0.8, 0.6]),
            require_front=True,
        )

        self.assertIsNotNone(result_left)
        self.assertIsNotNone(result_right)
        pos_left, _ = result_left
        pos_right, _ = result_right

        self.assertLess(
            pos_left[0],
            pos_right[0],
            f"Front target should bias front placement laterally: {pos_left} vs {pos_right}",
        )
        self.assertAlmostEqual(pos_left[0], -0.8, delta=0.05)
        self.assertAlmostEqual(pos_right[0], 0.8, delta=0.05)
        self.assertGreater(pos_left[1], 0.25)
        self.assertGreater(pos_right[1], 0.25)

    def test_front_required_fallback_stays_on_front_face(self):
        """Blocked front-center should still keep the robot on the front face."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        blocker = _make_mock_fixture(
            (0.0, 0.65, 0.0),
            size=(0.08, 0.4, 0.9),
            name="blocker",
        )
        placement = ContinuousPlacement({"target": target, "blocker": blocker}, sample_spacing=0.08)

        result = placement.find_placement(target, require_front=True)

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertGreater(pos_xy[1], 0.25, f"Expected front-face placement, got {pos_xy}")
        self.assertGreaterEqual(pos_xy[0], -1.05, f"Expected lateral front offset, got {pos_xy}")
        self.assertLessEqual(pos_xy[0], 1.05, f"Expected lateral front offset, got {pos_xy}")

    def test_front_required_does_not_fall_back_to_side_face(self):
        """If the entire front face is blocked, front-required placement should fail."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        blocker = _make_mock_fixture(
            (0.0, 0.50, 0.0),
            size=(2.6, 0.40, 0.9),
            name="front_blocker",
        )
        placement = ContinuousPlacement({"target": target, "blocker": blocker}, sample_spacing=0.08)

        result = placement.find_placement(target, require_front=True)

        self.assertIsNone(result, "Front-required placement should not fall back to a side face")

    def test_front_required_clamps_away_from_side_edge(self):
        """Front-required placement should reject edge-hugging corner approaches."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        placement = ContinuousPlacement({"target": target}, sample_spacing=0.08)

        result = placement.find_placement(
            target,
            ref_object_pos=np.array([1.2, 0.6]),
            require_front=True,
        )

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertLessEqual(pos_xy[0], 0.85, f"Expected side filtering to reject edge pose, got {pos_xy}")
        self.assertGreater(pos_xy[1], 0.25, f"Expected front-face placement, got {pos_xy}")

    def test_front_required_uses_target_side_over_fixture_rot(self):
        """A front target should override an incorrect rot-derived front face."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=0.0,
            name="fridge",
        )
        placement = ContinuousPlacement({"target": target}, sample_spacing=0.08)

        result = placement.find_placement(
            target,
            ref_object_pos=np.array([0.0, 0.7]),
            require_front=True,
        )

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertGreater(pos_xy[1], 0.25, f"Expected target-inferred front face, got {pos_xy}")
        self.assertGreater(abs(pos_xy[1]), abs(pos_xy[0]), f"Expected front along +Y, got {pos_xy}")

    def test_ref_object_pos_biases_surface_placement(self):
        """Surface placement should still honor reference-object bias."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="counter",
        )
        room_marker = _make_mock_fixture(
            (0.0, 2.0, 1.5),
            size=(0.2, 0.2, 0.2),
            name="room_marker",
        )
        placement = ContinuousPlacement(
            {"target": target, "room_marker": room_marker},
            sample_spacing=0.08,
        )

        result_left = placement.find_placement(target, ref_object_pos=np.array([-0.8, 0.5]))
        result_right = placement.find_placement(target, ref_object_pos=np.array([0.8, 0.5]))

        self.assertIsNotNone(result_left)
        self.assertIsNotNone(result_right)
        pos_left, _ = result_left
        pos_right, _ = result_right

        self.assertLess(
            pos_left[0],
            pos_right[0],
            f"Left ref should produce leftward placement: {pos_left[0]} vs {pos_right[0]}",
        )


if __name__ == "__main__":
    unittest.main()
