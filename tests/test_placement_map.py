from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robocasa.utils import placement_map as pm


class _FakeAxis:
    def __init__(self):
        self.patches = []
        self.plots = []
        self.texts = []

    def add_patch(self, patch):
        self.patches.append(patch)

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def text(self, x, y, label, **kwargs):
        self.texts.append((x, y, label, kwargs))
        return SimpleNamespace()


class PlacementMapTests(unittest.TestCase):
    def test_draw_robots_uses_agent_labels(self):
        ax = _FakeAxis()
        runner = SimpleNamespace(
            _num_robots=2,
            _get_robot_position=lambda ridx: np.array([float(ridx), float(ridx), 0.0]),
        )

        pm._draw_robots(ax, runner)

        self.assertEqual(
            [label for _, _, label, _ in ax.texts],
            ["agent 0", "agent 1"],
        )

    def test_draw_fixtures_keeps_full_names_and_offsets_overlapping_labels(self):
        ax = _FakeAxis()
        fixture_aabbs = {
            "counter_main_group": (
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 0.0]),
            ),
            "counter_main_group_2": (
                np.array([0.15, 0.05, 0.0]),
                np.array([1.15, 1.05, 0.0]),
            ),
        }
        fixtures = {name: object() for name in fixture_aabbs}

        with (
            patch.object(pm, "get_fixture_aabb", side_effect=lambda fixture: fixture_aabbs[next(
                name for name, candidate in fixtures.items() if candidate is fixture
            )]),
            patch.object(pm, "is_ground_obstacle", return_value=False),
        ):
            pm._draw_fixtures(ax, fixtures, label_fontsize=5)

        self.assertEqual(len(ax.texts), 2)
        labels = [label for _, _, label, _ in ax.texts]
        self.assertIn("counter main group", labels)
        self.assertIn("counter main group 2", labels)

        first_pos = np.array(ax.texts[0][:2], dtype=float)
        second_pos = np.array(ax.texts[1][:2], dtype=float)
        self.assertFalse(np.allclose(first_pos, second_pos))


if __name__ == "__main__":
    unittest.main()
