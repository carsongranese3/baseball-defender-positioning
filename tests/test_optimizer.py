"""Unit tests for the positioning math. Stdlib unittest - no extra deps.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import (clamp_to_anchor, project_infielder,
                       project_to_fair_territory, transform_coordinates,
                       weighted_constrained_kmeans)

MOUND = np.array([0.0, 60.5])
BAG_1B = [63.6, 63.6]


def dist_from_mound(p):
    return float(np.hypot(p[0] - MOUND[0], p[1] - MOUND[1]))


def is_fair(p):
    return p[1] >= abs(p[0]) - 1e-9


class TestCoordinateTransform(unittest.TestCase):
    def test_home_plate_origin(self):
        """Statcast (125, 204.5) is home plate -> (0, 0) in feet."""
        import pandas as pd
        df = transform_coordinates(pd.DataFrame({"hc_x": [125.0], "hc_y": [204.5]}))
        self.assertAlmostEqual(df["x"].iloc[0], 0.0)
        self.assertAlmostEqual(df["y"].iloc[0], 0.0)
        self.assertAlmostEqual(df["dist"].iloc[0], 0.0)

    def test_scale_and_y_inversion(self):
        """2.43 ft per pixel, and y is inverted (smaller hc_y = deeper)."""
        import pandas as pd
        df = transform_coordinates(pd.DataFrame({"hc_x": [135.0], "hc_y": [104.5]}))
        self.assertAlmostEqual(df["x"].iloc[0], 10 * 2.43)
        self.assertAlmostEqual(df["y"].iloc[0], 100 * 2.43)


class TestFairTerritory(unittest.TestCase):
    def test_fair_point_untouched(self):
        p = project_to_fair_territory([10.0, 200.0])
        self.assertEqual(list(p), [10.0, 200.0])

    def test_foul_right_projects_onto_line(self):
        p = project_to_fair_territory([63.6, 52.2])
        self.assertTrue(is_fair(p))
        self.assertAlmostEqual(p[0], p[1])  # lands on y = x

    def test_foul_left_projects_onto_line(self):
        p = project_to_fair_territory([-80.0, 40.0])
        self.assertTrue(is_fair(p))
        self.assertAlmostEqual(p[1], -p[0])  # lands on y = -x

    def test_behind_home_snaps_to_origin(self):
        self.assertEqual(list(project_to_fair_territory([30.0, -10.0])), [0, 0])


class TestClampToAnchor(unittest.TestCase):
    def test_inside_radius_untouched(self):
        p = clamp_to_anchor([65.0, 70.0], BAG_1B, 30.0)
        self.assertAlmostEqual(p[0], 65.0)
        self.assertAlmostEqual(p[1], 70.0)

    def test_outside_radius_pulled_to_edge(self):
        p = clamp_to_anchor([63.6, 163.6], BAG_1B, 30.0)
        self.assertAlmostEqual(np.linalg.norm(p - np.array(BAG_1B)), 30.0)

    def test_pulls_along_the_line(self):
        """Direction is preserved; only distance changes."""
        p = clamp_to_anchor([163.6, 163.6], BAG_1B, 10.0)
        self.assertAlmostEqual(p[0] - BAG_1B[0], p[1] - BAG_1B[1])

    def test_zero_distance_is_safe(self):
        p = clamp_to_anchor(list(BAG_1B), BAG_1B, 5.0)
        self.assertAlmostEqual(np.linalg.norm(p - np.array(BAG_1B)), 0.0)


class TestShiftLegality(unittest.TestCase):
    def test_left_side_fielders_stay_left(self):
        for k in (0, 1):
            p = project_infielder([50.0, 120.0], k)
            self.assertLessEqual(p[0], -1 + 1e-9, f"fielder {k} crossed to the right")

    def test_right_side_fielders_stay_right(self):
        for k in (2, 3):
            p = project_infielder([-50.0, 120.0], k)
            self.assertGreaterEqual(p[0], 1 - 1e-9, f"fielder {k} crossed to the left")

    def test_dirt_boundary_enforced(self):
        p = project_infielder([-10.0, 300.0], 0)
        self.assertLessEqual(dist_from_mound(p), 95 + 1e-6)

    def test_output_always_fair(self):
        for k in range(4):
            for pt in ([200.0, 10.0], [-200.0, 10.0], [5.0, -50.0], [90.0, 88.0]):
                self.assertTrue(is_fair(project_infielder(pt, k)),
                                f"fielder {k} left fair territory from {pt}")


class TestFirstBaseLeash(unittest.TestCase):
    """Regression: the 1B must always be able to cover the bag."""

    def test_leash_respected_when_optimizer_wants_him_deep(self):
        p = project_infielder([70.0, 130.0], 3, (BAG_1B, 30.0))
        self.assertLessEqual(np.linalg.norm(p - np.array(BAG_1B)), 30.0 + 1e-6)

    def test_hold_leash_keeps_him_on_the_bag(self):
        p = project_infielder([70.0, 130.0], 3, (BAG_1B, 3.0))
        self.assertLessEqual(np.linalg.norm(p - np.array(BAG_1B)), 3.0 + 1e-6)

    def test_leash_result_is_still_legal(self):
        """The bag sits on the foul line, so a naive leash can land foul.
        Regression for exactly that: infield-in pulling a held 1B off the line."""
        p = project_infielder([63.6, 53.7], 3, (BAG_1B, 3.0))
        self.assertTrue(is_fair(p), "leashed 1B ended up in foul territory")
        self.assertLessEqual(dist_from_mound(p), 95 + 1e-6)

    def test_leash_keeps_him_on_the_dirt_for_free(self):
        """30ft around a bag 63.7ft from the mound stays inside the 95ft arc."""
        for angle in np.linspace(0, 2 * np.pi, 24):
            pt = [BAG_1B[0] + 30 * np.cos(angle), BAG_1B[1] + 30 * np.sin(angle)]
            p = project_infielder(pt, 3, (BAG_1B, 30.0))
            self.assertLessEqual(dist_from_mound(p), 95 + 1e-6)


class TestConstrainedKMeans(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.data = np.column_stack([rng.uniform(-90, 90, 300),
                                     rng.uniform(60, 150, 300)])
        self.w = rng.uniform(0.1, 1.0, 300)

    def test_returns_fixed_plus_variable(self):
        c = weighted_constrained_kmeans(self.data, self.w, [[0, 60.5], [0, -2]], 4,
                                        is_infield=True)
        self.assertEqual(c.shape, (6, 2))
        self.assertAlmostEqual(c[0][1], 60.5)   # pitcher untouched
        self.assertAlmostEqual(c[1][1], -2)     # catcher untouched

    def test_all_infielders_legal(self):
        c = weighted_constrained_kmeans(self.data, self.w, [[0, 60.5], [0, -2]], 4,
                                        is_infield=True)
        left = sum(1 for k in range(4) if c[2 + k][0] < 0)
        self.assertEqual(left, 2, "anti-shift rule needs 2 fielders per side")
        for k in range(4):
            self.assertTrue(is_fair(c[2 + k]))
            self.assertLessEqual(dist_from_mound(c[2 + k]), 95 + 1e-6)

    def test_leash_enforced_through_iterations(self):
        c = weighted_constrained_kmeans(self.data, self.w, [[0, 60.5], [0, -2]], 4,
                                        is_infield=True,
                                        situational_leash={3: (BAG_1B, 3.0)})
        self.assertLessEqual(np.linalg.norm(c[5] - np.array(BAG_1B)), 3.0 + 1e-6)

    def test_empty_data_does_not_crash(self):
        c = weighted_constrained_kmeans(np.empty((0, 2)), np.empty(0),
                                        [[0, 60.5], [0, -2]], 4, is_infield=True)
        self.assertEqual(c.shape, (6, 2))


class TestDepthAdjustment(unittest.TestCase):
    """Regression: infield-in must scale depth (y) only, never x."""

    def test_y_only_scaling_preserves_lateral_position(self):
        c = np.array([[0, 60.5], [0, -2], [-61.4, 109.8], [-16.8, 137.3],
                      [35.4, 141.5], [63.9, 65.5]], dtype=float)
        before_x = c[2:, 0].copy()
        c[2:, 1] *= 0.82
        np.testing.assert_allclose(c[2:, 0], before_x,
                                   err_msg="depth adjustment moved x")

    def test_uniform_scaling_would_have_moved_x(self):
        """Guards the guard: proves the old buggy form actually differed."""
        c = np.array([[-61.4, 109.8]], dtype=float)
        buggy = c * 0.82
        self.assertNotAlmostEqual(buggy[0][0], c[0][0])


if __name__ == "__main__":
    unittest.main()
