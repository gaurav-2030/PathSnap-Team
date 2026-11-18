"""Unit tests for the geometry, Viterbi and scoring helpers in pathsnap/core.py.
Run from the repository root:  python -m unittest discover -s tests -v"""
import sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.core import EARTH_R, project, point_seg, log_transitions, viterbi, route_set_scores  # noqa: E402


class TestGeometry(unittest.TestCase):
    def test_projection_origin_is_zero(self):
        x, y = project(np.array([77.6]), np.array([12.9]), 77.6, 12.9)
        self.assertAlmostEqual(float(x[0]), 0.0)
        self.assertAlmostEqual(float(y[0]), 0.0)

    def test_one_degree_latitude_in_metres(self):
        _, y = project(np.array([0.0]), np.array([1.0]), 0.0, 0.0)
        self.assertAlmostEqual(float(y[0]), np.radians(1.0) * EARTH_R, places=3)

    def test_longitude_shrinks_with_latitude(self):
        x_eq, _ = project(np.array([1.0]), np.array([0.0]), 0.0, 0.0)
        x_60, _ = project(np.array([1.0]), np.array([60.0]), 0.0, 60.0)
        self.assertAlmostEqual(float(x_60[0]) / float(x_eq[0]), 0.5, places=6)

    def test_point_to_segment(self):
        d, t, t_raw = point_seg(np.array([5.0, -3.0, 14.0]), np.array([4.0, 0.0, 0.0]), 0.0, 0.0, 10.0, 0.0)
        np.testing.assert_allclose(d, [4.0, 3.0, 4.0])      # perpendicular, before start, after end
        np.testing.assert_allclose(t, [0.5, 0.0, 1.0])      # clipped position along the segment
        np.testing.assert_allclose(t_raw, [0.5, -0.3, 1.4])

    def test_zero_length_segment(self):
        d, t, _ = point_seg(np.array([3.0]), np.array([4.0]), 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(d[0]), 5.0)
        self.assertAlmostEqual(float(t[0]), 0.0)


def chain(n_obs, n_cand, logt_blocks):
    """Build the table layout used by viterbi(): n_obs observations with n_cand candidates each."""
    starts = np.arange(n_obs) * n_cand
    bnd = np.concatenate([[0], np.cumsum([b.size for b in logt_blocks])])
    return {"starts": starts, "ends": starts + n_cand, "n_obs": n_obs, "bnd": bnd}, np.concatenate([b.ravel() for b in logt_blocks])


class TestViterbi(unittest.TestCase):
    def test_follows_emissions_when_transitions_are_flat(self):
        tab, logt = chain(3, 2, [np.zeros((2, 2)), np.zeros((2, 2))])
        logp = np.log([0.9, 0.1, 0.2, 0.8, 0.7, 0.3])
        best, breaks = viterbi(logp, tab, logt)
        self.assertEqual(best.tolist(), [0, 3, 4])
        self.assertEqual(breaks, 0)

    def test_transitions_override_a_noisy_point(self):
        # the middle point slightly prefers candidate 1, but only candidate 0 is reachable from both neighbours
        stay = np.array([[0.0, -np.inf], [-np.inf, 0.0]])
        tab, logt = chain(3, 2, [stay, stay])
        logp = np.log([0.9, 0.1, 0.45, 0.55, 0.9, 0.1])
        best, breaks = viterbi(logp, tab, logt)
        self.assertEqual(best.tolist(), [0, 2, 4])
        self.assertEqual(breaks, 0)

    def test_disconnected_step_counts_a_chain_break(self):
        cut = np.full((2, 2), -np.inf)
        tab, logt = chain(3, 2, [np.zeros((2, 2)), cut])
        logp = np.log([0.9, 0.1, 0.8, 0.2, 0.3, 0.7])
        best, breaks = viterbi(logp, tab, logt)
        self.assertEqual(breaks, 1)
        self.assertEqual(best.tolist(), [0, 2, 5])

    def test_single_observation(self):
        tab = {"starts": np.array([0]), "ends": np.array([3]), "n_obs": 1, "bnd": np.array([0])}
        best, breaks = viterbi(np.log([0.2, 0.5, 0.3]), tab, np.array([]))
        self.assertEqual(best.tolist(), [1])
        self.assertEqual(breaks, 0)


class TestTransitions(unittest.TestCase):
    def test_log_transitions(self):
        tab = {"rd": np.array([10.0, 30.0, np.inf, 50.0]), "gc": np.array([10.0, 10.0, 10.0, 10.0]),
               "dt": np.array([1.0, 4.0, 1.0, 120.0])}
        logt = log_transitions(tab, beta=2.0, gap_s=60.0)
        self.assertEqual(logt[0], 0.0)                     # driving distance equals GPS distance
        self.assertAlmostEqual(logt[1], -20.0 / (2.0 * 2.0))  # tolerance scales with sqrt(dt)
        self.assertEqual(logt[2], -np.inf)                 # unreachable
        self.assertEqual(logt[3], 0.0)                     # long gap: continuity not enforced


class TestRouteScores(unittest.TestCase):
    def test_precision_recall_f1(self):
        p, r, f1 = route_set_scores(np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5, 6]))
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.6)
        self.assertAlmostEqual(f1, 2 * 0.75 * 0.6 / 1.35)

    def test_empty_prediction(self):
        self.assertEqual(route_set_scores(np.array([], dtype=int), np.array([1, 2])), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
