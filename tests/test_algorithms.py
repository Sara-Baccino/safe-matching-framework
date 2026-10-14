import numpy as np
from scipy.optimize import linear_sum_assignment

from safe_matching.matching.algorithms.sorted_dp import monotone_dp


def test_sorted_dp_matches_hungarian_cost_1d():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 12)); m = n + int(rng.integers(0, 12))
        a, b = np.sort(rng.normal(size=n)), np.sort(rng.normal(size=m))
        pairs = monotone_dp(a, b)
        C = np.abs(a[:, None] - b[None, :])
        r, c = linear_sum_assignment(C)
        assert len(pairs) == n
        assert abs(sum(C[i, j] for i, j in pairs) - C[r, c].sum()) < 1e-9


def test_sorted_dp_respects_infeasible_cells():
    a, b = np.array([0.0, 1.0]), np.array([0.1, 1.1, 5.0])
    feas = np.array([[True, False, False], [False, False, True]])
    assert monotone_dp(a, b, feas) == [(0, 0), (1, 2)]
