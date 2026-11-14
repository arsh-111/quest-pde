import numpy as np
from qpde.solver import _rhs_and_jac, DEFAULT, solve, cell_centres


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    n = 32
    u = rng.uniform(0.05, 0.9, n)
    dx = 1.0 / n
    f0, ab = _rhs_and_jac(u, DEFAULT, dx)
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = ab[1, i]
        if i + 1 < n:
            J[i, i + 1] = ab[0, i + 1]
            J[i + 1, i] = ab[2, i]
    Jfd = np.zeros((n, n))
    h = 1e-7
    for j in range(n):
        e = np.zeros(n); e[j] = h
        Jfd[:, j] = (_rhs_and_jac(u + e, DEFAULT, dx)[0] - _rhs_and_jac(u - e, DEFAULT, dx)[0]) / (2 * h)
    assert np.max(np.abs(J - Jfd)) / np.max(np.abs(Jfd)) < 1e-6


def test_constant_one_is_steady_without_right_outflow_violation():
    # u = sol everywhere: interior laplacian zero, left ghost = 1 -> zero; right ghost = 0 -> outflow.
    u = solve(np.ones(64), np.array([0.0, 1.0]))
    assert np.all(np.isfinite(u))
    assert u[-1, 0] > 0.99
