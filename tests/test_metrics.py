import numpy as np

from qpde import ics
from qpde.metrics import rel_l2, total_mass
from qpde.solver import solve, DEFAULT, EPS


def test_rel_l2_zero_for_identical():
    a = np.random.default_rng(0).random((3, 4, 5)) + 0.1
    assert np.allclose(rel_l2(a, a), 0)


def test_ics_are_nonnegative_and_sized():
    for kind in ics.GENERATORS:
        u = ics.make(kind, 256, seed=1)
        assert u.shape == (256,) and (u >= 0).all()


def test_total_mass_changes_only_through_boundary_fluxes():
    """Finite-volume balance on solver output: d/dt sum_i C(u_i) dx = flux_in - flux_out."""
    n, dx = 1024, 1 / 1024
    t = np.linspace(0, 100, 401)
    u = solve(np.full(n, 0.1), t, dt=0.05)
    c = (1 - DEFAULT.por) / DEFAULT.por * DEFAULT.rho_s * DEFAULT.k_f
    # cell sums (exact FV mass), the regulariser EPS matches the generator's (u + 1e-6)
    m = (u + c * (u + EPS) ** DEFAULT.n_f).sum(1) * dx
    D = DEFAULT.D
    fin = D * (DEFAULT.sol - u[:, 0]) / dx
    ghost = D * (u[:, -2] - u[:, -1]) / dx
    fout = D * (u[:, -1] - ghost) / dx
    dmdt = (m[2:] - m[:-2]) / (t[2:] - t[:-2])
    net = (fin - fout)[1:-1]
    k = t[1:-1] > 5  # skip the initial layer where the front is unresolved in time
    err = np.abs(dmdt[k] - net[k]) / np.abs(net[k])
    assert err.max() < 0.02, err.max()


def test_total_mass_helper_matches_cell_sum_roughly():
    x = (np.arange(1024) + 0.5) / 1024
    u = solve(np.full(1024, 0.1), np.array([0.0, 50.0]))[None]
    m_trap = total_mass(u, x)[0, 1]
    c = (1 - DEFAULT.por) / DEFAULT.por * DEFAULT.rho_s * DEFAULT.k_f
    m_sum = (u[0, 1] + c * (u[0, 1] + EPS) ** DEFAULT.n_f).sum() / 1024
    assert abs(m_trap - m_sum) / m_sum < 0.01
