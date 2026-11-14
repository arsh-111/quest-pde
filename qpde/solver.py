"""Independent implicit reference solver for the PDEBench 1D diffusion-sorption problem.

PDE (PDEBench, Takamoto et al. 2022):  du/dt = D / R(u) * d2u/dx2,  x in (0,1), t in (0,T]
    R(u) = 1 + (1-por)/por * rho_s * k_f * n_f * (u + 1e-6)^(n_f - 1)   (Freundlich retardation)
    BCs as in pdebench/data_gen/src/sim_diff_sorp.py:
      left : ghost cell value = sol (Dirichlet, u=1)
      right: ghost cell value = D * (u[N-2] - u[N-1]) / dx  (the generator's discretisation of
             u(t,1) = D du/dx(t,1))

The spatial discretisation matches the generator's (cell-centred finite volume, N cells), so
dataset trajectories can be reproduced. Time integration is independent of the generator:
the generator uses explicit adaptive RK45 at default tolerances, while this solver uses
implicit BDF2 (backward Euler for the first step) with Newton iterations on an analytic
tridiagonal Jacobian.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class PhysParams:
    D: float = 5e-4        # diffusion coefficient
    por: float = 0.29      # porosity
    rho_s: float = 2880.0  # dry bulk density
    k_f: float = 3.5e-4    # Freundlich constant
    n_f: float = 0.874     # Freundlich exponent
    sol: float = 1.0       # left boundary concentration (solubility)

    def to_dict(self):
        return asdict(self)


DEFAULT = PhysParams()
EPS = 1e-6  # the generator's regulariser inside (u + 1e-6)^(n_f - 1)


def cell_centres(nx: int) -> np.ndarray:
    dx = 1.0 / nx
    return np.linspace(dx / 2, 1 - dx / 2, nx)


def retardation(u, p: PhysParams = DEFAULT):
    # Guard against tiny negative values, which would give NaN in the fractional power.
    # The generator has no such guard; here it only matters for out-of-distribution inputs.
    us = np.maximum(u, 0.0) + EPS
    return 1.0 + (1 - p.por) / p.por * p.rho_s * p.k_f * p.n_f * us ** (p.n_f - 1)


def _rhs_and_jac(u, p: PhysParams, dx: float):
    """f(u) = D/R(u) * (L u + b(u)) and its tridiagonal Jacobian in banded form (3, N)."""
    n = u.size
    c = (1 - p.por) / p.por * p.rho_s * p.k_f * p.n_f
    us = np.maximum(u, 0.0) + EPS
    R = 1.0 + c * us ** (p.n_f - 1)
    dR = c * (p.n_f - 1) * us ** (p.n_f - 2) * (u > 0)  # zero where clipped
    g = p.D / R
    dg = -p.D * dR / R**2

    inv = 1.0 / dx**2
    Lu = np.empty(n)
    Lu[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) * inv
    Lu[0] = (-2 * u[0] + u[1]) * inv + p.sol * inv
    right_ghost = p.D * (u[-2] - u[-1]) / dx
    Lu[-1] = (u[-2] - 2 * u[-1]) * inv + right_ghost * inv
    f = g * Lu

    # dLu/du: tridiagonal; the last row also depends on the right ghost value
    main = np.full(n, -2 * inv)
    lower = np.full(n, inv)   # d Lu_i / d u_{i-1}
    upper = np.full(n, inv)   # d Lu_i / d u_{i+1}
    main[-1] += -p.D / dx * inv
    lower[-1] += p.D / dx * inv

    ab = np.zeros((3, n))
    ab[0, 1:] = g[:-1] * upper[:-1]          # super-diagonal J[i, i+1]
    ab[1, :] = g * main + dg * Lu            # diagonal
    ab[2, :-1] = g[1:] * lower[1:]           # sub-diagonal J[i+1, i]
    return f, ab


def solve(u0, t_out, params: PhysParams = DEFAULT, dt: float = 0.25,
          newton_tol: float = 1e-10, max_newton: int = 20, return_stats: bool = False):
    """Integrate from u0 (cell-centred, length N) and return u at the times t_out, shape (len(t_out), N).

    t_out must start at 0. Steps are adjusted so that every output time is hit exactly.
    """
    u = np.asarray(u0, dtype=np.float64).copy()
    n = u.size
    dx = 1.0 / n
    t_out = np.asarray(t_out, dtype=np.float64)
    out = np.empty((t_out.size, n))
    out[0] = u
    u_prev = None
    h_prev = None
    newton_iters = 0
    max_iters_seen = 0
    for k in range(1, t_out.size):
        span = t_out[k] - t_out[k - 1]
        m = max(1, int(np.ceil(span / dt - 1e-9)))
        h = span / m
        for _ in range(m):
            if u_prev is None or abs(h - h_prev) > 1e-12:
                # backward Euler: u - h f(u) = u_n
                a0, rhs_const = 1.0, u.copy()
                coef = h
            else:
                # BDF2 at constant step: 3/2 u - 2 u_n + 1/2 u_{n-1} = h f(u)
                a0 = 1.5
                rhs_const = 2.0 * u - 0.5 * u_prev
                coef = h
            v = u.copy()
            for it in range(max_newton):
                f, ab = _rhs_and_jac(v, params, dx)
                res = a0 * v - rhs_const - coef * f
                jab = -coef * ab
                jab[1] += a0
                dv = solve_banded((1, 1), jab, -res)
                v += dv
                if np.max(np.abs(dv)) < newton_tol * (1 + np.max(np.abs(v))):
                    break
            newton_iters += it + 1
            max_iters_seen = max(max_iters_seen, it + 1)
            u_prev, u, h_prev = u, v, h
        out[k] = u
    if return_stats:
        return out, {"newton_iters": newton_iters, "max_newton_per_step": max_iters_seen}
    return out


def pde_residual(u, t, params: PhysParams = DEFAULT):
    """Discrete residual of du/dt - D/R(u) (L u + b) evaluated with central differences in time.

    u has shape (Nt, Nx) on a cell-centred grid with Nx cells covering [0, 1]. Returns the residual
    on the interior time levels, shape (Nt-2, Nx), in units of du/dt.
    """
    u = np.asarray(u, dtype=np.float64)
    dx = 1.0 / u.shape[1]
    dudt = (u[2:] - u[:-2]) / (t[2:] - t[:-2])[:, None]
    f = np.stack([_rhs_and_jac(ui, params, dx)[0] for ui in u[1:-1]])
    return dudt - f
