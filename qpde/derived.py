"""Derived outputs computed from a trajectory u (T, N) on the cell-centred points x (N,).

All functions also work on batches (B, T, N). The grid may be the full 1024 cells or the 256-point strided cache;
x[0] is always the first fine cell centre (0.5/1024), so the inflow flux uses the fine cell width.
"""
from __future__ import annotations

import numpy as np

from qpde.solver import DEFAULT, EPS, PhysParams

FINE_DX = 1.0 / 1024


def conc_at(u, x, x_query: float):
    """u(t, x_query) by linear interpolation in x."""
    j = int(np.clip(np.searchsorted(x, x_query), 1, len(x) - 1))
    w = (x_query - x[j - 1]) / (x[j] - x[j - 1])
    w = float(np.clip(w, 0, 1))
    return (1 - w) * u[..., j - 1] + w * u[..., j]


def front_position(u, x, level: float = 0.5):
    """First x where u drops below `level` (linear interpolation). NaN where u never crosses the level."""
    u = np.asarray(u)
    below = u < level
    idx = np.argmax(below, axis=-1)                  # first index below the level
    has = below.any(-1) & (idx > 0) & (u[..., 0] >= level)
    i1 = np.clip(idx, 1, len(x) - 1)
    u0 = np.take_along_axis(u, (i1 - 1)[..., None], -1)[..., 0]
    u1 = np.take_along_axis(u, i1[..., None], -1)[..., 0]
    w = np.clip((u0 - level) / np.where(u0 - u1 == 0, 1, u0 - u1), 0, 1)
    xf = x[i1 - 1] + w * (x[i1] - x[i1 - 1])
    return np.where(has, xf, np.nan)


def total_conc(u, p: PhysParams = DEFAULT):
    c = (1 - p.por) / p.por * p.rho_s * p.k_f
    return u + c * (np.maximum(u, 0) + EPS) ** p.n_f


def uptake(u, x, p: PhysParams = DEFAULT):
    """Total (dissolved + sorbed) amount that has entered the column since t=0: integral of C(u(t)) - C(u(0)) dx."""
    C = total_conc(u, p)
    m = np.trapezoid(C, x, axis=-1)
    return m - m[..., :1]


def inflow_flux(u, p: PhysParams = DEFAULT):
    """Diffusive flux through x=0 as the generator discretises it: D (sol - u_cell0) / dx_fine."""
    return p.D * (p.sol - u[..., 0]) / FINE_DX


def all_outputs(u, x, x_query: float = 0.25, p: PhysParams = DEFAULT) -> dict:
    return {"conc_at_x": conc_at(u, x, x_query), "front_position": front_position(u, x),
            "uptake": uptake(u, x, p), "inflow_flux": inflow_flux(u, p)}
