"""Evaluation metrics on arrays shaped (B, T, N), with times t (T,) and cell positions x (N,)."""
from __future__ import annotations

import numpy as np

from qpde.solver import DEFAULT, PhysParams, EPS


def rel_l2(pred, true):
    d = (pred - true).reshape(len(pred), -1)
    return np.linalg.norm(d, axis=1) / np.linalg.norm(true.reshape(len(true), -1), axis=1)


def summary(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "worst": float(v.max()), "n": int(v.size)}


def error_vs_x(pred, true):
    """RMSE over samples and times at each x."""
    return np.sqrt(((pred - true) ** 2).mean(axis=(0, 1)))


def error_vs_t(pred, true):
    """Relative L2 over x, averaged over samples, at each time."""
    num = np.linalg.norm(pred - true, axis=2)
    den = np.linalg.norm(true, axis=2)
    return (num / den).mean(axis=0)


def boundary_errors(pred, true, k=4):
    """Errors at the first/last k grid points, the interior, and on the breakthrough curve u(t, x_last)."""
    e = np.abs(pred - true)
    bt = rel_l2(pred[:, :, -1:], true[:, :, -1:])
    return {"left_mae": float(e[..., :k].mean()), "right_mae": float(e[..., -k:].mean()),
            "interior_mae": float(e[..., k:-k].mean()),
            "breakthrough_rel_l2": summary(bt)}


def total_mass(u, x, p: PhysParams = DEFAULT):
    """Total (dissolved + sorbed) concentration integral: integral of u + (1-por)/por*rho_s*k_f*u^n_f dx.

    The PDE R(u) u_t = D u_xx conserves this quantity up to the boundary fluxes
    (d/du of c u^n_f equals the retardation term). Uses the trapezoid rule over the given points.
    """
    c = (1 - p.por) / p.por * p.rho_s * p.k_f
    tot = u + c * (np.maximum(u, 0) + EPS) ** p.n_f
    return np.trapezoid(tot, x, axis=-1)


def mass_error(pred, true, x):
    mp, mt = total_mass(pred, x), total_mass(true, x)
    return {"mean_rel_abs_mass_error": float((np.abs(mp - mt) / np.abs(mt)).mean()),
            "max_rel_abs_mass_error": float((np.abs(mp - mt) / np.abs(mt)).max())}


def flux_balance_residual(u, t, x, p: PhysParams = DEFAULT):
    """|dM/dt - (flux_in - flux_out)| / |flux_in| per sample (mean over time), from one-sided differences.

    flux_in = D (sol - u_0) / (x_0 - 0) at the left (Dirichlet), flux_out ~ D (u_{N-2} - u_{N-1}) / dx is
    replaced by the discrete outflow that follows from the generator's right-ghost rule. This is approximate on
    a strided grid, so it is only meaningful as a COMPARISON between the surrogate and the ground truth
    evaluated in the same way.
    """
    M = total_mass(u, x, p)
    dMdt = np.gradient(M, t, axis=1)
    dx = x[1] - x[0]
    fin = p.D * (p.sol - u[..., 0]) / (x[0] + 1e-12) / 2 * 1.0  # u_0 sits half a fine cell from the wall
    ghost = p.D * (u[..., -2] - u[..., -1]) / dx
    fout = p.D * (u[..., -1] - ghost) / dx
    r = np.abs(dMdt - (fin - fout))
    return (r / (np.abs(fin) + 1e-12)).mean(axis=1)


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(a))}
