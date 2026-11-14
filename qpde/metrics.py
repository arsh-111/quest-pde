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
    # u at the last point is ~1e-3 (outflow BC pins it near 0), so a relative error there is meaningless:
    # report the absolute error next to the typical magnitude instead.
    return {"left_mae": float(e[..., :k].mean()), "right_mae": float(e[..., -k:].mean()),
            "interior_mae": float(e[..., k:-k].mean()),
            "last_point_mae": float(e[..., -1].mean()), "last_point_mean_abs_u": float(np.abs(true[..., -1]).mean())}


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


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(a))}
