"""Model/solver service used by the FastAPI app. Everything here runs on the CPU."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from qpde import derived as Dv
from qpde import ics
from qpde.models import build
from qpde.solver import DEFAULT, PhysParams, solve
from qpde.train import make_inputs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
ASSETS = ROOT / "app" / "assets"
RESULTS = ROOT / "results"

NX_FINE = 1024
T = np.linspace(0.0, 500.0, 101)
C_RANGE = (0.0, 0.2)  # training IC levels: U(0, 0.2), observed (0.0012, 0.1999) in the inspected samples
SOLVER_DT = 1.0       # interactive solver step: 2.4e-4 rel L2 vs tight BDF (results/solver_validation.json)
PARAM_RANGES = {      # physics-shift lab slider ranges (factor ~0.25x .. 4x around the defaults, physically sensible)
    "D": (1.25e-4, 2e-3), "por": (0.1, 0.6), "rho_s": (1000.0, 4000.0),
    "k_f": (5e-5, 1.4e-3), "n_f": (0.5, 1.0), "sol": (0.5, 1.0),
}

torch.set_num_threads(4)


def x_grid(nx: int) -> np.ndarray:
    xf = (np.arange(NX_FINE) + 0.5) / NX_FINE
    stride = NX_FINE // nx
    return xf[::stride]


def _load(name: str):
    ck = torch.load(CKPT / f"{name}.pt", map_location="cpu", weights_only=False)
    m = build(ck["config"]["model"], **ck["config"]["kwargs"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["config"]


class Service:
    def __init__(self):
        self.summary = json.loads((ASSETS / "summary.json").read_text())  # built by scripts/build_assets.py
        self.ens_prefix = self.summary["ensemble"]
        self.ensemble = [_load(f"{self.ens_prefix}_s{i}")[0] for i in range(5)]
        self.fno, self.fno_cfg = _load(f"{self.ens_prefix}_s0")
        self.cnn, self.cnn_cfg = _load("cnn_s0")
        tab = np.load(ASSETS / "dataset_table.npz")
        self.table_c, self.table_u = tab["c"], tab["u"]  # dataset trajectories sorted by c, (K,), (K, 101, 256)

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def make_ic(spec: dict, nx: int = NX_FINE) -> np.ndarray:
        kind = spec.get("kind", "constant")
        if kind == "custom":
            v = np.asarray(spec["values"], dtype=np.float64)
            if v.ndim != 1 or v.size < 2 or not np.isfinite(v).all():
                raise ValueError("custom IC must be a finite 1-D list with at least 2 values")
            xs = (np.arange(v.size) + 0.5) / v.size
            return np.interp(x_grid(nx), xs, v)
        kw = {k: float(v) for k, v in spec.items() if k != "kind"}
        if "seed" in kw:
            kw["seed"] = int(kw["seed"])
        return ics.make(kind, nx, **kw)

    @staticmethod
    def ood_check(u0: np.ndarray, params: PhysParams) -> dict:
        reasons = []
        std = float(u0.std())
        lvl = float(u0.mean())
        if std > 1e-6:
            reasons.append(f"IC is not spatially constant (std {std:.3g}); training ICs were constant levels")
        if u0.min() < C_RANGE[0] - 1e-9 or u0.max() > C_RANGE[1] + 1e-9:
            reasons.append(f"IC values outside the training range {C_RANGE} (min {u0.min():.3g}, max {u0.max():.3g})")
        changed = [k for k, v in params.to_dict().items() if abs(v - getattr(DEFAULT, k)) > 1e-12 * max(1, abs(v))]
        if changed:
            reasons.append("physical parameters differ from training values: " + ", ".join(changed)
                           + " (the surrogate cannot see them)")
        return {"ood": bool(reasons), "reasons": reasons, "ic_level": lvl, "ic_std": std}

    # ------------------------------------------------------------------ models
    def _table_interp(self, level: float) -> tuple[np.ndarray, bool]:
        """Linear interpolation in c between stored dataset trajectories (256-point grid). Clamps outside the table."""
        lv = self.table_c
        clamped = not (lv[0] <= level <= lv[-1])
        c = float(np.clip(level, lv[0], lv[-1]))
        j = int(np.clip(np.searchsorted(lv, c), 1, len(lv) - 1))
        w = 0.0 if lv[j] == lv[j - 1] else float(np.clip((c - lv[j - 1]) / (lv[j] - lv[j - 1]), 0, 1))
        return ((1 - w) * self.table_u[j - 1] + w * self.table_u[j]).astype(np.float64), clamped

    def interp(self, u0_fine: np.ndarray, nx: int) -> dict:
        """The interpolation baseline: sees only the mean IC level; 1024 output is linear upsampling in x."""
        t0 = time.perf_counter()
        u, clamped = self._table_interp(float(u0_fine.mean()))
        if nx != 256:
            xs, xt = x_grid(256), x_grid(nx)
            u = np.stack([np.interp(xt, xs, row) for row in u])
        return {"u": u, "x": x_grid(nx), "ms": (time.perf_counter() - t0) * 1e3, "clamped": clamped}

    def predict(self, u0_fine: np.ndarray, model: str, nx: int) -> dict:
        if model == "interp":
            return self.interp(u0_fine, nx)
        x = x_grid(nx)
        u0 = u0_fine[:: NX_FINE // nx][None]
        X = make_inputs(u0, x)
        t0 = time.perf_counter()
        spread = None
        with torch.no_grad():
            if model == "ensemble":
                outs = np.stack([m(X).numpy()[0] for m in self.ensemble])
                pred, spread = outs.mean(0), outs.std(0)
            elif model == "fno":
                pred = self.fno(X).numpy()[0]
            elif model == "cnn":
                if nx % 4:
                    raise ValueError("cnn needs nx divisible by 4")
                pred = self.cnn(X).numpy()[0]
            else:
                raise ValueError(f"unknown model {model}")
        ms = (time.perf_counter() - t0) * 1e3
        u = np.concatenate([u0, pred], 0)
        out = {"u": u, "x": x, "ms": ms}
        if spread is not None:
            out["spread"] = np.concatenate([np.zeros_like(u0), spread], 0)
        return out

    def solve(self, u0_fine: np.ndarray, params: PhysParams, nx: int, dt: float = SOLVER_DT) -> dict:
        t0 = time.perf_counter()
        u = solve(u0_fine, T, params=params, dt=dt)
        ms = (time.perf_counter() - t0) * 1e3
        return {"u": u[:, :: NX_FINE // nx], "x": x_grid(nx), "ms": ms}

    def dataset_style(self, level: float, nx: int) -> dict | None:
        """Interpolate real PDEBench trajectories in c (constant ICs, default physics only); 256-point grid only."""
        if nx != 256 or not (self.table_c[0] <= level <= self.table_c[-1]):
            return None
        u, _ = self._table_interp(level)
        return {"u": u, "x": x_grid(256), "ms": 0.0}


def outputs(u: np.ndarray, x: np.ndarray, x_query: float, p: PhysParams) -> dict:
    o = Dv.all_outputs(u, x, x_query, p)
    return {k: _clean(v) for k, v in o.items()}


def _clean(a):
    a = np.asarray(a, dtype=np.float64)
    return [None if not np.isfinite(v) else round(float(v), 7) for v in a.ravel()] if a.ndim else float(a)


def arr(a, nd=5):  # 1e-5 resolution: far below every displayed error (smallest: interp vs data ~5e-4 relative)
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def rel_l2(a, b) -> float:
    return float(np.linalg.norm(a[1:] - b[1:]) / np.linalg.norm(b[1:]))


@lru_cache(maxsize=1)
def get_service() -> Service:
    return Service()
