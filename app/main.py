"""FastAPI service: JSON API + static no-build frontend.

Run: .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.service import (C_RANGE, PARAM_RANGES, ROOT, SOLVER_DT, T, arr, get_service, outputs, rel_l2)
from qpde.solver import DEFAULT, PhysParams

app = FastAPI(title="PDE solutions as a service: 1D diffusion-sorption", version="0.1")
STATIC = ROOT / "app" / "static"


class IC(BaseModel):
    kind: Literal["constant", "smooth", "noise", "pulse", "step", "custom"] = "constant"
    level: float | None = Field(None, ge=0.0, le=2.0)
    seed: int | None = Field(None, ge=0, le=10**6)
    base: float | None = Field(None, ge=0.0, le=2.0)
    amplitude: float | None = Field(None, ge=0.0, le=2.0)
    corr_len: float | None = Field(None, gt=0.0, le=1.0)
    noise_level: float | None = Field(None, ge=0.0, le=2.0)
    position: float | None = Field(None, ge=0.0, le=1.0)
    width: float | None = Field(None, gt=0.0, le=1.0)
    height: float | None = Field(None, ge=0.0, le=2.0)
    values: list[float] | None = Field(None, min_length=2, max_length=4096)

    def spec(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class Params(BaseModel):
    D: float = DEFAULT.D
    por: float = DEFAULT.por
    rho_s: float = DEFAULT.rho_s
    k_f: float = DEFAULT.k_f
    n_f: float = DEFAULT.n_f
    sol: float = DEFAULT.sol

    @field_validator("*")
    @classmethod
    def in_range(cls, v, info):
        lo, hi = PARAM_RANGES[info.field_name]
        if not (lo - 1e-12 <= v <= hi + 1e-12):
            raise ValueError(f"{info.field_name} must be in [{lo}, {hi}]")
        return v

    def phys(self) -> PhysParams:
        return PhysParams(**self.model_dump())


class Req(BaseModel):
    ic: IC = IC(kind="constant", level=0.1)
    params: Params = Params()
    model: Literal["ensemble", "fno", "cnn"] = "ensemble"
    nx: Literal[256, 1024] = 256
    x_query: float = Field(0.25, ge=0.0, le=1.0)


class SweepReq(BaseModel):
    param: Literal["D", "por", "rho_s", "k_f", "n_f", "sol"] = "D"
    level: float = Field(0.1, ge=0.0, le=0.2)
    n: int = Field(7, ge=3, le=11)


def _ic(req: Req):
    try:
        return get_service().make_ic(req.ic.spec())
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, f"invalid initial condition: {e}") from e


@app.get("/api/health")
def health():
    s = get_service()
    return {"status": "ok", "ensemble_members": len(s.ensemble), "ensemble": s.ens_prefix}


@app.get("/api/meta")
def meta():
    s = get_service()
    return {
        "t": T.tolist(), "nx_options": [256, 1024], "c_range": C_RANGE,
        "dataset_table_c_range": [float(s.table_c[0]), float(s.table_c[-1])],
        "params_default": DEFAULT.to_dict(), "param_ranges": PARAM_RANGES, "solver_dt": SOLVER_DT,
        "models": {"ensemble": f"5 x FNO ({s.ens_prefix}), mean and spread", "fno": f"single FNO ({s.ens_prefix}_s0)",
                   "cnn": "CNN/U-Net baseline (cnn_s0)"},
        "evaluation": s.summary,
    }


@app.post("/api/predict")
def predict(req: Req):
    s = get_service()
    u0 = _ic(req)
    p = req.params.phys()
    r = s.predict(u0, req.model, req.nx)
    out = {"t": T.tolist(), "x": arr(r["x"]), "u": arr(r["u"]), "timing_ms": round(r["ms"], 2), "model": req.model,
           "outputs": outputs(r["u"], r["x"], req.x_query, p), "ood": s.ood_check(u0, p)}
    if "spread" in r:
        out["spread"] = arr(r["spread"])
        out["spread_mean"] = float(r["spread"][1:].mean())
    return out


@app.post("/api/solve")
def solve_ep(req: Req):
    s = get_service()
    u0 = _ic(req)
    p = req.params.phys()
    r = s.solve(u0, p, req.nx)
    if not np.isfinite(r["u"]).all():
        raise HTTPException(500, "solver produced non-finite values")
    return {"t": T.tolist(), "x": arr(r["x"]), "u": arr(r["u"]), "timing_ms": round(r["ms"], 2),
            "solver": f"implicit BDF2 + Newton, 1024 cells, dt={SOLVER_DT}",
            "outputs": outputs(r["u"], r["x"], req.x_query, p)}


@app.post("/api/compare")
def compare(req: Req):
    s = get_service()
    u0 = _ic(req)
    p = req.params.phys()
    t0 = time.perf_counter()
    sur = s.predict(u0, req.model, req.nx)
    sol = s.solve(u0, p, req.nx)
    ood = s.ood_check(u0, p)
    ds = None
    if not ood["ood"]:
        ds = s.dataset_style(ood["ic_level"], req.nx)
    res = {
        "t": T.tolist(), "x": arr(sur["x"]), "model": req.model, "ood": ood,
        "surrogate": {"u": arr(sur["u"]), "timing_ms": round(sur["ms"], 2),
                      "outputs": outputs(sur["u"], sur["x"], req.x_query, DEFAULT)},
        "solver": {"u": arr(sol["u"]), "timing_ms": round(sol["ms"], 2),
                   "outputs": outputs(sol["u"], sol["x"], req.x_query, p)},
        "abs_error": arr(np.abs(sur["u"] - sol["u"])),
        "metrics": {"rel_l2_surrogate_vs_solver": rel_l2(sur["u"], sol["u"]),
                    "max_abs_surrogate_vs_solver": float(np.abs(sur["u"] - sol["u"]).max())},
    }
    if "spread" in sur:
        res["surrogate"]["spread_mean"] = float(sur["spread"][1:].mean())
    if ds is not None:
        res["dataset_style"] = {"u": arr(ds["u"]), "outputs": outputs(ds["u"], ds["x"], req.x_query, p),
                                "how": "interpolation in c between stored PDEBench trajectories (RK45, default tolerance)"}
        res["metrics"]["rel_l2_surrogate_vs_dataset_style"] = rel_l2(sur["u"], ds["u"])
        res["metrics"]["rel_l2_dataset_style_vs_solver"] = rel_l2(ds["u"], sol["u"])
    res["total_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    return res


@app.post("/api/physics_sweep")
def physics_sweep(req: SweepReq):
    """Surrogate (fixed physics) vs solver while one physical parameter moves across its range."""
    s = get_service()
    lo, hi = PARAM_RANGES[req.param]
    default = getattr(DEFAULT, req.param)
    vals = np.unique(np.r_[np.geomspace(lo, hi, req.n) if lo > 0 else np.linspace(lo, hi, req.n), default])
    u0 = s.make_ic({"kind": "constant", "level": req.level})
    sur = s.predict(u0, "ensemble", 256)
    rows = []
    for v in vals:
        p = PhysParams(**{**DEFAULT.to_dict(), req.param: float(v)})
        sol = s.solve(u0, p, 256)
        rows.append({"value": float(v), "rel_l2_surrogate_vs_solver": rel_l2(sur["u"], sol["u"]),
                     "is_default": bool(abs(v - default) < 1e-12 * max(1, abs(default)))})
    return {"param": req.param, "default": default, "level": req.level, "rows": rows}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
