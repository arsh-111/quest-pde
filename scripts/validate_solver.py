"""Validate the implicit reference solver (qpde.solver) on held-out dataset trajectories.

Held-out keys 3600.. (the test split; never used for training) at full resolution (1024 cells x 101 times).
For each sample:
  * qpde.solver at dt in DTS vs the PDEBench data
  * qpde.solver vs a tight-tolerance BDF solution of the SAME semi-discrete ODE (scipy, rtol 1e-8, analytic Jacobian)
    -> the solver's own discretisation error
  * the tight BDF solution vs the data -> the dataset's own integration error
Diagnostics in scripts/diag/ show that the generator's verbatim RK45 reproduces the data to ~2e-8 and that its RHS equals
qpde.solver's to ~5e-14, so the remaining gap is the data's RK45 (default rtol=1e-3) error.
Writes results/solver_validation.json and figures/solver_validation.png.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpde.solver import solve, _rhs_and_jac, DEFAULT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DTS = [2.0, 1.0, 0.5, 0.25]
dx = 1 / 1024


def rhs(t, y):
    return _rhs_and_jac(y, DEFAULT, dx)[0]


def jac(t, y):
    ab = _rhs_and_jac(y, DEFAULT, dx)[1]
    return diags([ab[2, :-1], ab[1], ab[0, 1:]], [-1, 0, 1], format="csc")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rel_t(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


res = {"keys": [], "per_sample": []}
curves = {"data_vs_ref_by_t": [], "mine_vs_data_by_t": []}
with h5py.File(ROOT / "data" / "raw" / "1D_diff-sorp_NA_NA.h5", "r") as f:
    keys = sorted(f.keys())[3600:3600 + N_SAMPLES]
    t = f[keys[0]]["grid/t"][:].astype(np.float64)
    for k in keys:
        data = f[k]["data"][:, :, 0].astype(np.float64)
        u0 = data[0]
        s = time.perf_counter()
        ref = solve_ivp(rhs, (0, t[-1]), u0, t_eval=t, method="BDF", rtol=1e-8, atol=1e-10, jac=jac).y.T
        rec = {"key": k, "u0": float(u0.mean()), "tight_bdf_vs_data": rel(ref, data),
               "tight_bdf_seconds": time.perf_counter() - s, "implicit": {}}
        for dt in DTS:
            s = time.perf_counter()
            u = solve(u0, t, dt=dt)
            rec["implicit"][str(dt)] = {"vs_data": rel(u, data), "vs_tight_ref": rel(u, ref),
                                        "max_abs_vs_data": float(np.abs(u - data).max()),
                                        "seconds": time.perf_counter() - s}
            if dt == 1.0:
                keep = (data, u, ref)
                curves["mine_vs_data_by_t"].append(rel_t(u[1:], data[1:]))
        curves["data_vs_ref_by_t"].append(rel_t(data[1:], ref[1:]))
        res["per_sample"].append(rec)
        print(k, f"u0={rec['u0']:.4f} bdf_vs_data={rec['tight_bdf_vs_data']:.5f}",
              {d: (round(v["vs_data"], 5), f"{v['vs_tight_ref']:.1e}") for d, v in rec["implicit"].items()}, flush=True)

summ = {"n_samples": len(keys), "keys": [keys[0], keys[-1]]}
for dt in DTS:
    g = lambda m: np.array([s["implicit"][str(dt)][m] for s in res["per_sample"]])  # noqa: E731
    summ[f"implicit_dt{dt}"] = {"vs_data_mean": float(g("vs_data").mean()), "vs_data_max": float(g("vs_data").max()),
                                "vs_tight_ref_mean": float(g("vs_tight_ref").mean()),
                                "vs_tight_ref_max": float(g("vs_tight_ref").max()),
                                "seconds_mean_1024cells_101outputs": float(g("seconds").mean())}
b = np.array([s["tight_bdf_vs_data"] for s in res["per_sample"]])
summ["tight_bdf_vs_data"] = {"mean": float(b.mean()), "min": float(b.min()), "max": float(b.max())}
summ["dataset_error_by_t_mean"] = {f"t={t[i + 1]:.0f}": float(np.mean(curves["data_vs_ref_by_t"], 0)[i])
                                   for i in [0, 1, 4, 19, 49, 99]}
res["summary"] = summ
(ROOT / "results" / "solver_validation.json").write_text(json.dumps(res, indent=1))
print(json.dumps(summ, indent=1))

data, u, ref = keep
x = (np.arange(1024) + 0.5) / 1024
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for tk in [1, 10, 40, 100]:
    ax[0].plot(x, data[tk], "k-", lw=2.5, alpha=.35)
    ax[0].plot(x, u[tk], "--", lw=1.2, label=f"t={t[tk]:.0f}")
ax[0].set(title=f"key {keys[-1]}: data (grey) vs implicit solver dt=1 (dashed)", xlabel="x", ylabel="u")
ax[0].legend()
im = ax[1].imshow(u - data, aspect="auto", origin="lower", extent=[0, 1, 0, t[-1]], cmap="RdBu_r")
fig.colorbar(im, ax=ax[1]); ax[1].set(title="implicit - data (signed)", xlabel="x", ylabel="t")
ax[2].plot(t[1:], np.mean(curves["data_vs_ref_by_t"], 0), label="data vs tight BDF (dataset error)")
ax[2].plot(t[1:], np.mean(curves["mine_vs_data_by_t"], 0), "--", label="implicit dt=1 vs data")
ax[2].set(title=f"relative L2 vs t (mean over {len(keys)} held-out samples)", xlabel="t", yscale="log"); ax[2].legend()
fig.tight_layout(); fig.savefig(ROOT / "figures" / "solver_validation.png", dpi=110)
