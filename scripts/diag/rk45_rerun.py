"""Diagnostic: re-run the PDEBench generator's ODE (verbatim copy of Simulator.rc_ode) with solve_ivp RK45 at default
tolerances on one held-out sample, and locate the discrepancy against the data and against qpde.solver."""
import sys, time, json
import numpy as np, h5py
from scipy.integrate import solve_ivp
from scipy.sparse import diags
sys.path.insert(0, ".")
from qpde.solver import solve

key = sys.argv[1] if len(sys.argv) > 1 else "3600"
D, por, rho_s, k_f, n_f, sol, Nx = 5e-4, 0.29, 2880, 3.5e-4, 0.874, 1.0, 1024
dx = 1.0 / Nx
lap = diags([-2 * np.ones(Nx) / dx**2, np.ones(Nx - 1) / dx**2, np.ones(Nx - 1) / dx**2], [0, -1, 1]).tocsr()
rhs = np.zeros(Nx)
def rc_ode(t, y):  # verbatim logic of pdebench sim_diff_sorp.Simulator.rc_ode
    left_BC = sol
    right_BC = (y[-2] - y[-1]) / dx * D
    retardation = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (y + 1e-6) ** (n_f - 1)
    rhs[0] = D / retardation[0] / (dx**2) * left_BC
    rhs[-1] = D / retardation[-1] / (dx**2) * right_BC
    return D / retardation * (lap @ y) + rhs

f = h5py.File("data/raw/1D_diff-sorp_NA_NA.h5", "r")
d = f[key]["data"][:, :, 0].astype(float); t = f[key]["grid/t"][:].astype(float)
s = time.perf_counter()
# the generator integrated on (0, 500) with t_eval = linspace(0, 500, 501); dense output makes t_eval irrelevant
r = solve_ivp(rc_ode, (0, 500.0), d[0], t_eval=np.linspace(0, 500, 501))
rk = r.y.T[::5]; rk_sec = time.perf_counter() - s
mine = solve(d[0], t, dt=0.5)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
def rel_t(a, b): return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
out = {"key": key, "rk45_nfev": int(r.nfev), "rk45_seconds": round(rk_sec, 1),
       "rk45_rerun_vs_data": rel(rk, d), "mine_vs_data": rel(mine, d), "mine_vs_rk45_rerun": rel(mine, rk),
       "max_abs_rk45_vs_data": float(np.abs(rk - d).max())}
for name, a, b in [("rk45_vs_data", rk, d), ("mine_vs_data", mine, d), ("mine_vs_rk45", mine, rk)]:
    e = rel_t(a[1:], b[1:]); out[name + "_by_t"] = {f"t={t[i+1]:.0f}": round(float(e[i]), 5) for i in [0, 1, 4, 19, 49, 99]}
print(json.dumps(out, indent=1))
