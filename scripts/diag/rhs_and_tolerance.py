"""Diagnostic: (1) does qpde.solver's RHS equal the generator's rc_ode pointwise on data states?
(2) does the generator's own RHS, integrated at tight tolerance, converge to the data or to qpde.solver?"""
import sys, json, time
import numpy as np, h5py
from scipy.integrate import solve_ivp
from scipy.sparse import diags
sys.path.insert(0, ".")
from qpde.solver import solve, _rhs_and_jac, DEFAULT
np.seterr(invalid="ignore")
D, por, rho_s, k_f, n_f, sol, Nx = 5e-4, 0.29, 2880, 3.5e-4, 0.874, 1.0, 1024
dx = 1.0 / Nx
lap = diags([-2 * np.ones(Nx) / dx**2, np.ones(Nx - 1) / dx**2, np.ones(Nx - 1) / dx**2], [0, -1, 1]).tocsr()
def rc_ode(t, y):
    rhs = np.zeros(Nx)
    R = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (y + 1e-6) ** (n_f - 1)
    rhs[0] = D / R[0] / dx**2 * sol
    rhs[-1] = D / R[-1] / dx**2 * ((y[-2] - y[-1]) / dx * D)
    return D / R * (lap @ y) + rhs
def jac(t, y):
    ab = _rhs_and_jac(y, DEFAULT, dx)[1]
    return diags([ab[2, :-1], ab[1], ab[0, 1:]], [-1, 0, 1], format="csc")
f = h5py.File("data/raw/1D_diff-sorp_NA_NA.h5", "r")
d = f["3600/data"][:, :, 0].astype(float); t = f["3600/grid/t"][:].astype(float)
out = {}
rel_rhs = [np.linalg.norm(rc_ode(0, y) - _rhs_and_jac(y, DEFAULT, dx)[0]) / np.linalg.norm(rc_ode(0, y)) for y in d[1:]]
out["max_rel_rhs_difference_on_data_states"] = float(max(rel_rhs))
out["min_u_in_sample"] = float(d.min())
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
mine = solve(d[0], t, dt=0.1)
for meth, rtol, atol in [("BDF", 1e-8, 1e-10), ("Radau", 1e-10, 1e-12), ("RK45", 1e-6, 1e-9)]:
    s = time.perf_counter()
    kw = {"jac": jac} if meth != "RK45" else {}
    y = solve_ivp(rc_ode, (0, 500.0), d[0], t_eval=t, method=meth, rtol=rtol, atol=atol, **kw).y.T
    out[f"{meth}_rtol{rtol}"] = {"vs_data": rel(y, d), "vs_mine_dt0.1": rel(y, mine), "seconds": round(time.perf_counter() - s, 1)}
    print(meth, out[f"{meth}_rtol{rtol}"], flush=True)
print(json.dumps(out, indent=1))
json.dump(out, open("results/diag_rhs_tolerance.json", "w"), indent=1)
