"""Which derived outputs does the data support, and how well does the surrogate reproduce them?

For each candidate (u at x=0.1/0.25/0.5, front position, uptake, inflow flux at x=0), on the 400 test samples:
  * sensitivity to the IC level c: across-sample std / mean magnitude (time-averaged) -> does the output carry
    information about c at all?
  * errors of the ensemble and the interpolation baseline vs the dataset and vs the accurate solver
Also sizes the "dataset-style" lookup: interpolating K dataset trajectories in c, error vs held-out data, for several K.
Writes results/derived_outputs.json and figures/derived_outputs.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpde import derived as Dv  # noqa: E402
from qpde.models import build  # noqa: E402
from qpde.train import make_inputs  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = ROOT / "data" / "cache"
ens_prefix = json.loads((ROOT / "results" / "padding_choice.json").read_text())["chosen"]
x, t = np.load(C / "x256.npy"), np.load(C / "t101.npy")
train, test = np.load(C / "train.npy"), np.load(C / "test.npy")
test_sol = np.load(C / "test_solver_dt0.5.npy")


def load(name):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    m = build(ck["config"]["model"], **ck["config"]["kwargs"]); m.load_state_dict(ck["state_dict"])
    return m.eval()


with torch.no_grad():
    X = make_inputs(test[:, 0], x)
    ens = np.mean([np.concatenate([test[:, None, 0], load(f"{ens_prefix}_s{i}")(X).numpy()], 1) for i in range(5)], 0)

c_train = train[:, 0].mean(1)


def interp_table(levels_idx, c_query):
    lv, tab = c_train[levels_idx], train[levels_idx]
    o = np.argsort(lv); lv, tab = lv[o], tab[o]
    out = []
    for c in c_query:
        j = int(np.clip(np.searchsorted(lv, c), 1, len(lv) - 1))
        w = float(np.clip((c - lv[j - 1]) / (lv[j] - lv[j - 1]), 0, 1)) if lv[j] > lv[j - 1] else 0.0
        out.append((1 - w) * tab[j - 1] + w * tab[j])
    return np.stack(out)


c_test = test[:, 0].mean(1)
interp_all = interp_table(np.arange(len(train)), c_test)


def outs(u):
    o = {f"u_at_x={q}": Dv.conc_at(u, x, q) for q in [0.1, 0.25, 0.5]}
    o["front_position_u0.5"] = Dv.front_position(u, x)
    o["uptake"] = Dv.uptake(u, x)
    o["inflow_flux_x0"] = Dv.inflow_flux(u)
    return o


O = {"data": outs(test), "solver": outs(test_sol), "ensemble": outs(ens), "interp_c": outs(interp_all)}
res = {"ensemble_prefix": ens_prefix, "outputs": {}}
for k in O["data"]:
    d = O["data"][k][:, 1:]
    mag = np.nanmean(np.abs(d))
    sens = float(np.nanmean(np.nanstd(d, axis=0)) / mag)
    r = {"sensitivity_to_c_std_over_mean": sens, "mean_magnitude": float(mag)}
    for src, ref in [("ensemble", "data"), ("ensemble", "solver"), ("interp_c", "data"), ("data", "solver")]:
        a, b = O[src][k][:, 1:], O[ref][k][:, 1:]
        ok = np.isfinite(a) & np.isfinite(b)
        r[f"{src}_vs_{ref}_rel_l2"] = float(np.linalg.norm(np.where(ok, a - b, 0)) / np.linalg.norm(np.where(ok, b, 0)))
        r[f"{src}_vs_{ref}_nan_mismatch"] = int((np.isfinite(a) != np.isfinite(b)).sum())
    # signal-to-error: how many "error units" the c-dependence spans
    r["c_signal_over_ensemble_error"] = sens / max(r["ensemble_vs_data_rel_l2"], 1e-12)
    res["outputs"][k] = r
    print(f"{k:22s} sens {sens:.4f}  ens-vs-data {r['ensemble_vs_data_rel_l2']:.4f}  "
          f"ens-vs-solver {r['ensemble_vs_solver_rel_l2']:.4f}  data-vs-solver {r['data_vs_solver_rel_l2']:.4f}", flush=True)

res["dataset_lookup_table"] = {}
rng = np.random.default_rng(0)
for K in [32, 64, 128, 256, 512, len(train)]:
    idx = np.linspace(0, len(train) - 1, K).astype(int) if K < len(train) else np.arange(len(train))
    idx = np.argsort(c_train)[idx] if K < len(train) else idx
    p = interp_table(idx, c_test)
    e = np.linalg.norm((p - test)[:, 1:].reshape(len(test), -1), axis=1) / np.linalg.norm(test[:, 1:].reshape(len(test), -1), axis=1)
    res["dataset_lookup_table"][str(K)] = {"mean_rel_l2_vs_data": float(e.mean()), "max": float(e.max()),
                                           "size_mb_float32": round(K * 101 * 256 * 4 / 2**20, 1)}
    print("lookup K", K, res["dataset_lookup_table"][str(K)], flush=True)

(ROOT / "results" / "derived_outputs.json").write_text(json.dumps(res, indent=1))

i_lo, i_hi = np.argmin(c_test), np.argmax(c_test)
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for a, k in zip(ax, ["u_at_x=0.25", "front_position_u0.5", "uptake", "inflow_flux_x0"]):
    for i, col in [(i_lo, "C0"), (i_hi, "C3")]:
        a.plot(t, O["data"][k][i], "-", color=col, lw=2.5, alpha=.4, label=f"data c={c_test[i]:.3f}")
        a.plot(t, O["solver"][k][i], ":", color=col, lw=1.5, label="accurate solver")
        a.plot(t, O["ensemble"][k][i], "--", color=col, lw=1, label="FNO ensemble")
    a.set(title=k, xlabel="t")
    if k == "inflow_flux_x0":
        a.set_yscale("log")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "derived_outputs.png", dpi=110)
