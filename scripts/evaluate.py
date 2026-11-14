"""Phase-1 evaluation. Every number reported in the README/UI comes from results/eval_metrics.json written here.

python -I scripts/evaluate.py --ensemble fno_pad25 [--single fno_pad0_s0 fno_pad25_s0 cnn_s0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpde import metrics as M  # noqa: E402
from qpde.models import build  # noqa: E402
from qpde.solver import solve, DEFAULT, retardation, PhysParams  # noqa: E402
from qpde.train import make_inputs  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ensemble", default="fno_pad25", help="checkpoint prefix; uses <prefix>_s0..s4")
ap.add_argument("--single", nargs="*", default=["cnn_s0", "fno_pad0_s0", "fno_pad25_s0"])
ap.add_argument("--solver_dt", type=float, default=0.5)
args = ap.parse_args()

C = ROOT / "data" / "cache"
FIG = ROOT / "figures"
torch.set_num_threads(4)
x256, x1024, t = np.load(C / "x256.npy"), np.load(C / "x1024.npy"), np.load(C / "t101.npy")
train, test, test_full = np.load(C / "train.npy"), np.load(C / "test.npy"), np.load(C / "test_full.npy")
ood = np.load(ROOT / "data" / "ood" / "ood.npz")
ood_u, ood_cat = ood["u"], ood["category"]
R = {"meta": {"n_test": len(test), "n_test_full": len(test_full), "n_ood": len(ood_u),
              "solver_dt": args.solver_dt, "torch": torch.__version__}}


def load(name):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    m = build(ck["config"]["model"], **ck["config"]["kwargs"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def predict(m, u0, x, bs=100):
    out = [m(make_inputs(u0[i:i + bs], x)).numpy() for i in range(0, len(u0), bs)]
    return np.concatenate([u0[:, None, :], np.concatenate(out)], 1)  # prepend the IC -> (B, 101, N)


# ----------------------------------------------------------------------------------------------------------------
# Ground truth from the validated solver for the test split (to measure the ~1.4% data-vs-solver floor)
sol_path = C / f"test_solver_dt{args.solver_dt}.npy"
if not sol_path.exists():
    s = time.time()
    raw = []
    for u0 in test[:, 0, :]:
        u0f = np.full(1024, float(u0.mean()))  # ICs are constant (verified), so the 1024 IC is exact
        raw.append(solve(u0f, t, dt=args.solver_dt)[:, ::4].astype(np.float32))
    np.save(sol_path, np.stack(raw))
    print(f"solver truth for test: {time.time() - s:.0f}s", flush=True)
test_sol = np.load(sol_path)
R["data_vs_solver_on_test"] = M.summary(M.rel_l2(test[:, 1:], test_sol[:, 1:]))

# ----------------------------------------------------------------------------------------------------------------
# Baselines that exploit / ignore the scalar IC
u0_train = train[:, 0, :].mean(1)
order = np.argsort(u0_train)
def interp_baseline(u0_levels):
    """Linear interpolation between the two training trajectories with the nearest IC level."""
    lv = u0_train[order]
    out = []
    for c in u0_levels:
        j = np.clip(np.searchsorted(lv, c), 1, len(lv) - 1)
        a, b = order[j - 1], order[j]
        w = 0.0 if lv[j] == lv[j - 1] else np.clip((c - lv[j - 1]) / (lv[j] - lv[j - 1]), 0, 1)
        out.append((1 - w) * train[a] + w * train[b])
    return np.stack(out)


preds_test = {"interp_u0": interp_baseline(test[:, 0, :].mean(1)),
              "mean_trajectory": np.broadcast_to(train.mean(0), test.shape).copy()}
preds_ood = {"interp_u0": interp_baseline(ood_u[:, 0, :].mean(1)),
             "mean_trajectory": np.broadcast_to(train.mean(0), ood_u.shape).copy()}
models = {n: load(n) for n in args.single}
ens_names = [f"{args.ensemble}_s{i}" for i in range(5)]
ens_names = [n for n in ens_names if (ROOT / "checkpoints" / f"{n}.pt").exists()]
ens = [load(n) for n in ens_names]
R["meta"]["ensemble_members"] = ens_names
for n, m in models.items():
    preds_test[n] = predict(m, test[:, 0], x256)
    preds_ood[n] = predict(m, ood_u[:, 0], x256)
E_test = np.stack([predict(m, test[:, 0], x256) for m in ens])
E_ood = np.stack([predict(m, ood_u[:, 0], x256) for m in ens])
preds_test["ensemble_mean"], preds_ood["ensemble_mean"] = E_test.mean(0), E_ood.mean(0)
spread_test, spread_ood = E_test[:, :, 1:].std(0).mean((1, 2)), E_ood[:, :, 1:].std(0).mean((1, 2))
R["ensemble_member_val_rel_l2"] = {}
for n in ens_names:
    R["ensemble_member_val_rel_l2"][n] = json.loads((ROOT / "results" / "train" / f"{n}.json").read_text())["config"]["best_val_rel_l2"]

# ----------------------------------------------------------------------------------------------------------------
# In-distribution accuracy (t_1..t_100; the IC itself is excluded), against data and against the solver
R["test"] = {}
curves = {}
for n, p in preds_test.items():
    e_data = M.rel_l2(p[:, 1:], test[:, 1:])
    R["test"][n] = {
        "rel_l2_vs_data": M.summary(e_data),
        "rel_l2_vs_solver": M.summary(M.rel_l2(p[:, 1:], test_sol[:, 1:])),
        "boundary": M.boundary_errors(p[:, 1:], test[:, 1:]),
        "mass": M.mass_error(p[:, 1:], test[:, 1:], x256),
    }
    curves[n] = {"x": M.error_vs_x(p[:, 1:], test[:, 1:]), "t": M.error_vs_t(p[:, 1:], test[:, 1:])}
    print(f"test {n:16s} vs data {R['test'][n]['rel_l2_vs_data']['mean']:.5f}  "
          f"vs solver {R['test'][n]['rel_l2_vs_solver']['mean']:.5f}", flush=True)
R["error_vs_t_at"] = {n: {f"t={t[i + 1]:.0f}": float(c["t"][i]) for i in [0, 4, 19, 99]} for n, c in curves.items()}


# ----------------------------------------------------------------------------------------------------------------
# PDE residual on the strided grid (interior points only; boundary columns excluded because the strided points are not
# the cell centres of a 256-cell finite-volume grid). Reported as ratio to the residual of the ground truth itself.
def residual_norm(u, p: PhysParams = DEFAULT):
    h = x256[1] - x256[0]
    ui = u[:, 1:-1, 1:-1]
    uxx = (u[:, 1:-1, :-2] - 2 * ui + u[:, 1:-1, 2:]) / h**2
    dudt = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (t[2:] - t[:-2])[None, :, None]
    r = dudt - p.D / retardation(ui, p) * uxx
    return np.sqrt((r[:, 1:] ** 2).mean((1, 2)))  # skip the first interior level (front still sub-grid)


res_true_test, res_true_ood = residual_norm(test), residual_norm(ood_u)
R["pde_residual_rms"] = {"ground_truth_test": M.summary(res_true_test), "ground_truth_ood": M.summary(res_true_ood)}
for n in preds_test:
    R["pde_residual_rms"][n + "_test"] = M.summary(residual_norm(preds_test[n]))
    R["pde_residual_rms"][n + "_ood"] = M.summary(residual_norm(preds_ood[n]))

# ----------------------------------------------------------------------------------------------------------------
# Out-of-distribution (truth = validated solver). const_in is a solver-generated in-distribution control.
R["ood"] = {}
cats = list(dict.fromkeys(ood_cat.tolist()))
for n, p in preds_ood.items():
    e = M.rel_l2(p[:, 1:], ood_u[:, 1:])
    R["ood"][n] = {c: M.summary(e[ood_cat == c]) for c in cats}
    print(f"ood  {n:16s} " + " ".join(f"{c}={R['ood'][n][c]['mean']:.4f}" for c in cats), flush=True)
ood_err_ens = M.rel_l2(preds_ood["ensemble_mean"][:, 1:], ood_u[:, 1:])
test_err_ens = M.rel_l2(preds_test["ensemble_mean"][:, 1:], test[:, 1:])

# ----------------------------------------------------------------------------------------------------------------
# Trust signals: do ensemble spread / PDE residual rank the true error?
res_ens_test, res_ens_ood = residual_norm(preds_test["ensemble_mean"]), residual_norm(preds_ood["ensemble_mean"])
R["trust"] = {
    "spread_vs_error_test": M.spearman(spread_test, test_err_ens),
    "spread_vs_error_ood": M.spearman(spread_ood, ood_err_ens),
    "spread_vs_error_pooled": M.spearman(np.r_[spread_test, spread_ood], np.r_[test_err_ens, ood_err_ens]),
    "residual_vs_error_test": M.spearman(res_ens_test, test_err_ens),
    "residual_vs_error_ood": M.spearman(res_ens_ood, ood_err_ens),
    "residual_vs_error_pooled": M.spearman(np.r_[res_ens_test, res_ens_ood], np.r_[test_err_ens, ood_err_ens]),
    "spread_test_summary": M.summary(spread_test), "spread_ood_summary": M.summary(spread_ood),
    "spread_ood_by_category": {c: M.summary(spread_ood[ood_cat == c]) for c in cats},
}
# a simple detector: flag "untrusted" if spread exceeds the 99th percentile of in-distribution spread
thr = float(np.percentile(spread_test, 99))
big = ood_err_ens > 2 * np.percentile(test_err_ens, 99)
R["trust"]["spread_threshold_p99_test"] = thr
R["trust"]["ood_flagged_fraction_by_category"] = {c: float((spread_ood[ood_cat == c] > thr).mean()) for c in cats}
R["trust"]["detector_on_ood"] = {"n_large_error": int(big.sum()),
                                 "recall_large_error": float((spread_ood[big] > thr).mean()) if big.any() else None,
                                 "flag_rate_small_error": float((spread_ood[~big] > thr).mean()) if (~big).any() else None}

# ----------------------------------------------------------------------------------------------------------------
# Resolution transfer: models trained at 256 points evaluated at the full 1024 resolution
R["resolution_transfer"] = {}
for n, m in list(models.items()) + [("ensemble_mean", None)]:
    if n == "ensemble_mean":
        p = np.mean([predict(e, test_full[:, 0], x1024) for e in ens], 0)
    else:
        p = predict(m, test_full[:, 0], x1024)
    e1024 = M.rel_l2(p[:, 1:], test_full[:, 1:])
    e256 = M.rel_l2(preds_test[n][:100, 1:], test[:100, 1:])
    R["resolution_transfer"][n] = {"rel_l2_at_1024": M.summary(e1024), "rel_l2_at_256_same_samples": M.summary(e256)}

# ----------------------------------------------------------------------------------------------------------------
# IC vs boundary influence (solver at 1024 cells): how much does the trajectory change when the IC or the boundary moves?
base = solve(np.full(1024, 0.1), t, dt=1.0)
ic_hi = solve(np.full(1024, 0.2), t, dt=1.0)
ic_lo = solve(np.full(1024, 0.0), t, dt=1.0)
bc_hi = solve(np.full(1024, 0.1), t, dt=1.0, params=PhysParams(sol=1.1))
def rel_traj(a, b):
    return float(np.linalg.norm(a[1:] - b[1:]) / np.linalg.norm(b[1:]))
R["ic_vs_bc"] = {
    "rel_change_ic_0.1_to_0.2": rel_traj(ic_hi, base),
    "rel_change_ic_0.0_to_0.2_full_range": rel_traj(ic_hi, ic_lo),
    "rel_change_bc_1.0_to_1.1": rel_traj(bc_hi, base),
    "ic_full_range_by_t": {f"t={t[i]:.0f}": float(np.linalg.norm(ic_hi[i] - ic_lo[i]) / np.linalg.norm(ic_lo[i]))
                           for i in [1, 5, 20, 100]},
    "mean_trajectory_baseline_test_rel_l2": R["test"]["mean_trajectory"]["rel_l2_vs_data"]["mean"],
}

# ----------------------------------------------------------------------------------------------------------------
# Timing on CPU (4 threads): surrogate vs solver, batch 1 and batch 100
def tm(fn, reps):
    fn()
    s = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - s) / reps
u0b = test[:100, 0]
X1 = make_inputs(test[:1, 0], x256)
X100 = make_inputs(u0b, x256)
timing = {}
with torch.no_grad():
    for n, m in models.items():
        timing[n] = {"batch1_s": tm(lambda: m(X1), 20), "batch100_per_sample_s": tm(lambda: m(X100), 3) / 100}
    timing["ensemble5"] = {"batch1_s": tm(lambda: [e(X1) for e in ens], 10),
                           "batch100_per_sample_s": tm(lambda: [e(X100) for e in ens], 2) / 100}
u0f = np.full(1024, 0.1)
timing["solver_1024_dt0.5"] = {"per_sample_s": tm(lambda: solve(u0f, t, dt=0.5), 3)}
timing["solver_1024_dt1"] = {"per_sample_s": tm(lambda: solve(u0f, t, dt=1.0), 3)}
timing["solver_256_dt1"] = {"per_sample_s": tm(lambda: solve(np.full(256, 0.1), t, dt=1.0), 3)}
R["timing_cpu"] = timing

(ROOT / "results" / "eval_metrics.json").write_text(json.dumps(R, indent=1))
np.savez(ROOT / "results" / "eval_curves.npz", **{f"{n}_x": c["x"] for n, c in curves.items()},
         **{f"{n}_t": c["t"] for n, c in curves.items()})
print(json.dumps({k: R[k] for k in ["ic_vs_bc", "timing_cpu", "data_vs_solver_on_test"]}, indent=1))
print(json.dumps(R["trust"], indent=1))
print(json.dumps(R["resolution_transfer"], indent=1))

# ----------------------------------------------------------------------------------------------------------------
# Figures
show = [n for n in ["interp_u0", "cnn_s0", "fno_pad0_s0", "fno_pad25_s0", "ensemble_mean"] if n in curves]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for n in show:
    ax[0].plot(x256, curves[n]["x"], label=n)
    ax[1].plot(t[1:], curves[n]["t"], label=n)
ax[0].set(title="RMSE vs x (test, vs data)", xlabel="x", yscale="log"); ax[0].legend()
ax[1].set(title="relative L2 vs t (test, vs data)", xlabel="t", yscale="log"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "error_vs_x_t.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
w = 0.8 / len(show)
for i, n in enumerate(show):
    ax.bar(np.arange(len(cats)) + i * w, [R["ood"][n][c]["mean"] for c in cats], w, label=n)
ax.set_xticks(np.arange(len(cats)) + 0.4 - w / 2, cats); ax.set_yscale("log")
ax.set(title="OOD mean relative L2 vs validated solver (const_in = in-distribution control)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "ood_by_category.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].scatter(spread_test, test_err_ens, s=6, label="test", alpha=.6)
for c in cats:
    k = ood_cat == c
    ax[0].scatter(spread_ood[k], ood_err_ens[k], s=18, label=c)
ax[0].set(xscale="log", yscale="log", xlabel="ensemble spread", ylabel="rel L2 of ensemble mean",
          title=f"spread vs error (Spearman test {R['trust']['spread_vs_error_test']['rho']:.2f}, "
                f"OOD {R['trust']['spread_vs_error_ood']['rho']:.2f})")
ax[0].legend(fontsize=7)
ax[1].scatter(res_ens_test, test_err_ens, s=6, alpha=.6, label="test")
for c in cats:
    k = ood_cat == c
    ax[1].scatter(res_ens_ood[k], ood_err_ens[k], s=18, label=c)
ax[1].set(xscale="log", yscale="log", xlabel="PDE residual RMS", ylabel="rel L2",
          title=f"residual vs error (Spearman test {R['trust']['residual_vs_error_test']['rho']:.2f}, "
                f"OOD {R['trust']['residual_vs_error_ood']['rho']:.2f})")
fig.tight_layout(); fig.savefig(FIG / "trust_scatter.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for row, (name, truth, pred, title) in enumerate([
        ("test", test[0], preds_test["ensemble_mean"][0], f"test key 3600 (u0={test[0, 0].mean():.3f})"),
        ("ood", ood_u[ood_cat == "pulse"][0], preds_ood["ensemble_mean"][ood_cat == "pulse"][0], "OOD pulse IC")]):
    for tk in [0, 2, 10, 40, 100]:
        ax[row, 0].plot(x256, truth[tk], "k-", alpha=.35, lw=2.5)
        ax[row, 0].plot(x256, pred[tk], "--", label=f"t={t[tk]:.0f}")
    ax[row, 0].set(title=title + ": truth (grey) vs ensemble"); ax[row, 0].legend(fontsize=7)
    im = ax[row, 1].imshow(pred - truth, aspect="auto", origin="lower", extent=[0, 1, 0, 500], cmap="RdBu_r")
    fig.colorbar(im, ax=ax[row, 1]); ax[row, 1].set(title="ensemble - truth", xlabel="x", ylabel="t")
    ax[row, 2].plot(t, truth[:, 128], "k-", alpha=.35, lw=2.5, label="truth x=0.5")
    ax[row, 2].plot(t, pred[:, 128], "--", label="ensemble x=0.5")
    ax[row, 2].set(title="concentration at x=0.5 vs t"); ax[row, 2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "examples.png", dpi=110); plt.close(fig)
print("figures written")
