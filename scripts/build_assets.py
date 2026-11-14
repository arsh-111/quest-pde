"""Build the app's small serving assets from cached data and result files (no numbers are typed by hand).

  app/assets/dataset_table.npz : K PDEBench trajectories (train split, 256 x 101) spread evenly over c, for the
                                 "dataset-style" curve (K is argv[1]; chosen from results/derived_outputs.json)
  app/assets/summary.json      : numbers and texts shown in the UI, all read from results/*.json
Trust indicator rule (fixed before the evaluation was looked at): it is shown only if Spearman rho(ensemble spread,
true error) >= 0.5 with p < 0.01 on BOTH the held-out test set and the OOD set; otherwise the raw spread is shown with
a caveat.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
A = ROOT / "app" / "assets"
A.mkdir(parents=True, exist_ok=True)
K = int(sys.argv[1]) if len(sys.argv) > 1 else 256

if "--table" in sys.argv or not (A / "dataset_table.npz").exists():  # opt-in: avoids re-committing 22 MB
    train = np.load(ROOT / "data" / "cache" / "train.npy", mmap_mode="r")
    c = np.asarray(train[:, 0].mean(1))
    order = np.argsort(c)
    pick = order[np.linspace(0, len(order) - 1, K).astype(int)]
    np.savez_compressed(A / "dataset_table.npz", c=c[pick].astype(np.float64), u=np.asarray(train[pick], dtype=np.float32))

ev = json.loads((R / "eval_metrics.json").read_text())
sv = json.loads((R / "solver_validation.json").read_text())["summary"]
dv = json.loads((R / "derived_outputs.json").read_text())
pc = json.loads((R / "padding_choice.json").read_text())
ens = "ensemble_mean"
T = ev["test"]
floor = ev["data_vs_solver_on_test"]["mean"]

tr = ev["trust"]
sp_t, sp_o = tr["spread_vs_error_test"], tr["spread_vs_error_ood"]
supported = all(s["rho"] >= 0.5 and s["p"] < 0.01 for s in (sp_t, sp_o))
cats_order = ["const_in", "const_high", "smooth", "noise", "pulse", "step"]
ood_table = [{"category": k, **{m: ev["ood"][ens][k][m] for m in ("mean", "worst", "n")}}
             for k in cats_order if k in ev["ood"][ens]]

pct = lambda v: f"{100 * v:.3f}%" if v < 0.01 else f"{100 * v:.2f}%"  # noqa: E731  (same rule as app.js)
summary = {
    "ensemble": pc["chosen"],
    "test": {"ensemble_vs_data": T[ens]["rel_l2_vs_data"], "ensemble_vs_solver": T[ens]["rel_l2_vs_solver"],
             "interp_vs_data": T["interp_u0"]["rel_l2_vs_data"], "cnn_vs_data": T["cnn_s0"]["rel_l2_vs_data"],
             "mean_trajectory_vs_data": T["mean_trajectory"]["rel_l2_vs_data"]},
    "data_vs_solver_floor": ev["data_vs_solver_on_test"],
    "ood_table": ood_table,
    "ood_note": ("const_in is a solver-generated in-distribution control; its error is the dataset-vs-solver floor "
                 f"(~{pct(floor)}) plus the surrogate's own error."),
    "trust": {"supported": supported, "spearman_test": sp_t, "spearman_ood": sp_o,
              "card_note": (f"evidence-backed: Spearman vs error {sp_t['rho']:.2f} (test), {sp_o['rho']:.2f} (OOD)"
                            if supported else
                            f"not an error estimate; above {tr['spread_threshold_p99_test']:.1e} = unfamiliar input"),
              "spread_html": (
                  f"<b>Ensemble spread</b> is shown as a raw number, not as a trust score. In-distribution it does not rank the "
                  f"true error (Spearman {sp_t['rho']:.2f}, p = {sp_t['p']:.2f}); on the OOD set it does ({sp_o['rho']:.2f}). "
                  f"Above {tr['spread_threshold_p99_test']:.1e} (99th percentile of test spread) it flagged "
                  + ", ".join(f"{k} {v:.0%}" for k, v in tr['ood_flagged_fraction_by_category'].items())
                  + " of the held-out OOD set, so read a large spread as 'unlike the training data', not as an error size."),
              "flag_threshold": tr["spread_threshold_p99_test"]},
    "timing_cpu": ev["timing_cpu"],
    "explainer_html": (
        f"<b>Accurate solver</b> (black): the PDEBench equations solved with an implicit BDF2/Newton scheme on the same "
        f"1024-cell grid; it agrees with a tight-tolerance reference to {sv['implicit_dt1.0']['vs_tight_ref_mean']:.1e} "
        f"(relative L2) at the step used here. "
        f"<b>Dataset-style</b> (orange): interpolation between stored PDEBench trajectories. The generator integrated "
        f"with SciPy RK45 at default tolerances, which leaves a {pct(sv['tight_bdf_vs_data']['mean'])} relative-L2 "
        f"time-integration error that grows with t. "
        f"<b>Surrogate</b> (blue): trained on the dataset, so it reproduces the dataset ({pct(T[ens]['rel_l2_vs_data']['mean'])} "
        f"mean on held-out test samples) and inherits its offset from the accurate solution "
        f"({pct(T[ens]['rel_l2_vs_solver']['mean'])}). Shown only for spatially constant ICs at the training physics."),
    "observables_html": (
        "The breakthrough curve at x = 1 is not shown: the outflow boundary u = D·∂u/∂x with D = 5e-4 keeps u(1) ≈ 0, "
        "so it is flat in every sample. Instead: the concentration at a chosen x, the position of the u = 0.5 front, "
        "and the total amount that has entered the column (uptake). The inflow flux at x = 0 is not shown either: it "
        "depends on 1 − u in the first cell, and even the PDEBench data differs from the accurate solver there by "
        f"{pct(dv['outputs']['inflow_flux_x0']['data_vs_solver_rel_l2'])} (the surrogate: "
        f"{pct(dv['outputs']['inflow_flux_x0']['ensemble_vs_data_rel_l2'])} vs data). Uptake is its time integral and is "
        f"robust: ensemble vs data {pct(dv['outputs']['uptake']['ensemble_vs_data_rel_l2'])}."),
    "footer_html": (f"Ensemble: 5 × FNO ({pc['chosen']}), trained on 3,200 PDEBench samples (keys 0000–3199); "
                    "test keys 3600–3999. All numbers on this page come from results/*.json produced in this project."),
}
# ---- where each model helps (A1): in-range test and OOD families, measured in results/eval_metrics.json
O = ev["ood"]
lk = dv["dataset_lookup_table"]
summary["model_guide"] = {
    "in_range": {
        "ensemble_vs_data": T[ens]["rel_l2_vs_data"]["mean"], "fno_vs_data": T["fno_pad0_s0"]["rel_l2_vs_data"]["mean"],
        "cnn_vs_data": T["cnn_s0"]["rel_l2_vs_data"]["mean"],
        "interp_full_vs_data": T["interp_u0"]["rel_l2_vs_data"]["mean"],
        "interp_app_table_vs_data": lk[str(K)]["mean_rel_l2_vs_data"], "app_table_levels": K,
        "interp_full_levels": json.loads((ROOT / "data" / "cache" / "meta.json").read_text())["splits"]["train"]["n"],
    },
    "ood_rows": [{"family": c, "ensemble": O[ens][c]["mean"], "interp": O["interp_u0"][c]["mean"],
                  "cnn": O["cnn_s0"][c]["mean"]} for c in cats_order if c in O[ens]],
}
g = summary["model_guide"]
hi = next(r for r in g["ood_rows"] if r["family"] == "const_high")
fno_better = [r for r in g["ood_rows"] if r["family"] != "const_in" and r["ensemble"] < r["interp"]]
itp_better = [r for r in g["ood_rows"] if r["family"] != "const_in" and r["ensemble"] >= r["interp"]]
fam = lambda rows: ", ".join(f"{r['family']} ({pct(r['ensemble'])} vs {pct(r['interp'])})" for r in rows)  # noqa: E731
summary["model_guide_html"] = (
    f"<b>In range (flat IC, 0 &lt; c &lt; 0.2): interpolation wins.</b> Interpolating stored trajectories in c is "
    f"{pct(g['in_range']['interp_full_vs_data'])} from held-out data with all {g['in_range']['interp_full_levels']:,} "
    f"training trajectories ({pct(g['in_range']['interp_app_table_vs_data'])} with the {K}-level table this app serves); "
    f"the FNO ensemble is {pct(g['in_range']['ensemble_vs_data'])}. "
    f"<b>Out of distribution, the FNO ensemble is better for:</b> {fam(fno_better) or 'none'} "
    f"(ensemble vs interpolation, against the accurate solver). "
    + (f"<b>Interpolation from the mean level is still better for:</b> {fam(itp_better)}, because those ICs smooth out "
       "within the first output steps. " if itp_better else "")
    + "Interpolation cannot extrapolate: above its table it clamps to the edge. The OOD set has only 6–8 samples per "
    "family, so small differences (e.g. smooth, pulse) are within noise.")
summary["two_errors_html"] = (
    "Two errors, always shown together: <b>vs dataset</b> is how well the model reproduces PDEBench's stored "
    "trajectories (what it was trained on); <b>vs accurate solver</b> is the error against the equations solved "
    f"accurately. They differ by the dataset's own RK45 tolerance error (~{pct(floor)}).")
summary["test_two_errors"] = {n: {"vs_data": T[n]["rel_l2_vs_data"]["mean"], "vs_solver": T[n]["rel_l2_vs_solver"]["mean"]}
                              for n in ["interp_u0", ens, "fno_pad0_s0", "cnn_s0"]}
bb = R / "batch_benchmark.json"
if bb.exists():
    summary["batch_benchmark"] = json.loads(bb.read_text())
pd = R / "padding_3seeds.json"
if pd.exists():
    P = json.loads(pd.read_text())["configs"]
    summary["padding"] = {k: v["mean_std"] for k, v in P.items()}
    q = lambda v: f"{100 * v:.3f}%"  # noqa: E731
    a0, a1 = P["fno_pad0"]["mean_std"], P["fno_pad25"]["mean_std"]
    summary["padding_note"] = (
        f"Padding comparison, 3 seeds each (mean ± std): test rel L2 pad0 {q(a0['test_rel_l2']['mean'])} ± "
        f"{q(a0['test_rel_l2']['std'])} vs pad 0.25 {q(a1['test_rel_l2']['mean'])} ± {q(a1['test_rel_l2']['std'])}; "
        f"right-boundary MAE {a0['right_mae']['mean']:.1e} vs {a1['right_mae']['mean']:.1e} (padding helps only there).")
else:
    summary["padding_note"] = "Padding comparison: single seed (pad0 vs pad 0.25)."
summary["trust"]["card_note"] = (f"informational only; Spearman with true error: {sp_t['rho']:.2f} in range, "
                                 f"{sp_o['rho']:.2f} on OOD")
(A / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ["test", "trust"]}, indent=1))
print("dataset table", K, "levels", f"{(A / 'dataset_table.npz').stat().st_size / 2**20:.1f} MB")
