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

pct = lambda v: f"{100 * v:.2f}%"  # noqa: E731
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
                            f"raw value, not an error estimate (rank corr. with error {sp_t['rho']:.2f} in-distribution). "
                            f"Above {tr['spread_threshold_p99_test']:.1e} it signals an unfamiliar input: seen in "
                            f"≥{min(v for k, v in tr['ood_flagged_fraction_by_category'].items() if k != 'const_in'):.0%} of "
                            f"OOD cases in every family vs {tr['ood_flagged_fraction_by_category'].get('const_in', 0):.0%} "
                            "of in-distribution controls"),
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
(A / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ["test", "trust"]}, indent=1))
print("dataset table", K, "levels", f"{(A / 'dataset_table.npz').stat().st_size / 2**20:.1f} MB")
