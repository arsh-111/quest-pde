"""Fill the README's generated blocks (<!-- GEN:name --> ... <!-- /GEN:name -->) from results/*.json only."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ld = lambda n: json.loads((R / n).read_text(encoding="utf-8"))  # noqa: E731
ev, sv, dv = ld("eval_metrics.json"), ld("solver_validation.json")["summary"], ld("derived_outputs.json")
pad, bb, lat, gz = ld("padding_3seeds.json")["configs"], ld("batch_benchmark.json"), ld("latency.json"), ld("gzip_tradeoff.json")
p = lambda v: f"{100 * v:.3f}%" if v < 0.01 else f"{100 * v:.2f}%"  # noqa: E731
B = {}

L = ["| solver | vs tight BDF reference (mean / max) | vs PDEBench data (mean) | s per run (1024 cells) |", "|---|---|---|---|"]
for dt in ["2.0", "1.0", "0.5", "0.25"]:
    s = sv[f"implicit_dt{dt}"]
    L.append(f"| implicit BDF2 + Newton, dt = {dt} | {s['vs_tight_ref_mean']:.1e} / {s['vs_tight_ref_max']:.1e} | "
             f"{p(s['vs_data_mean'])} | {s['seconds_mean_1024cells_101outputs']:.2f} |")
b = sv["tight_bdf_vs_data"]
L += ["", f"Tight BDF (rtol 1e-8) vs PDEBench data, 20 held-out samples: mean **{p(b['mean'])}** (min {p(b['min'])}, "
      f"max {p(b['max'])}); by time " + ", ".join(f"{k} {p(v)}" for k, v in sv["dataset_error_by_t_mean"].items()) + "."]
B["solver"] = L

names = {"interp_u0": "interpolation in c (3,200 trajectories, no learning)", "ensemble_mean": "FNO ensemble (5 seeds, pad 0)",
         "fno_pad0_s0": "single FNO, pad 0", "fno_pad25_s0": "single FNO, pad 0.25", "cnn_s0": "CNN / U-Net baseline",
         "mean_trajectory": "mean trajectory (ignores the IC)"}
cats = ["const_in", "const_high", "smooth", "noise", "pulse", "step"]
L = ["| model | test vs **dataset** (mean / p95 / worst) | test vs **accurate solver** (mean) | left / interior / right MAE | "
     + " | ".join(f"OOD {c}" for c in cats) + " |", "|---" * (4 + len(cats)) + "|"]
for k in ["interp_u0", "ensemble_mean", "fno_pad0_s0", "fno_pad25_s0", "cnn_s0", "mean_trajectory"]:
    t, bd = ev["test"][k], ev["test"][k]["boundary"]
    d = t["rel_l2_vs_data"]
    L.append(f"| {names[k]} | {p(d['mean'])} / {p(d['p95'])} / {p(d['worst'])} | {p(t['rel_l2_vs_solver']['mean'])} | "
             f"{bd['left_mae']:.1e} / {bd['interior_mae']:.1e} / {bd['right_mae']:.1e} | "
             + " | ".join(p(ev["ood"][k][c]["mean"]) for c in cats) + " |")
f = ev["data_vs_solver_on_test"]
L += ["", f"Test = 400 held-out samples (keys 3600-3999), relative L2 over t = 5…500 on 256 points. OOD = 46 solver-generated "
      f"trajectories (6-8 per family), error vs the accurate solver; const_in is an in-range control. Dataset vs accurate "
      f"solver on the same test samples: {p(f['mean'])}. Zero-shot at 1024 points: "
      + ", ".join(f"{names.get(k, k)} {p(v['rel_l2_at_1024']['mean'])}" for k, v in ev["resolution_transfer"].items()) + "."]
B["models"] = L

L = ["| setting (3 seeds each) | test rel L2 | right-boundary MAE | left-boundary MAE | OOD mean |", "|---|---|---|---|---|"]
for k, lab in [("fno_pad0", "pad 0"), ("fno_pad25", "pad 0.25")]:
    m = pad[k]["mean_std"]
    L.append(f"| {lab} | {p(m['test_rel_l2']['mean'])} ± {p(m['test_rel_l2']['std'])} | {m['right_mae']['mean']:.1e} ± "
             f"{m['right_mae']['std']:.0e} | {m['left_mae']['mean']:.2e} | {p(m['ood_mean_rel_l2']['mean'])} |")
B["padding"] = L

sw = sorted((json.loads(q.read_text())["config"] for q in (R / "train").glob("sweep_*.json")), key=lambda c: c["n_train"])
B["data_eff"] = ["| n_train | " + " | ".join(str(c["n_train"]) for c in sw) + " |", "|---" * (len(sw) + 1) + "|",
                 "| best val rel L2 | " + " | ".join(p(c["best_val_rel_l2"]) for c in sw) + " |"]

tr = ev["trust"]
B["trust"] = ["| signal | Spearman with true error: test (n=400) | OOD (n=46) |", "|---|---|---|"] + [
    f"| {lab} | {tr[f'{k}_vs_error_test']['rho']:.2f} (p = {tr[f'{k}_vs_error_test']['p']:.2g}) | "
    f"{tr[f'{k}_vs_error_ood']['rho']:.2f} (p = {tr[f'{k}_vs_error_ood']['p']:.2g}) |"
    for k, lab in [("spread", "ensemble spread"), ("residual", "PDE residual")]]

rows = bb["rows"]
meth = list(dict.fromkeys(r["method"] for r in rows))
bs = sorted({r["batch"] for r in rows})
L = ["| method | " + " | ".join(f"batch {b} (ms/sample)" for b in bs) + " |", "|---" * (len(bs) + 1) + "|"]
for m in meth:
    L.append(f"| {m} | " + " | ".join(next((f"{r['ms_per_sample']:.2f}" for r in rows if r["method"] == m and r["batch"] == b), "–")
                                    for b in bs) + " |")
sl = json.loads((ROOT / "app" / "assets" / "summary.json").read_text(encoding="utf-8"))["timing_slowdown"]
L += ["", f"{bb['statistic']}, {bb['threads']} CPU threads, measured {bb['measured']} with {sl['background']}; batch-1 timings "
      f"were {sl['min']}–{sl['max']}× slower than the earlier quiet-machine measurement in `results/eval_metrics.json` "
      f"(single FNO {ev['timing_cpu']['fno_pad0_s0']['batch1_s'] * 1e3:.1f} ms, solver dt=1 {ev['timing_cpu']['solver_1024_dt1']['per_sample_s'] * 1e3:.0f} ms)."]
B["bench"] = L

L = ["| endpoint (localhost, gzip) | HTTP median | server compute | transferred | uncompressed |", "|---|---|---|---|---|"]
for k, v in lat.items():
    sc = v["server_side_median"]
    comp = (f"{sc['timing_ms']} ms" if "timing_ms" in sc else
            f"{sc['surrogate_ms']} ms model + {sc['solver_ms']} ms solver" if "solver_ms" in sc else "–")
    L.append(f"| {k} | {v['http_ms_median']} ms | {comp} | {v['transferred_kb_gzip']} KB | {v['response_kb_uncompressed']} KB |")
L += ["", f"Gzip trade-off on /api/compare (`results/gzip_tradeoff.json`): {gz['plain']['transferred_kb']} KB → "
      f"{gz['gzip']['transferred_kb']} KB on the wire for +{gz['gzip']['outside_compute_ms_median'] - gz['plain']['outside_compute_ms_median']:.0f} ms "
      "of server CPU on localhost."]
B["latency"] = L

lk = dv["dataset_lookup_table"]
B["derived"] = ["| output | sensitivity to c (std/mean) | ensemble vs data | data vs accurate solver |", "|---|---|---|---|"] + [
    f"| {k} | {v['sensitivity_to_c_std_over_mean']:.3f} | {p(v['ensemble_vs_data_rel_l2'])} | {p(v['data_vs_solver_rel_l2'])} |"
    for k, v in dv["outputs"].items()] + ["", "Dataset-style lookup (interpolating K stored trajectories) vs held-out data: "
                                          + ", ".join(f"K={k}: {p(v['mean_rel_l2_vs_data'])}" for k, v in lk.items()) + "."]

ib = ev["ic_vs_bc"]
B["icbc"] = [f"- IC level 0.1 → 0.2 changes the trajectory by {p(ib['rel_change_ic_0.1_to_0.2'])}; boundary value 1.0 → 1.1 "
             f"by {p(ib['rel_change_bc_1.0_to_1.1'])}.",
             "- Full IC range 0 → 0.2 by time: " + ", ".join(f"{k} {p(v)}" for k, v in ib["ic_full_range_by_t"].items()) + ".",
             f"- Ignoring the IC (mean trajectory) costs {p(ib['mean_trajectory_baseline_test_rel_l2'])} on the test set."]

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for k, lines in B.items():
    pat = re.compile(rf"(<!-- GEN:{k} -->\n).*?(<!-- /GEN:{k} -->)", re.S)
    if not pat.search(readme):
        raise SystemExit(f"README is missing block GEN:{k}")
    readme = pat.sub(lambda m: m.group(1) + "\n".join(lines) + "\n" + m.group(2), readme)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("filled blocks:", ", ".join(B))
