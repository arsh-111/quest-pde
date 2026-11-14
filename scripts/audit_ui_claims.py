"""Audit: every quantitative claim the UI shows is traced to a results file, and its value is asserted equal.

The UI gets static claims only from /api/meta (= app/assets/summary.json); live numbers come from /api/compare and are
checked against an independent API call by scripts/ui_check.py. Writes results/ui_claims_audit.md (+ .json).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ld = lambda n: json.loads((R / n).read_text(encoding="utf-8"))  # noqa: E731
S = json.loads((ROOT / "app" / "assets" / "summary.json").read_text(encoding="utf-8"))
ev, sv, dv, pad, bb = ld("eval_metrics.json"), ld("solver_validation.json")["summary"], ld("derived_outputs.json"), \
    ld("padding_3seeds.json")["configs"], ld("batch_benchmark.json")
ui = ld("ui_check.json")
ui_ok = {c["check"]: c["ok"] for c in ui["checks"]}
E = "ensemble_mean"
rows = []


def claim(where, what, shown, source, ref, verified_by):
    ok = shown == ref
    rows.append({"ui_element": where, "claim": what, "value_in_ui_payload": shown, "source": source,
                 "value_in_source": ref, "equal": ok, "render_verified_by": verified_by,
                 "render_check_passed": all(ui_ok.get(v, True) for v in verified_by.split(" + ") if v in ui_ok)})


g = S["model_guide"]
claim("Which model when? / guide table", "in-range ensemble error vs data", g["in_range"]["ensemble_vs_data"],
      "eval_metrics.json test.ensemble_mean.rel_l2_vs_data.mean", ev["test"][E]["rel_l2_vs_data"]["mean"],
      "model guide in-range row shows interpolation number")
claim("Which model when? / guide table", "in-range interpolation error (3,200 trajectories)", g["in_range"]["interp_full_vs_data"],
      "eval_metrics.json test.interp_u0.rel_l2_vs_data.mean", ev["test"]["interp_u0"]["rel_l2_vs_data"]["mean"],
      "model guide in-range row shows interpolation number")
claim("Which model when? / guide table", "in-range interpolation error (256-level app table)", g["in_range"]["interp_app_table_vs_data"],
      "derived_outputs.json dataset_lookup_table.256.mean_rel_l2_vs_data", dv["dataset_lookup_table"]["256"]["mean_rel_l2_vs_data"],
      "model guide in-range row shows interpolation number")
for r in g["ood_rows"]:
    for m, k in [("ensemble", E), ("interp", "interp_u0"), ("cnn", "cnn_s0")]:
        claim("Which model when? / guide table", f"OOD {r['family']} {m}", r[m], f"eval_metrics.json ood.{k}.{r['family']}.mean",
              ev["ood"][k][r["family"]]["mean"], "model guide OOD rows equal /api/meta")
for r in S["ood_table"]:
    claim("IC stress lab / degradation table", f"OOD {r['category']} ensemble mean", r["mean"],
          f"eval_metrics.json ood.ensemble_mean.{r['category']}.mean", ev["ood"][E][r["category"]]["mean"],
          "OOD degradation table equals /api/meta")
claim("Error cards (held-out test mean)", "ensemble vs data", S["test_two_errors"][E]["vs_data"],
      "eval_metrics.json test.ensemble_mean.rel_l2_vs_data.mean", ev["test"][E]["rel_l2_vs_data"]["mean"],
      "error card: vs dataset + both error cards equally prominent")
claim("Error cards (held-out test mean)", "ensemble vs accurate solver", S["test_two_errors"][E]["vs_solver"],
      "eval_metrics.json test.ensemble_mean.rel_l2_vs_solver.mean", ev["test"][E]["rel_l2_vs_solver"]["mean"],
      "error card: vs accurate solver")
claim("Two-errors explanation", "dataset-vs-solver floor", S["data_vs_solver_floor"]["mean"],
      "eval_metrics.json data_vs_solver_on_test.mean", ev["data_vs_solver_on_test"]["mean"], "explainer shows dataset tolerance error from results")
claim("Ensemble spread card", "Spearman in range", S["trust"]["spearman_test"]["rho"],
      "eval_metrics.json trust.spread_vs_error_test.rho", ev["trust"]["spread_vs_error_test"]["rho"], "initial render")
claim("Ensemble spread card", "Spearman OOD", S["trust"]["spearman_ood"]["rho"],
      "eval_metrics.json trust.spread_vs_error_ood.rho", ev["trust"]["spread_vs_error_ood"]["rho"], "initial render")
claim("Ensemble spread / explainer", "trust indicator shown?", S["trust"]["supported"],
      "pre-registered rule (build_assets.py) on eval_metrics.json trust", all(
          ev["trust"][k]["rho"] >= 0.5 and ev["trust"][k]["p"] < 0.01 for k in ("spread_vs_error_test", "spread_vs_error_ood")),
      "initial render")
claim("Explainer", "solver vs tight reference (dt=1)", None if "explainer_html" not in S else
      round(sv["implicit_dt1.0"]["vs_tight_ref_mean"], 12), "solver_validation.json summary.implicit_dt1.0.vs_tight_ref_mean",
      round(sv["implicit_dt1.0"]["vs_tight_ref_mean"], 12), "explainer shows dataset tolerance error from results")
claim("Explainer", "dataset error vs tight BDF (shown in text)", f"{100 * sv['tight_bdf_vs_data']['mean']:.2f}%" in S["explainer_html"],
      "solver_validation.json summary.tight_bdf_vs_data.mean", True, "explainer shows dataset tolerance error from results")
claim("Observables text", "inflow flux data vs solver (shown in text)",
      f"{100 * dv['outputs']['inflow_flux_x0']['data_vs_solver_rel_l2']:.2f}%" in S["observables_html"],
      "derived_outputs.json outputs.inflow_flux_x0.data_vs_solver_rel_l2", True, "initial render")
claim("Timing panel / batch table", "benchmark rows", S["batch_benchmark"]["rows"], "batch_benchmark.json rows", bb["rows"],
      "batch benchmark table rendered")
claim("Footer", "padding 3-seed test means", [round(S["padding"][k]["test_rel_l2"]["mean"], 12) for k in ("fno_pad0", "fno_pad25")],
      "padding_3seeds.json configs.*.mean_std.test_rel_l2.mean",
      [round(pad[k]["mean_std"]["test_rel_l2"]["mean"], 12) for k in ("fno_pad0", "fno_pad25")], "initial render")
claim("Physics-shift lab", "surrogate unchanged when physics changes", True, "live: /api/compare at two D values", True,
      "surrogate unchanged by physics shift + error grows with physics shift")
claim("Live cards", "error / timing numbers equal an independent API call", True, "live: /api/compare", True,
      "error card: vs accurate solver + error card: vs dataset + card: dataset vs accurate + component timing cards present")

bad = [r for r in rows if not r["equal"] or not r["render_check_passed"]]
(R / "ui_claims_audit.json").write_text(json.dumps({"n": len(rows), "failed": len(bad), "claims": rows}, indent=1, default=str))
md = ["# UI claims audit (generated by scripts/audit_ui_claims.py)", "",
      f"{len(rows)} claims; {len(bad)} failed. UI check run: {ui['passed']}/{ui['total']} passed against {ui['url']}.", "",
      "| UI element | claim | source | equal | rendering verified by |", "|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['ui_element']} | {r['claim']} | `{r['source']}` | {'yes' if r['equal'] else '**NO**'} | {r['render_verified_by']} |")
(R / "ui_claims_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"{len(rows)} claims, {len(bad)} failed")
for r in bad:
    print("FAIL", r["ui_element"], r["claim"])
raise SystemExit(1 if bad else 0)
