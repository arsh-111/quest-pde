"""API tests: run against the real checkpoints and assets (no mocks)."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from qpde.solver import solve

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["ensemble_members"] == 5


def test_meta_numbers_come_from_result_files():
    m = client.get("/api/meta").json()
    ev = json.loads((ROOT / "results" / "eval_metrics.json").read_text())
    assert m["evaluation"]["test"]["ensemble_vs_data"] == ev["test"]["ensemble_mean"]["rel_l2_vs_data"]
    assert m["evaluation"]["data_vs_solver_floor"] == ev["data_vs_solver_on_test"]
    assert len(m["t"]) == 101 and m["c_range"] == [0.0, 0.2]


def test_predict_shapes_and_ic_row():
    r = client.post("/api/predict", json={"ic": {"kind": "constant", "level": 0.1}, "model": "ensemble"}).json()
    u = np.array(r["u"])
    assert u.shape == (101, 256) and np.allclose(u[0], 0.1)
    assert not r["ood"]["ood"] and "spread" in r
    assert set(r["outputs"]) == {"conc_at_x", "front_position", "uptake", "inflow_flux"}


@pytest.mark.parametrize("model", ["fno", "cnn"])
def test_predict_other_models(model):
    r = client.post("/api/predict", json={"ic": {"kind": "constant", "level": 0.05}, "model": model})
    assert r.status_code == 200 and np.array(r.json()["u"]).shape == (101, 256)


def test_predict_1024_resolution():
    r = client.post("/api/predict", json={"ic": {"kind": "constant", "level": 0.1}, "model": "fno", "nx": 1024}).json()
    assert np.array(r["u"]).shape == (101, 1024)


def test_solve_matches_library_solver():
    r = client.post("/api/solve", json={"ic": {"kind": "constant", "level": 0.1}}).json()
    ref = solve(np.full(1024, 0.1), np.linspace(0, 500, 101), dt=1.0)[:, ::4]
    assert np.abs(np.array(r["u"]) - ref).max() < 1e-5


def test_compare_in_distribution_has_three_curves_and_consistent_metric():
    r = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.1}}).json()
    assert "dataset_style" in r and not r["ood"]["ood"]
    s, v = np.array(r["surrogate"]["u"]), np.array(r["solver"]["u"])
    rel = np.linalg.norm(s[1:] - v[1:]) / np.linalg.norm(v[1:])
    assert abs(rel - r["metrics"]["rel_l2_surrogate_vs_solver"]) < 1e-4
    # the dataset-style curve differs from the accurate solver by roughly the generator's tolerance error
    assert 0.005 < r["metrics"]["rel_l2_dataset_style_vs_solver"] < 0.03


def test_compare_flags_ood_level_and_noise_ic():
    hi = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.35}}).json()
    assert hi["ood"]["ood"] and "dataset_style" not in hi
    nz = client.post("/api/compare", json={"ic": {"kind": "noise", "seed": 1, "base": 0.1, "noise_level": 0.05}}).json()
    assert nz["ood"]["ood"] and any("not spatially constant" in s for s in nz["ood"]["reasons"])


def test_physics_shift_flags_and_changes_solver_only():
    base = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.1}}).json()
    shifted = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.1},
                                                "params": {"D": 1e-3}}).json()
    assert shifted["ood"]["ood"] and any("D" in s for s in shifted["ood"]["reasons"])
    assert np.allclose(base["surrogate"]["u"], shifted["surrogate"]["u"])           # surrogate physics is fixed
    assert shifted["metrics"]["rel_l2_surrogate_vs_solver"] > base["metrics"]["rel_l2_surrogate_vs_solver"]


def test_physics_sweep_minimum_at_default():
    r = client.post("/api/physics_sweep", json={"param": "D", "level": 0.1, "n": 5}).json()
    errs = {row["value"]: row["rel_l2_surrogate_vs_solver"] for row in r["rows"]}
    assert min(errs, key=errs.get) == pytest.approx(r["default"])


def test_custom_csv_ic_and_validation_errors():
    ok = client.post("/api/predict", json={"ic": {"kind": "custom", "values": [0.1, 0.12, 0.08, 0.1]}})
    assert ok.status_code == 200
    assert client.post("/api/predict", json={"ic": {"kind": "custom", "values": [0.1]}}).status_code == 422
    assert client.post("/api/predict", json={"ic": {"kind": "constant", "level": -1}}).status_code == 422
    assert client.post("/api/solve", json={"params": {"D": 1.0}}).status_code == 422
    assert client.post("/api/predict", json={"model": "nope"}).status_code == 422


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "Sorption Surrogate Lab" in r.text


def test_interp_model_and_component_timings():
    r = client.post("/api/predict", json={"ic": {"kind": "constant", "level": 0.1}, "model": "interp"}).json()
    u = np.array(r["u"])
    assert u.shape == (101, 256) and np.allclose(u[0], 0.1, atol=1e-3)
    c = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.1}, "model": "ensemble"}).json()
    assert set(c["component_ms"]) == {"surrogate", "interp", "solver"}
    assert c["component_ms"]["solver"] > c["component_ms"]["surrogate"]


def test_interp_flags_clamping_outside_table():
    c = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.35}, "model": "interp"}).json()
    assert any("clamps" in s for s in c["ood"]["reasons"])


def test_gzip_and_rounding():
    r = client.post("/api/compare", json={"ic": {"kind": "constant", "level": 0.1}}, headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"
    vals = r.json()["solver"]["u"][50][:50]
    assert all(abs(v * 1e5 - round(v * 1e5)) < 1e-6 for v in vals)


def test_live_benchmark_endpoint():
    rows = client.post("/api/benchmark").json()["rows"]
    methods = {row["method"] for row in rows}
    assert {"fno", "ensemble", "cnn", "interp"} <= methods and any(m.startswith("solver") for m in methods)


def test_plotly_is_served_locally():
    r = client.get("/static/vendor/plotly-cartesian-2.35.2.min.js")
    assert r.status_code == 200 and "plotly.js" in r.text[:200]
    assert "cdn." not in client.get("/").text


def test_model_guide_numbers_come_from_result_files():
    g = client.get("/api/meta").json()["evaluation"]["model_guide"]
    ev = json.loads((ROOT / "results" / "eval_metrics.json").read_text())
    assert g["in_range"]["interp_full_vs_data"] == ev["test"]["interp_u0"]["rel_l2_vs_data"]["mean"]
    hi = next(r for r in g["ood_rows"] if r["family"] == "const_high")
    assert hi["ensemble"] == ev["ood"]["ensemble_mean"]["const_high"]["mean"]
    assert hi["interp"] == ev["ood"]["interp_u0"]["const_high"]["mean"]
