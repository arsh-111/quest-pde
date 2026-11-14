"""Measure API latency over HTTP (10 calls each, localhost, CPU). Writes results/latency.json."""
import json, statistics, sys, time, urllib.request
from pathlib import Path
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
import gzip
SIZES = {}
def call(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Accept-Encoding": "gzip"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read(); enc = r.headers.get("Content-Encoding")
    ms = (time.perf_counter() - t) * 1e3
    SIZES[path] = len(raw)
    return ms, (gzip.decompress(raw) if enc == "gzip" else raw)
res = {}
cases = {
    "predict_ensemble_256": ("/api/predict", {"ic": {"kind": "constant", "level": 0.1}, "model": "ensemble"}),
    "predict_fno_256": ("/api/predict", {"ic": {"kind": "constant", "level": 0.1}, "model": "fno"}),
    "predict_fno_1024": ("/api/predict", {"ic": {"kind": "constant", "level": 0.1}, "model": "fno", "nx": 1024}),
    "solve_256": ("/api/solve", {"ic": {"kind": "constant", "level": 0.1}}),
    "compare_ensemble_256": ("/api/compare", {"ic": {"kind": "constant", "level": 0.1}}),
    "compare_ensemble_1024": ("/api/compare", {"ic": {"kind": "constant", "level": 0.1}, "nx": 1024}),
    "predict_interp_256": ("/api/predict", {"ic": {"kind": "constant", "level": 0.1}, "model": "interp"}),
    "physics_sweep_7": ("/api/physics_sweep", {"param": "D", "level": 0.1, "n": 7}),
}
for name, (path, body) in cases.items():
    call(path, body)  # warm-up
    n = 3 if "sweep" in name else 10
    ms, server = [], []
    for _ in range(n):
        m, raw = call(path, body); ms.append(m); j = json.loads(raw)
        server.append({k: j.get(k) for k in ["timing_ms", "total_ms"] if k in j} |
                      ({"surrogate_ms": j["surrogate"]["timing_ms"], "solver_ms": j["solver"]["timing_ms"]} if "surrogate" in j else {}))
    agg = {k: round(statistics.median([s[k] for s in server]), 1) for k in server[0]}
    res[name] = {"http_ms_median": round(statistics.median(ms), 1), "http_ms_max": round(max(ms), 1),
                 "response_kb_uncompressed": round(len(raw) / 1024, 1), "transferred_kb_gzip": round(SIZES[path] / 1024, 1), "server_side_median": agg, "n": n}
    print(name, res[name], flush=True)
Path(sys.argv[2] if len(sys.argv) > 2 else "results/latency.json").write_text(json.dumps(res, indent=1))
