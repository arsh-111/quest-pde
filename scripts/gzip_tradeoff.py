"""Measure /api/compare with and without gzip (alternating blocks, 8 calls each). Writes results/gzip_tradeoff.json."""
import gzip, json, statistics, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
B = json.dumps({"ic": {"kind": "constant", "level": 0.1}}).encode()
def one(gz):
    h = {"Content-Type": "application/json"} | ({"Accept-Encoding": "gzip"} if gz else {})
    t = time.perf_counter(); raw = urllib.request.urlopen(urllib.request.Request(URL + "/api/compare", data=B, headers=h)).read()
    ms = (time.perf_counter() - t) * 1e3; j = json.loads(gzip.decompress(raw) if gz else raw)
    return ms, ms - j["total_ms"], len(raw)
res = {"plain": [], "gzip": []}
for gz in (False, True, False, True):
    res["gzip" if gz else "plain"] += [one(gz) for _ in range(8)]
out = {k: {"http_ms_median": round(statistics.median(x[0] for x in v), 1),
           "outside_compute_ms_median": round(statistics.median(x[1] for x in v), 1),
           "transferred_kb": round(v[0][2] / 1024, 1), "n": len(v)} for k, v in res.items()}
out["note"] = "localhost; outside_compute = HTTP time minus server-side compute (JSON encoding, gzip, transfer)"
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "results/gzip_tradeoff.json", "w"), indent=1)
print(json.dumps(out, indent=1))
