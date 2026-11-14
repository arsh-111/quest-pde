"""CPU batch benchmark (4 threads): per-sample cost of each method vs batch size. Writes results/batch_benchmark.json.

Methods: single FNO, 5-FNO ensemble, CNN, interpolation in c (256-level table, as served), implicit solver
(1024 cells; dt=1 as served, dt=0.5 as used for ground truth; run serially, so per-sample cost does not drop with batch).
Run on an otherwise idle machine.
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.service import Service, x_grid  # noqa: E402
from qpde.solver import solve  # noqa: E402
from qpde.train import make_inputs  # noqa: E402

torch.set_num_threads(4)
s = Service()
x = x_grid(256)
t = np.linspace(0, 500, 101)
rng = np.random.default_rng(0)
REPS = {1: 50, 10: 20, 100: 7}
rows = []


def bench(fn, reps):
    """Median of individually timed repetitions (robust to background-load spikes)."""
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


for bs, reps in REPS.items():
    lv = rng.uniform(0, 0.2, bs)
    X = make_inputs(lv[:, None] * np.ones((1, 256)), x)
    with torch.no_grad():
        for name, fn in [("single FNO", lambda: s.fno(X)), ("5-FNO ensemble", lambda: [m(X) for m in s.ensemble]),
                         ("CNN", lambda: s.cnn(X))]:
            rows.append({"method": name, "batch": bs, "ms_per_sample": bench(fn, reps) * 1e3 / bs})
    rows.append({"method": "interpolation in c", "batch": bs,
                 "ms_per_sample": bench(lambda: [s._table_interp(float(c)) for c in lv], reps) * 1e3 / bs})
    print(bs, [f"{r['method']}={r['ms_per_sample']:.3f}" for r in rows if r["batch"] == bs], flush=True)
for dt in (1.0, 0.5):
    sec = bench(lambda: solve(np.full(1024, 0.1), t, dt=dt), 7)
    rows.append({"method": f"implicit solver (1024 cells, dt={dt})", "batch": 1, "ms_per_sample": sec * 1e3})
    print("solver", dt, f"{sec * 1e3:.1f} ms", flush=True)
import psutil  # noqa: E402
out = {"statistic": "median of individually timed repetitions",
       "background_cpu_percent_after": psutil.cpu_percent(interval=1.0), "threads": torch.get_num_threads(), "cpu": platform.processor(), "torch": torch.__version__, "rows": rows,
       "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
(ROOT / "results" / "batch_benchmark.json").write_text(json.dumps(out, indent=1))
