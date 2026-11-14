"""Generate an out-of-distribution test set with the validated implicit solver.

Every IC is built on the 1024-cell grid, solved at 1024 cells with dt = argv[1], and stored strided to 256 x 101
exactly as the training cache is. Categories:
  const_in    spatially constant, level in (0, 0.2)  (the training distribution: a control)
  const_high  constant, level 0.25 .. 0.9            (amplitude outside (0, 0.2))
  smooth      smooth random field around 0.1
  noise       i.i.d. uniform noise around 0.1        (the "random noise" IC the brief describes)
  pulse       Gaussian pulse on a low base
  step        step in the left or right half
Writes data/ood/ood.npz and results/ood_manifest.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpde import ics  # noqa: E402
from qpde.solver import solve  # noqa: E402

DT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
rng = np.random.default_rng(20261114)
spec = []
for lv in rng.uniform(0.005, 0.195, 8):
    spec.append(("const_in", "constant", dict(level=float(lv))))
for lv in [0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
    spec.append(("const_high", "constant", dict(level=lv)))
for i in range(8):
    spec.append(("smooth", "smooth", dict(seed=100 + i, base=0.1, amplitude=float(rng.uniform(0.03, 0.09)),
                                         corr_len=float(rng.uniform(0.03, 0.3)))))
for i in range(8):
    spec.append(("noise", "noise", dict(seed=200 + i, base=0.1, noise_level=float(rng.uniform(0.02, 0.1)))))
for i in range(8):
    spec.append(("pulse", "pulse", dict(base=float(rng.uniform(0.0, 0.1)), position=float(rng.uniform(0.15, 0.85)),
                                       width=float(rng.uniform(0.02, 0.1)), height=float(rng.uniform(0.2, 0.8)))))
for i in range(6):
    spec.append(("step", "step", dict(base=float(rng.uniform(0.0, 0.1)), position=float(rng.uniform(0.2, 0.8)),
                                     height=float(rng.uniform(0.1, 0.4)))))

t = np.linspace(0, 500, 101)
U, cats, man = [], [], []
t0 = time.time()
for cat, kind, kw in spec:
    u0 = ics.make(kind, 1024, **kw)
    s = time.perf_counter()
    u = solve(u0, t, dt=DT)
    U.append(u[:, ::4].astype(np.float32))
    cats.append(cat)
    man.append({"category": cat, "kind": kind, "params": kw, "u0_min": float(u0.min()), "u0_max": float(u0.max()),
                "solve_seconds": round(time.perf_counter() - s, 3)})
    print(cat, kind, {k: round(v, 3) for k, v in kw.items()}, f"{man[-1]['solve_seconds']}s", flush=True)
(ROOT / "data" / "ood").mkdir(parents=True, exist_ok=True)
np.savez(ROOT / "data" / "ood" / "ood.npz", u=np.stack(U), category=np.array(cats))
(ROOT / "results" / "ood_manifest.json").write_text(json.dumps({"dt": DT, "seed": 20261114, "n": len(spec),
                                                               "seconds": round(time.time() - t0, 1),
                                                               "samples": man}, indent=1))
print("done", len(spec), f"{time.time() - t0:.0f}s")
