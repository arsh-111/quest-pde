"""Padding comparison with 3 seeds (pad 0 vs pad 0.25; seeds 0-2; identical training settings).
Test (400 held-out, vs data) and OOD (46, vs solver). Writes results/padding_3seeds.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qpde import metrics as M  # noqa: E402
from qpde.models import build  # noqa: E402
from qpde.train import make_inputs  # noqa: E402

C = ROOT / "data" / "cache"
x, test = np.load(C / "x256.npy"), np.load(C / "test.npy")
ood = np.load(ROOT / "data" / "ood" / "ood.npz")
ood_u, ood_cat = ood["u"], ood["category"]


def run(name, u0):
    ck = torch.load(ROOT / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    m = build(ck["config"]["model"], **ck["config"]["kwargs"]); m.load_state_dict(ck["state_dict"]); m.eval()
    with torch.no_grad():
        return np.concatenate([u0[:, None], m(make_inputs(u0, x)).numpy()], 1), ck["config"]["best_val_rel_l2"]


res = {"seeds": [0, 1, 2], "configs": {}}
for tag in ["fno_pad0", "fno_pad25"]:
    per = []
    for s in range(3):
        p, val = run(f"{tag}_s{s}", test[:, 0])
        q, _ = run(f"{tag}_s{s}", ood_u[:, 0])
        b = M.boundary_errors(p[:, 1:], test[:, 1:])
        e_ood = M.rel_l2(q[:, 1:], ood_u[:, 1:])
        per.append({"seed": s, "val": val, "test_rel_l2": float(M.rel_l2(p[:, 1:], test[:, 1:]).mean()),
                    "left_mae": b["left_mae"], "right_mae": b["right_mae"], "interior_mae": b["interior_mae"],
                    "ood_mean_rel_l2": float(e_ood.mean()),
                    "ood_by_family": {c: float(e_ood[ood_cat == c].mean()) for c in dict.fromkeys(ood_cat.tolist())}})
    agg = {k: {"mean": float(np.mean([r[k] for r in per])), "std": float(np.std([r[k] for r in per], ddof=1))}
           for k in ["val", "test_rel_l2", "left_mae", "right_mae", "interior_mae", "ood_mean_rel_l2"]}
    res["configs"][tag] = {"per_seed": per, "mean_std": agg}
    print(tag, {k: f"{v['mean']:.3e}±{v['std']:.1e}" for k, v in agg.items()}, flush=True)
(ROOT / "results" / "padding_3seeds.json").write_text(json.dumps(res, indent=1))
