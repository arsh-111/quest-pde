"""Inspect the PDEBench diffusion-sorption HDF5 lazily (never loads the whole file).

Usage: python -I scripts/inspect_data.py data/raw/1D_diff-sorp_NA_NA.h5
Writes results/data_inspection.json and figures/sample_trajectories.png.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1])
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

info = {"file": path.name, "size_bytes": path.stat().st_size}
with h5py.File(path, "r") as f:
    keys = list(f.keys())
    info["n_top_level_keys"] = len(keys)
    info["first_keys"] = keys[:5]
    info["last_keys"] = keys[-3:]
    info["root_attrs"] = {k: str(v)[:200] for k, v in f.attrs.items()}
    g = f[keys[0]]
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            info.setdefault("sample_layout", {})[name] = {
                "shape": obj.shape, "dtype": str(obj.dtype), "chunks": obj.chunks,
                "compression": obj.compression}
    g.visititems(walk)
    info["sample_attrs_keys"] = list(g.attrs.keys())
    cfg = g.attrs.get("config")
    if cfg is not None:
        info["sample0_config_yaml"] = str(cfg)
    x = g["grid/x"][:]
    t = g["grid/t"][:]
    info["x"] = {"n": int(x.size), "first": float(x[0]), "last": float(x[-1]), "dx": float(x[1] - x[0])}
    info["t"] = {"n": int(t.size), "first": float(t[0]), "last": float(t[-1]), "dt": float(t[1] - t[0])}

    # Statistics over a spread of samples: how spatially varying is the IC? ranges? NaNs?
    idx = np.linspace(0, len(keys) - 1, 200).astype(int)
    ic_std, ic_mean, umin, umax, nonfinite = [], [], [], [], 0
    for i in idx:
        d = f[keys[i]]["data"]
        u0 = d[0, :, 0]
        ic_std.append(float(u0.std()))
        ic_mean.append(float(u0.mean()))
        arr = d[::10, :, 0]
        nonfinite += int((~np.isfinite(arr)).sum())
        umin.append(float(np.nanmin(arr)))
        umax.append(float(np.nanmax(arr)))
    info["ic_stats_200_samples"] = {
        "max_spatial_std_of_u0": max(ic_std),
        "mean_u0_min": min(ic_mean), "mean_u0_max": max(ic_mean),
        "u_min_over_samples": min(umin), "u_max_over_samples": max(umax),
        "nonfinite_values_in_checked_slices": nonfinite,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, i in enumerate(idx[:4]):
        u = f[keys[i]]["data"][:, :, 0]
        for tk, ls in zip([0, len(t)//10, len(t)//2, len(t)-1], ["-", "--", "-.", ":"]):
            axes[0].plot(x, u[tk], ls, color=f"C{j}", lw=1, label=f"{keys[i]} t={t[tk]:.0f}" if j == 0 else None)
        axes[2].plot(t, u[:, -1], color=f"C{j}", label=keys[i])
    u = f[keys[idx[0]]]["data"][:, :, 0]
    im = axes[1].imshow(u, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]])
    fig.colorbar(im, ax=axes[1])
    axes[0].set(title="profiles (colour = sample)", xlabel="x", ylabel="u"); axes[0].legend(fontsize=7)
    axes[1].set(title=f"sample {keys[idx[0]]}", xlabel="x", ylabel="t")
    axes[2].set(title="breakthrough u(t, x_last)", xlabel="t", ylabel="u"); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sample_trajectories.png", dpi=110)

(ROOT / "results" / "data_inspection.json").write_text(json.dumps(info, indent=2, default=str))
print(json.dumps({k: v for k, v in info.items() if k != "sample0_config_yaml"}, indent=2, default=str))
