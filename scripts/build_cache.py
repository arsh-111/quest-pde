"""Build the float32 training cache from the raw HDF5, one sample at a time.

The split is by sample key index, fixed BEFORE any shuffling:
  train = keys[0:3200], val = keys[3200:3600], test = keys[3600:4000]
  test_full = keys[3600:3700] at the full 1024 spatial resolution (resolution transfer)
Spatial stride 4 (1024 -> 256 cells), temporal stride 5 (501 -> 101 times, dt = 5 s).
Usage: python -I scripts/build_cache.py data/raw/1D_diff-sorp_NA_NA.h5 data/cache
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
XS, TS = 4, 5
SPLITS = {"train": (0, 3200), "val": (3200, 3600), "test": (3600, 4000)}
FULL = (3600, 3700)

t0 = time.time()
with h5py.File(src, "r") as f:
    keys = sorted(f.keys())
    assert len(keys) >= 4000, len(keys)
    x_full = f[keys[0]]["grid/x"][:].astype(np.float64)
    t_full = f[keys[0]]["grid/t"][:].astype(np.float64)
    assert x_full.size == 1024 and t_full.size == 501, (x_full.size, t_full.size)
    meta = {"source": src.name, "x_stride": XS, "t_stride": TS, "splits": {}, "full_res_test": None}
    for name, (a, b) in SPLITS.items():
        arr = np.empty((b - a, 101, 256), dtype=np.float32)
        for j, k in enumerate(keys[a:b]):
            d = f[k]["data"]
            assert d.shape == (501, 1024, 1), d.shape
            arr[j] = d[::TS, ::XS, 0]
        assert np.isfinite(arr).all(), name
        np.save(out / f"{name}.npy", arr)
        meta["splits"][name] = {"keys": [keys[a], keys[b - 1]], "n": b - a, "shape": arr.shape}
        print(name, arr.shape, f"{time.time() - t0:.0f}s", flush=True)
    a, b = FULL
    arr = np.empty((b - a, 101, 1024), dtype=np.float32)
    for j, k in enumerate(keys[a:b]):
        arr[j] = f[k]["data"][::TS, :, 0]
    np.save(out / "test_full.npy", arr)
    meta["full_res_test"] = {"keys": [keys[a], keys[b - 1]], "n": b - a, "shape": arr.shape}
np.save(out / "x256.npy", x_full[::XS]); np.save(out / "x1024.npy", x_full); np.save(out / "t101.npy", t_full[::TS])
meta["seconds"] = round(time.time() - t0, 1)
(out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(json.dumps(meta, indent=1, default=str))
