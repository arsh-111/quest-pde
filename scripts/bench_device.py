"""Measure seconds per training epoch (3200 samples, batch 32) on CPU and CUDA for each model."""
import json, time, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpde.models import build, n_params
from qpde.train import rel_l2

torch.manual_seed(0)
res = {"torch_threads": torch.get_num_threads()}
X = torch.rand(3200, 2, 256); Y = torch.rand(3200, 100, 256)
cfgs = {"fno_pad0": ("fno", dict(n_out=100, width=32, modes=16, layers=4, pad_frac=0.0)),
        "fno_pad0.25": ("fno", dict(n_out=100, width=32, modes=16, layers=4, pad_frac=0.25)),
        "cnn": ("cnn", dict(n_out=100, width=32))}
for dev in ["cpu", "cuda"]:
    for name, (k, kw) in cfgs.items():
        m = build(k, **kw).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
        Xd, Yd = X.to(dev), Y.to(dev)
        def epoch(nb):
            for i in range(nb):
                b = slice(32 * i, 32 * i + 32)
                loss = rel_l2(m(Xd[b]), Yd[b]).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if dev == "cuda": torch.cuda.synchronize()
        epoch(5)  # warm-up
        t = time.time(); epoch(30); dt = (time.time() - t) / 30 * 100
        res[f"{dev}/{name}"] = {"sec_per_epoch": round(dt, 2), "params": n_params(m)}
        print(dev, name, res[f"{dev}/{name}"], flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/device_benchmark.json").write_text(json.dumps(res, indent=1))
