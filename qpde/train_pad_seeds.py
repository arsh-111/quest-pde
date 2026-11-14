"""Padding comparison with 3 seeds: trains fno_pad25_s1 and _s2 (fno_pad0_s0..s2 and fno_pad25_s0 already exist).
Same settings as train_many (150 epochs, batch 32, lr 1e-3); one process, data loaded once, sequential."""
import json

import torch

from qpde.train import load_data, mem_report
from qpde.train_many import log, run

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"=== pad seeds start {json.dumps(mem_report(dev))}")
data = load_data(dev)
for s in (1, 2):
    run({"name": f"fno_pad25_s{s}", "model": "fno", "seed": s, "epochs": 150, "pad": 0.25}, data, dev)
log("=== pad seeds finished")
