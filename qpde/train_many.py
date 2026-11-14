"""Run every Phase-1 training job in ONE process, strictly sequentially, with the data loaded once.

Order: FNO pad0 s0 -> FNO pad25 s0 -> ensemble seeds 1..4 of the padding with the lower validation error
       -> data-efficiency sweep (same config, seed 0, n_train in SWEEP_N, fixed step budget).
Finished jobs (checkpoint present) are skipped, so a restart continues where it stopped; an interrupted job
resumes from its last 10-epoch state in checkpoints_tmp/. Memory is logged before each run, and the process
aborts (does not retry) if available system RAM falls below MIN_AVAILABLE_GB.
Log: runs/train_many.log
"""
from __future__ import annotations

import json
import sys
import time

import torch

from qpde.train import ROOT, load_data, mem_report, train_run

EPOCHS = 150
SWEEP_N = [25, 50, 100, 200, 400, 800, 1600, 3200]
SWEEP_STEPS = 6000
MIN_AVAILABLE_GB = 1.5
LOG = ROOT / "runs" / "train_many.log"


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(name: str) -> dict | None:
    p = ROOT / "results" / "train" / f"{name}.json"
    return json.loads(p.read_text())["config"] if (ROOT / "checkpoints" / f"{name}.pt").exists() and p.exists() else None


def run(spec, data, dev):
    prev = done(spec["name"])
    if prev:
        log(f"[{spec['name']}] already finished (best val {prev['best_val_rel_l2']:.4e}), skipping")
        return prev
    m = mem_report(dev)
    log(f"[{spec['name']}] start {json.dumps(spec)} mem {json.dumps(m)}")
    if m["sys_available_gb"] < MIN_AVAILABLE_GB:
        log(f"ABORT: available RAM {m['sys_available_gb']} GB < {MIN_AVAILABLE_GB} GB before {spec['name']}")
        sys.exit(3)
    cfg = train_run(spec, data, dev, log=log)
    log(f"[{spec['name']}] finished best val {cfg['best_val_rel_l2']:.4e} in {cfg['train_seconds']}s "
        f"mem after cleanup {json.dumps(mem_report(dev))}")
    return cfg


def main():
    LOG.parent.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"=== train_many start, device {dev}, mem before data load {json.dumps(mem_report(dev))}")
    data = load_data(dev)
    log(f"data loaded: train {tuple(data['train'][1].shape)} val {tuple(data['val'][1].shape)} "
        f"mem {json.dumps(mem_report(dev))}")
    a = run({"name": "fno_pad0_s0", "model": "fno", "seed": 0, "epochs": EPOCHS, "pad": 0.0}, data, dev)
    b = run({"name": "fno_pad25_s0", "model": "fno", "seed": 0, "epochs": EPOCHS, "pad": 0.25}, data, dev)
    pad, tag = (0.0, "fno_pad0") if a["best_val_rel_l2"] <= b["best_val_rel_l2"] else (0.25, "fno_pad25")
    log(f"padding choice by validation: {tag} (pad0 {a['best_val_rel_l2']:.4e} vs pad25 {b['best_val_rel_l2']:.4e})")
    (ROOT / "results" / "padding_choice.json").write_text(json.dumps(
        {"chosen": tag, "val_pad0": a["best_val_rel_l2"], "val_pad25": b["best_val_rel_l2"]}, indent=1))
    for s in range(1, 5):
        run({"name": f"{tag}_s{s}", "model": "fno", "seed": s, "epochs": EPOCHS, "pad": pad}, data, dev)
    for n in SWEEP_N:
        run({"name": f"sweep_{tag}_n{n}", "model": "fno", "seed": 0, "steps": SWEEP_STEPS, "pad": pad,
             "n_train": n}, data, dev)
    log("=== train_many finished")


if __name__ == "__main__":
    main()
