"""Training core: resumable single runs on data that is loaded once and kept on the device.

Single run:  python -m qpde.train --model fno --seed 0 --epochs 150 --pad 0.25 --name fno_pad25_s0
Many runs (one process, strictly sequential): python -m qpde.train_many (see that module)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from qpde.models import build, n_params

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
TMP = ROOT / "checkpoints_tmp"  # resumable state, gitignored


def make_inputs(u0, x) -> torch.Tensor:
    """u0: (B, N) -> (B, 2, N) with channels [u0, x]."""
    u0 = torch.as_tensor(np.ascontiguousarray(u0), dtype=torch.float32)
    xx = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32).expand_as(u0)
    return torch.stack([u0, xx], 1)


def rel_l2(pred, true):
    """Per-sample relative L2 over all (t, x) entries."""
    d = (pred - true).flatten(1).norm(dim=1)
    return d / true.flatten(1).norm(dim=1)


def mem_report(dev) -> dict:
    import psutil
    vm = psutil.virtual_memory()
    r = {"proc_rss_gb": round(psutil.Process().memory_info().rss / 2**30, 3),
         "sys_available_gb": round(vm.available / 2**30, 3)}
    if dev.type == "cuda":
        r["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        r["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
    return r


def load_data(dev) -> dict:
    """Load the cache once, move it to the device and drop the CPU copies."""
    x = np.load(CACHE / "x256.npy")
    out = {"x": x}
    for split in ["train", "val"]:
        a = np.load(CACHE / f"{split}.npy", mmap_mode="r")
        X = make_inputs(a[:, 0], x).to(dev)
        Y = torch.from_numpy(np.ascontiguousarray(a[:, 1:])).to(dev)
        out[split] = (X, Y)
        del a
    gc.collect()
    return out


def train_run(spec: dict, data: dict, dev, log=print) -> dict:
    """Train one model; resumes from TMP/<name>.state.pt if present. Returns the final config dict."""
    name = spec["name"]
    seed, bs, lr = spec.get("seed", 0), spec.get("batch", 32), spec.get("lr", 1e-3)
    Xtr, Ytr = data["train"]
    n_train = spec.get("n_train") or len(Xtr)
    Xtr, Ytr = Xtr[:n_train], Ytr[:n_train]
    Xva, Yva = data["val"]
    spe = math.ceil(n_train / bs)
    epochs = spec.get("epochs") or math.ceil(spec["steps"] / spe)
    val_every = max(1, math.ceil(100 / spe))          # validate about every 100 optimiser steps
    ckpt_every = max(10, math.ceil(1000 / spe))       # resumable state about every 1000 steps (>= 10 epochs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = dict(n_out=Ytr.shape[1], width=spec.get("width", 32))
    if spec["model"] == "fno":
        kw.update(modes=spec.get("modes", 16), layers=spec.get("layers", 4), pad_frac=spec.get("pad", 0.0))
    model = build(spec["model"], **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=spe)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hist, best, best_state, start_ep, t_prev = [], float("inf"), None, 0, 0.0

    TMP.mkdir(exist_ok=True)
    state_path = TMP / f"{name}.state.pt"
    if state_path.exists():
        # load on the CPU: the RNG state must stay a CPU ByteTensor; load_state_dict moves model/optimizer tensors
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        gen.set_state(st["gen"]); hist, best, best_state = st["hist"], st["best"], st["best_state"]
        start_ep, t_prev = st["epoch"], st["seconds"]
        log(f"[{name}] resumed at epoch {start_ep}")

    t_start = time.time() - t_prev
    for ep in range(start_ep, epochs):
        model.train()
        te = time.time()
        perm = torch.randperm(n_train, generator=gen)
        tot = 0.0
        for i in range(0, n_train, bs):
            b = perm[i:i + bs].to(dev)
            loss = rel_l2(model(Xtr[b]), Ytr[b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - te
        last = ep + 1 == epochs
        if (ep + 1) % val_every and not last:
            continue
        model.eval()
        with torch.no_grad():
            v = torch.cat([rel_l2(model(Xva[i:i + 100]), Yva[i:i + 100]) for i in range(0, len(Xva), 100)]).mean().item()
        hist.append({"epoch": ep + 1, "train_rel_l2": tot / n_train, "val_rel_l2": v, "epoch_seconds": round(ep_time, 3)})
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if (ep + 1) % ckpt_every == 0 or last:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "gen": gen.get_state(), "hist": hist, "best": best, "best_state": best_state,
                        "epoch": ep + 1, "seconds": time.time() - t_start}, state_path)
            log(f"[{name}] ep {ep + 1}/{epochs} train {tot / n_train:.4e} val {v:.4e} best {best:.4e} "
                f"({ep_time:.1f}s/ep) {json.dumps(mem_report(dev))}")

    cfg = {"model": spec["model"], "kwargs": kw, "seed": seed, "epochs_run": epochs, "steps": epochs * spe,
           "batch": bs, "lr": lr, "n_train": n_train, "device": str(dev), "n_params": n_params(model),
           "best_val_rel_l2": best, "train_seconds": round(time.time() - t_start, 1)}
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dict": best_state}, ROOT / "checkpoints" / f"{name}.pt")
    (ROOT / "results" / "train" / f"{name}.json").write_text(json.dumps({"config": cfg, "history": hist}, indent=1))
    state_path.unlink(missing_ok=True)
    del model, opt, sched, best_state
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fno")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    spec = {"name": a.name, "model": a.model, "seed": a.seed, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
            "pad": a.pad, "n_train": a.n_train}
    print(json.dumps(train_run(spec, load_data(dev), dev)))


if __name__ == "__main__":
    main()
