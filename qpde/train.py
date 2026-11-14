"""Train one surrogate on the cached data.

python -m qpde.train --model fno --seed 0 --epochs 100 --pad 0.0 --device cuda --name fno_s0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qpde.models import build, n_params

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def make_inputs(u0: np.ndarray, x: np.ndarray) -> torch.Tensor:
    """u0: (B, N) -> (B, 2, N) with channels [u0, x]."""
    u0 = torch.as_tensor(u0, dtype=torch.float32)
    xx = torch.as_tensor(x, dtype=torch.float32).expand_as(u0)
    return torch.stack([u0, xx], 1)


def rel_l2(pred, true):
    """Per-sample relative L2 over all (t, x) entries."""
    d = (pred - true).flatten(1).norm(dim=1)
    return d / true.flatten(1).norm(dim=1)


def load_split(name):
    a = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    return np.ascontiguousarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fno")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max_seconds", type=float, default=0, help="stop early after this wall time")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    x = np.load(CACHE / "x256.npy")
    tr, va = load_split("train"), load_split("val")
    if args.n_train:
        tr = tr[: args.n_train]
    Xtr, Ytr = make_inputs(tr[:, 0], x), torch.as_tensor(tr[:, 1:])
    Xva, Yva = make_inputs(va[:, 0], x).to(dev), torch.as_tensor(va[:, 1:]).to(dev)
    Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)

    kw = dict(n_out=Ytr.shape[1], width=args.width)
    if args.model == "fno":
        kw.update(modes=args.modes, layers=args.layers, pad_frac=args.pad)
    model = build(args.model, **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=int(np.ceil(len(Xtr) / args.batch)))
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    hist, best, best_state = [], float("inf"), None
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        te = time.time()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(Xtr), args.batch):
            b = perm[i:i + args.batch].to(dev)
            loss = rel_l2(model(Xtr[b]), Ytr[b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - te
        model.eval()
        with torch.no_grad():
            v = torch.cat([rel_l2(model(Xva[i:i + 100]), Yva[i:i + 100]) for i in range(0, len(Xva), 100)])
        v = v.mean().item()
        hist.append({"epoch": ep + 1, "train_rel_l2": tot / len(Xtr), "val_rel_l2": v, "epoch_seconds": ep_time})
        if v < best:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{args.name}] ep {ep + 1} train {tot / len(Xtr):.4e} val {v:.4e} ({ep_time:.1f}s/ep)", flush=True)
        if args.max_seconds and time.time() - t_start > args.max_seconds:
            print(f"[{args.name}] stopping at the wall-time budget after {ep + 1} epochs", flush=True)
            break
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "results" / "train").mkdir(parents=True, exist_ok=True)
    cfg = {"model": args.model, "kwargs": kw, "seed": args.seed, "epochs_run": len(hist),
           "batch": args.batch, "lr": args.lr, "n_train": len(Xtr), "device": args.device,
           "n_params": n_params(model), "best_val_rel_l2": best,
           "train_seconds": round(time.time() - t_start, 1)}
    torch.save({"config": cfg, "state_dict": best_state}, ROOT / "checkpoints" / f"{args.name}.pt")
    (ROOT / "results" / "train" / f"{args.name}.json").write_text(json.dumps({"config": cfg, "history": hist}, indent=1))
    print(json.dumps(cfg), flush=True)


if __name__ == "__main__":
    main()
