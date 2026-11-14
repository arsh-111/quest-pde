# Plan: PDE solutions as a service (PDEBench 1D diffusion-sorption)

Start: 2026-11-14 08:55:59 +05:30 (recorded with Get-Date at the first prompt).
Hard limit: 72 h after start, but the submission deadline of 15 Nov 2026 binds first.
The deadline's time zone is unconfirmed, so I assume the earliest possible reading:
end of 15 Nov in UTC+14 = **2026-11-15 15:29 IST**. That leaves about 30.5 h in total.

## Phase 0: setup
- repo, venv, .gitignore, PLAN.md, DECISIONS.md
- PDEBench cloned (shallow, gitignored) as a read-only reference for constants and URLs
- data download in the background (3.92 GB, MD5 checked against PDEBench's url csv)

## Phase 1: data, solver, models, evaluation (stop after this phase)
1. Inspect the HDF5 with h5py (layout, shapes, dtype, grids, IC statistics); plot a sample.
2. Cache: up to 4000 samples at 256 x 101 (strided) float32, split by sample index (seeded), plus about 100 test samples at 1024.
3. Reference solver (implicit, Newton, tridiagonal), validated against held-out trajectories before it is used.
4. Measure CPU vs MX450 epoch time; then train the CNN baseline, the FNO (padding on/off) and a 5-seed FNO ensemble.
5. Evaluation: rel-L2 statistics, error vs x/t, boundary error, mass/flux, PDE residual, timing,
   OOD set from the solver, IC-vs-BC sensitivity, Spearman(spread or residual, error).
6. Results JSON and figures in `results/`, small checkpoints in `checkpoints/`.

## Phase 2 (after the user's go-ahead): FastAPI + static frontend + Physics-shift lab.
