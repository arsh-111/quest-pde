# Plan: PDE solutions as a service (PDEBench 1D diffusion-sorption)

Start: 2026-11-14 08:55:59 +05:30 (recorded with Get-Date at the first prompt).
Hard limit: 72 h after start, but the submission deadline of 15 Nov 2026 binds first.
The deadline's time zone is unconfirmed, so I assume the earliest possible reading:
end of 15 Nov in UTC+14 = **2026-11-15 15:29 IST**.

## What the data turned out to be (it overrides the original brief; see DECISIONS.md 7, 9, 12)
- The IC is a spatially constant level c ∈ (0, 0.2), not noise, so the problem is a one-parameter family c → u(x, t).
- 101 stored times (dt = 5 s), 1024 cells, 10,000 samples.
- u(x≈1) ≈ 0 for t > ~5 s, so a breakthrough curve at x = 1 carries no information.
- The dataset carries ~1.44% relative-L2 RK45 tolerance error against the accurately solved equations.

## Phase 0: done
Repo, venv, .gitignore, PDEBench reference clone (gitignored), MD5-verified download.

## Phase 1: data, solver, models, evaluation
1. [done] HDF5 inspection, cache (split by key index), solver validated (8e-5 vs tight BDF at dt=0.5).
2. [done] CNN baseline (killed stage kept its finished checkpoint).
3. [running] Sequential trainer: FNO pad0, FNO pad25, 4 more ensemble seeds, data-efficiency sweep.
4. Evaluation with error vs the dataset AND vs the accurate solver, plus the trivial baselines
   (interpolation in c, mean trajectory), OOD, trust, IC-vs-BC sensitivity, timing.

## Phase 2: application (no deployment yet)
- FastAPI: /api/health, /api/meta, /api/predict, /api/solve, /api/compare; static vanilla JS + Plotly frontend.
- The main control is the level c over the training range, with an OOD warning outside it.
- An IC "stress lab" (noise, pulse, smooth, step, CSV) presented as out of distribution, with the measured degradation.
- Outputs chosen from what the data supports (concentration at a selectable x, front position, inflow/uptake at x=0)
  instead of the x=1 breakthrough curve.
- Three curves where relevant: surrogate, dataset-style solution (RK45, default tolerance, like the generator),
  and the accurately solved solution, with an explainer of the ~1.4% gap.
- Physics-shift lab and a trust indicator, each only if the evidence supports it.
- pytest API tests; UI verification by loading it in a browser (screenshots, console errors, numbers vs API).
