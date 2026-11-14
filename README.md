# PDE solutions as a service: PDEBench 1D diffusion–sorption

**Sorption Surrogate Lab** is a FastAPI service with a static (vanilla JS + Plotly) frontend. It serves a neural-operator
surrogate (a 5-seed FNO ensemble) for PDEBench's 1D diffusion–sorption dataset and compares it live against an
independently written, validated implicit solver. An interpolation baseline, an out-of-distribution stress lab and a
physics-shift lab are included.

- **Deployment status: not deployed.** Hugging Face refused to create the Docker Space: HTTP 402, because free CPU Docker Spaces now require a PRO subscription (`results/deploy_attempt.json`). The user chose not to deploy elsewhere. The Dockerfile and deploy scripts are included and were smoke-tested from the staged build folder; see [Deployment](#deployment).
- **Numbers:** every number in the tables below is generated from `results/*.json` by `scripts/make_readme_tables.py`; prose numbers are checked against those files by `scripts/audit_readme.py`. Decisions and their evidence are in [`DECISIONS.md`](DECISIONS.md), and the full metrics in [`results/RESULTS.md`](results/RESULTS.md).

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # or the cu124 wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765
```

Serving needs only what is committed: `app/` (code, static files, assets), `qpde/`, and 6 checkpoints in `checkpoints/`.
The raw data is needed only to re-run the pipeline (see [Reproduce](#reproduce)).

Docker (not built locally, because Docker is not installed here):

```
python scripts/stage_space.py build_ctx && docker build -t sorption-lab build_ctx && docker run -p 7860:7860 sorption-lab
```

## Architecture

```
browser (app/static: index.html, app.js, style.css, vendor/plotly-cartesian-2.35.2.min.js)
   │  JSON over HTTP (gzip)
FastAPI  app/main.py ── /api/health  /api/meta  /api/predict  /api/solve  /api/compare  /api/physics_sweep  /api/benchmark
   │
app/service.py ── FNO ensemble, single FNO, CNN (PyTorch, CPU)
   │             ├─ interpolation in c over 256 stored PDEBench trajectories (app/assets/dataset_table.npz)
   │             ├─ OOD check (non-constant IC, level outside (0, 0.2), changed physics)
   │             └─ app/assets/summary.json: every evaluation number shown in the UI (built by scripts/build_assets.py)
qpde/ ── solver.py (implicit BDF2 + Newton, tridiagonal), models.py (FNO1d, CNN1d), train.py / train_many.py,
          metrics.py, derived.py (observables), ics.py (IC generators)
```

- **`/api/compare`** runs the selected model, interpolation and the solver for one IC and parameter set. It returns all fields, the dataset-style curve (in range only), both error types, the per-component timings and the OOD reasons.
- **The Physics-shift lab** sends changed parameters. Only the solver responds, because the surrogate's physics is fixed.

## What the data actually is

PDE (constants from PDEBench's `sim_diff_sorp.py` and `configs/diff-sorp.yaml`):

du/dt = D/R(u) · d²u/dx², with R(u) = 1 + (1-φ)/φ · ρ_s · k_f · n_f · (u + 1e-6)^(n_f-1)

- D = 5e-4, φ = 0.29, ρ_s = 2880, k_f = 3.5e-4, n_f = 0.874.
- Cell-centred finite-volume grid of 1024 cells.
- Left boundary: Dirichlet u = 1, applied on a ghost cell.
- Right boundary: ghost value D·(u[N-2] − u[N-1])/dx.

| Assumption in the original brief | What the data shows (`results/data_inspection.json`) |
|---|---|
| ICs are random noise in (0, 0.2) | Each IC is a **spatially constant** level c ∈ (0, 0.2). The largest per-sample spatial std is 3e-8, so the dataset is a one-parameter family c → u(x,t). |
| 501 time steps, plus a 101-step version | Only **101 stored times** (dt = 5 s), although each sample's config says tdim = 501. |
| A useful breakthrough curve at x = 1 | The outflow condition keeps **u(x≈1) ≈ 0**, so the curve is flat. |
| Standard HDF5 layout | 10,000 groups `0000…9999`, each with `data` (101, 1024, 1) float32 plus `grid/x` and `grid/t` (PDEBench loads it with `single_file=False`). |
| Data = ground truth | The data carries about **1.4% time-integration error** (next section). |

How much the IC matters, measured with the accurate solver (`results/eval_metrics.json`):

<!-- GEN:icbc -->
- IC level 0.1 → 0.2 changes the trajectory by 18.52%; boundary value 1.0 → 1.1 by 9.38%.
- Full IC range 0 → 0.2 by time: t=5 148.39%, t=25 93.91%, t=100 59.55%, t=500 25.24%.
- Ignoring the IC (mean trajectory) costs 9.44% on the test set.
<!-- /GEN:icbc -->

## Solver validation

The solver in `qpde/solver.py` was written independently: implicit BDF2 with a backward-Euler start, Newton iterations,
an analytic tridiagonal Jacobian (checked against finite differences in `tests/`), and SciPy's banded solve. It is
validated on 20 held-out samples at the full 1024 × 101 resolution:

<!-- GEN:solver -->
| solver | vs tight BDF reference (mean / max) | vs PDEBench data (mean) | s per run (1024 cells) |
|---|---|---|---|
| implicit BDF2 + Newton, dt = 2.0 | 6.6e-04 / 7.6e-04 | 1.44% | 0.23 |
| implicit BDF2 + Newton, dt = 1.0 | 2.4e-04 / 3.5e-04 | 1.44% | 0.33 |
| implicit BDF2 + Newton, dt = 0.5 | 8.0e-05 / 1.7e-04 | 1.44% | 0.64 |
| implicit BDF2 + Newton, dt = 0.25 | 4.0e-05 / 9.2e-05 | 1.45% | 1.13 |

Tight BDF (rtol 1e-8) vs PDEBench data, 20 held-out samples: mean **1.45%** (min 1.37%, max 1.49%); by time t=5 0.548%, t=10 0.625%, t=25 0.796%, t=100 1.14%, t=250 1.43%, t=500 1.70%.
<!-- /GEN:solver -->

**Why the solver and the data disagree by about 1.4%** (`results/diag_rk45_rerun_3600.json`, `results/diag_rhs_tolerance.json`, `scripts/diag/`):
- A verbatim copy of the generator's RK45 reproduces the stored data to 2.3e-8.
- This repo's right-hand side equals the generator's to 5e-14.
- Integrated accurately (BDF, Radau or tight RK45), the generator's own equations agree with this solver to 1e-5.

The gap is therefore the **dataset's** own error from `solve_ivp` at default tolerance (rtol 1e-3), and it grows with t.
Ground truth here is the solver at dt = 0.5; the app uses dt = 1.

## Results

All errors are relative L2.

**"vs dataset"** measures how well a model reproduces the PDEBench trajectories it was trained on. **"vs accurate
solver"** measures the error against the equations solved accurately. They differ by the dataset's own ~1.4% error.

<!-- GEN:models -->
| model | test vs **dataset** (mean / p95 / worst) | test vs **accurate solver** (mean) | left / interior / right MAE | OOD const_in | OOD const_high | OOD smooth | OOD noise | OOD pulse | OOD step |
|---|---|---|---|---|---|---|---|---|---|
| interpolation in c (3,200 trajectories, no learning) | 0.009% / 0.043% / 0.134% | 1.44% | 8.8e-05 / 2.0e-06 / 5.9e-10 | 1.43% | 37.29% | 3.43% | 1.54% | 7.45% | 9.68% |
| FNO ensemble (5 seeds, pad 0) | 0.124% / 0.153% / 0.408% | 1.44% | 1.7e-03 / 1.9e-04 / 3.7e-04 | 1.43% | 8.62% | 2.90% | 2.15% | 6.79% | 6.81% |
| single FNO, pad 0 | 0.127% / 0.155% / 0.381% | 1.44% | 1.7e-03 / 2.1e-04 / 4.7e-04 | 1.43% | 7.71% | 3.23% | 1.92% | 7.04% | 7.33% |
| single FNO, pad 0.25 | 0.131% / 0.166% / 0.435% | 1.44% | 1.7e-03 / 2.3e-04 / 2.8e-04 | 1.43% | 10.34% | 2.40% | 1.88% | 6.32% | 7.45% |
| CNN / U-Net baseline | 0.139% / 0.173% / 0.290% | 1.44% | 1.7e-03 / 2.7e-04 / 5.1e-04 | 1.43% | 7.41% | 3.84% | 2.43% | 14.64% | 9.41% |
| mean trajectory (ignores the IC) | 9.44% / 18.37% / 20.69% | 9.76% | 2.2e-03 / 3.3e-02 / 2.1e-03 | 12.39% | 49.57% | 3.32% | 1.37% | 10.20% | 16.54% |

Test = 400 held-out samples (keys 3600-3999), relative L2 over t = 5…500 on 256 points. OOD = 46 solver-generated trajectories (6-8 per family), error vs the accurate solver; const_in is an in-range control. Dataset vs accurate solver on the same test samples: 1.44%. Zero-shot at 1024 points: CNN / U-Net baseline 6.01%, single FNO, pad 0 0.643%, single FNO, pad 0.25 0.632%, FNO ensemble (5 seeds, pad 0) 0.594%.
<!-- /GEN:models -->

Observables (the app shows the first three types and drops inflow flux; `results/derived_outputs.json`):

<!-- GEN:derived -->
| output | sensitivity to c (std/mean) | ensemble vs data | data vs accurate solver |
|---|---|---|---|
| u_at_x=0.1 | 0.033 | 0.025% | 1.60% |
| u_at_x=0.25 | 0.116 | 0.078% | 1.41% |
| u_at_x=0.5 | 0.352 | 0.183% | 0.852% |
| front_position_u0.5 | 0.083 | 0.050% | 1.69% |
| uptake | 0.151 | 0.113% | 1.83% |
| inflow_flux_x0 | 0.182 | 19.63% | 22.54% |

Dataset-style lookup (interpolating K stored trajectories) vs held-out data: K=32: 0.127%, K=64: 0.115%, K=128: 0.093%, K=256: 0.051%, K=512: 0.027%, K=3200: 0.009%.
<!-- /GEN:derived -->

Padding, with 3 seeds per setting (`results/padding_3seeds.json`):

<!-- GEN:padding -->
| setting (3 seeds each) | test rel L2 | right-boundary MAE | left-boundary MAE | OOD mean |
|---|---|---|---|---|
| pad 0 | 0.129% ± 0.003% | 4.4e-04 ± 3e-05 | 1.71e-03 | 5.21% |
| pad 0.25 | 0.135% ± 0.004% | 2.7e-04 ± 5e-06 | 1.73e-03 | 5.01% |
<!-- /GEN:padding -->

Data efficiency: FNO, seed 0, 6000 optimiser steps at every size; one seed per size (`results/train/sweep_*.json`):

<!-- GEN:data_eff -->
| n_train | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|---|
| best val rel L2 | 0.159% | 0.169% | 0.171% | 0.165% | 0.170% | 0.166% | 0.168% | 0.165% |
<!-- /GEN:data_eff -->

Trust signals (`results/eval_metrics.json`). The rule was fixed before the numbers were seen: a trust indicator would be
shown only if ρ ≥ 0.5 with p < 0.01 on both sets. It failed, so the UI shows the spread only as an informational number
next to these correlations.

<!-- GEN:trust -->
| signal | Spearman with true error: test (n=400) | OOD (n=46) |
|---|---|---|
| ensemble spread | 0.04 (p = 0.44) | 0.97 (p = 5.4e-28) |
| PDE residual | 0.70 (p = 2.5e-61) | 0.29 (p = 0.051) |
<!-- /GEN:trust -->

CPU batch benchmark (`results/batch_benchmark.json`):

<!-- GEN:bench -->
| method | batch 1 (ms/sample) | batch 10 (ms/sample) | batch 100 (ms/sample) |
|---|---|---|---|
| single FNO | 7.79 | 2.33 | 0.88 |
| 5-FNO ensemble | 54.83 | 10.57 | 3.94 |
| CNN | 8.35 | 3.37 | 1.81 |
| interpolation in c | 0.26 | 0.32 | 0.61 |
| implicit solver (1024 cells, dt=1.0) | 914.15 | – | – |
| implicit solver (1024 cells, dt=0.5) | 1245.20 | – | – |

median of individually timed repetitions, 4 CPU threads, measured 2026-11-14 12:29:19 with Brave browser running (user-confirmed), ~11% CPU; batch-1 timings were 1.54–4.06× slower than the earlier quiet-machine measurement in `results/eval_metrics.json` (single FNO 5.0 ms, solver dt=1 341 ms).
<!-- /GEN:bench -->

API latency (`results/latency.json`, median of 10 calls; physics sweep 3 calls):

<!-- GEN:latency -->
| endpoint (localhost, gzip) | HTTP median | server compute | transferred | uncompressed |
|---|---|---|---|---|
| predict_ensemble_256 | 286.0 ms | 39.9 ms | 92.4 KB | 380.8 KB |
| predict_fno_256 | 136.2 ms | 6.6 ms | 71.8 KB | 204.9 KB |
| predict_fno_1024 | 406.0 ms | 8.1 ms | 263.9 KB | 805.6 KB |
| solve_256 | 521.1 ms | 433.6 ms | 70.3 KB | 202.3 KB |
| compare_ensemble_256 | 864.5 ms | 21.6 ms model + 479.6 ms solver | 245.9 KB | 792.9 KB |
| compare_ensemble_1024 | 984.4 ms | 28.2 ms model + 360.2 ms solver | 627.0 KB | 2372.5 KB |
| predict_interp_256 | 78.1 ms | 0.3 ms | 70.2 KB | 202.1 KB |
| physics_sweep_7 | 2527.4 ms | – | 0.8 KB | 0.8 KB |

Gzip trade-off on /api/compare (`results/gzip_tradeoff.json`): 792.9 KB → 245.9 KB on the wire for +54 ms of server CPU on localhost.
<!-- /GEN:latency -->

## The app

- **Level c** (main control). A slider with an OOD warning above 0.2. It shows:
  - space–time heatmaps of the model, the solver and their absolute error;
  - a profile at a chosen time with three curves: model, dataset-style (interpolated PDEBench trajectories, in range only) and accurate solver;
  - the observables: u at a chosen x, the u = 0.5 front position, and uptake.
- **Model selector:** FNO ensemble, single FNO, CNN, or interpolation in c.
- **"Which model when?"** A guide generated from the per-family results: where the FNO helps, and where interpolation is better.
- **IC stress lab:** pulse, smooth, noise, step and CSV upload. All flagged OOD, and shown next to the measured degradation table.
- **Physics-shift lab:** sliders for D, φ, ρ_s, k_f, n_f and the boundary value move only the solver, plus a one-parameter sweep of the error.
- **Timing panel:** this call's surrogate, interpolation and solver times, the session benchmark table, and a live-benchmark button.
- **Verification:** `scripts/ui_check.py` drives Microsoft Edge with Playwright and checks 38 items, including:
  - no console errors and no requests to other hosts;
  - every displayed error number equals an independent API call;
  - the guide and OOD tables equal `/api/meta`;
  - the OOD banners appear;
  - the surrogate does not change under a physics shift;
  - no horizontal scroll at 390 px wide;
  - dark mode renders.

  Results are in `results/ui_check.json`, screenshots in `figures/ui/`. The final run was against a server started from a fresh clone of this repository.
- **Claim audit:** `scripts/audit_ui_claims.py` traces each quantitative UI claim to its results file and asserts equality (`results/ui_claims_audit.md`).

## Limitations

- **The physics is fixed.** The surrogate was trained at one parameter set. Away from it the error grows fast; the Physics-shift lab shows this, it does not solve it.
- **The training distribution is scalar.** Every training IC is a flat level in (0, 0.2), so any non-flat IC or level above 0.2 is extrapolation.
- **In range, interpolation beats the FNO.** Interpolating stored trajectories is more accurate and cheaper there (table above).
  - The FNO is clearly better only above c = 0.2.
  - It is narrowly better for smooth, pulse and step ICs.
  - It is worse than interpolation for noise ICs.
- **Ensemble spread does not predict error in range.** It ranks error well only out of distribution.
- **The OOD set is thin:** 46 trajectories, 6–8 per family. Small per-family differences are within noise.
- **Seed counts are small:** 5 for the ensemble, 3 per padding setting, 1 per data-efficiency size.
- **The page does not show the surrogate's speed.** It runs the solver on every update, so it responds at solver speed.
- **Timings vary on this laptop** with background load: with the Brave browser running, batch-1 timings were 1.54–4.06× slower than in a quiet run (`app/assets/summary.json`, `timing_slowdown`). Both measurements are reported.
- **The left boundary dominates the error:** the ensemble's absolute error in the first 4 points is 8.8× its interior error (table above).

## What was wrong (brief and own work)

**In the original brief** (each contradicted by the data, and the user confirmed after the Phase-1 checkpoint):
- Noise ICs, when the ICs are in fact constant levels.
- 501 stored times, when only 101 are stored.
- A useful breakthrough curve at x = 1, when it is flat.
- "The boundary may dominate": both matter. The IC moves the trajectory more than a 10% change in the boundary value.
- That the Hugging Face CLI login was configured and that free Docker Spaces were available: neither held at deploy time.

**In my own work, caught during the session:**

| # | Mistake | How it was caught | Fix |
|---|---|---|---|
| 1 | First commit used a placeholder author identity instead of the configured one | Noticed right after committing | Amended before any further commits |
| 2 | Inspection and cache scripts assumed 501 time steps | `IndexError` on the first run | Read the shape from the file; strides adjusted |
| 3 | Validation called Radau without a Jacobian (dense finite differences on 1024 unknowns) and hung for over 10 min | No output after 10 min | Analytic sparse Jacobian; slow RK45 moved to a one-sample diagnostic |
| 4 | Ran training in parallel with heavy CPU jobs; separate processes each held CPU data copies | Training was killed twice under system memory pressure (the main cause was the browser) | One sequential trainer with data on the GPU, memory logging, and resumable checkpoints |
| 5 | Resume path loaded the RNG state onto CUDA | The first real resume crashed | Load trainer state on the CPU |
| 6 | Wrote an unverified flux-balance metric, and a relative error at x = 1 where u ≈ 1e-3 (reads as 87%) | Review of the dry-run evaluation | Metric removed; x = 1 error reported as absolute |
| 7 | Plotly CDN URL did not exist (404) | Playwright: no charts, console error | Verified npm build, then vendored locally |
| 8 | Physics sliders snapped off the training values, so training values were flagged OOD | Playwright check | Store the snapped default position |
| 9 | Guide text claimed "the FNO helps for non-flat ICs", which is false for noise ICs | Reading the UI screenshot against the table | Guide generated from per-family results |
| 10 | Rounding hid differences ("0.01%" for 0.009%; "0.13% ± 0.00%") | Reading the generated text | Values under 1% shown with 3 decimals |
| 11 | Stored the dataset's config, including a third-party cluster path | Pre-publication review | Only the physics block is kept (history left as is, per user) |
| 12 | Quoted a number (18.5%) from a screenshot before checking a results file | Self-check before commit | Cited `results/ui_check.json` |
| 13 | Deploy script imported `requests`, which `huggingface_hub` 2.x no longer installs | Crash at start | Switched to `httpx` |
| 14 | A memory-probe filter matched its own shell's command line and terminated it | Exit code 15 | Matched only the uvicorn process |
| 15 | First batch benchmark ran under background load without saying so | Numbers well above the earlier run's | Medians, conditions recorded, both runs reported |
| 17 | Quoted the slowdown as "2–3×" and the boundary ratio as "about 8×" without computing them (actual: 1.54–4.06×, 8.8×) | The README audit flagged ratio claims it could not verify | Both ratios computed into `summary.json`; UI, README and DECISIONS use the computed values |
| 18 | The serving table `app/assets/dataset_table.npz` was never committed: the `*.npz` ignore rule meant for data caches also matched it, and every local and staged test read it from the working tree | Fresh-clone test in the final audit: 17 of 25 tests failed | Explicit `.gitignore` exceptions for `app/assets/*.npz` and `results/*.npz`; file committed; clone re-tested |
| 16 | The dry-run evaluation output was committed once | Noticed in the commit list | Overwritten by the final run; the README cites only final files |

## Deployment

- **Attempt:** Hugging Face Space `HarshalShirke/sorption-surrogate-lab`, Docker SDK, cpu-basic, public. Creation was refused with HTTP 402 ("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). Nothing was created or uploaded (`results/deploy_attempt.json`).
- **Memory:** the server used 514 MB RSS under load, which rules out 512 MB free container tiers without changes.
- **Decision:** the user chose not to deploy.
- **Kept for later:** `deploy/Dockerfile` (CPU-only torch wheel, port 7860), `scripts/stage_space.py` (minimal build context, about 27 MB) and `scripts/deploy_space.py`. The deploy script uploads through the Hub API, so local git history is not pushed, and it prints the build log on failure.

## Reproduce

```powershell
# data: 1D_diff-sorp_NA_NA.h5 (DaRUS doi 10.18419/darus-2986; URL and MD5 in PDEBench's pdebench_data_urls.csv) -> data/raw/
python -I scripts/inspect_data.py data/raw/1D_diff-sorp_NA_NA.h5
python -I scripts/build_cache.py data/raw/1D_diff-sorp_NA_NA.h5 data/cache
python -I scripts/validate_solver.py 20
python -I scripts/make_ood.py 0.5
python -m qpde.train --model cnn --epochs 150 --name cnn_s0
python -m qpde.train_many            # FNO pad0/pad25, ensemble seeds 1-4, data-efficiency sweep
python -m qpde.train_pad_seeds       # pad 0.25 seeds 1-2 (3-seed padding comparison)
python -I scripts/evaluate.py --ensemble fno_pad0
python -I scripts/eval_derived.py
python -I scripts/eval_padding.py
python -I scripts/batch_benchmark.py
python -I scripts/build_assets.py 256 --table
python -I scripts/make_report.py ; python -I scripts/make_readme_tables.py ; python -I scripts/audit_readme.py
python -I scripts/audit_ui_claims.py   # after ui_check: every UI number traced to a results file
python -m pytest -q tests
python scripts/ui_check.py http://127.0.0.1:8765        # server running; needs Microsoft Edge (Playwright)
python scripts/latency.py http://127.0.0.1:8765 ; python scripts/gzip_tradeoff.py http://127.0.0.1:8765
```

## Repository layout

| path | contents |
|---|---|
| `qpde/` | solver, models, training, metrics, observables, IC generators |
| `app/` | FastAPI app, static UI, vendored Plotly, serving assets |
| `scripts/` | pipeline steps, evaluation, UI check, latency, deploy tooling; `scripts/diag/` holds the generator diagnostics |
| `results/` | every metric (JSON) and `RESULTS.md` |
| `figures/` | evaluation plots and UI screenshots |
| `checkpoints/` | 17 small `.pt` files: 6 are served, the rest are for the padding and data-efficiency comparisons |
| `deploy/` | Dockerfile, Space card, slim requirements |
