# PDE solutions as a service: PDEBench 1D diffusion–sorption

A neural-operator surrogate (5 × FNO ensemble) for PDEBench's 1D diffusion–sorption dataset is verified against an
independently written, validated implicit solver. Both are served through a FastAPI JSON API and a static
(vanilla JS + Plotly) frontend.

All numbers below come from files produced by runs in this repository. The full tables are generated into
[`results/RESULTS.md`](results/RESULTS.md) by `scripts/make_report.py`. Decisions and their evidence are in
[`DECISIONS.md`](DECISIONS.md).

## The problem (as the data actually is)

du/dt = D / R(u) · d²u/dx², x ∈ (0,1), t ∈ (0, 500], with R(u) = 1 + (1-φ)/φ · ρ_s · k_f · n_f · (u + 1e-6)^(n_f-1).

- The constants are from PDEBench's generator: D = 5e-4, φ = 0.29, ρ_s = 2880, k_f = 3.5e-4, n_f = 0.874.
- Left boundary: u = 1 on the ghost cell. Right boundary: u = D·∂u/∂x on the ghost cell.

The data differ from the brief this project started from in three ways (verified in `results/data_inspection.json`):

- **Initial condition:** each sample's IC is a *spatially constant* level c ~ U(0, 0.2), not noise. The dataset is a one-parameter family c → u(x,t).
- **Time steps:** the DaRUS file stores 101 times (dt = 5 s), not 501.
- **Right boundary:** the outflow condition keeps u(x≈1) ≈ 0, so a breakthrough curve at x = 1 is flat and carries no information.

## Key findings

| | result |
|---|---|
| Solver error vs a tight-tolerance reference (dt = 0.5 / 1.0) | 8.0e-05 / 2.4e-04 relative L2 |
| **PDEBench data vs the accurately solved equations** | **1.446% mean** (1.37–1.49%), growing from 0.548% at t=5 to 1.696% at t=500 |
| FNO ensemble vs data, 400 held-out samples | 0.124% mean, 0.153% p95, 0.408% worst |
| FNO ensemble vs accurate solver | 1.441% (the dataset floor) |
| **Linear interpolation in c (no learning) vs data** | **0.009%**: 14× better than any neural model in distribution |
| CNN/U-Net baseline vs data | 0.139% |
| Ignoring the IC (mean trajectory) | 9.438% |
| Zero-shot at 1024 points (trained at 256) | ensemble 0.594%, CNN 6.007% |
| OOD vs solver, ensemble: c > 0.2 / smooth / noise / pulse / step | 8.62% / 2.90% / 2.15% / 6.79% / 6.81% |
| OOD c > 0.2, interpolation (it can only clamp) | 37.29% |
| Data efficiency (6000 steps each): best val, n_train = 25 vs 3200 | 0.159% vs 0.165% |
| Ensemble spread vs true error, Spearman (test / OOD) | 0.04 / 0.97: no trust indicator is shown |

**What the solver check showed.** The verbatim generator code (SciPy RK45 at default tolerances) reproduces the data to
2.3e-8. Integrated accurately (BDF, Radau, or tight RK45), the generator's equations agree with this repo's solver to
1e-5. So the ~1.4% gap is the *dataset's* own integration error, not a solver bug (see `results/diag_rhs_tolerance.json`).

## The app

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765
```

- **Level c (main control).** A slider over the training range, with an OOD warning above 0.2. It shows three curves:
  - the surrogate;
  - the dataset-style solution: interpolation of 256 stored PDEBench trajectories, 0.051% from held-out data;
  - the accurately solved solution, with an explainer of the ~1.4% gap.
- **IC stress lab.** Pulse, smooth, noise, step and CSV-upload ICs, all flagged as out of distribution and shown next to the measured degradation on the held-out OOD set.
- **Physics-shift lab.** Sliders for D, φ, ρ_s, k_f, n_f and the boundary value drive only the solver; the surrogate's physics is fixed. A one-parameter sweep shows the error is smallest at the training value and grows quickly away from it, for example 18.5% when D is doubled (`results/ui_check.json`).
- **Observables:**
  - u at a chosen x, the u = 0.5 front position, and uptake (total amount entered).
  - Inflow flux at x = 0 is not shown: even the dataset is 22.5% from the accurate solver there.
- **API:**

  | endpoint | purpose |
  |---|---|
  | `GET /api/health`, `GET /api/meta` | status; grid, ranges and evaluation summary |
  | `POST /api/predict` | surrogate only |
  | `POST /api/solve` | numerical solver only |
  | `POST /api/compare` | surrogate + solver + dataset-style curve + metrics + OOD flag |
  | `POST /api/physics_sweep` | surrogate error across one physical parameter |

**Latency** (`results/latency.json`, CPU, localhost, median over 10 calls):

| call | HTTP time | of which compute |
|---|---|---|
| predict, single FNO | 66 ms | 4.9 ms |
| predict, ensemble | 132 ms | 15.6 ms |
| solve | 294 ms | 244 ms |
| compare (what the page calls) | 463 ms | 317 ms solver + 17 ms surrogate |

## Limitations and what could mislead

- **The surrogate is not the best in-distribution model.** For the dataset as given (scalar IC), interpolating stored trajectories is more accurate and cheaper. The FNO earns its place only for extrapolation (c > 0.2) and non-constant ICs, and there its errors are 2–9% on average and up to 20.8% worst case.
- **"Error vs the dataset" (0.12%) and "error vs the true equations" (1.44%) are different things.** The surrogate faithfully learns the dataset's RK45 error.
- **Ensemble spread is not an error bar.** It does not rank in-distribution error (ρ = 0.04). It only signals unfamiliar inputs.
- **The page does not show the surrogate's speed advantage.** The interactive comparison always runs the solver too, so it updates at solver speed (~0.5 s).
- **The OOD set is small** (46 trajectories, 6–8 per family), so per-family numbers are rough.
- **Data efficiency and padding comparisons use one seed each.**
- **The frontend loads Plotly from jsDelivr,** so it needs internet access.

## Reproduce

```powershell
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt  # torch: CUDA wheel, see file
# data: 1D_diff-sorp_NA_NA.h5 (DaRUS doi 10.18419/darus-2986, URL in PDEBench's pdebench_data_urls.csv) -> data/raw/
python -I scripts/inspect_data.py data/raw/1D_diff-sorp_NA_NA.h5
python -I scripts/build_cache.py data/raw/1D_diff-sorp_NA_NA.h5 data/cache
python -I scripts/validate_solver.py 20
python -I scripts/make_ood.py 0.5
python -m qpde.train --model cnn --epochs 150 --name cnn_s0
python -m qpde.train_many                       # FNO pad0/pad25, ensemble seeds, data-efficiency sweep
python -I scripts/evaluate.py --ensemble fno_pad0
python -I scripts/eval_derived.py
python -I scripts/build_assets.py 256 --table
python -I scripts/make_report.py
python -m pytest -q tests
python scripts/ui_check.py http://127.0.0.1:8765  # needs the server running and Microsoft Edge (Playwright)
```

Layout:

| path | contents |
|---|---|
| `qpde/` | solver, models, training, metrics, ICs, derived outputs |
| `scripts/` | pipeline steps and diagnostics |
| `app/` | FastAPI app and static UI |
| `results/` | JSON metrics, `RESULTS.md` |
| `figures/` | plots and UI screenshots |
| `checkpoints/` | 15 small `.pt` files |
