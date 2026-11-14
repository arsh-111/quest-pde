# Decisions

Format: decision | alternatives | evidence.

1. **Repo location `D:\AgenticAI\quest-pde`**, not `D:\quest-pde` as the brief says | brief path, session cwd | the user chose this folder when asked at the start of the session.
2. **Deadline read as 2026-11-15 15:29 IST** (end of 15 Nov in UTC+14, the earliest any time zone finishes the day) | IST end of day, 72 h rule | the time zone is unconfirmed, so I took the most conservative reading.
3. **Constants and discretisation are taken from `pdebench/data_gen/src/sim_diff_sorp.py` and `configs/diff-sorp.yaml`** rather than from the brief: D=5e-4, por=0.29, rho_s=2880, k_f=3.5e-4, n_f=0.874, sol=1, T=500, 501 outputs, 1024 cells, cell-centred FVM grid, `solve_ivp` (RK45) at default tolerances.
   - Left BC: ghost cell = 1.
   - Right BC: ghost value = D·(u[N-2]-u[N-1])/dx.
   - R(u) = 1 + (1-por)/por·rho_s·k_f·n_f·(u+1e-6)^(n_f-1).
4. **Possible conflict with the brief:** the generator code makes the initial condition spatially *constant*, `u0 = ones * U(0, 0.2)`, not random noise. This is to be verified against the data.
5. **h5py pinned to 3.11.0** | latest h5py, a pure-Python HDF5 reader | the latest wheel failed with "DLL load failed while importing h5pl: An Application Control policy has blocked this file"; 3.11.0 imports cleanly, and 3.10.0 fails with a numpy ABI error.
6. **torch 2.6.0+cu124** | CPU build | `torch.cuda.is_available()` = True on the MX450; whether to train on CPU or GPU will be decided from measured epoch times.
7. **The data contradicts the brief (verified, `results/data_inspection.json`).**
   - The ICs are spatially constant: the largest per-sample std of u0 over 200 samples is 3.0e-8, and the means span (0.0012, 0.1999). Each trajectory is determined by one scalar. (This confirms item 4.)
   - The DaRUS file holds only 101 times (dt = 5 s), although the per-sample config says tdim=501. There is no 501-step version.
   - Layout: 10000 groups `0000..9999`, each with `data` (101,1024,1) float32, uncompressed, plus `grid/x` (cell centres) and `grid/t`.
   - The right BC u = D·u_x with D = 5e-4 makes u(x≈1) ≈ 0 after the first few seconds, so a breakthrough curve at x=1 is nearly flat.
8. **Cache:** spatial stride 4 (256 points), no time stride.
   - Split by key index before any shuffle: train 0000-3199, val 3200-3599, test 3600-3999; full-resolution test 3600-3699.
   - Because the ICs are scalar, test samples are interpolations between neighbouring training ICs, so a 1-D interpolation baseline in u0 is added to the evaluation.
9. **The PDEBench data carries about 1.4% time-integration error; the reference solver is still trusted** | treat the data as exact, or match the data's RK45 | `results/solver_validation.json` and `results/diag_rhs_tolerance.json`:
   - The verbatim generator RK45 reproduces the data to 2.3e-8 rel L2 (key 3600).
   - My RHS equals the generator's rc_ode to 5e-14 on the data states.
   - The generator's ODE integrated accurately (BDF rtol 1e-8, Radau rtol 1e-10, RK45 rtol 1e-6) agrees with my solver to 1e-5, and all of them sit 1.42-1.45% from the data.
   - The data's error grows with t (0.55% at t=5, 1.70% at t=500), from `solve_ivp` default rtol=1e-3.

   Consequences:
   - The surrogate is trained on the data as given.
   - Surrogate-vs-solver comparisons have a floor of about 1.4%, which is measured directly with a solver-generated in-distribution control set.
   - Metrics are reported against both the data and the solver.
10. **Reference solver dt = 0.5 for ground truth (8e-5 rel L2 vs tight BDF, 0.64 s per 1024-cell run); dt = 1 for interactive use (2.4e-4, 0.33 s).**
11. **Train on the GPU:** measured FNO 7.5 s/epoch on CUDA vs 33.8 s on CPU; CNN 5.7 vs 58.8 s (`results/device_benchmark.json`).
12. **The brief's premise was wrong on three points; the evidence wins (the user confirmed this after the Phase-1 checkpoint).**

    | Brief assumed | Data / generator shows | Evidence |
    |---|---|---|
    | IC = random noise in (0, 0.2) | IC = spatially constant level c ~ U(0, 0.2); per-sample spatial std ≤ 3e-8 | `results/data_inspection.json`, `sim_diff_sorp.py` (`u0 = ones * uniform(0, 0.2)`) |
    | 501 time steps plus a 101-step version | only 101 stored times (dt = 5 s), although the config says tdim=501 | `results/data_inspection.json` |
    | a useful breakthrough curve at x = 1 | the right BC u = D·u_x (D = 5e-4) pins u(x≈1) ≈ 0; the last cell falls below ~0.005 within the first output step and stays near 0.001 | `figures/sample_trajectories.png` |

    Consequences:
    - The learning problem is a one-parameter family c → u(x, t).
    - Strong trivial baselines (interpolation in c, the mean trajectory) belong in every table.
    - Noise, pulse and smooth ICs are out of distribution by construction.
    - The product's main control becomes the level c, and the outputs become ones the data supports (see the Phase 2 decisions).
13. **Two kinds of error are always reported separately:**
    - error against the dataset (what the surrogate was trained on);
    - error against the accurately solved equations (the implicit solver at dt = 0.5, 8e-5 from tight BDF).

    The two differ by the dataset's own ~1.44% RK45 tolerance error (decision 9).
14. **Training runs strictly sequentially in one process (`qpde/train_many.py`) with the data loaded once onto the GPU** | separate processes per run | the first stage-1 run was killed by Claude Code under system memory pressure (Brave was using ~4 GB; 3.2 GB physical free at restart).
    - The new trainer peaks at ~1.0 GB RSS with the 0.35 GB of data on the GPU.
    - It logs memory before each run, saves resumable state about every 1000 steps, and aborts (no retry) below 1.5 GB of available RAM.
    - The finished CNN baseline (`cnn_s0`, best val 1.38e-3) from the killed stage is kept.
15. **Data-efficiency sweep** | none | because the input is a scalar, few samples may suffice. FNO, seed 0, n_train ∈ {25 … 3200}, fixed budget of 6000 optimiser steps for every size, so only the data size changes.
16. **Padding: pad0 is used for the ensemble, but it is effectively a tie** | pad 0.25 | single-seed best val 1.257e-3 (pad0) vs 1.285e-3 (pad25); test 0.127% vs 0.131%.
    - Padding does reduce the right-boundary error (right-4-point MAE 2.8e-4 vs 4.7e-4).
    - The left boundary, where the steep front sits (MAE 1.7e-3 for both), dominates the total error.
    - Source: `results/RESULTS.md`.
17. **The neural surrogates are not the most accurate in-distribution model, and this is reported plainly.**
    - Linear interpolation in c between training trajectories: 0.009% mean test rel L2.
    - FNO ensemble: 0.124%; CNN: 0.139%.
    - All of them are 1.44% from the accurate solver, which is the dataset floor.
    - The FNO's value is in extrapolation (c > 0.2: 8.6% vs 37% for interpolation, which can only clamp) and in non-constant ICs.
18. **No "trust" indicator in the UI.** The rule fixed in `scripts/build_assets.py` before the evaluation was looked at was not met.
    - Spread vs error, Spearman: 0.04 (p=0.44) on test and 0.97 on OOD.
    - Residual vs error: 0.71 on test and 0.29 (p=0.05) on OOD.
    - Spread above the test p99 (2.5e-4) flagged 100% of every OOD family and 0% of the in-distribution controls.
    - The UI therefore shows the raw spread, labelled as an unfamiliar-input signal and not an error estimate. The rule-based OOD flag stays primary.
19. **Observables shown: u at a selectable x, front position (u = 0.5), uptake. Not shown: breakthrough at x=1, inflow flux** | `results/derived_outputs.json`.
    - Inflow flux at x=0 is unreliable even in the dataset (22.5% from the accurate solver; the surrogate is 19.6% from the data).
    - Uptake, its time integral, is robust: ensemble vs data 0.11%.
20. **Dataset-style curve = interpolation of K=256 stored trajectories in c** | K=128 (0.093%), K=3200 (316 MB) | K=256: 0.051% from held-out data, far below the 1.44% gap it illustrates; 22 MB compressed. Total committed assets: 32 MB.
21. **Data efficiency: 25 training samples are enough at this step budget.** With 6000 steps for every size, best val stays at 0.159-0.171% for n_train 25…3200. Error is limited by optimisation and model capacity, not data. Single seed per size.
22. **Padding re-run with 3 seeds (about 15 min of compute, within the 20-minute budget).** `results/padding_3seeds.json`.
    - Test rel L2: pad0 0.129% ± 0.003%, pad 0.25 0.135% ± 0.004%.
    - Right-boundary MAE: 4.4e-4 vs 2.7e-4.
    - OOD mean: 5.2% vs 5.0% (within noise).
    - The single-seed conclusion (decision 16) holds: padding helps only at the non-periodic right boundary.
23. **The interpolation baseline is selectable in the app, and the model guide is generated from the per-family results, not a blanket claim** | "FNO helps for non-flat ICs" | `results/eval_metrics.json`.
    - That blanket claim was false for noise ICs: interpolation from the mean level is better there (1.54% vs 2.15%), because noise smooths out within the first steps. The guide text now lists, family by family, where each model wins.
24. **Payloads: gzip level 5 and 5-decimal rounding** | no gzip | `results/gzip_tradeoff.json`, `results/latency.json`.
    - /api/compare goes from 793 KB to 246 KB on the wire.
    - Gzip costs about 54 ms of CPU on localhost. It is kept because it saves bandwidth on any real network (break-even about 65 Mbit/s).
25. **Plotly vendored locally: the cartesian partial build 2.35.2 (1.33 MB, 450 KB gzipped)** | CDN.
    - The CDN link I first used (cdnjs) was a 404 for this version.
    - The npm tarball's SHA-512 was verified against the registry's `dist.integrity` before extraction.
26. **Batch benchmark uses medians of individually timed repetitions** | mean of a batch.
    - Your browser reopened during the run, and laptop timings were 2-3× slower than in the quiet earlier measurement. Medians do not remove a systematic slowdown, so the UI and README state the measurement conditions.
27. **Not deployed (user decision).**
    - Hugging Face refused to create the Docker Space with HTTP 402: free CPU Docker/Gradio Spaces now need PRO (`results/deploy_attempt.json`).
    - The server measured 514 MB RSS under load, leaving no headroom on 512 MB free container tiers.
    - Options offered to the user: PRO, Render free tier, an in-browser static rewrite, no deploy. They chose no deploy.
    - The Dockerfile, the staged-context script and the deploy script are kept and were smoke-tested locally from the staged folder.
