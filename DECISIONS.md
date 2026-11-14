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
