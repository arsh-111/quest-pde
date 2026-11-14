# Decisions

Format: decision | alternatives | evidence.

1. **Repo location `D:\AgenticAI\quest-pde`**, not `D:\quest-pde` as the brief says | brief path, session cwd | the user chose this folder when asked at the start of the session.
2. **Deadline read as 2026-11-15 15:29 IST** (end of 15 Nov in UTC+14, the earliest any time zone finishes the day) | IST end of day, 72 h rule | the time zone is unconfirmed, so I took the most conservative reading.
3. **Constants and discretisation are taken from `pdebench/data_gen/src/sim_diff_sorp.py` and `configs/diff-sorp.yaml`** rather than from the brief: D=5e-4, por=0.29, rho_s=2880, k_f=3.5e-4, n_f=0.874, sol=1, T=500, 501 outputs, 1024 cells, cell-centred FVM grid, `solve_ivp` (RK45) at default tolerances.
   - Left BC: ghost cell = 1.
   - Right BC: ghost value = D·(u[N-2]-u[N-1])/dx.
   - R(u) = 1 + (1-por)/por·rho_s·k_f·n_f·(u+1e-6)^(n_f-1).
4. **Possible conflict with the brief:** the generator code makes the initial condition spatially *constant*, `u0 = ones * U(0, 0.2)`, not random noise. This is to be verified against the data.
