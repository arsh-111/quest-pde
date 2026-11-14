---
title: Sorption Surrogate Lab
emoji: 🧪
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: FNO surrogate vs validated solver, PDEBench diffusion-sorption
---

# Sorption Surrogate Lab

PDEBench 1D diffusion–sorption: a 5 × FNO ensemble surrogate compared live against an independently written, validated
implicit solver, with an interpolation baseline, an out-of-distribution stress lab and a physics-shift lab.

The surrogate's physics is fixed at the training values. All evaluation numbers shown in the app come from result files
produced in the source project.

API: `GET /api/health`, `GET /api/meta`, `POST /api/predict`, `POST /api/solve`, `POST /api/compare`,
`POST /api/physics_sweep`, `POST /api/benchmark`.
