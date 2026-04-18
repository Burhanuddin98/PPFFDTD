# PPFFDTD Hardware Requirements

## Overview

PPFFDTD wraps Brian Hamilton's PFFDTD wave solver with a non-intrusive ROM. The computational cost scales with room volume and target frequency. This document details the hardware requirements for each BRAS benchmark room.

## Scaling rules

- Grid voxels: N ~ V × (f_max / c)³ × PPW³
- VRAM: ~16 bytes per voxel (2 pressure fields × 8 bytes)
- Boundary state: ~12 × MMb × N_boundary × 8 bytes
- ROM training: 33 FDTD runs (Smolyak level 2, 3D parameter space)

Default settings: PPW = 6, c = 343 m/s, CFL = 1/√3

## BRAS S09 — Seminar Room

| Parameter | Value |
|-----------|-------|
| Volume | 169 m³ (8.4 × 6.7 × 3.0 m) |
| Surfaces | 6 |
| Source-receiver pairs | 10 |

| f_max | Grid | Voxels | VRAM | FDTD time (GPU) | ROM training |
|-------|------|--------|------|------------------|--------------|
| 500 Hz | 79×64×32 | 162K | 0.1 GB | ~15s | ~8 min |
| 1000 Hz | 157×127×62 | 1.24M | 0.5 GB | ~77s | ~42 min |
| 2000 Hz | 313×253×123 | 9.8M | 3.5 GB | ~10 min | ~5.5 hrs |
| 4000 Hz | 625×505×245 | 77M | 28 GB | ~80 min | ~44 hrs |

**Minimum GPU:** RTX 2060 (6 GB) at 1000 Hz
**Recommended GPU:** RTX 3060 (12 GB) at 2000 Hz

## BRAS S10 — Lecture Hall

| Parameter | Value |
|-----------|-------|
| Volume | ~300 m³ |
| Surfaces | 6 |
| Source-receiver pairs | 10 |

| f_max | Voxels | VRAM | FDTD time (GPU) | ROM training |
|-------|--------|------|------------------|--------------|
| 500 Hz | ~290K | 0.1 GB | ~25s | ~14 min |
| 1000 Hz | ~2.3M | 0.9 GB | ~150s | ~82 min |
| 2000 Hz | ~18M | 6.5 GB | ~20 min | ~11 hrs |

**Minimum GPU:** RTX 2060 (6 GB) at 1000 Hz
**Recommended GPU:** RTX 3090 (24 GB) at 2000 Hz

## BRAS S11 — Auditorium

| Parameter | Value |
|-----------|-------|
| Volume | 15,764 m³ |
| Surfaces | Multiple (complex geometry) |
| Source-receiver pairs | 10 |

| f_max | Voxels | VRAM | FDTD time (GPU) | ROM training |
|-------|--------|------|------------------|--------------|
| 250 Hz | ~1.5M | 0.6 GB | ~90s | ~50 min |
| 500 Hz | ~12M | 4.5 GB | ~12 min | ~6.5 hrs |
| 1000 Hz | ~100M | 36 GB | ~100 min | ~55 hrs |
| 2000 Hz | ~790M | 285 GB | infeasible | infeasible |

**Minimum GPU:** RTX 3090 (24 GB) at 500 Hz
**Required for 1000 Hz:** A100 (80 GB) or multi-GPU

## Cloud options for S11

| Provider | GPU | VRAM | Cost | S11 at 500 Hz | S11 at 1000 Hz |
|----------|-----|------|------|---------------|----------------|
| Kaggle (free) | T4 | 16 GB | Free | Possible | No |
| Colab Pro | A100 | 40 GB | $10/month | Yes | Possible |
| Lambda Cloud | A100 | 80 GB | $1.10/hr | Yes | Yes |
| vast.ai | A100 | 80 GB | $0.80/hr | Yes | Yes |
| AWS p4d | A100 ×8 | 640 GB | $32/hr | Yes | Yes (fast) |

## System RAM requirements

| Room | f_max | Voxelizer RAM | Simulation RAM | ROM training RAM |
|------|-------|---------------|----------------|------------------|
| S09 | 1000 Hz | 2 GB | 4 GB | 8 GB |
| S10 | 1000 Hz | 4 GB | 6 GB | 12 GB |
| S11 | 500 Hz | 8 GB | 12 GB | 24 GB |
| S11 | 1000 Hz | 32 GB | 48 GB | 96 GB |

## Software requirements

- Python 3.10+
- numpy, scipy, numba, h5py, gmsh, matplotlib, resampy
- scikit-learn (ROM GP regression)
- CuPy with matching CUDA version (GPU acceleration)

## Current validated configuration

| Component | Detail |
|-----------|--------|
| OS | Windows 11 |
| CPU | 12-thread (numba parallel) |
| GPU | NVIDIA RTX 2060 (6 GB VRAM) |
| RAM | 16 GB |
| CUDA | 12.4 |
| CuPy | Latest |
| Tested rooms | BRAS S09 at 1000 Hz, CHORAS MeasurementRoom at 1000 Hz |
| FDTD time | 77s (GPU), 207s (CPU) on S09 |
| ROM training | 19 min (33 cases) on MeasurementRoom |
| ROM accuracy | 1% T30 error on unseen materials |

## Notes

- VRAM estimates include state arrays, boundary filter state, operator matrices, and CuPy workspace. Actual usage may be 20-30% higher due to temporary allocations.
- ROM training times assume 33 Smolyak cases. Can be reduced to 13 cases (level 1) at the cost of accuracy in the parameter space corners.
- The voxelizer runs on CPU (single-threaded on Windows due to multiprocessing limitations). Voxelization time is typically 5-30% of total pipeline time.
- All times assume double precision (float64). Single precision (float32) halves memory and roughly doubles throughput but requires stability safeguards (see PFFDTD README).
