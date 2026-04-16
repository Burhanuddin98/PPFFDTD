# PPFFDTD Algorithm Specification

Complete step-by-step specification of the PFFDTD algorithm (Hamilton 2021).
Reference: `sim_fdtd.py` in the original PFFDTD repository.

## Overview

FDTD solves the 3D wave equation on a regular Cartesian grid with
frequency-dependent impedance boundary conditions modelled as parallel
RLC circuits (IIR digital filters).

## Grid

- Dimensions: `Nx x Ny x Nz` (includes 2-cell halo on each side)
- Grid spacing: `h` (meters)
- Time step: `Ts` (seconds), constrained by CFL: `l2 = (c*Ts/h)^2 <= 1/3`
- Stability factor: `l = c*Ts/h`
- Flat indexing: `idx = ix*Ny*Nz + iy*Nz + iz`

## Node types

1. **Air nodes**: interior, not touching any boundary. 6-point stencil.
2. **Boundary nodes** (`bn_ixyz`): touching at least one wall face.
   - **Rigid** (`mat_bn == -1`): perfectly reflecting, no filter state.
   - **Lossy** (`mat_bn >= 0`): frequency-dependent absorption via IIR filter.
3. **Halo nodes**: 2-cell padding around the grid. Extrapolated each step.
4. **ABC nodes** (`bna_ixyz`): outermost interior layer (ix==2 or ix==Nx-3, etc.).
   Absorbing boundary condition applied after leapfrog.

## Material model

Each material k has `Mb[k]` parallel RLC branches, each defined by a DEF triplet:
- `D[m]`: inductance-like (kg/m^4)
- `E[m]`: resistance-like (Pa*s/m^3)
- `F[m]`: capacitance-like (Pa/m)

The continuous-time admittance is:
```
Y(s) = sum_m  s / (D[m]*s^2 + E[m]*s + F[m])
```

## Coefficient setup (once, before time loop)

For each material k with Mb branches:
```
Dh[m] = D[m] / Ts
Eh[m] = E[m]
Fh[m] = F[m] * Ts

b[m]   = 1.0 / (2*Dh[m] + Eh[m] + 0.5*Fh[m])
bd[m]  = b[m] * (2*Dh[m] - Eh[m] - 0.5*Fh[m])
bDh[m] = b[m] * Dh[m]
bFh[m] = b[m] * Fh[m]
beta   = sum(b[:Mb])
```

## Time loop (for each step n = 0, 1, ..., Nt-1)

### Step 1: Save ABC previous values
```
u2ba[i] = u0[bna_ixyz[i]]     for i = 0..Nba-1
```

### Step 2: Halo extrapolation
Copy interior values to halo (2-cell deep on each face):
```
u1[:,:,0]  = u1[:,:,2]        (z-low halo)
u1[:,:,1]  = u1[:,:,2]
u1[:,:,-1] = u1[:,:,-3]       (z-high halo)
u1[:,:,-2] = u1[:,:,-3]
(same for y and x faces)
```

### Step 3: Air stencil (Laplacian at air nodes)
For each interior node NOT on boundary:
```
Lu1[ix,iy,iz] = u1[ix+1,iy,iz] + u1[ix-1,iy,iz]
              + u1[ix,iy+1,iz] + u1[ix,iy-1,iz]
              + u1[ix,iy,iz+1] + u1[ix,iy,iz-1]
              - 6.0 * u1[ix,iy,iz]
```

### Step 4: Boundary stencil (Laplacian at boundary nodes)
For each boundary node i with adjacency `adj_bn[i, 0..5]`:
```
K = sum(adj_bn[i,:])
Lu1[bn_ixyz[i]] = sum_d(adj_bn[i,d] * u1[bn_ixyz[i] + offset[d]]) - K * u1[bn_ixyz[i]]
```
where offsets = [+NyNz, -NyNz, +Nz, -Nz, +1, -1] for the 6 neighbors.
`adj_bn[i,d]` is 0 or 1 (0 = wall, 1 = air neighbor).

### Step 5: Save lossy boundary previous values
```
u2b[i] = u0[bnl_ixyz[i]]      for i = 0..Nbl-1
```

### Step 6: Leapfrog update (ALL interior nodes)
```
u0[ix,iy,iz] = 2*u1[ix,iy,iz] - u0[ix,iy,iz] + l2 * Lu1[ix,iy,iz]
```
for ix in [1, Nx-2], iy in [1, Ny-2], iz in [1, Nz-2].

### Step 7: Boundary IIR filter (lossy nodes only)

For each lossy boundary node i (mat_bn[i] >= 0):

```python
k = mat_bnl[i]                          # material index
ib = bnl_ixyz[i]                        # grid index

# 7a. Subtract filter feedback
u0[ib] -= l * ssaf_bnl[i] * sum_m(2*bDh[k,m]*vh1[i,m] - bFh[k,m]*gh1[i,m])

# 7b. Admittance normalization
lo2Kbg = 0.5 * l * ssaf_bnl[i] * beta[k]
u0[ib] = (u0[ib] + lo2Kbg * u2b[i]) / (1.0 + lo2Kbg)

# 7c. Update branch velocity
vh0[i,m] = b[k,m] * (u0[ib] - u2b[i]) + bd[k,m] * vh1[i,m] - 2*bFh[k,m] * gh1[i,m]

# 7d. Integrate to strain
gh1[i,m] += 0.5*vh0[i,m] + 0.5*vh1[i,m]
```

### Step 8: ABC update
For each ABC node i:
```
lQ = l * Q_bna[i]
u0[bna_ixyz[i]] = (u0[bna_ixyz[i]] + lQ * u2ba[i]) / (1.0 + lQ)
```
where `Q_bna[i]` = 1 (face), 2 (edge), or 3 (corner).

### Step 9: Source injection
```
u0[in_ixyz[s]] += in_sigs[s, n]         for s = 0..Ns-1
```

### Step 10: Record output
```
u_out[r, n] = u1[out_ixyz[r]]           for r = 0..Nr-1
```
Note: records from u1 (previous step), not u0 (current step).

### Step 11: Swap
```
u0, u1 = u1, u0
vh0, vh1 = vh1, vh0
```

## Surface area correction

`ssaf_bnl[i]` is the surface-area-to-volume ratio correction for boundary node i.
For Cartesian grids: `ssaf_bnl = saf_bnl` (raw surface area fraction from voxelizer).
For FCC grids: `ssaf_bnl = saf_bnl * 0.5 / sqrt(2)`.

The voxelizer computes `saf_bn[i]` as the effective surface area fraction
based on the dot product between the actual surface normal and the grid face normal.
This corrects for staircasing artifacts.

## ABC nodes

ABC nodes are on the second-to-last layer of the grid interior:
```
Q = 0
if ix == 2 or ix == Nx-3: Q += 1
if iy == 2 or iy == Ny-3: Q += 1
if iz == 2 or iz == Nz-3: Q += 1
if Q > 0: this is an ABC node with Q_bna = Q
```

## Sided materials

PFFDTD uses one-sided materials: only one side of a surface is lossy.
The voxelizer marks boundary nodes as lossy (mat_bn >= 0) or rigid (mat_bn == -1)
based on which side of the surface they fall on.
The lossy side carries the IIR filter state; the rigid side reflects perfectly.
This halves the boundary computation cost.
