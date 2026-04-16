"""
GPU-accelerated PFFDTD engine via CuPy RawKernel.

Fused CUDA kernels + zero CPU roundtrips per step:
  - Source signals pre-uploaded to GPU
  - Receiver output buffered on GPU, transferred once at end
  - All operations via RawKernel (no CuPy array-op overhead)
"""

import numpy as np
import time as _time

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


# ---- CUDA kernel source code (pure ASCII) ----

_AIR_KERNEL = r"""
extern "C" __global__
void air_stencil_leapfrog(
    double* __restrict__ u0,
    const double* __restrict__ u1,
    const bool* __restrict__ bn_mask,
    const double l2,
    const int Nx, const int Ny, const int Nz)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int NyNz = Ny * Nz;
    int total = Nx * NyNz;
    if (idx >= total) return;

    int ix = idx / NyNz;
    int iy = (idx % NyNz) / Nz;
    int iz = idx % Nz;

    if (ix < 2 || ix >= Nx-2 || iy < 2 || iy >= Ny-2 || iz < 2 || iz >= Nz-2)
        return;
    if (bn_mask[idx])
        return;

    double Lu = -6.0 * u1[idx]
              + u1[idx + NyNz] + u1[idx - NyNz]
              + u1[idx + Nz]   + u1[idx - Nz]
              + u1[idx + 1]    + u1[idx - 1];

    u0[idx] = 2.0 * u1[idx] - u0[idx] + l2 * Lu;
}
"""

_BN_KERNEL = r"""
extern "C" __global__
void boundary_update(
    double* __restrict__ u0,
    const double* __restrict__ u1,
    const long long* __restrict__ bn_ixyz,
    const double* __restrict__ adj_bn,
    const int Nbn,
    const long long* __restrict__ bnl_ixyz,
    const double* __restrict__ ssaf_bnl,
    double* __restrict__ vh0,
    double* __restrict__ vh1,
    double* __restrict__ gh1,
    double* __restrict__ u2b,
    const double* __restrict__ b_g,
    const double* __restrict__ bd_g,
    const double* __restrict__ bDh_g,
    const double* __restrict__ bFh_g,
    const double* __restrict__ lo2Kbg,
    const int Nbl,
    const int MMb,
    const double l2,
    const double l,
    const int Ny, const int Nz,
    const int total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Phase 1: boundary stencil + leapfrog (all bn nodes)
    if (i < Nbn) {
        long long ib = bn_ixyz[i];
        double K = 0.0;
        double nb_sum = 0.0;
        int NyNz = Ny * Nz;
        long long offsets[6] = {NyNz, -NyNz, Nz, -Nz, 1, -1};
        for (int d = 0; d < 6; d++) {
            double a = adj_bn[i * 6 + d];
            K += a;
            long long nb = ib + offsets[d];
            if (nb >= 0 && nb < total)
                nb_sum += a * u1[nb];
        }
        double Lu = nb_sum - K * u1[ib];
        u0[ib] = 2.0 * u1[ib] - u0[ib] + l2 * Lu;
    }

    // Phase 2: IIR boundary filter (lossy bn nodes only)
    if (i < Nbl) {
        long long ib = bnl_ixyz[i];
        double u0_bn = u0[ib];
        double ssaf = ssaf_bnl[i];

        double feedback = 0.0;
        for (int m = 0; m < MMb; m++) {
            int idx = i * MMb + m;
            feedback += 2.0 * bDh_g[idx] * vh1[idx] - bFh_g[idx] * gh1[idx];
        }
        u0_bn -= l * ssaf * feedback;

        double lkb = lo2Kbg[i];
        u0_bn = (u0_bn + lkb * u2b[i]) / (1.0 + lkb);

        u0[ib] = u0_bn;

        double u2b_i = u2b[i];
        double diff = u0_bn - u2b_i;
        for (int m = 0; m < MMb; m++) {
            int idx = i * MMb + m;
            double vh0_new = b_g[idx] * diff + bd_g[idx] * vh1[idx] - 2.0 * bFh_g[idx] * gh1[idx];
            gh1[idx] += 0.5 * vh0_new + 0.5 * vh1[idx];
            vh0[idx] = vh0_new;
        }
    }
}
"""

_ABC_KERNEL = r"""
extern "C" __global__
void abc_update(
    double* __restrict__ u0,
    const double* __restrict__ u2ba,
    const long long* __restrict__ bna_ixyz,
    const signed char* __restrict__ Q_bna,
    const double l,
    const int Nba)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= Nba) return;

    long long ib = bna_ixyz[i];
    double lQ = l * (double)Q_bna[i];
    u0[ib] = (u0[ib] + lQ * u2ba[i]) / (1.0 + lQ);
}
"""

_HALO_KERNEL = r"""
extern "C" __global__
void halo_copy(
    double* __restrict__ u,
    const int Nx, const int Ny, const int Nz)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int NyNz = Ny * Nz;

    // Z faces
    int Nxy = Nx * Ny;
    if (tid < Nxy) {
        int ix = tid / Ny;
        int iy = tid % Ny;
        int base = ix * NyNz + iy * Nz;
        u[base + 0] = u[base + 2];
        u[base + 1] = u[base + 2];
        u[base + Nz - 1] = u[base + Nz - 3];
        u[base + Nz - 2] = u[base + Nz - 3];
        return;
    }
    tid -= Nxy;

    // Y faces
    int Nxz = Nx * Nz;
    if (tid < Nxz) {
        int ix = tid / Nz;
        int iz = tid % Nz;
        int base_x = ix * NyNz;
        u[base_x + 0 * Nz + iz] = u[base_x + 2 * Nz + iz];
        u[base_x + 1 * Nz + iz] = u[base_x + 2 * Nz + iz];
        u[base_x + (Ny-1) * Nz + iz] = u[base_x + (Ny-3) * Nz + iz];
        u[base_x + (Ny-2) * Nz + iz] = u[base_x + (Ny-3) * Nz + iz];
        return;
    }
    tid -= Nxz;

    // X faces
    if (tid < NyNz) {
        int yz = tid;
        u[0 * NyNz + yz] = u[2 * NyNz + yz];
        u[1 * NyNz + yz] = u[2 * NyNz + yz];
        u[(Nx-1) * NyNz + yz] = u[(Nx-3) * NyNz + yz];
        u[(Nx-2) * NyNz + yz] = u[(Nx-3) * NyNz + yz];
    }
}
"""

_SAVE_BN_KERNEL = r"""
extern "C" __global__
void save_bn(
    const double* __restrict__ u,
    double* __restrict__ u2b,
    const long long* __restrict__ ixyz,
    const int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    u2b[i] = u[ixyz[i]];
}
"""

_SRC_INJECT_KERNEL = r"""
extern "C" __global__
void src_inject(
    double* __restrict__ u0,
    const long long* __restrict__ in_ixyz,
    const double* __restrict__ in_sigs,
    const int Ns,
    const int Nt,
    const int n)
{
    int si = blockDim.x * blockIdx.x + threadIdx.x;
    if (si >= Ns) return;
    double val = in_sigs[si * Nt + n];
    if (val != 0.0) {
        u0[in_ixyz[si]] += val;
    }
}
"""

_REC_KERNEL = r"""
extern "C" __global__
void record_output(
    const double* __restrict__ u1,
    double* __restrict__ u_out,
    const long long* __restrict__ out_ixyz,
    const int Nr,
    const int Nt,
    const int n)
{
    int ri = blockDim.x * blockIdx.x + threadIdx.x;
    if (ri >= Nr) return;
    u_out[ri * Nt + n] = u1[out_ixyz[ri]];
}
"""


_DFT_BATCH_KERNEL = r"""
extern "C" __global__
void dft_accumulate_batch(
    const double* __restrict__ u1,
    double* __restrict__ dft_cos,
    double* __restrict__ dft_sin,
    const double* __restrict__ cos_vals,
    const double* __restrict__ sin_vals,
    const int N_total,
    const int N_freqs)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N_total) return;
    double p = u1[i];
    for (int fi = 0; fi < N_freqs; fi++) {
        long long offset = (long long)fi * N_total + i;
        dft_cos[offset] += p * cos_vals[fi];
        dft_sin[offset] += p * sin_vals[fi];
    }
}
"""


def run_gpu_time_snapshots(engine, n_snapshots=200):
    """Run PFFDTD on GPU, save full pressure field at regular intervals.

    Returns snapshots: (n_snapshots, N_total) real array.
    These capture transient wavefront dynamics for time-domain POD basis.
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy required")

    Nx, Ny, Nz = engine.Nx, engine.Ny, engine.Nz
    Nt = engine.Nt
    l2 = float(engine.l2)
    l = float(engine.l)
    total = Nx * Ny * Nz
    MMb = engine.vh0.shape[1]
    Ns = engine.Ns
    Nr = engine.Nr

    save_every = max(1, Nt // n_snapshots)
    actual_snaps = len(range(0, Nt, save_every))
    snap_bytes = actual_snaps * total * 8
    print(f"  Time snapshots: save every {save_every} steps, "
          f"{actual_snaps} snapshots, {snap_bytes/1e9:.1f} GB CPU RAM")

    # Compile kernels
    air_kern = cp.RawKernel(_AIR_KERNEL, 'air_stencil_leapfrog')
    bn_kern = cp.RawKernel(_BN_KERNEL, 'boundary_update')
    abc_kern = cp.RawKernel(_ABC_KERNEL, 'abc_update')
    halo_kern = cp.RawKernel(_HALO_KERNEL, 'halo_copy')
    save_kern = cp.RawKernel(_SAVE_BN_KERNEL, 'save_bn')
    src_kern = cp.RawKernel(_SRC_INJECT_KERNEL, 'src_inject')
    rec_kern = cp.RawKernel(_REC_KERNEL, 'record_output')

    # Transfer FDTD state to GPU
    u0 = cp.asarray(engine.u0.ravel()).astype(cp.float64)
    u1 = cp.asarray(engine.u1.ravel()).astype(cp.float64)
    bn_mask = cp.asarray(engine.bn_mask.ravel()).astype(cp.bool_)

    bn_ixyz = cp.asarray(engine.bn_ixyz).astype(cp.int64)
    adj_bn = cp.asarray(engine.adj_bn.astype(np.float64)).ravel().astype(cp.float64)
    Nbn = engine.bn_ixyz.size

    bnl_ixyz = cp.asarray(engine.bnl_ixyz).astype(cp.int64)
    ssaf_bnl = cp.asarray(engine.ssaf_bnl).astype(cp.float64)
    Nbl = engine.bnl_ixyz.size

    vh0 = cp.asarray(engine.vh0.ravel()).astype(cp.float64)
    vh1 = cp.asarray(engine.vh1.ravel()).astype(cp.float64)
    gh1 = cp.asarray(engine.gh1.ravel()).astype(cp.float64)
    u2b = cp.asarray(engine.u2b).astype(cp.float64)

    bna_ixyz = cp.asarray(engine.bna_ixyz).astype(cp.int64)
    Q_bna = cp.asarray(engine.Q_bna).astype(cp.int8)
    u2ba = cp.asarray(engine.u2ba).astype(cp.float64)
    Nba = engine.bna_ixyz.size

    mat_bnl = engine.mat_bnl
    mcs = engine.mat_coeffs_struct
    b_arr = np.zeros((Nbl, MMb))
    bd_arr = np.zeros((Nbl, MMb))
    bDh_arr = np.zeros((Nbl, MMb))
    bFh_arr = np.zeros((Nbl, MMb))
    beta_arr = np.zeros(Nbl)
    for i in range(Nbl):
        k = mat_bnl[i]
        if k >= 0:
            b_arr[i] = mcs[k]['b']
            bd_arr[i] = mcs[k]['bd']
            bDh_arr[i] = mcs[k]['bDh']
            bFh_arr[i] = mcs[k]['bFh']
            beta_arr[i] = mcs[k]['beta']
    b_g = cp.asarray(b_arr.ravel()).astype(cp.float64)
    bd_g = cp.asarray(bd_arr.ravel()).astype(cp.float64)
    bDh_g = cp.asarray(bDh_arr.ravel()).astype(cp.float64)
    bFh_g = cp.asarray(bFh_arr.ravel()).astype(cp.float64)
    lo2Kbg = (0.5 * l * cp.asarray(engine.ssaf_bnl) * cp.asarray(beta_arr)).astype(cp.float64)

    in_ixyz_g = cp.asarray(engine.in_ixyz).astype(cp.int64)
    in_sigs_g = cp.asarray(engine.in_sigs).astype(cp.float64)
    out_ixyz_g = cp.asarray(engine.out_ixyz).astype(cp.int64)
    u_out_g = cp.zeros((Nr, Nt), dtype=cp.float64)

    BLOCK = 256
    block = (BLOCK,)
    air_grid = ((total + BLOCK - 1) // BLOCK,)
    bn_grid = ((max(Nbn, Nbl) + BLOCK - 1) // BLOCK,)
    abc_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    halo_grid = (((Nx*Ny + Nx*Nz + Ny*Nz) + BLOCK - 1) // BLOCK,)
    save_bnl_grid = ((max(1, Nbl) + BLOCK - 1) // BLOCK,)
    save_bna_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    src_grid = ((max(1, Ns) + BLOCK - 1) // BLOCK,)
    rec_grid = ((max(1, Nr) + BLOCK - 1) // BLOCK,)

    i_Nx, i_Ny, i_Nz = np.int32(Nx), np.int32(Ny), np.int32(Nz)
    i_Nbn, i_Nbl, i_MMb = np.int32(Nbn), np.int32(Nbl), np.int32(MMb)
    i_Nba, i_Ns, i_Nr, i_Nt = np.int32(Nba), np.int32(Ns), np.int32(Nr), np.int32(Nt)
    i_total = np.int32(total)
    f_l2, f_l = np.float64(l2), np.float64(l)

    # Collect snapshots on CPU (GPU→CPU transfer each save step)
    snapshots = np.zeros((actual_snaps, total), dtype=np.float64)
    snap_idx = 0

    t0 = _time.perf_counter()
    cp.cuda.Stream.null.synchronize()

    for n in range(Nt):
        # FDTD step (identical to run_gpu)
        halo_kern(halo_grid, block, (u1, i_Nx, i_Ny, i_Nz))
        save_kern(save_bna_grid, block, (u0, u2ba, bna_ixyz, i_Nba))
        save_kern(save_bnl_grid, block, (u0, u2b, bnl_ixyz, i_Nbl))
        air_kern(air_grid, block, (u0, u1, bn_mask, f_l2, i_Nx, i_Ny, i_Nz))
        bn_kern(bn_grid, block,
                (u0, u1, bn_ixyz, adj_bn, i_Nbn,
                 bnl_ixyz, ssaf_bnl, vh0, vh1, gh1, u2b,
                 b_g, bd_g, bDh_g, bFh_g, lo2Kbg,
                 i_Nbl, i_MMb, f_l2, f_l, i_Ny, i_Nz, i_total))
        if Nba > 0:
            abc_kern(abc_grid, block, (u0, u2ba, bna_ixyz, Q_bna, f_l, i_Nba))
        src_kern(src_grid, block, (u0, in_ixyz_g, in_sigs_g, i_Ns, i_Nt, np.int32(n)))
        rec_kern(rec_grid, block, (u1, u_out_g, out_ixyz_g, i_Nr, i_Nt, np.int32(n)))

        # Save snapshot every save_every steps
        # Save u0 (just updated with boundary effects) BEFORE swap
        if n % save_every == 0 and snap_idx < actual_snaps:
            cp.cuda.Stream.null.synchronize()
            snapshots[snap_idx] = cp.asnumpy(u0)
            snap_idx += 1

        u0, u1 = u1, u0
        vh0, vh1 = vh1, vh0

        if (n + 1) % max(1, Nt // 10) == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = _time.perf_counter() - t0
            rate = (n + 1) / elapsed
            eta = (Nt - n - 1) / rate
            print(f"  GPU+snap: {n+1}/{Nt} ({elapsed:.0f}s, ~{eta:.0f}s left)",
                  flush=True)

    cp.cuda.Stream.null.synchronize()
    engine.u_out = cp.asnumpy(u_out_g)
    engine.u0 = cp.asnumpy(u0).reshape(Nx, Ny, Nz)
    engine.u1 = cp.asnumpy(u1).reshape(Nx, Ny, Nz)
    engine.vh0 = cp.asnumpy(vh0).reshape(Nbl, MMb)
    engine.vh1 = cp.asnumpy(vh1).reshape(Nbl, MMb)
    engine.gh1 = cp.asnumpy(gh1).reshape(Nbl, MMb)

    elapsed = _time.perf_counter() - t0
    print(f"  GPU+snap done: {Nt} steps in {elapsed:.0f}s, "
          f"{snap_idx} snapshots saved")
    return snapshots[:snap_idx]


def run_gpu_snapshots(engine, target_freqs):
    """Run PFFDTD on GPU and accumulate running DFT at target frequencies.

    Returns snapshots: complex array of shape (N_freqs, N_total).
    Each row is the frequency-domain pressure field U(f_i) at all grid points.
    These are exact Helmholtz solutions from the FDTD FOM.
    """
    if not HAS_GPU:
        raise RuntimeError("CuPy required for GPU snapshots")

    Nx, Ny, Nz = engine.Nx, engine.Ny, engine.Nz
    Nt = engine.Nt
    l2 = float(engine.l2)
    l = float(engine.l)
    total = Nx * Ny * Nz
    MMb = engine.vh0.shape[1]
    Ns = engine.Ns
    Nr = engine.Nr
    Ts = engine.Ts
    N_freqs = len(target_freqs)

    print(f"  GPU snapshots: {total:,} voxels x {N_freqs} frequencies "
          f"({total * N_freqs * 16 / 1e9:.1f} GB DFT buffers)")

    # Compile kernels
    air_kern = cp.RawKernel(_AIR_KERNEL, 'air_stencil_leapfrog')
    bn_kern = cp.RawKernel(_BN_KERNEL, 'boundary_update')
    abc_kern = cp.RawKernel(_ABC_KERNEL, 'abc_update')
    halo_kern = cp.RawKernel(_HALO_KERNEL, 'halo_copy')
    save_kern = cp.RawKernel(_SAVE_BN_KERNEL, 'save_bn')
    src_kern = cp.RawKernel(_SRC_INJECT_KERNEL, 'src_inject')
    rec_kern = cp.RawKernel(_REC_KERNEL, 'record_output')
    dft_kern = cp.RawKernel(_DFT_BATCH_KERNEL, 'dft_accumulate_batch')

    # Transfer FDTD state to GPU (same as run_gpu)
    u0 = cp.asarray(engine.u0.ravel()).astype(cp.float64)
    u1 = cp.asarray(engine.u1.ravel()).astype(cp.float64)
    bn_mask = cp.asarray(engine.bn_mask.ravel()).astype(cp.bool_)

    bn_ixyz = cp.asarray(engine.bn_ixyz).astype(cp.int64)
    adj_bn = cp.asarray(engine.adj_bn.astype(np.float64)).ravel().astype(cp.float64)
    Nbn = engine.bn_ixyz.size

    bnl_ixyz = cp.asarray(engine.bnl_ixyz).astype(cp.int64)
    ssaf_bnl = cp.asarray(engine.ssaf_bnl).astype(cp.float64)
    Nbl = engine.bnl_ixyz.size

    vh0 = cp.asarray(engine.vh0.ravel()).astype(cp.float64)
    vh1 = cp.asarray(engine.vh1.ravel()).astype(cp.float64)
    gh1 = cp.asarray(engine.gh1.ravel()).astype(cp.float64)
    u2b = cp.asarray(engine.u2b).astype(cp.float64)

    bna_ixyz = cp.asarray(engine.bna_ixyz).astype(cp.int64)
    Q_bna = cp.asarray(engine.Q_bna).astype(cp.int8)
    u2ba = cp.asarray(engine.u2ba).astype(cp.float64)
    Nba = engine.bna_ixyz.size

    # Material coefficients
    mat_bnl = engine.mat_bnl
    mcs = engine.mat_coeffs_struct

    b_arr = np.zeros((Nbl, MMb))
    bd_arr = np.zeros((Nbl, MMb))
    bDh_arr = np.zeros((Nbl, MMb))
    bFh_arr = np.zeros((Nbl, MMb))
    beta_arr = np.zeros(Nbl)
    for i in range(Nbl):
        k = mat_bnl[i]
        if k >= 0:
            b_arr[i] = mcs[k]['b']
            bd_arr[i] = mcs[k]['bd']
            bDh_arr[i] = mcs[k]['bDh']
            bFh_arr[i] = mcs[k]['bFh']
            beta_arr[i] = mcs[k]['beta']

    b_g = cp.asarray(b_arr.ravel()).astype(cp.float64)
    bd_g = cp.asarray(bd_arr.ravel()).astype(cp.float64)
    bDh_g = cp.asarray(bDh_arr.ravel()).astype(cp.float64)
    bFh_g = cp.asarray(bFh_arr.ravel()).astype(cp.float64)
    lo2Kbg = (0.5 * l * cp.asarray(engine.ssaf_bnl) * cp.asarray(beta_arr)).astype(cp.float64)

    # Source/receiver on GPU
    in_ixyz_g = cp.asarray(engine.in_ixyz).astype(cp.int64)
    in_sigs_g = cp.asarray(engine.in_sigs).astype(cp.float64)
    out_ixyz_g = cp.asarray(engine.out_ixyz).astype(cp.int64)
    u_out_g = cp.zeros((Nr, Nt), dtype=cp.float64)

    # DFT accumulators on GPU: (N_freqs, total) for cos and sin
    dft_cos = cp.zeros((N_freqs, total), dtype=cp.float64)
    dft_sin = cp.zeros((N_freqs, total), dtype=cp.float64)

    # Precompute angular frequencies + GPU scratch for cos/sin values
    omegas = 2.0 * np.pi * np.array(target_freqs)
    cos_vals_g = cp.zeros(N_freqs, dtype=cp.float64)
    sin_vals_g = cp.zeros(N_freqs, dtype=cp.float64)

    # Kernel config
    BLOCK = 256
    block = (BLOCK,)
    air_grid = ((total + BLOCK - 1) // BLOCK,)
    bn_grid = ((max(Nbn, Nbl) + BLOCK - 1) // BLOCK,)
    abc_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    halo_grid = (((Nx*Ny + Nx*Nz + Ny*Nz) + BLOCK - 1) // BLOCK,)
    save_bnl_grid = ((max(1, Nbl) + BLOCK - 1) // BLOCK,)
    save_bna_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    src_grid = ((max(1, Ns) + BLOCK - 1) // BLOCK,)
    rec_grid = ((max(1, Nr) + BLOCK - 1) // BLOCK,)
    dft_grid = ((total + BLOCK - 1) // BLOCK,)

    i_Nx, i_Ny, i_Nz = np.int32(Nx), np.int32(Ny), np.int32(Nz)
    i_Nbn, i_Nbl, i_MMb = np.int32(Nbn), np.int32(Nbl), np.int32(MMb)
    i_Nba, i_Ns, i_Nr, i_Nt = np.int32(Nba), np.int32(Ns), np.int32(Nr), np.int32(Nt)
    i_total = np.int32(total)
    f_l2, f_l = np.float64(l2), np.float64(l)

    t0 = _time.perf_counter()
    cp.cuda.Stream.null.synchronize()

    for n in range(Nt):
        # FDTD step (identical to run_gpu)
        halo_kern(halo_grid, block, (u1, i_Nx, i_Ny, i_Nz))
        save_kern(save_bna_grid, block, (u0, u2ba, bna_ixyz, i_Nba))
        save_kern(save_bnl_grid, block, (u0, u2b, bnl_ixyz, i_Nbl))
        air_kern(air_grid, block, (u0, u1, bn_mask, f_l2, i_Nx, i_Ny, i_Nz))
        bn_kern(bn_grid, block,
                (u0, u1, bn_ixyz, adj_bn, i_Nbn,
                 bnl_ixyz, ssaf_bnl, vh0, vh1, gh1, u2b,
                 b_g, bd_g, bDh_g, bFh_g, lo2Kbg,
                 i_Nbl, i_MMb, f_l2, f_l, i_Ny, i_Nz, i_total))
        if Nba > 0:
            abc_kern(abc_grid, block, (u0, u2ba, bna_ixyz, Q_bna, f_l, i_Nba))
        src_kern(src_grid, block, (u0, in_ixyz_g, in_sigs_g, i_Ns, i_Nt, np.int32(n)))
        rec_kern(rec_grid, block, (u1, u_out_g, out_ixyz_g, i_Nr, i_Nt, np.int32(n)))

        # DFT accumulation: one batched kernel for all frequencies
        t_n = n * Ts
        cos_vals_np = np.cos(omegas * t_n)
        sin_vals_np = np.sin(omegas * t_n)
        cos_vals_g[:] = cp.asarray(cos_vals_np)
        sin_vals_g[:] = cp.asarray(sin_vals_np)
        dft_kern(dft_grid, block,
                 (u1, dft_cos, dft_sin, cos_vals_g, sin_vals_g,
                  i_total, np.int32(N_freqs)))

        # Swap
        u0, u1 = u1, u0
        vh0, vh1 = vh1, vh0

        if (n + 1) % max(1, Nt // 10) == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = _time.perf_counter() - t0
            rate = (n + 1) / elapsed
            eta = (Nt - n - 1) / rate
            print(f"  GPU+DFT: {n+1}/{Nt} ({elapsed:.0f}s, ~{eta:.0f}s left)",
                  flush=True)

    cp.cuda.Stream.null.synchronize()

    # Transfer DFT results to CPU as complex snapshots
    cos_np = cp.asnumpy(dft_cos)  # (N_freqs, total)
    sin_np = cp.asnumpy(dft_sin)
    snapshots = (cos_np - 1j * sin_np) * Ts  # scale by dt for proper DFT

    # Also save receiver output
    engine.u_out = cp.asnumpy(u_out_g)
    engine.u0 = cp.asnumpy(u0).reshape(Nx, Ny, Nz)
    engine.u1 = cp.asnumpy(u1).reshape(Nx, Ny, Nz)
    engine.vh0 = cp.asnumpy(vh0).reshape(Nbl, MMb)
    engine.vh1 = cp.asnumpy(vh1).reshape(Nbl, MMb)
    engine.gh1 = cp.asnumpy(gh1).reshape(Nbl, MMb)

    elapsed = _time.perf_counter() - t0
    print(f"  GPU+DFT done: {Nt} steps in {elapsed:.0f}s, "
          f"{N_freqs} snapshots collected", flush=True)

    return snapshots


def run_gpu(engine):
    """Run full PFFDTD simulation on GPU with fused CUDA kernels."""
    if not HAS_GPU:
        print("No CuPy, using CPU", flush=True)
        engine.run_all()
        return

    Nx, Ny, Nz = engine.Nx, engine.Ny, engine.Nz
    Nt = engine.Nt
    l2 = float(engine.l2)
    l = float(engine.l)
    total = Nx * Ny * Nz
    MMb = engine.vh0.shape[1]
    Ns = engine.Ns
    Nr = engine.Nr

    # Compile kernels
    air_kern = cp.RawKernel(_AIR_KERNEL, 'air_stencil_leapfrog')
    bn_kern = cp.RawKernel(_BN_KERNEL, 'boundary_update')
    abc_kern = cp.RawKernel(_ABC_KERNEL, 'abc_update')
    halo_kern = cp.RawKernel(_HALO_KERNEL, 'halo_copy')
    save_kern = cp.RawKernel(_SAVE_BN_KERNEL, 'save_bn')
    src_kern = cp.RawKernel(_SRC_INJECT_KERNEL, 'src_inject')
    rec_kern = cp.RawKernel(_REC_KERNEL, 'record_output')

    # Transfer everything to GPU
    u0 = cp.asarray(engine.u0.ravel()).astype(cp.float64)
    u1 = cp.asarray(engine.u1.ravel()).astype(cp.float64)
    bn_mask = cp.asarray(engine.bn_mask.ravel()).astype(cp.bool_)

    bn_ixyz = cp.asarray(engine.bn_ixyz).astype(cp.int64)
    adj_bn = cp.asarray(engine.adj_bn.astype(np.float64)).ravel().astype(cp.float64)
    Nbn = engine.bn_ixyz.size

    bnl_ixyz = cp.asarray(engine.bnl_ixyz).astype(cp.int64)
    ssaf_bnl = cp.asarray(engine.ssaf_bnl).astype(cp.float64)
    Nbl = engine.bnl_ixyz.size

    vh0 = cp.asarray(engine.vh0.ravel()).astype(cp.float64)
    vh1 = cp.asarray(engine.vh1.ravel()).astype(cp.float64)
    gh1 = cp.asarray(engine.gh1.ravel()).astype(cp.float64)
    u2b = cp.asarray(engine.u2b).astype(cp.float64)

    bna_ixyz = cp.asarray(engine.bna_ixyz).astype(cp.int64)
    Q_bna = cp.asarray(engine.Q_bna).astype(cp.int8)
    u2ba = cp.asarray(engine.u2ba).astype(cp.float64)
    Nba = engine.bna_ixyz.size

    # Material coefficients
    mat_bnl = engine.mat_bnl
    mcs = engine.mat_coeffs_struct

    b_arr = np.zeros((Nbl, MMb))
    bd_arr = np.zeros((Nbl, MMb))
    bDh_arr = np.zeros((Nbl, MMb))
    bFh_arr = np.zeros((Nbl, MMb))
    beta_arr = np.zeros(Nbl)

    for i in range(Nbl):
        k = mat_bnl[i]
        if k >= 0:
            b_arr[i] = mcs[k]['b']
            bd_arr[i] = mcs[k]['bd']
            bDh_arr[i] = mcs[k]['bDh']
            bFh_arr[i] = mcs[k]['bFh']
            beta_arr[i] = mcs[k]['beta']

    b_g = cp.asarray(b_arr.ravel()).astype(cp.float64)
    bd_g = cp.asarray(bd_arr.ravel()).astype(cp.float64)
    bDh_g = cp.asarray(bDh_arr.ravel()).astype(cp.float64)
    bFh_g = cp.asarray(bFh_arr.ravel()).astype(cp.float64)
    lo2Kbg = (0.5 * l * cp.asarray(engine.ssaf_bnl) * cp.asarray(beta_arr)).astype(cp.float64)

    # Source/receiver fully on GPU
    in_ixyz_g = cp.asarray(engine.in_ixyz).astype(cp.int64)
    in_sigs_g = cp.asarray(engine.in_sigs).astype(cp.float64)  # (Ns, Nt) on GPU
    out_ixyz_g = cp.asarray(engine.out_ixyz).astype(cp.int64)
    u_out_g = cp.zeros((Nr, Nt), dtype=cp.float64)  # record on GPU

    # Kernel launch config
    BLOCK = 256

    air_grid = ((total + BLOCK - 1) // BLOCK,)
    bn_N = max(Nbn, Nbl)
    bn_grid = ((bn_N + BLOCK - 1) // BLOCK,)
    abc_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    halo_total = Nx * Ny + Nx * Nz + Ny * Nz
    halo_grid = ((halo_total + BLOCK - 1) // BLOCK,)
    save_bnl_grid = ((max(1, Nbl) + BLOCK - 1) // BLOCK,)
    save_bna_grid = ((max(1, Nba) + BLOCK - 1) // BLOCK,)
    src_grid = ((max(1, Ns) + BLOCK - 1) // BLOCK,)
    rec_grid = ((max(1, Nr) + BLOCK - 1) // BLOCK,)

    block = (BLOCK,)

    # Pre-compute int32 constants
    i_Nx = np.int32(Nx)
    i_Ny = np.int32(Ny)
    i_Nz = np.int32(Nz)
    i_Nbn = np.int32(Nbn)
    i_Nbl = np.int32(Nbl)
    i_MMb = np.int32(MMb)
    i_Nba = np.int32(Nba)
    i_Ns = np.int32(Ns)
    i_Nr = np.int32(Nr)
    i_Nt = np.int32(Nt)
    i_total = np.int32(total)
    f_l2 = np.float64(l2)
    f_l = np.float64(l)

    t0 = _time.perf_counter()
    cp.cuda.Stream.null.synchronize()

    for n in range(Nt):
        # 1. Halo
        halo_kern(halo_grid, block, (u1, i_Nx, i_Ny, i_Nz))

        # 2. Save boundary previous
        save_kern(save_bna_grid, block, (u0, u2ba, bna_ixyz, i_Nba))
        save_kern(save_bnl_grid, block, (u0, u2b, bnl_ixyz, i_Nbl))

        # 3. Air stencil + leapfrog
        air_kern(air_grid, block, (u0, u1, bn_mask, f_l2, i_Nx, i_Ny, i_Nz))

        # 4. Boundary stencil + leapfrog + IIR filter
        bn_kern(bn_grid, block,
                (u0, u1, bn_ixyz, adj_bn, i_Nbn,
                 bnl_ixyz, ssaf_bnl,
                 vh0, vh1, gh1, u2b,
                 b_g, bd_g, bDh_g, bFh_g, lo2Kbg,
                 i_Nbl, i_MMb, f_l2, f_l, i_Ny, i_Nz, i_total))

        # 5. ABC
        if Nba > 0:
            abc_kern(abc_grid, block, (u0, u2ba, bna_ixyz, Q_bna, f_l, i_Nba))

        # 6. Source injection (GPU kernel, no CPU roundtrip)
        src_kern(src_grid, block, (u0, in_ixyz_g, in_sigs_g, i_Ns, i_Nt, np.int32(n)))

        # 7. Record receiver (GPU kernel, stays on GPU)
        rec_kern(rec_grid, block, (u1, u_out_g, out_ixyz_g, i_Nr, i_Nt, np.int32(n)))

        # 8. Swap
        u0, u1 = u1, u0
        vh0, vh1 = vh1, vh0

        if (n + 1) % max(1, Nt // 10) == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = _time.perf_counter() - t0
            rate = (n + 1) / elapsed
            eta = (Nt - n - 1) / rate
            print(f"  GPU: {n+1}/{Nt} ({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

    cp.cuda.Stream.null.synchronize()

    # Transfer back
    engine.u0 = cp.asnumpy(u0).reshape(Nx, Ny, Nz)
    engine.u1 = cp.asnumpy(u1).reshape(Nx, Ny, Nz)
    engine.vh0 = cp.asnumpy(vh0).reshape(Nbl, MMb)
    engine.vh1 = cp.asnumpy(vh1).reshape(Nbl, MMb)
    engine.gh1 = cp.asnumpy(gh1).reshape(Nbl, MMb)
    engine.u_out = cp.asnumpy(u_out_g)

    elapsed = _time.perf_counter() - t0
    print(f"  GPU done: {Nt} steps in {elapsed:.0f}s ({Nt/elapsed:.0f} steps/s)", flush=True)
