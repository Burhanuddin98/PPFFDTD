"""
PPFFDTD core FDTD engine.

Exact reimplementation of PFFDTD's sim_fdtd.py algorithm.
Every step matches Hamilton's reference — same ranges, same order.
See docs/algorithm.md for the specification.
"""

import numpy as np
import time as _time

MMb = 12  # max RLC branches per material


class FDTDEngine:
    """Pure-Python FDTD engine for room acoustics.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions (including halo).
    h : float
        Grid spacing (meters).
    c : float
        Speed of sound (m/s).
    Ts : float
        Time step (seconds). Must satisfy CFL: (c*Ts/h)^2 <= 1/3.
    """

    def __init__(self, Nx, Ny, Nz, h, c=343.0, Ts=None):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.h = h
        self.c = c
        self.N = Nx * Ny * Nz
        self.NyNz = Ny * Nz

        if Ts is None:
            Ts = h / (c * np.sqrt(3))
        self.Ts = Ts
        self.l = c * Ts / h
        self.l2 = self.l ** 2
        assert self.l2 <= 1.0 / 3.0 + 1e-10, f"CFL violated: l2={self.l2}"

        # State (flat arrays)
        self.u0 = np.zeros(self.N)
        self.u1 = np.zeros(self.N)

        # Boundary data (set via set_boundary)
        self.bn_ixyz = np.array([], dtype=np.int64)
        self.adj_bn = np.empty((0, 6), dtype=np.float64)
        self.bnl_ixyz = np.array([], dtype=np.int64)
        self.mat_bnl = np.array([], dtype=np.int8)
        self.ssaf_bnl = np.array([], dtype=np.float64)
        self.Nbn = 0
        self.Nbl = 0
        self.bn_mask = np.zeros(self.N, dtype=bool)

        self._compute_abc()
        self.sources = []
        self.receivers = []

    # ---- Setup ----

    def set_boundary(self, bn_ixyz, adj_bn, mat_bn, saf_bn):
        self.bn_ixyz = bn_ixyz.astype(np.int64)
        self.adj_bn = adj_bn.astype(np.float64)
        self.Nbn = bn_ixyz.size

        self.bn_mask[:] = False
        self.bn_mask[bn_ixyz] = True

        lossy = mat_bn > -1
        self.bnl_ixyz = bn_ixyz[lossy].astype(np.int64)
        self.mat_bnl = mat_bn[lossy].astype(np.int8)
        self.ssaf_bnl = saf_bn[lossy].astype(np.float64)
        self.Nbl = self.bnl_ixyz.size

        self.vh0 = np.zeros((self.Nbl, MMb))
        self.vh1 = np.zeros((self.Nbl, MMb))
        self.gh1 = np.zeros((self.Nbl, MMb))
        self.u2b = np.zeros(self.Nbl)

    def set_materials(self, DEF, Mb):
        self.Nmat = len(Mb)
        self.Mb = Mb
        Ts = self.Ts

        self.b = np.zeros((self.Nmat, MMb))
        self.bd = np.zeros((self.Nmat, MMb))
        self.bDh = np.zeros((self.Nmat, MMb))
        self.bFh = np.zeros((self.Nmat, MMb))
        self.beta = np.zeros(self.Nmat)

        for k in range(self.Nmat):
            M = Mb[k]
            if M == 0:
                continue
            D, E, F = DEF[k, :M].T
            Dh = D / Ts
            Fh = F * Ts
            b = 1.0 / (2.0 * Dh + E + 0.5 * Fh)
            d = 2.0 * Dh - E - 0.5 * Fh

            self.b[k, :M] = b
            self.bd[k, :M] = b * d
            self.bDh[k, :M] = b * Dh
            self.bFh[k, :M] = b * Fh
            self.beta[k] = np.sum(b)

        # Expand to per-node
        Nbl = self.Nbl
        self.b_n = np.zeros((Nbl, MMb))
        self.bd_n = np.zeros((Nbl, MMb))
        self.bDh_n = np.zeros((Nbl, MMb))
        self.bFh_n = np.zeros((Nbl, MMb))
        self.lo2Kbg = np.zeros(Nbl)

        for i in range(Nbl):
            k = self.mat_bnl[i]
            self.b_n[i] = self.b[k]
            self.bd_n[i] = self.bd[k]
            self.bDh_n[i] = self.bDh[k]
            self.bFh_n[i] = self.bFh[k]
            self.lo2Kbg[i] = 0.5 * self.l * self.ssaf_bnl[i] * self.beta[k]

    def add_source(self, idx, signal):
        self.sources.append((int(idx), np.asarray(signal, dtype=np.float64)))

    def add_receiver(self, idx):
        self.receivers.append(int(idx))

    def _compute_abc(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        abc, q = [], []
        for ix in range(1, Nx - 1):
            for iy in range(1, Ny - 1):
                for iz in range(1, Nz - 1):
                    Q = 0
                    if ix == 1 or ix == Nx - 2: Q += 1
                    if iy == 1 or iy == Ny - 2: Q += 1
                    if iz == 1 or iz == Nz - 2: Q += 1
                    if Q > 0:
                        abc.append(ix * Ny * Nz + iy * Nz + iz)
                        q.append(Q)
        self.bna_ixyz = np.array(abc, dtype=np.int64)
        self.Q_bna = np.array(q, dtype=np.int8)
        self.Nba = len(abc)
        self.u2ba = np.zeros(self.Nba)

    # ---- Run ----

    def run(self, Nt, callback=None):
        """Run FDTD for Nt steps. Returns u_out (Nr, Nt)."""
        u0 = self.u0
        u1 = self.u1
        l2, l = self.l2, self.l
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        NyNz = self.NyNz
        N = self.N

        bn_mask = self.bn_mask
        bn_ixyz = self.bn_ixyz
        adj_bn = self.adj_bn
        bnl_ixyz = self.bnl_ixyz
        ssaf_bnl = self.ssaf_bnl
        vh0, vh1, gh1 = self.vh0, self.vh1, self.gh1
        u2b = self.u2b
        b_n, bd_n = self.b_n, self.bd_n
        bDh_n, bFh_n = self.bDh_n, self.bFh_n
        lo2Kbg = self.lo2Kbg

        bna_ixyz = self.bna_ixyz
        Q_bna = self.Q_bna.astype(np.float64)
        u2ba = self.u2ba

        offsets = np.array([NyNz, -NyNz, Nz, -Nz, 1, -1], dtype=np.int64)
        K_bn = np.sum(adj_bn, axis=1)

        Nr = len(self.receivers)
        u_out = np.zeros((Nr, Nt))

        # 3D views (no copy — same memory as flat)
        bn_mask_3d = bn_mask.reshape(Nx, Ny, Nz)

        # Interior slice [1, Nx-2] — matches PFFDTD exactly
        si = slice(1, -1)

        t0 = _time.perf_counter()

        for n in range(Nt):
            # Reshape for 3D operations (views, no copy)
            u0_3d = u0.reshape(Nx, Ny, Nz)
            u1_3d = u1.reshape(Nx, Ny, Nz)

            # Step 1: Save ABC previous
            u2ba[:] = u0[bna_ixyz]

            # Step 2: Halo — copy index 2 → 0, Nx-3 → Nx-1
            # (matches nb_flip_halos exactly: ONE cell, not two)
            u1_3d[:, :, 0] = u1_3d[:, :, 2]
            u1_3d[:, :, -1] = u1_3d[:, :, -3]
            u1_3d[:, 0, :] = u1_3d[:, 2, :]
            u1_3d[:, -1, :] = u1_3d[:, -3, :]
            u1_3d[0, :, :] = u1_3d[2, :, :]
            u1_3d[-1, :, :] = u1_3d[-3, :, :]

            # Step 3: Air Laplacian — range [1, Nx-2], skip boundary nodes
            # Lu1[ix,iy,iz] = u1[ix+1]+u1[ix-1]+u1[iy+1]+u1[iy-1]+u1[iz+1]+u1[iz-1] - 6*u1
            Lu1 = np.zeros(N)
            Lu1_3d = Lu1.reshape(Nx, Ny, Nz)
            Lu1_3d[si, si, si] = (
                u1_3d[2:, 1:-1, 1:-1] + u1_3d[:-2, 1:-1, 1:-1] +
                u1_3d[1:-1, 2:, 1:-1] + u1_3d[1:-1, :-2, 1:-1] +
                u1_3d[1:-1, 1:-1, 2:] + u1_3d[1:-1, 1:-1, :-2] -
                6.0 * u1_3d[si, si, si]
            ) * (~bn_mask_3d[si, si, si])

            # Step 4: Boundary Laplacian — sparse adjacency
            bn_val = -K_bn * u1[bn_ixyz]
            for d in range(6):
                nb = np.clip(bn_ixyz + offsets[d], 0, N - 1)
                bn_val += adj_bn[:, d] * u1[nb]
            Lu1[bn_ixyz] = bn_val

            # Step 5: Save lossy boundary previous
            u2b[:] = u0[bnl_ixyz]

            # Step 6: Leapfrog — range [1, Nx-2]
            u0_3d[si, si, si] = (2.0 * u1_3d[si, si, si]
                                 - u0_3d[si, si, si]
                                 + l2 * Lu1_3d[si, si, si])

            # Step 7: Boundary IIR filter
            if self.Nbl > 0:
                # 7a: feedback
                feedback = np.sum(2.0 * bDh_n * vh1 - bFh_n * gh1, axis=1)
                u0[bnl_ixyz] -= l * ssaf_bnl * feedback

                # 7b: admittance
                u0[bnl_ixyz] = (u0[bnl_ixyz] + lo2Kbg * u2b) / (1.0 + lo2Kbg)

                # 7c: branch velocity
                diff = u0[bnl_ixyz] - u2b
                vh0[:] = (b_n * diff[:, None]
                          + bd_n * vh1
                          - 2.0 * bFh_n * gh1)

                # 7d: strain integration
                gh1 += 0.5 * vh0 + 0.5 * vh1

            # Step 8: ABC
            if self.Nba > 0:
                lQ = l * Q_bna
                u0[bna_ixyz] = (u0[bna_ixyz] + lQ * u2ba) / (1.0 + lQ)

            # Step 9: Source injection
            for src_idx, sig in self.sources:
                if n < len(sig):
                    u0[src_idx] += sig[n]

            # Step 10: Record from u1 (previous step)
            for ri, rec_idx in enumerate(self.receivers):
                u_out[ri, n] = u1[rec_idx]

            # Step 11: Swap
            u0, u1 = u1, u0
            vh0, vh1 = vh1, vh0

            if callback and (n + 1) % max(1, Nt // 10) == 0:
                callback(n + 1, Nt)

        self.u0, self.u1 = u0, u1
        self.vh0, self.vh1 = vh0, vh1

        elapsed = _time.perf_counter() - t0
        print(f"PPFFDTD: {Nt} steps in {elapsed:.1f}s "
              f"({Nt / elapsed:.0f} steps/s, "
              f"{N * Nt / elapsed / 1e6:.1f} MVox/s)")
        return u_out

    @property
    def fs(self):
        return 1.0 / self.Ts

    @property
    def room_dims(self):
        return ((self.Nx - 5) * self.h,
                (self.Ny - 5) * self.h,
                (self.Nz - 5) * self.h)
