"""
PPFFDTD core FDTD engine.

Implements the complete PFFDTD time-stepping algorithm:
  - 7-point Cartesian Laplacian stencil
  - Leapfrog (Verlet) time integration
  - Frequency-dependent IIR boundary filters (parallel RLC)
  - First-order absorbing boundary conditions (ABC)
  - Surface area correction for staircase boundaries

Every step follows docs/algorithm.md exactly.
"""

import numpy as np
import time as _time

MMb = 12  # max RLC branches per material


class FDTDEngine:
    """Pure-Python FDTD engine for room acoustics.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions (including 2-cell halo on each side).
    h : float
        Grid spacing (meters).
    c : float
        Speed of sound (m/s), default 343.
    """

    def __init__(self, Nx, Ny, Nz, h, c=343.0):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.h = h
        self.c = c
        self.N = Nx * Ny * Nz
        self.NyNz = Ny * Nz

        # CFL
        self.Ts = h / (c * np.sqrt(3))  # maximum stable Ts for Cartesian
        self.l = c * self.Ts / h         # should be 1/sqrt(3)
        self.l2 = self.l ** 2            # should be 1/3

        # State arrays
        self.u0 = np.zeros(self.N)       # pressure at step n+1
        self.u1 = np.zeros(self.N)       # pressure at step n

        # Boundary data (set via set_boundary)
        self.bn_ixyz = np.array([], dtype=np.int64)
        self.adj_bn = np.empty((0, 6), dtype=np.float64)
        self.bnl_ixyz = np.array([], dtype=np.int64)
        self.mat_bnl = np.array([], dtype=np.int8)
        self.ssaf_bnl = np.array([], dtype=np.float64)
        self.Nbn = 0
        self.Nbl = 0

        # Boundary mask
        self.bn_mask = np.zeros(self.N, dtype=bool)

        # ABC
        self._compute_abc()

        # Source / receiver
        self.sources = []
        self.receivers = []

        # Materials (set via set_materials)
        self.Nmat = 0
        self.mat_coeffs = None

    # ---- Setup ----

    def set_boundary(self, bn_ixyz, adj_bn, mat_bn, saf_bn):
        """Set boundary nodes from voxelizer output.

        Parameters
        ----------
        bn_ixyz : (Nbn,) int64 — flat indices of all boundary nodes
        adj_bn : (Nbn, 6) float64 — adjacency (0 or 1 per neighbor direction)
        mat_bn : (Nbn,) int8 — material index (-1 = rigid, >=0 = lossy)
        saf_bn : (Nbn,) float64 — surface area fraction
        """
        self.bn_ixyz = bn_ixyz.astype(np.int64)
        self.adj_bn = adj_bn.astype(np.float64)
        self.Nbn = bn_ixyz.size

        # Boundary mask
        self.bn_mask[:] = False
        self.bn_mask[bn_ixyz] = True

        # Lossy subset
        lossy = mat_bn > -1
        self.bnl_ixyz = bn_ixyz[lossy].astype(np.int64)
        self.mat_bnl = mat_bn[lossy].astype(np.int8)
        self.ssaf_bnl = saf_bn[lossy].astype(np.float64)
        self.Nbl = self.bnl_ixyz.size

        # Filter state
        self.vh0 = np.zeros((self.Nbl, MMb))
        self.vh1 = np.zeros((self.Nbl, MMb))
        self.gh1 = np.zeros((self.Nbl, MMb))
        self.u2b = np.zeros(self.Nbl)

    def set_materials(self, DEF, Mb):
        """Set material DEF triplets and compute filter coefficients.

        Parameters
        ----------
        DEF : (Nmat, MMb, 3) — D, E, F triplets per material per branch
        Mb : (Nmat,) int8 — number of active branches per material
        """
        self.Nmat = len(Mb)
        Ts = self.Ts

        # Precompute per-material coefficients
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
            Eh = E
            Fh = F * Ts

            b = 1.0 / (2.0 * Dh + Eh + 0.5 * Fh)
            d = 2.0 * Dh - Eh - 0.5 * Fh

            self.b[k, :M] = b
            self.bd[k, :M] = b * d
            self.bDh[k, :M] = b * Dh
            self.bFh[k, :M] = b * Fh
            self.beta[k] = np.sum(b)

        # Precompute per-node coefficients (expand material -> node)
        self._expand_coeffs()

    def _expand_coeffs(self):
        """Expand per-material coefficients to per-boundary-node arrays."""
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
        """Add a point source.

        Parameters
        ----------
        idx : int — flat grid index
        signal : (Nt,) array — source signal (pre-scaled)
        """
        self.sources.append((int(idx), np.asarray(signal, dtype=np.float64)))

    def add_receiver(self, idx):
        """Add a point receiver.

        Parameters
        ----------
        idx : int — flat grid index
        """
        self.receivers.append(int(idx))

    def _compute_abc(self):
        """Identify ABC nodes (second-to-last interior layer)."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        abc_list = []
        q_list = []
        for ix in range(1, Nx - 1):
            for iy in range(1, Ny - 1):
                for iz in range(1, Nz - 1):
                    Q = 0
                    if ix == 1 or ix == Nx - 2: Q += 1
                    if iy == 1 or iy == Ny - 2: Q += 1
                    if iz == 1 or iz == Nz - 2: Q += 1
                    if Q > 0:
                        abc_list.append(ix * Ny * Nz + iy * Nz + iz)
                        q_list.append(Q)
        self.bna_ixyz = np.array(abc_list, dtype=np.int64)
        self.Q_bna = np.array(q_list, dtype=np.int8)
        self.Nba = len(abc_list)
        self.u2ba = np.zeros(self.Nba)

    # ---- Time stepping ----

    def run(self, Nt, callback=None):
        """Run FDTD simulation for Nt time steps.

        Parameters
        ----------
        Nt : int — number of time steps
        callback : callable(n, engine), optional — called every 10%

        Returns
        -------
        u_out : (Nr, Nt) array — recorded impulse responses
        """
        u0 = self.u0
        u1 = self.u1
        l2 = self.l2
        l = self.l
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        NyNz = self.NyNz

        bn_mask = self.bn_mask
        bn_ixyz = self.bn_ixyz
        adj_bn = self.adj_bn
        bnl_ixyz = self.bnl_ixyz
        ssaf_bnl = self.ssaf_bnl
        vh0, vh1, gh1 = self.vh0, self.vh1, self.gh1
        u2b = self.u2b
        b_n, bd_n, bDh_n, bFh_n = self.b_n, self.bd_n, self.bDh_n, self.bFh_n
        lo2Kbg = self.lo2Kbg

        bna_ixyz = self.bna_ixyz
        Q_bna = self.Q_bna
        u2ba = self.u2ba

        offsets = np.array([NyNz, -NyNz, Nz, -Nz, 1, -1], dtype=np.int64)

        Nr = len(self.receivers)
        u_out = np.zeros((Nr, Nt))

        t0 = _time.perf_counter()

        for n in range(Nt):
            # Step 1: Save ABC previous
            u2ba[:] = u0[bna_ixyz]

            # Step 2: Halo extrapolation
            u1_3d = u1.reshape(Nx, Ny, Nz)
            u1_3d[:, :, 0] = u1_3d[:, :, 2]
            u1_3d[:, :, 1] = u1_3d[:, :, 2]
            u1_3d[:, :, -1] = u1_3d[:, :, -3]
            u1_3d[:, :, -2] = u1_3d[:, :, -3]
            u1_3d[:, 0, :] = u1_3d[:, 2, :]
            u1_3d[:, 1, :] = u1_3d[:, 2, :]
            u1_3d[:, -1, :] = u1_3d[:, -3, :]
            u1_3d[:, -2, :] = u1_3d[:, -3, :]
            u1_3d[0, :, :] = u1_3d[2, :, :]
            u1_3d[1, :, :] = u1_3d[2, :, :]
            u1_3d[-1, :, :] = u1_3d[-3, :, :]
            u1_3d[-2, :, :] = u1_3d[-3, :, :]

            # Step 3: Air stencil (vectorized)
            s = slice(2, -2)  # interior excluding halo
            Lu1 = np.zeros(self.N)
            Lu1_3d = Lu1.reshape(Nx, Ny, Nz)
            air_3d = (~bn_mask.reshape(Nx, Ny, Nz))[s, s, s]
            Lu1_3d[s, s, s] = (
                u1_3d[4:, 2:-2, 2:-2] + u1_3d[:-4, 2:-2, 2:-2] +
                u1_3d[2:-2, 4:, 2:-2] + u1_3d[2:-2, :-4, 2:-2] +
                u1_3d[2:-2, 2:-2, 4:] + u1_3d[2:-2, 2:-2, :-4] -
                6.0 * u1_3d[s, s, s]
            ) * air_3d

            # Step 4: Boundary stencil
            K_bn = np.sum(adj_bn, axis=1)
            bn_val = -K_bn * u1[bn_ixyz]
            for d in range(6):
                nb = np.clip(bn_ixyz + offsets[d], 0, self.N - 1)
                bn_val += adj_bn[:, d] * u1[nb]
            Lu1[bn_ixyz] = bn_val

            # Step 5: Save lossy boundary previous
            u2b[:] = u0[bnl_ixyz]

            # Step 6: Leapfrog
            s1 = slice(1, -1)
            u0_3d = u0.reshape(Nx, Ny, Nz)
            u0_3d[s1, s1, s1] = (2.0 * u1_3d[s1, s1, s1]
                                 - u0_3d[s1, s1, s1]
                                 + l2 * Lu1_3d[s1, s1, s1])

            # Step 7: Boundary IIR filter
            # 7a: subtract filter feedback
            feedback = np.sum(2.0 * bDh_n * vh1 - bFh_n * gh1, axis=1)
            u0[bnl_ixyz] -= l * ssaf_bnl * feedback

            # 7b: admittance normalization
            u0[bnl_ixyz] = (u0[bnl_ixyz] + lo2Kbg * u2b) / (1.0 + lo2Kbg)

            # 7c: update branch velocity
            diff = u0[bnl_ixyz] - u2b
            vh0[:] = b_n * diff[:, None] + bd_n * vh1 - 2.0 * bFh_n * gh1

            # 7d: integrate to strain
            gh1 += 0.5 * vh0 + 0.5 * vh1

            # Step 8: ABC
            lQ = l * Q_bna.astype(np.float64)
            u0[bna_ixyz] = (u0[bna_ixyz] + lQ * u2ba) / (1.0 + lQ)

            # Step 9: Source injection
            for src_idx, sig in self.sources:
                if n < len(sig):
                    u0[src_idx] += sig[n]

            # Step 10: Record (from u1, previous step)
            for ri, rec_idx in enumerate(self.receivers):
                u_out[ri, n] = u1[rec_idx]

            # Step 11: Swap
            u0, u1 = u1, u0
            vh0, vh1 = vh1, vh0

            if callback and (n + 1) % max(1, Nt // 10) == 0:
                callback(n + 1, self)

        # Rebind after swaps
        self.u0 = u0
        self.u1 = u1
        self.vh0 = vh0
        self.vh1 = vh1

        elapsed = _time.perf_counter() - t0
        print(f"PPFFDTD: {Nt} steps in {elapsed:.1f}s "
              f"({Nt/elapsed:.0f} steps/s, "
              f"{self.N*Nt/elapsed/1e6:.1f} MVox/s)")

        return u_out

    def get_ir(self, u_out, rec_idx=0):
        """Get impulse response for a receiver."""
        return u_out[rec_idx, :]

    @property
    def fs(self):
        return 1.0 / self.Ts

    @property
    def room_dims(self):
        """Interior room dimensions (excluding halo)."""
        return ((self.Nx - 5) * self.h,
                (self.Ny - 5) * self.h,
                (self.Nz - 5) * self.h)
