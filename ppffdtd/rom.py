"""
Non-intrusive ROM for PFFDTD.

PFFDTD is treated as a black box. Multiple training runs with different
materials produce a set of IRs. POD compresses them into a compact basis.
RBF interpolation maps new material parameters to POD coefficients.

Offline (once, ~17 min):
  - Run PFFDTD 13 times with perturbed materials
  - POD on 13 IRs -> compact basis (r ~ 10)
  - Build RBF interpolation model

Online (instant):
  - New material parameters -> interpolate POD coefficients -> reconstruct IR
"""

import numpy as np
import time as _time
from pathlib import Path


class NonIntrusiveROM:
    """Black-box ROM built from multiple PFFDTD runs.

    Parameters
    ----------
    data_dir : str or Path
        PFFDTD simulation data directory
    pffdtd_python_dir : str or Path
        Path to PFFDTD python/ directory (for imports)
    """

    def __init__(self, data_dir, pffdtd_python_dir):
        self.data_dir = Path(data_dir)
        self.pffdtd_dir = Path(pffdtd_python_dir)
        self._load_baseline()

    def _load_baseline(self):
        """Load baseline material data and grid info."""
        import h5py
        d = self.data_dir

        with h5py.File(d / 'sim_consts.h5', 'r') as f:
            self.c = float(f['c'][()])
            self.Ts = float(f['Ts'][()])
            self.h = float(f['h'][()])
        with h5py.File(d / 'comms_out.h5', 'r') as f:
            self.Nt = int(f['Nt'][()])
            self.Nr = int(f['Nr'][()])
            self.Ns = int(f['Ns'][()])
        with h5py.File(d / 'sim_mats.h5', 'r') as f:
            Nmat = int(f['Nmat'][()])
            Mb = f['Mb'][...]
            DEF = np.zeros((Nmat, 12, 3))
            for i in range(Nmat):
                ds = f[f'mat_{i:02d}_DEF'][...]
                DEF[i, :Mb[i]] = ds
            self.DEF_baseline = DEF.copy()
            self.Mb = Mb
            self.Nmat = Nmat

        self.fs = 1.0 / self.Ts

    def _run_fdtd(self, DEF, use_gpu=True, verbose=True):
        """Run one PFFDTD simulation with given DEF triplets.

        Returns u_out (Nr, Nt) array.
        """
        import sys
        pdir = str(self.pffdtd_dir)
        if pdir not in sys.path:
            sys.path.insert(0, pdir)

        from fdtd.sim_fdtd import SimEngine

        engine = SimEngine(str(self.data_dir), energy_on=False, nthreads=12)
        engine.load_h5_data()

        # Override materials
        engine.DEF = DEF.copy()

        engine.setup_mask()
        engine.allocate_mem()
        engine.set_coeffs()
        engine.checks()

        if use_gpu:
            from fdtd.gpu_engine import run_gpu
            run_gpu(engine)
        else:
            engine.run_all()

        return engine.u_out.copy(), engine.out_reorder.copy()

    def train(self, scales=None, rec_idx=0, use_gpu=True):
        """Run training cases and build ROM.

        Parameters
        ----------
        scales : list of float, optional
            Perturbation scales for each material. Default: [0.5, 2.0]
        rec_idx : int
            Receiver index for IR collection
        use_gpu : bool
            Use GPU acceleration
        """
        if scales is None:
            scales = [0.5, 2.0]

        t0 = _time.perf_counter()

        # Generate training configurations
        configs = []
        param_vectors = []

        # Baseline (all scales = 1.0)
        configs.append(('baseline', self.DEF_baseline.copy()))
        param_vectors.append(np.ones(self.Nmat))

        # Perturb each material independently
        for mat_idx in range(self.Nmat):
            for scale in scales:
                label = f'mat{mat_idx}_x{scale}'
                DEF = self.DEF_baseline.copy()
                # Scale the E coefficient (resistance) — this changes absorption
                DEF[mat_idx, :, 1] *= scale
                configs.append((label, DEF))
                params = np.ones(self.Nmat)
                params[mat_idx] = scale
                param_vectors.append(params)

        n_train = len(configs)
        print(f"NI-ROM: {n_train} training cases ({n_train * 77 / 60:.0f} min estimated)")

        # Run all training cases
        irs = []
        for i, (label, DEF) in enumerate(configs):
            t1 = _time.perf_counter()
            print(f"\n  [{i+1}/{n_train}] {label}...")
            u_out, out_reorder = self._run_fdtd(DEF, use_gpu=use_gpu)
            ir = u_out[out_reorder[rec_idx], :]
            irs.append(ir)
            dt = _time.perf_counter() - t1
            elapsed = _time.perf_counter() - t0
            eta = elapsed / (i+1) * (n_train - i - 1)
            print(f"    done ({dt:.0f}s, total {elapsed:.0f}s, ~{eta:.0f}s left)")

        self.training_irs = np.array(irs)  # (n_train, Nt)
        self.training_params = np.array(param_vectors)  # (n_train, Nmat)
        self.n_train = n_train

        # Build POD basis from training IRs
        self._build_pod()

        # Build interpolation model
        self._build_interpolation()

        elapsed = _time.perf_counter() - t0
        print(f"\nNI-ROM training complete: {elapsed:.0f}s "
              f"({n_train} runs, r={self.r} modes)")

    def _build_pod(self):
        """POD on training IRs -> compact basis."""
        X = self.training_irs.T  # (Nt, n_train)
        # Subtract mean for better compression
        self.ir_mean = np.mean(X, axis=1)
        X_centered = X - self.ir_mean[:, None]

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        energy = np.cumsum(S**2) / np.sum(S**2)
        r = np.searchsorted(energy, 0.9999) + 1
        r = min(r, len(S))

        self.Phi_ir = U[:, :r]  # (Nt, r)
        self.r = r

        # Project training IRs onto basis
        self.training_coeffs = (X_centered.T @ self.Phi_ir)  # (n_train, r)

        print(f"  POD: r={r}, energy={energy[r-1]*100:.4f}%, "
              f"sigma_1={S[0]:.2e}, sigma_r={S[r-1]:.2e}")

    def _build_interpolation(self):
        """Build RBF interpolation: parameters -> POD coefficients."""
        from scipy.interpolate import RBFInterpolator

        # Log-scale the parameters for better interpolation
        log_params = np.log(self.training_params)

        self.rbf_models = []
        for j in range(self.r):
            rbf = RBFInterpolator(log_params, self.training_coeffs[:, j],
                                  kernel='thin_plate_spline', smoothing=0.0)
            self.rbf_models.append(rbf)

        print(f"  RBF interpolation: {self.Nmat}D -> {self.r} coefficients")

    def evaluate(self, material_scales=None):
        """Evaluate ROM with new material parameters.

        Parameters
        ----------
        material_scales : array-like of length Nmat, optional
            Scale factors for each material's E coefficient.
            Default: all 1.0 (baseline).

        Returns
        -------
        ir : ndarray
            Impulse response (Nt,)
        """
        if material_scales is None:
            material_scales = np.ones(self.Nmat)
        material_scales = np.asarray(material_scales, dtype=float)

        log_params = np.log(material_scales).reshape(1, -1)

        # Predict POD coefficients
        coeffs = np.array([rbf(log_params)[0] for rbf in self.rbf_models])

        # Reconstruct IR
        ir = self.ir_mean + self.Phi_ir @ coeffs
        return ir

    def evaluate_ir(self, material_scales=None):
        """Evaluate and return as ImpulseResponse object."""
        from romacoustics.ir import ImpulseResponse
        ir = self.evaluate(material_scales)
        return ImpulseResponse(ir, self.fs)

    def save(self, path):
        """Save trained ROM to disk."""
        np.savez_compressed(path,
                            ir_mean=self.ir_mean,
                            Phi_ir=self.Phi_ir,
                            training_irs=self.training_irs,
                            training_params=self.training_params,
                            training_coeffs=self.training_coeffs,
                            DEF_baseline=self.DEF_baseline,
                            Mb=self.Mb,
                            fs=self.fs, Nt=self.Nt, Nmat=self.Nmat)
        print(f"ROM saved to {path}")

    def load(self, path):
        """Load trained ROM from disk."""
        from scipy.interpolate import RBFInterpolator

        d = np.load(path)
        self.ir_mean = d['ir_mean']
        self.Phi_ir = d['Phi_ir']
        self.training_irs = d['training_irs']
        self.training_params = d['training_params']
        self.training_coeffs = d['training_coeffs']
        self.DEF_baseline = d['DEF_baseline']
        self.Mb = d['Mb']
        self.fs = float(d['fs'])
        self.Nt = int(d['Nt'])
        self.Nmat = int(d['Nmat'])
        self.r = self.Phi_ir.shape[1]
        self.n_train = self.training_params.shape[0]

        self._build_interpolation()
        print(f"ROM loaded: r={self.r}, {self.n_train} training cases")

    @staticmethod
    def t30(ir, fs):
        ir2 = ir**2
        cum = np.cumsum(ir2[::-1])[::-1]
        cum /= (cum[0] + 1e-30)
        edc = 10 * np.log10(cum + 1e-30)
        t = np.arange(len(edc)) / fs
        i5, i35 = np.argmax(edc < -5), np.argmax(edc < -35)
        if i35 > i5 + 10:
            return -60 / np.polyfit(t[i5:i35], edc[i5:i35], 1)[0]
        return float('nan')
