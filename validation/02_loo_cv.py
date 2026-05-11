"""Step 2 of the PPFFDTD ROM verification suite.

Full leave-one-out cross-validation of the trained ROM. For each of the 33
training points i:

    1. Hold out IR_i and params_i.
    2. Recompute ir_mean and Phi from the remaining 32 IRs (full POD per fold).
    3. Project the 32 IRs onto Phi to get the 32x12 coefficient matrix.
    4. Fit 12 independent GPs (Matern-5/2 + ARD + WhiteKernel) on params[~i] -> coeffs[~i].
    5. Predict at params_i, getting mean coefficients c_hat and posterior std sigma_hat.
    6. Reconstruct IR_hat = ir_mean + Phi @ c_hat.
    7. Score against the held-out IR_i:
        - Pearson correlation between IR_i and IR_hat
        - Peak-amplitude ratio
        - Per-band T30 error (octave bands 125, 250, 500, 1000, 2000 Hz)
        - Per-coefficient predicted sigma_hat (for the calibration step 4)

Output:
    docs/figures/02_loo_cv.png            -- summary plots
    docs/figures/02_loo_cv_data.npz       -- raw per-fold arrays for step 4
    Console table -- median / IQR / max per metric

This uses the FULL LOO protocol (POD basis recomputed per fold), not the
fast variant that keeps the all-33 Phi fixed. Full LOO is the honest test:
it shows the entire pipeline's prediction skill, not just GP regression
on a pre-fixed basis.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# Import iso_metrics from the staged choras_integration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "choras_integration" / "pffdtd_method"))
from iso_metrics import _bandpass, _t30  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)        # sklearn convergence chatter
warnings.filterwarnings("ignore", category=RuntimeWarning)


OCTAVE_BANDS = [125, 250, 500, 1000, 2000]


def _per_band_t30(ir, fs, bands=OCTAVE_BANDS):
    """T30 per octave band; None if EDC fit fails."""
    out = []
    for fc in bands:
        ir_b = _bandpass(ir, fs, fc)
        out.append(_t30(ir_b, fs))
    return out


def main():
    here = Path(__file__).resolve().parent
    rom_path = here.parent / "common" / "pffdtd_data" / "rom_v2.npz"
    fig_path = here.parent / "docs" / "figures" / "02_loo_cv.png"
    data_path = here.parent / "docs" / "figures" / "02_loo_cv_data.npz"

    print(f"Loading {rom_path}")
    d = np.load(rom_path, allow_pickle=True)

    training_irs    = d["training_irs"]     # (33, n_t)
    training_params = d["training_params"]  # (33, 3) — scale factors per (floor, ceiling, walls)
    fs_out          = int(d["fs_out"])
    r_saved         = int(d["Phi"].shape[1])

    n_train, n_t = training_irs.shape
    n_dim = training_params.shape[1]
    print(f"n_train={n_train}, n_t={n_t}, n_dim={n_dim}, fs_out={fs_out}, r_saved={r_saved}")

    # Per-fold storage
    corrs       = np.zeros(n_train)
    peak_ratios = np.zeros(n_train)
    t30_true    = np.full((n_train, len(OCTAVE_BANDS)), np.nan)
    t30_pred    = np.full((n_train, len(OCTAVE_BANDS)), np.nan)
    coeffs_true = np.zeros((n_train, r_saved))
    coeffs_pred = np.zeros((n_train, r_saved))
    coeffs_sig  = np.zeros((n_train, r_saved))   # GP posterior std
    log_params  = np.log(training_params + 1e-30)  # GP works in log-scale per rom.py

    print(f"\nRunning {n_train}-fold LOO with full POD re-fitting...")
    for i in range(n_train):
        keep = np.ones(n_train, dtype=bool)
        keep[i] = False

        train_irs_i = training_irs[keep]       # (32, n_t)
        train_lp_i  = log_params[keep]         # (32, 3)
        held_ir     = training_irs[i]
        held_lp     = log_params[i:i+1]        # (1, 3)

        # POD on the 32 retained snapshots
        ir_mean_i = np.mean(train_irs_i, axis=0)
        X_i = (train_irs_i - ir_mean_i).T       # (n_t, 32)
        U, S, _ = np.linalg.svd(X_i, full_matrices=False)
        Phi_i = U[:, :r_saved]                  # (n_t, r)
        coeffs_i = (train_irs_i - ir_mean_i) @ Phi_i  # (32, r)

        # True coefficients for the held-out IR, projected onto this fold's basis
        coeffs_true[i] = (held_ir - ir_mean_i) @ Phi_i

        # Fit r independent GPs over log-params
        for j in range(r_saved):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(nu=2.5, length_scale=np.ones(n_dim),
                               length_scale_bounds=(1e-2, 1e2))
                      + WhiteKernel(noise_level=1e-5,
                                    noise_level_bounds=(1e-10, 1e-1)))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
            gp.fit(train_lp_i, coeffs_i[:, j])
            mu, sd = gp.predict(held_lp, return_std=True)
            coeffs_pred[i, j] = mu[0]
            coeffs_sig[i, j]  = sd[0]

        # Reconstruct IR at the held-out point
        ir_hat = ir_mean_i + Phi_i @ coeffs_pred[i]

        # Scoring
        c = np.corrcoef(held_ir, ir_hat)[0, 1]
        corrs[i] = c

        p_true = float(np.max(np.abs(held_ir)))
        p_hat  = float(np.max(np.abs(ir_hat)))
        peak_ratios[i] = p_hat / p_true if p_true > 0 else np.nan

        t30_true[i] = _per_band_t30(held_ir, fs_out)
        t30_pred[i] = _per_band_t30(ir_hat,  fs_out)

        scales = training_params[i]
        print(f"  fold {i+1:>2}/{n_train}  scales=({scales[0]:.2f},{scales[1]:.2f},{scales[2]:.2f})  "
              f"corr={c:.5f}  peakRatio={peak_ratios[i]:.3f}")

    # ── Aggregate ──
    print("\n=== LOO summary ===")
    print(f"IR correlation:  median={np.median(corrs):.5f}  IQR=[{np.quantile(corrs,0.25):.5f}, {np.quantile(corrs,0.75):.5f}]  min={corrs.min():.5f}")
    print(f"Peak ratio:      median={np.median(peak_ratios):.4f}  IQR=[{np.quantile(peak_ratios,0.25):.4f}, {np.quantile(peak_ratios,0.75):.4f}]")

    # T30 error %: 100 * |pred - true| / true, ignoring NaNs
    t30_err = np.where(
        (t30_true > 0) & np.isfinite(t30_true) & np.isfinite(t30_pred),
        100.0 * np.abs(t30_pred - t30_true) / np.maximum(t30_true, 1e-9),
        np.nan,
    )
    print("Per-band T30 error % (median across folds):")
    for j, fc in enumerate(OCTAVE_BANDS):
        col = t30_err[:, j]
        m = np.nanmedian(col)
        q1 = np.nanquantile(col, 0.25)
        q3 = np.nanquantile(col, 0.75)
        mx = np.nanmax(col)
        n_valid = np.sum(np.isfinite(col))
        print(f"  {fc:>5} Hz:  median={m:5.2f}%  IQR=[{q1:5.2f}, {q3:5.2f}]  max={mx:5.2f}%  (valid {n_valid}/{n_train})")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) IR correlation distribution
    ax = axes[0, 0]
    ax.hist(corrs, bins=20, color="C0", alpha=0.85, edgecolor="white")
    ax.axvline(np.median(corrs), color="C3", linestyle="--", linewidth=1.0,
               label=f"median = {np.median(corrs):.5f}")
    ax.set_xlabel("LOO IR correlation")
    ax.set_ylabel("Fold count")
    ax.set_title("(a) LOO IR correlation (33 folds)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Per-band T30 error box plot
    ax = axes[0, 1]
    data = [t30_err[:, j][np.isfinite(t30_err[:, j])] for j in range(len(OCTAVE_BANDS))]
    ax.boxplot(data, labels=[str(f) for f in OCTAVE_BANDS], showmeans=True)
    ax.set_xlabel("Octave band center (Hz)")
    ax.set_ylabel("LOO T30 error (%)")
    ax.set_title("(b) Per-band T30 error distribution")
    ax.grid(True, alpha=0.3)

    # (c) Coefficient-wise prediction skill
    ax = axes[1, 0]
    for j in range(r_saved):
        ax.scatter(coeffs_true[:, j], coeffs_pred[:, j], alpha=0.5, s=18)
    lo = min(coeffs_true.min(), coeffs_pred.min())
    hi = max(coeffs_true.max(), coeffs_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("True coefficient (LOO basis)")
    ax.set_ylabel("GP predicted coefficient")
    ax.set_title(f"(c) GP prediction vs truth (all r={r_saved} coeffs, 33 folds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # (d) Peak-amplitude ratio
    ax = axes[1, 1]
    ax.hist(peak_ratios, bins=20, color="C2", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="k", linestyle="--", linewidth=0.7, label="perfect = 1.0")
    ax.axvline(np.median(peak_ratios), color="C3", linestyle="--", linewidth=1.0,
               label=f"median = {np.median(peak_ratios):.4f}")
    ax.set_xlabel("Peak amp ratio (predicted / true)")
    ax.set_ylabel("Fold count")
    ax.set_title("(d) Peak amplitude calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("PPFFDTD ROM — full leave-one-out cross-validation (33 folds, POD refit per fold)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    np.savez_compressed(data_path,
                        corrs=corrs,
                        peak_ratios=peak_ratios,
                        t30_true=t30_true,
                        t30_pred=t30_pred,
                        t30_err_pct=t30_err,
                        coeffs_true=coeffs_true,
                        coeffs_pred=coeffs_pred,
                        coeffs_sig=coeffs_sig,
                        training_params=training_params,
                        octave_bands=np.array(OCTAVE_BANDS))
    print(f"Data saved:   {data_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
