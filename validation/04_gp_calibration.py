"""Step 4 of the PPFFDTD ROM verification suite.

GP posterior calibration / reliability diagram. Uses the LOO results saved in
docs/figures/02_loo_cv_data.npz (which include per-coefficient predicted std
sigma_hat as well as the true coefficients).

We answer two questions:

    1. Are the GP's predicted standard deviations sigma_hat informative about
       the actual prediction errors?

       Standardized residual: z = (c_true - c_pred) / sigma_hat.
       If the GP is well-calibrated AND its (assumed Gaussian) posterior is
       correct, z ~ N(0, 1).

       We check this by:
         - Plotting the empirical CDF of z against the standard normal CDF.
         - Reporting empirical coverage at +/- 1 sigma and +/- 2 sigma
           (theoretical: 68.27% and 95.45%).

    2. Is sigma_hat correlated with the actual |error|?

       Reliability diagram: bin (sigma_hat) into K quantile bins; in each bin
       compute mean(sigma_hat) (predicted spread) and rms(c_true - c_pred)
       (observed spread). On a calibrated GP the bin centers lie on y = x.

Output:
    docs/figures/04_gp_calibration.png
    docs/figures/04_gp_calibration_data.npz
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    loo_data_path = REPO / "docs" / "figures" / "02_loo_cv_data.npz"
    fig_path      = REPO / "docs" / "figures" / "04_gp_calibration.png"
    data_out      = REPO / "docs" / "figures" / "04_gp_calibration_data.npz"

    print(f"Loading {loo_data_path}")
    d = np.load(loo_data_path, allow_pickle=True)
    c_true = d["coeffs_true"]    # (33, r)
    c_pred = d["coeffs_pred"]    # (33, r)
    sig    = d["coeffs_sig"]     # (33, r)
    n_fold, r = c_true.shape
    print(f"shapes: n_fold={n_fold}, r={r}")

    # Standardized residuals (z-scores)
    err = c_true - c_pred
    # Guard against numerically zero sigma (sklearn's WhiteKernel can drive sigma
    # arbitrarily small if the data is essentially noise-free in a fold).
    sig_clip = np.maximum(sig, 1e-12)
    z = err / sig_clip
    z_flat = z.ravel()
    err_flat = err.ravel()
    sig_flat = sig.ravel()
    print(f"residuals: mean={err_flat.mean():.4e}, std={err_flat.std():.4e}")
    print(f"sigma_hat: mean={sig_flat.mean():.4e}, median={np.median(sig_flat):.4e}, "
          f"min={sig_flat.min():.4e}, max={sig_flat.max():.4e}")

    # Empirical coverage at +/- 1, 2, 3 sigma
    for k_sigma in [1.0, 2.0, 3.0]:
        cov = float(np.mean(np.abs(z_flat) <= k_sigma)) * 100
        thr = (1.0 - 2 * norm.sf(k_sigma)) * 100
        print(f"  |z| <= {k_sigma}: empirical {cov:5.2f}%   theoretical {thr:5.2f}%")

    # Reliability diagram: bin sigma_hat into K quantile bins
    K = 8
    order = np.argsort(sig_flat)
    sig_sorted = sig_flat[order]
    err_sorted = err_flat[order]
    bins = np.array_split(np.arange(len(sig_sorted)), K)
    bin_sigma_mean = np.array([sig_sorted[b].mean() for b in bins])
    bin_err_rms    = np.array([np.sqrt((err_sorted[b] ** 2).mean()) for b in bins])
    bin_count      = np.array([len(b) for b in bins])

    print(f"\nReliability bins (K={K}, sorted by sigma_hat):")
    for i, (sm, er, n) in enumerate(zip(bin_sigma_mean, bin_err_rms, bin_count)):
        print(f"  bin {i+1}: sigma_hat={sm:.4e}  rms_err={er:.4e}  ratio={er/sm:.3f}  (n={n})")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) Empirical CDF of z vs N(0,1)
    ax = axes[0]
    grid = np.linspace(min(-5, z_flat.min() - 0.5), max(5, z_flat.max() + 0.5), 401)
    ecdf = np.searchsorted(np.sort(z_flat), grid) / len(z_flat)
    ax.plot(grid, ecdf, color="C0", linewidth=1.6, label="empirical")
    ax.plot(grid, norm.cdf(grid), color="k", linestyle="--", linewidth=0.9, label="N(0, 1)")
    ax.set_xlabel(r"Standardized residual $z = (c_\mathrm{true} - c_\mathrm{pred}) / \hat{\sigma}$")
    ax.set_ylabel("CDF")
    ax.set_title("(a) Empirical CDF of standardized residuals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Histogram of |z| with theoretical coverage markers
    ax = axes[1]
    ax.hist(np.clip(np.abs(z_flat), 0, 5), bins=40, color="C2", alpha=0.85, edgecolor="white")
    for k_sigma in [1.0, 2.0, 3.0]:
        ax.axvline(k_sigma, color="grey", linestyle=":", linewidth=0.8)
        thr = (1.0 - 2 * norm.sf(k_sigma)) * 100
        cov = float(np.mean(np.abs(z_flat) <= k_sigma)) * 100
        ax.text(k_sigma, ax.get_ylim()[1] * 0.95, f"  {k_sigma:.0f}σ\n  emp {cov:.1f}%\n  exp {thr:.1f}%",
                fontsize=7, color="grey", va="top")
    ax.set_xlabel(r"$|z|$")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) |z| distribution ({len(z_flat)} GP-predicted coeffs)")
    ax.grid(True, alpha=0.3)

    # (c) Reliability diagram
    ax = axes[2]
    ax.loglog(bin_sigma_mean, bin_err_rms, "o-", color="C3", markersize=7, linewidth=1.2,
              label="reliability bins")
    lo = float(min(bin_sigma_mean.min(), bin_err_rms.min())) * 0.5
    hi = float(max(bin_sigma_mean.max(), bin_err_rms.max())) * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, label="perfectly calibrated")
    ax.set_xlabel("Predicted spread (mean $\\hat{\\sigma}$ in bin)")
    ax.set_ylabel("Observed RMS error in bin")
    ax.set_title(f"(c) Reliability diagram (K={K} bins)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle("PPFFDTD ROM — GP posterior calibration (LOO over r=12 coefficients x 33 folds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    np.savez_compressed(data_out,
                        z=z, err=err, sig=sig,
                        bin_sigma_mean=bin_sigma_mean,
                        bin_err_rms=bin_err_rms,
                        bin_count=bin_count)
    print(f"Data saved:   {data_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
