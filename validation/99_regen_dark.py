"""Regenerate all 5 verification plots in dark mode from cached *_data.npz files.

Writes <name>_dark.png next to the original light-mode <name>.png.
No FDTD or GP work — pure plot regeneration. Should finish in seconds.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


# ── Dark theme ──
DARK_BG = "#0d1117"   # GitHub dark
PANEL_BG = "#161b22"
FG = "#e6edf3"        # foreground text
GRID = "#30363d"
ACCENTS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#79c0ff", "#ffa657", "#7ee787", "#ff7b72"]


def apply_dark_style():
    mpl.rcParams.update({
        "figure.facecolor":   DARK_BG,
        "axes.facecolor":     PANEL_BG,
        "savefig.facecolor":  DARK_BG,
        "savefig.edgecolor":  DARK_BG,
        "axes.edgecolor":     FG,
        "axes.labelcolor":    FG,
        "axes.titlecolor":    FG,
        "xtick.color":        FG,
        "ytick.color":        FG,
        "text.color":         FG,
        "grid.color":         GRID,
        "grid.linestyle":     "--",
        "grid.alpha":         0.5,
        "axes.grid":          True,
        "legend.facecolor":   PANEL_BG,
        "legend.edgecolor":   GRID,
        "legend.labelcolor":  FG,
        "axes.prop_cycle":    mpl.cycler(color=ACCENTS),
        "figure.titlesize":   12,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
    })


def fig01_pod(data_path: Path, out_path: Path):
    d = np.load(data_path)
    S = d["singular_values"]
    cum = d["energy_fraction"]
    r_saved = int(d["r_saved"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    k = np.arange(1, len(S) + 1)
    ax_a.semilogy(k, S, "o-", color=ACCENTS[0], markersize=5, linewidth=1.0)
    ax_a.axvline(r_saved, color=ACCENTS[2], linestyle="--", linewidth=1.0,
                 label=f"r = {r_saved} (saved)")
    ax_a.set_xlabel("Basis index k")
    ax_a.set_ylabel(r"Singular value $\sigma_k$")
    ax_a.set_title("(a) POD singular spectrum")
    ax_a.legend(loc="upper right")

    ax_b.plot(k, cum, "o-", color=ACCENTS[1], markersize=5, linewidth=1.0)
    for th in (0.99, 0.999, 0.9999, 0.99999):
        ax_b.axhline(th, color=GRID, linestyle=":", linewidth=0.8)
        ax_b.text(len(S) * 0.98, th, f"  {th}", ha="right", va="bottom",
                  fontsize=7, color="#8b949e")
    ax_b.axvline(r_saved, color=ACCENTS[2], linestyle="--", linewidth=1.0,
                 label=f"r = {r_saved}  (cum E = {cum[r_saved-1]:.6f})")
    ax_b.set_xlabel("Basis index k")
    ax_b.set_ylabel("Cumulative energy fraction")
    ax_b.set_title("(b) Cumulative energy E(r)")
    ax_b.set_ylim(0.0, 1.005)
    ax_b.legend(loc="lower right")

    fig.suptitle("PPFFDTD ROM — POD basis convergence on the 33-point Smolyak training set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig02_loo(data_path: Path, out_path: Path):
    d = np.load(data_path)
    corrs       = d["corrs"]
    peak_ratios = d["peak_ratios"]
    t30_err     = d["t30_err_pct"]
    c_true      = d["coeffs_true"]
    c_pred      = d["coeffs_pred"]
    bands       = d["octave_bands"]
    r_saved     = c_true.shape[1]
    n_train     = c_true.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(corrs, bins=20, color=ACCENTS[0], alpha=0.9, edgecolor=PANEL_BG)
    ax.axvline(np.median(corrs), color=ACCENTS[2], linestyle="--",
               label=f"median = {np.median(corrs):.5f}")
    ax.set_xlabel("LOO IR correlation"); ax.set_ylabel("Fold count")
    ax.set_title("(a) LOO IR correlation (33 folds)"); ax.legend()

    ax = axes[0, 1]
    data = [t30_err[:, j][np.isfinite(t30_err[:, j])] for j in range(len(bands))]
    bp = ax.boxplot(data, tick_labels=[str(f) for f in bands], showmeans=True,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENTS[0]); patch.set_alpha(0.6); patch.set_edgecolor(FG)
    for elem in ("whiskers", "caps", "medians"):
        for line in bp[elem]:
            line.set_color(FG)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(ACCENTS[2]); flier.set_markeredgecolor(FG)
    ax.set_xlabel("Octave band center (Hz)"); ax.set_ylabel("LOO T30 error (%)")
    ax.set_title("(b) Per-band T30 error distribution")

    ax = axes[1, 0]
    for j in range(r_saved):
        ax.scatter(c_true[:, j], c_pred[:, j], alpha=0.6, s=18,
                   color=ACCENTS[j % len(ACCENTS)])
    lo = min(c_true.min(), c_pred.min())
    hi = max(c_true.max(), c_pred.max())
    ax.plot([lo, hi], [lo, hi], color=FG, linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlabel("True coefficient (LOO basis)"); ax.set_ylabel("GP predicted coefficient")
    ax.set_title(f"(c) GP prediction vs truth (all r={r_saved} coeffs, {n_train} folds)")
    ax.legend(); ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    ax.hist(peak_ratios, bins=20, color=ACCENTS[1], alpha=0.9, edgecolor=PANEL_BG)
    ax.axvline(1.0, color=FG, linestyle="--", linewidth=0.7, label="perfect = 1.0")
    ax.axvline(np.median(peak_ratios), color=ACCENTS[2], linestyle="--",
               label=f"median = {np.median(peak_ratios):.4f}")
    ax.set_xlabel("Peak amp ratio (predicted / true)"); ax.set_ylabel("Fold count")
    ax.set_title("(d) Peak amplitude calibration"); ax.legend()

    fig.suptitle(f"PPFFDTD ROM — full leave-one-out cross-validation ({n_train} folds, POD refit per fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig03_unseen(data_path: Path, out_path: Path):
    d = np.load(data_path)
    unseen   = d["unseen_params"]
    irs_t    = d["irs_truth"]
    irs_r    = d["irs_rom"]
    corrs    = d["corrs"]
    t30_t    = d["t30_truth"]
    t30_r    = d["t30_rom"]
    t30_err  = d["t30_err_pct"]
    bands    = d["octave_bands"]
    fs       = 48000
    n        = unseen.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.bar(range(1, n + 1), corrs, color=ACCENTS[0], alpha=0.9, edgecolor=PANEL_BG)
    ax.set_xticks(range(1, n + 1))
    ax.set_ylim(0.99, 1.0001)
    ax.set_xlabel("Unseen point index"); ax.set_ylabel("IR correlation")
    ax.set_title("(a) ROM vs FDTD ground-truth IR correlation")

    ax = axes[0, 1]
    data = [t30_err[:, j][np.isfinite(t30_err[:, j])] for j in range(len(bands))]
    bp = ax.boxplot(data, tick_labels=[str(f) for f in bands], showmeans=True,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENTS[0]); patch.set_alpha(0.6); patch.set_edgecolor(FG)
    for elem in ("whiskers", "caps", "medians"):
        for line in bp[elem]:
            line.set_color(FG)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(ACCENTS[2]); flier.set_markeredgecolor(FG)
    ax.set_xlabel("Octave band center (Hz)"); ax.set_ylabel("T30 error (%)")
    ax.set_title("(b) Per-band T30 error vs FDTD")

    ax = axes[1, 0]
    for j, fc in enumerate(bands):
        valid = np.isfinite(t30_t[:, j]) & np.isfinite(t30_r[:, j])
        if valid.any():
            ax.scatter(t30_t[valid, j], t30_r[valid, j], label=f"{fc} Hz",
                       alpha=0.85, s=28, color=ACCENTS[j % len(ACCENTS)])
    all_vals = np.concatenate([t30_t.ravel(), t30_r.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals):
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], color=FG, linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlabel("FDTD truth T30 (s)"); ax.set_ylabel("ROM-predicted T30 (s)")
    ax.set_title("(c) Per-band T30 scatter (all unseen points)")
    ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    worst = int(np.argmin(corrs))
    t = np.arange(irs_t.shape[1]) / fs
    n_show = int(0.4 * fs)
    ax.plot(t[:n_show], irs_t[worst, :n_show], color=FG,
            linewidth=0.6, label="FDTD truth")
    ax.plot(t[:n_show], irs_r[worst, :n_show], color=ACCENTS[2],
            linewidth=0.6, alpha=0.85, label="ROM predicted")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pressure")
    s = unseen[worst]
    ax.set_title(f"(d) Worst-corr case: idx {worst+1}, "
                 f"scales=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f}), corr={corrs[worst]:.5f}")
    ax.legend(fontsize=8)

    fig.suptitle(f"PPFFDTD ROM — unseen-point validation ({n} fresh CPU FDTD draws)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig04_calibration(data_path: Path, out_path: Path):
    d = np.load(data_path)
    z   = d["z"]
    err = d["err"]
    sig = d["sig"]
    bin_sig = d["bin_sigma_mean"]
    bin_err = d["bin_err_rms"]
    z_flat = z.ravel()
    err_flat = err.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    grid = np.linspace(min(-5, z_flat.min() - 0.5), max(5, z_flat.max() + 0.5), 401)
    ecdf = np.searchsorted(np.sort(z_flat), grid) / len(z_flat)
    ax.plot(grid, ecdf, color=ACCENTS[0], linewidth=1.8, label="empirical")
    ax.plot(grid, norm.cdf(grid), color=FG, linestyle="--", linewidth=0.9, label="N(0, 1)")
    ax.set_xlabel(r"Standardized residual $z = (c_\mathrm{true} - c_\mathrm{pred}) / \hat{\sigma}$")
    ax.set_ylabel("CDF")
    ax.set_title("(a) Empirical CDF of standardized residuals")
    ax.legend()

    ax = axes[1]
    ax.hist(np.clip(np.abs(z_flat), 0, 5), bins=40, color=ACCENTS[1], alpha=0.9,
            edgecolor=PANEL_BG)
    for k_sigma in (1.0, 2.0, 3.0):
        ax.axvline(k_sigma, color=GRID, linestyle=":", linewidth=0.8)
        thr = (1.0 - 2 * norm.sf(k_sigma)) * 100
        cov = float(np.mean(np.abs(z_flat) <= k_sigma)) * 100
        ax.text(k_sigma, ax.get_ylim()[1] * 0.95,
                f"  {k_sigma:.0f}σ\n  emp {cov:.1f}%\n  exp {thr:.1f}%",
                fontsize=7, color="#8b949e", va="top")
    ax.set_xlabel(r"$|z|$"); ax.set_ylabel("Count")
    ax.set_title(f"(b) |z| distribution ({len(z_flat)} GP-predicted coeffs)")

    ax = axes[2]
    ax.loglog(bin_sig, bin_err, "o-", color=ACCENTS[2], markersize=8, linewidth=1.4,
              label="reliability bins")
    lo = float(min(bin_sig.min(), bin_err.min())) * 0.5
    hi = float(max(bin_sig.max(), bin_err.max())) * 2.0
    ax.plot([lo, hi], [lo, hi], color=FG, linestyle="--", linewidth=0.9,
            label="perfectly calibrated")
    ax.set_xlabel("Predicted spread (mean $\\hat{\\sigma}$ in bin)")
    ax.set_ylabel("Observed RMS error in bin")
    ax.set_title("(c) Reliability diagram")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle("PPFFDTD ROM — GP posterior calibration (LOO over r=12 coefficients × 33 folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig05_smolyak(data_path: Path, out_path: Path):
    d = np.load(data_path)
    corr_L1 = d["corr_L1"]; corr_L2 = d["corr_L2"]
    err_L1  = d["err_L1_pct"]; err_L2 = d["err_L2_pct"]
    bands   = d["octave_bands"]
    n1 = int(d["n_train_L1"]); n2 = int(d["n_train_L2"])
    n_unseen = len(corr_L1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    width = 0.4
    x = np.arange(1, n_unseen + 1)
    ax.bar(x - width/2, corr_L1, width, label=f"L1 ({n1} points)",
           color=ACCENTS[2], alpha=0.9, edgecolor=PANEL_BG)
    ax.bar(x + width/2, corr_L2, width, label=f"L2 ({n2} points)",
           color=ACCENTS[0], alpha=0.9, edgecolor=PANEL_BG)
    ax.set_xticks(x); ax.set_ylim(0.97, 1.0005)
    ax.set_xlabel("Unseen point index"); ax.set_ylabel("IR correlation")
    ax.set_title("(a) IR correlation per unseen point")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(bands)); width = 0.4
    m1 = np.array([np.nanmedian(err_L1[:, j]) for j in range(len(bands))])
    m2 = np.array([np.nanmedian(err_L2[:, j]) for j in range(len(bands))])
    ax.bar(x - width/2, m1, width, label=f"L1 ({n1} points)",
           color=ACCENTS[2], alpha=0.9, edgecolor=PANEL_BG)
    ax.bar(x + width/2, m2, width, label=f"L2 ({n2} points)",
           color=ACCENTS[0], alpha=0.9, edgecolor=PANEL_BG)
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in bands])
    ax.set_xlabel("Octave band center (Hz)"); ax.set_ylabel("Median T30 error (%)")
    ax.set_title("(b) Per-band T30 error: L1 vs L2 median")
    ax.legend(fontsize=8)

    ax = axes[2]
    avg_L1 = float(np.nanmedian(err_L1)); avg_L2 = float(np.nanmedian(err_L2))
    max_L1 = float(np.nanmax(err_L1)); max_L2 = float(np.nanmax(err_L2))
    ax.plot([n1, n2], [avg_L1, avg_L2], "o-", color=ACCENTS[0], markersize=9,
            linewidth=1.5, label="median T30 error (all bands)")
    ax.plot([n1, n2], [max_L1, max_L2], "s--", color=ACCENTS[2], markersize=9,
            linewidth=1.5, label="max T30 error (all bands)")
    ax.set_xlabel("Training-set size (# FDTD runs)")
    ax.set_ylabel("T30 error (%)")
    ax.set_title("(c) Convergence with Smolyak level")
    ax.set_xticks([n1, n2]); ax.set_xticklabels([f"L1 ({n1})", f"L2 ({n2})"])
    ax.legend(fontsize=8)

    fig.suptitle("PPFFDTD ROM — Smolyak level convergence on 10 unseen FDTD points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    apply_dark_style()
    figdir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    print(f"Regenerating dark-mode plots into {figdir}")
    fig01_pod(figdir / "01_pod_spectrum_data.npz",   figdir / "01_pod_spectrum_dark.png")
    fig02_loo(figdir / "02_loo_cv_data.npz",         figdir / "02_loo_cv_dark.png")
    fig03_unseen(figdir / "03_unseen_data.npz",      figdir / "03_unseen_dark.png")
    fig04_calibration(figdir / "04_gp_calibration_data.npz", figdir / "04_gp_calibration_dark.png")
    fig05_smolyak(figdir / "05_smolyak_data.npz",    figdir / "05_smolyak_dark.png")
    print("Done.")


if __name__ == "__main__":
    main()
