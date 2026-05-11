"""Step 3 of the PPFFDTD ROM verification suite.

Unseen-point validation. Draw N random parameter points from the training box
(uniform in log-space, away from any Smolyak training node), and for each:

    a) Compute the ground-truth IR with a fresh full FDTD run on the GPU.
    b) Predict the IR with the trained ROM (33-Smolyak GP load).
    c) Compare: IR correlation, peak amplitude ratio, per-band T30 error.

This is the headline number for the talk. The LOO study (Step 2) shows the ROM
predicts held-out training points; this study shows it predicts BRAND NEW
points the basis has never seen.

Output:
    docs/figures/03_unseen.png
    docs/figures/03_unseen_data.npz
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Path setup so we can import PFFDTD + our ROM + iso_metrics
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "pffdtd" / "python"))
sys.path.insert(0, str(REPO / "ppffdtd"))
sys.path.insert(0, str(REPO / "choras_integration" / "pffdtd_method"))

from rom import NonIntrusiveROM, postprocess_ir  # noqa: E402
from iso_metrics import _bandpass, _t30           # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


OCTAVE_BANDS = [125, 250, 500, 1000, 2000]
N_UNSEEN     = 10           # number of unseen draws
SEED         = 20260511     # today's date for reproducibility
LOG_LO       = np.log(0.30) # training box lower bound (per absorption_grid in rom.py)
LOG_HI       = np.log(3.00) # training box upper bound
MIN_LOG_DIST = 0.15         # reject candidate if within this log-distance of any training point


def _per_band_t30(ir, fs, bands=OCTAVE_BANDS):
    return [_t30(_bandpass(ir, fs, fc), fs) for fc in bands]


def draw_unseen_points(training_params, n, seed):
    """Uniform-in-log-space draws inside [0.3, 3.0]^3, rejecting near-training points."""
    rng = np.random.default_rng(seed)
    log_train = np.log(training_params + 1e-30)  # (33, 3)
    chosen = []
    attempts = 0
    while len(chosen) < n and attempts < n * 30:
        log_pt = rng.uniform(LOG_LO + 0.05, LOG_HI - 0.05, size=3)
        dists = np.linalg.norm(log_train - log_pt[None, :], axis=1)
        if dists.min() >= MIN_LOG_DIST:
            chosen.append(np.exp(log_pt))
        attempts += 1
    if len(chosen) < n:
        raise RuntimeError(f"Could only sample {len(chosen)} unseen points; reduce MIN_LOG_DIST or n.")
    return np.asarray(chosen)


def main():
    rom_path  = REPO / "common" / "pffdtd_data" / "rom_v2.npz"
    data_dir  = REPO / "common" / "pffdtd_data"
    pffdtd_py = REPO / "pffdtd" / "python"
    input_json = REPO / "common" / "exampleInput_PFFDTD.json"
    fig_path  = REPO / "docs" / "figures" / "03_unseen.png"
    data_out  = REPO / "docs" / "figures" / "03_unseen_data.npz"

    # ── Load baselines + ROM ──
    with open(input_json, "r") as f:
        cfg = json.load(f)
    baseline_alphas = cfg["absorption_coefficients"]
    print(f"Baseline materials ({len(baseline_alphas)} surfaces):")
    for name, alpha_str in baseline_alphas.items():
        print(f"  {name:>10}: {alpha_str}")

    rom = NonIntrusiveROM(str(data_dir), str(pffdtd_py))
    rom.load(str(rom_path))
    print(f"\nROM loaded: r={rom.Phi.shape[1]}, n_train={rom.training_params.shape[0]}, fs_out={rom.fs_out}")

    # ── Draw unseen points ──
    unseen = draw_unseen_points(rom.training_params, n=N_UNSEEN, seed=SEED)
    print(f"\nDrew {len(unseen)} unseen points (seed={SEED}, min_log_dist={MIN_LOG_DIST}):")
    for i, s in enumerate(unseen):
        print(f"  {i+1:>2}: scales=(floor={s[0]:.3f}, ceil={s[1]:.3f}, walls={s[2]:.3f})")

    # ── Run FDTD ground truth + ROM prediction for each ──
    n = len(unseen)
    fs = int(rom.fs_out)
    n_t_max = rom.Phi.shape[0]

    irs_truth = np.zeros((n, n_t_max), dtype=np.float64)
    irs_rom   = np.zeros((n, n_t_max), dtype=np.float64)
    fdtd_secs = np.zeros(n)
    corrs       = np.zeros(n)
    peak_ratios = np.zeros(n)
    t30_truth = np.full((n, len(OCTAVE_BANDS)), np.nan)
    t30_rom   = np.full((n, len(OCTAVE_BANDS)), np.nan)

    for i, scales in enumerate(unseen):
        # Expand 3D -> Nmat (mirrors rom.py:319-326)
        if rom.dim == 3 and rom.Nmat == 6:
            full_scales = np.array([scales[0], scales[1], scales[2], scales[2], scales[2], scales[2]])
        else:
            full_scales = scales

        # FDTD ground truth
        print(f"\n[{i+1}/{n}] scales=(floor={scales[0]:.3f}, ceil={scales[1]:.3f}, walls={scales[2]:.3f})")
        t0 = time.perf_counter()
        DEF = rom._alpha_to_DEF(full_scales, baseline_alphas)
        # CPU FDTD: training was on CPU (rom_v2 was trained with use_gpu=False); the GPU
        # engine produces meaningfully different IRs from CPU in this version of PFFDTD,
        # so for an apples-to-apples comparison against the trained ROM we must use CPU.
        ir_raw = rom._run_single_fdtd(DEF, use_gpu=False)
        ir_truth = postprocess_ir(ir_raw, rom.fs_native, rom.fmax_grid, rom.fs_out)
        fdtd_secs[i] = time.perf_counter() - t0
        print(f"    FDTD: {fdtd_secs[i]:.1f}s")

        # ROM prediction
        ir_pred, _unc = rom.evaluate(scales)

        # Pad / truncate to common length
        L = min(len(ir_truth), len(ir_pred), n_t_max)
        irs_truth[i, :L] = ir_truth[:L]
        irs_rom[i,   :L] = ir_pred[:L]

        # Scoring
        corrs[i] = np.corrcoef(irs_truth[i, :L], irs_rom[i, :L])[0, 1]
        p_t = float(np.max(np.abs(irs_truth[i, :L])))
        p_r = float(np.max(np.abs(irs_rom[i,   :L])))
        peak_ratios[i] = p_r / p_t if p_t > 0 else np.nan
        t30_truth[i] = _per_band_t30(irs_truth[i, :L], fs)
        t30_rom[i]   = _per_band_t30(irs_rom[i,   :L], fs)

        print(f"    corr={corrs[i]:.5f}  peakRatio={peak_ratios[i]:.3f}")
        for j, fc in enumerate(OCTAVE_BANDS):
            tt, tp = t30_truth[i, j], t30_rom[i, j]
            if tt and tp and np.isfinite(tt) and np.isfinite(tp):
                err_pct = 100 * abs(tp - tt) / tt
                print(f"      {fc:>5} Hz:  truth T30={tt:.3f}s  rom T30={tp:.3f}s  err={err_pct:5.2f}%")

    # ── Aggregate ──
    print("\n=== Unseen-point summary ===")
    print(f"FDTD time: mean {fdtd_secs.mean():.1f}s, total {fdtd_secs.sum():.0f}s")
    print(f"IR correlation:  median={np.median(corrs):.5f}  IQR=[{np.quantile(corrs,0.25):.5f}, {np.quantile(corrs,0.75):.5f}]  min={corrs.min():.5f}")
    print(f"Peak ratio:      median={np.median(peak_ratios):.4f}  IQR=[{np.quantile(peak_ratios,0.25):.4f}, {np.quantile(peak_ratios,0.75):.4f}]")
    t30_err = np.where(
        (t30_truth > 0) & np.isfinite(t30_truth) & np.isfinite(t30_rom),
        100.0 * np.abs(t30_rom - t30_truth) / np.maximum(t30_truth, 1e-9),
        np.nan,
    )
    print("Per-band T30 error % (across unseen points):")
    for j, fc in enumerate(OCTAVE_BANDS):
        col = t30_err[:, j]
        nv = int(np.sum(np.isfinite(col)))
        m  = np.nanmedian(col); q1 = np.nanquantile(col, 0.25); q3 = np.nanquantile(col, 0.75); mx = np.nanmax(col)
        print(f"  {fc:>5} Hz:  median={m:5.2f}%  IQR=[{q1:5.2f}, {q3:5.2f}]  max={mx:5.2f}%  (valid {nv}/{n})")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.bar(range(1, n + 1), corrs, color="C0", alpha=0.85)
    ax.set_xticks(range(1, n + 1))
    ax.set_ylabel("IR correlation")
    ax.set_ylim(0.99, 1.0001)
    ax.set_xlabel("Unseen point index")
    ax.set_title("(a) ROM vs FDTD ground-truth IR correlation")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    data = [t30_err[:, j][np.isfinite(t30_err[:, j])] for j in range(len(OCTAVE_BANDS))]
    ax.boxplot(data, tick_labels=[str(f) for f in OCTAVE_BANDS], showmeans=True)
    ax.set_xlabel("Octave band center (Hz)")
    ax.set_ylabel("T30 error (%)")
    ax.set_title("(b) Per-band T30 error vs FDTD")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for j, fc in enumerate(OCTAVE_BANDS):
        valid = np.isfinite(t30_truth[:, j]) & np.isfinite(t30_rom[:, j])
        if valid.any():
            ax.scatter(t30_truth[valid, j], t30_rom[valid, j], label=f"{fc} Hz", alpha=0.75, s=24)
    all_vals = np.concatenate([t30_truth.ravel(), t30_rom.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals):
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("FDTD truth T30 (s)")
    ax.set_ylabel("ROM-predicted T30 (s)")
    ax.set_title("(c) Per-band T30 scatter (across all unseen points)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # Show one truth vs predicted IR (the worst-correlation case)
    worst = int(np.argmin(corrs))
    t = np.arange(n_t_max) / fs
    ax.plot(t[:int(0.4 * fs)], irs_truth[worst, :int(0.4 * fs)], color="k",
            linewidth=0.6, label="FDTD truth")
    ax.plot(t[:int(0.4 * fs)], irs_rom[worst, :int(0.4 * fs)], color="C3",
            linewidth=0.6, alpha=0.7, label="ROM predicted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure")
    s = unseen[worst]
    ax.set_title(f"(d) Worst-corr case: idx {worst+1}, "
                 f"scales=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f}), corr={corrs[worst]:.5f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"PPFFDTD ROM — unseen-point validation ({n} fresh FDTD-on-GPU draws)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    np.savez_compressed(data_out,
                        unseen_params=unseen,
                        irs_truth=irs_truth.astype(np.float32),
                        irs_rom=irs_rom.astype(np.float32),
                        corrs=corrs,
                        peak_ratios=peak_ratios,
                        t30_truth=t30_truth,
                        t30_rom=t30_rom,
                        t30_err_pct=t30_err,
                        fdtd_secs=fdtd_secs,
                        octave_bands=np.array(OCTAVE_BANDS),
                        seed=SEED)
    print(f"Data saved:   {data_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
