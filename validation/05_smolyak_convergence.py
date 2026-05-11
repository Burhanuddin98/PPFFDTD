"""Step 5 of the PPFFDTD ROM verification suite.

Smolyak sparse-grid convergence. Train a SECOND ROM at level 1 (which gives 7
training points in 3D, vs level 2's 33), and evaluate both ROMs at the same 10
unseen points used in Step 3. The L1 ROM has ~5x less training data; comparing
its accuracy with the L2 ROM gives a concrete convergence-with-training-budget
slope.

The 10 ground-truth FDTD IRs were saved in 03_unseen_data.npz, so we don't need
to re-run them.

Output:
    docs/figures/05_smolyak.png
    docs/figures/05_smolyak_data.npz
    common/pffdtd_data/rom_L1.npz   (new L1 ROM artifact — committed alongside L2)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "pffdtd" / "python"))
sys.path.insert(0, str(REPO / "ppffdtd"))
sys.path.insert(0, str(REPO / "choras_integration" / "pffdtd_method"))

from rom import NonIntrusiveROM, postprocess_ir  # noqa: E402
from iso_metrics import _bandpass, _t30           # noqa: E402

OCTAVE_BANDS = [125, 250, 500, 1000, 2000]


def _per_band_t30(ir, fs, bands=OCTAVE_BANDS):
    return [_t30(_bandpass(ir, fs, fc), fs) for fc in bands]


def main():
    data_dir   = REPO / "common" / "pffdtd_data"
    pffdtd_py  = REPO / "pffdtd" / "python"
    rom_L2_path = data_dir / "rom_v2.npz"
    rom_L1_path = data_dir / "rom_L1.npz"
    unseen_path = REPO / "docs" / "figures" / "03_unseen_data.npz"
    fig_path    = REPO / "docs" / "figures" / "05_smolyak.png"
    data_out    = REPO / "docs" / "figures" / "05_smolyak_data.npz"
    input_json  = REPO / "common" / "exampleInput_PFFDTD.json"

    with open(input_json, "r") as f:
        cfg = json.load(f)
    baseline_alphas = cfg["absorption_coefficients"]

    # Train L1 ROM (only if not already cached)
    if rom_L1_path.exists():
        print(f"L1 ROM already cached at {rom_L1_path}, loading")
        rom_L1 = NonIntrusiveROM(str(data_dir), str(pffdtd_py))
        rom_L1.load(str(rom_L1_path))
    else:
        print("Training L1 ROM (7 FDTD runs at level-1 Smolyak grid, CPU)...")
        t0 = time.perf_counter()
        rom_L1 = NonIntrusiveROM(str(data_dir), str(pffdtd_py))
        rom_L1.train(baseline_alphas, dim=3, level=1, use_gpu=False)
        rom_L1.save(str(rom_L1_path))
        print(f"  L1 training: {time.perf_counter()-t0:.0f}s total, {rom_L1.n_train} points")

    # Load L2 ROM
    rom_L2 = NonIntrusiveROM(str(data_dir), str(pffdtd_py))
    rom_L2.load(str(rom_L2_path))

    print(f"\nL1 ROM: n_train={rom_L1.n_train}, r={rom_L1.Phi.shape[1]}")
    print(f"L2 ROM: n_train={rom_L2.n_train}, r={rom_L2.Phi.shape[1]}")

    # Reuse step 3's unseen points + ground-truth IRs
    d = np.load(unseen_path, allow_pickle=True)
    unseen_params = d["unseen_params"]
    irs_truth     = d["irs_truth"].astype(np.float64)
    t30_truth_step3 = d["t30_truth"]
    fs = 48000
    n_unseen = unseen_params.shape[0]
    print(f"\nReusing {n_unseen} unseen points from step 3")

    # Evaluate both ROMs at each unseen point
    irs_L1 = np.zeros_like(irs_truth)
    irs_L2 = np.zeros_like(irs_truth)
    t30_L1 = np.full((n_unseen, len(OCTAVE_BANDS)), np.nan)
    t30_L2 = np.full((n_unseen, len(OCTAVE_BANDS)), np.nan)
    corr_L1 = np.zeros(n_unseen)
    corr_L2 = np.zeros(n_unseen)
    for i, scales in enumerate(unseen_params):
        ir1, _ = rom_L1.evaluate(scales)
        ir2, _ = rom_L2.evaluate(scales)
        L = min(len(ir1), len(ir2), irs_truth.shape[1])
        irs_L1[i, :L] = ir1[:L]
        irs_L2[i, :L] = ir2[:L]
        corr_L1[i] = np.corrcoef(irs_truth[i, :L], ir1[:L])[0, 1]
        corr_L2[i] = np.corrcoef(irs_truth[i, :L], ir2[:L])[0, 1]
        t30_L1[i] = _per_band_t30(ir1[:L], fs)
        t30_L2[i] = _per_band_t30(ir2[:L], fs)

    # T30 errors
    def _err_pct(t_pred, t_true):
        return np.where(
            (t_true > 0) & np.isfinite(t_true) & np.isfinite(t_pred),
            100.0 * np.abs(t_pred - t_true) / np.maximum(t_true, 1e-9),
            np.nan,
        )
    err_L1 = _err_pct(t30_L1, t30_truth_step3)
    err_L2 = _err_pct(t30_L2, t30_truth_step3)

    print("\n=== Smolyak convergence summary ===")
    print(f"{'':>10}   {'L1 (7 pts)':>14}   {'L2 (33 pts)':>14}")
    print(f"{'IR corr':>10}   median={np.median(corr_L1):.5f}   median={np.median(corr_L2):.5f}")
    print(f"{'IR corr':>10}   min   ={corr_L1.min():.5f}   min   ={corr_L2.min():.5f}")
    print("Per-band T30 error % (median across 10 unseen):")
    for j, fc in enumerate(OCTAVE_BANDS):
        m1 = float(np.nanmedian(err_L1[:, j]))
        m2 = float(np.nanmedian(err_L2[:, j]))
        x1 = float(np.nanmax(err_L1[:, j]))
        x2 = float(np.nanmax(err_L2[:, j]))
        print(f"  {fc:>5} Hz:  L1 median={m1:5.2f}% max={x1:5.2f}%  |  "
              f"L2 median={m2:5.2f}% max={x2:5.2f}%  |  L2 / L1 ratio = {m2/m1 if m1>0 else float('nan'):.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) IR correlation per fold
    ax = axes[0]
    width = 0.4
    x = np.arange(1, n_unseen + 1)
    ax.bar(x - width/2, corr_L1, width, label=f"L1 ({rom_L1.n_train} points)", color="C1", alpha=0.85)
    ax.bar(x + width/2, corr_L2, width, label="L2 (33 points)", color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xlabel("Unseen point index")
    ax.set_ylabel("IR correlation")
    ax.set_ylim(0.97, 1.0005)
    ax.set_title("(a) ROM vs FDTD IR correlation per unseen point")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Per-band T30 error medians
    ax = axes[1]
    x = np.arange(len(OCTAVE_BANDS))
    width = 0.4
    med_L1 = np.array([np.nanmedian(err_L1[:, j]) for j in range(len(OCTAVE_BANDS))])
    med_L2 = np.array([np.nanmedian(err_L2[:, j]) for j in range(len(OCTAVE_BANDS))])
    ax.bar(x - width/2, med_L1, width, label=f"L1 ({rom_L1.n_train} points)", color="C1", alpha=0.85)
    ax.bar(x + width/2, med_L2, width, label="L2 (33 points)", color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in OCTAVE_BANDS])
    ax.set_xlabel("Octave band center (Hz)")
    ax.set_ylabel("Median T30 error (%)")
    ax.set_title("(b) Per-band T30 error: L1 vs L2 median")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Overall error as function of training-set size
    ax = axes[2]
    # Average per-band median across bands
    avg_L1 = float(np.nanmedian(err_L1))
    avg_L2 = float(np.nanmedian(err_L2))
    max_L1 = float(np.nanmax(err_L1))
    max_L2 = float(np.nanmax(err_L2))
    ax.plot([rom_L1.n_train, rom_L2.n_train], [avg_L1, avg_L2], "o-",
            color="C0", markersize=8, label="median T30 error (all bands)")
    ax.plot([rom_L1.n_train, rom_L2.n_train], [max_L1, max_L2], "s--",
            color="C3", markersize=8, label="max T30 error (all bands)")
    ax.set_xlabel("Training-set size (# FDTD runs)")
    ax.set_ylabel("T30 error (%)")
    ax.set_title("(c) Convergence with Smolyak level")
    ax.set_xticks([rom_L1.n_train, rom_L2.n_train])
    ax.set_xticklabels([f"L1 ({rom_L1.n_train})", f"L2 ({rom_L2.n_train})"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("PPFFDTD ROM — Smolyak level convergence on 10 unseen FDTD points",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    np.savez_compressed(data_out,
                        unseen_params=unseen_params,
                        n_train_L1=rom_L1.n_train, n_train_L2=rom_L2.n_train,
                        corr_L1=corr_L1, corr_L2=corr_L2,
                        t30_L1=t30_L1, t30_L2=t30_L2,
                        err_L1_pct=err_L1, err_L2_pct=err_L2,
                        octave_bands=np.array(OCTAVE_BANDS))
    print(f"Data saved:   {data_out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
