"""Step 1 of the PPFFDTD ROM verification suite.

Plots the POD singular-value spectrum of the trained ROM. Two panels:

    (a) Singular values sigma_k (semilog-y) vs basis index k
        - Shows the geometric decay characteristic of low-rank physical systems.
        - Reveals if our r=12 truncation captures the "elbow".

    (b) Cumulative energy fraction E(r) = (sum_{k=1..r} sigma_k^2) / (sum_k sigma_k^2)
        - Justifies r=12 quantitatively: the threshold typically aimed for is 0.9999.
        - Reports the actual energy retained at r=12 and the smallest r that
          captures 0.9999.

Input:  ../common/pffdtd_data/rom_v2.npz  (trained ROM artifact)
Output: ../docs/figures/01_pod_spectrum.png
        ../docs/figures/01_pod_spectrum_data.npz  (raw singular values for citation)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).resolve().parent
    rom_path = here.parent / "common" / "pffdtd_data" / "rom_v2.npz"
    fig_path = here.parent / "docs" / "figures" / "01_pod_spectrum.png"
    data_path = here.parent / "docs" / "figures" / "01_pod_spectrum_data.npz"

    print(f"Loading {rom_path}")
    d = np.load(rom_path, allow_pickle=True)

    training_irs = d["training_irs"]   # (33, 48001)
    ir_mean      = d["ir_mean"]        # (48001,)
    r_saved      = int(d["Phi"].shape[1])

    n_train, n_t = training_irs.shape
    print(f"training_irs: shape={training_irs.shape}, dtype={training_irs.dtype}")
    print(f"ir_mean:      shape={ir_mean.shape}")
    print(f"r (saved):    {r_saved}")

    # Center the snapshots (rom.py uses X = training_irs.T - ir_mean[:, None])
    X = training_irs.T - ir_mean[:, None]   # (n_t, n_train) = (48001, 33)

    # Full SVD over the training set
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    print(f"singular values: {len(S)} total, min={S.min():.4e}, max={S.max():.4e}")

    # Energy / cumulative energy
    energy = S ** 2
    total_energy = energy.sum()
    cum_energy = np.cumsum(energy) / total_energy

    # Where does cum_energy first reach 0.9999?
    thresholds = (0.99, 0.999, 0.9999, 0.99999)
    r_at = {th: int(np.searchsorted(cum_energy, th) + 1) for th in thresholds}
    print("\nr required to reach cumulative energy threshold:")
    for th, r in r_at.items():
        print(f"  {th:8.5f} -> r={r}")
    print(f"\nAt r={r_saved} (the value stored in rom_v2.npz):")
    print(f"  cum_energy = {cum_energy[r_saved - 1]:.7f}")

    # ── Plot ──
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))

    k = np.arange(1, len(S) + 1)
    ax_a.semilogy(k, S, "o-", color="C0", markersize=5, linewidth=1.0)
    ax_a.axvline(r_saved, color="C3", linestyle="--", linewidth=1.0,
                 label=f"r = {r_saved} (saved)")
    ax_a.set_xlabel("Basis index k")
    ax_a.set_ylabel(r"Singular value $\sigma_k$")
    ax_a.set_title("(a) POD singular spectrum")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(loc="upper right")

    ax_b.plot(k, cum_energy, "o-", color="C2", markersize=5, linewidth=1.0)
    for th, r in r_at.items():
        if r <= len(S):
            ax_b.axhline(th, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
            ax_b.text(len(S) * 0.98, th, f"  {th}", ha="right", va="bottom",
                      fontsize=7, color="grey")
    ax_b.axvline(r_saved, color="C3", linestyle="--", linewidth=1.0,
                 label=f"r = {r_saved}  (cum E = {cum_energy[r_saved - 1]:.6f})")
    ax_b.set_xlabel("Basis index k")
    ax_b.set_ylabel("Cumulative energy fraction")
    ax_b.set_title("(b) Cumulative energy E(r)")
    ax_b.set_ylim(0.0, 1.005)
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(loc="lower right")

    fig.suptitle("PPFFDTD ROM — POD basis convergence on the 33-point Smolyak training set",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    np.savez_compressed(data_path,
                        singular_values=S,
                        energy_fraction=cum_energy,
                        r_saved=r_saved,
                        thresholds=np.array(thresholds),
                        r_at_threshold=np.array(list(r_at.values())))
    print(f"Data saved:   {data_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
