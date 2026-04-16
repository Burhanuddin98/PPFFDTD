"""Validate PPFFDTD against Hamilton's PFFDTD on BRAS S09."""
import sys
import numpy as np

sys.path.insert(0, '.')
from ppffdtd.pffdtd_loader import load_from_pffdtd

DATA_DIR = r'C:\Users\bsaka\AppData\Local\Temp\pffdtd_bras_8fymsxx6\sim_out_v4'

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

print("=" * 60)
print("PPFFDTD vs PFFDTD — BRAS S09")
print("=" * 60)

# Load
engine, Nt = load_from_pffdtd(DATA_DIR)

# Progress callback
def progress(n, eng):
    print(f"  step {n}/{Nt} ({n/Nt*100:.0f}%)", flush=True)

# Run
print(f"\nRunning PPFFDTD ({Nt} steps)...")
u_out = engine.run(Nt, callback=progress)

# Compare with PFFDTD output
import h5py
with h5py.File(DATA_DIR + '/sim_outs.h5', 'r') as f:
    u_out_ref = f['u_out'][...]

print(f"\n{'Receiver':<12} {'T30_PPFFDTD':<14} {'T30_PFFDTD':<14} {'Error':<10}")
print("-" * 50)
for r in range(min(u_out.shape[0], u_out_ref.shape[0])):
    ir_pp = u_out[engine.out_reorder[r], :]
    ir_ref = u_out_ref[r, :]
    t30_pp = t30(ir_pp, engine.fs)
    t30_ref = t30(ir_ref, engine.fs)
    err = abs(t30_pp - t30_ref) / t30_ref * 100
    corr = np.corrcoef(ir_pp[:5000], ir_ref[:5000])[0, 1]
    print(f"  {r:<10} {t30_pp:<14.3f} {t30_ref:<14.3f} {err:<10.1f}%  corr={corr:.4f}")

print(f"\nMeasured BRAS S09 LS1-MP1: T30 = 1.677s")
