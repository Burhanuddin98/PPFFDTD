"""
Dark-mode neon FDTD visualization with dense early snapshots.

Runs a fresh FDTD on the MeasurementRoom (35s) collecting snapshots
every 5 steps for the first 500 steps (catch wavefront), then every
100 steps for the rest (catch decay).
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
import h5py
import time as _time

sys.path.insert(0, 'pffdtd/python')

plt.style.use('dark_background')

# Neon colormap
neon_cmap = LinearSegmentedColormap.from_list('neon', [
    (0.00, '#00EEFF'),
    (0.25, '#002266'),
    (0.50, '#080808'),
    (0.75, '#660022'),
    (1.00, '#FF00FF'),
])

DATA_DIR = Path('common/pffdtd_data')
OUT_DIR = Path('vis')
OUT_DIR.mkdir(exist_ok=True)

# ---- Run FDTD with dense snapshot collection ----
print("Running FDTD with snapshot collection...")
from fdtd.sim_fdtd import SimEngine

engine = SimEngine(str(DATA_DIR), energy_on=False, nthreads=12)
engine.load_h5_data()
engine.setup_mask()
engine.allocate_mem()
engine.set_coeffs()
engine.checks()

Nx, Ny, Nz = engine.Nx, engine.Ny, engine.Nz
Nt = engine.Nt
NyNz = Ny * Nz
N = Nx * Ny * Nz

# Source/receiver
src = engine.in_ixyz[0]
sx, sy, sz = src // NyNz, (src % NyNz) // Nz, src % Nz
rec = engine.out_ixyz[0]
rx, ry, rz = rec // NyNz, (rec % NyNz) // Nz, rec % Nz

xv, yv, zv = engine.xv, engine.yv, engine.zv
bn_mask_3d = engine.bn_mask

print(f"Grid: {Nx}x{Ny}x{Nz}, Nt={Nt}")
print(f"Source: ({xv[sx]:.2f}, {yv[sy]:.2f}, {zv[sz]:.2f})")
print(f"Receiver: ({xv[rx]:.2f}, {yv[ry]:.2f}, {zv[rz]:.2f})")

# Collect snapshots: dense early, sparse late
snap_steps = list(range(1, 500, 5))  # every 5 steps for first 500
snap_steps += list(range(500, Nt, 200))  # every 200 after
snap_steps = [s for s in snap_steps if s < Nt]
n_total_snaps = len(snap_steps)
snap_set = set(snap_steps)

# Store only the 3 slices (not full field — saves memory)
slices_xy = np.zeros((n_total_snaps, Nx, Ny))
slices_xz = np.zeros((n_total_snaps, Nx, Nz))
slices_yz = np.zeros((n_total_snaps, Ny, Nz))
snap_idx = 0

t0 = _time.perf_counter()

# Time-step with snapshot collection
from fdtd.sim_fdtd import (nb_flip_halos, nb_stencil_air_cart,
                            nb_stencil_bn_cart, nb_leapfrog_update,
                            nb_update_bnl_fd, nb_update_abc, nb_save_bn)

for n in range(Nt):
    nb_save_bn(engine.u0, engine.u2ba, engine.bna_ixyz)
    nb_flip_halos(engine.u1)
    nb_stencil_air_cart(engine.Lu1, engine.u1, engine.bn_mask)
    nb_stencil_bn_cart(engine.Lu1, engine.u1, engine.bn_ixyz, engine.adj_bn)
    nb_save_bn(engine.u0, engine.u2b, engine.bnl_ixyz)
    nb_leapfrog_update(engine.u0, engine.u1, engine.Lu1, engine.l2)
    nb_update_bnl_fd(engine.u0, engine.u2b, engine.l,
                     engine.bnl_ixyz, engine.ssaf_bnl,
                     engine.vh0, engine.vh1, engine.gh1,
                     engine.mat_bnl, engine.mat_coeffs_struct)
    nb_update_abc(engine.u0, engine.u2ba, engine.l,
                  engine.bna_ixyz, engine.Q_bna)
    engine.u0.flat[engine.in_ixyz] += engine.in_sigs[:, n]

    if n in snap_set and snap_idx < n_total_snaps:
        slices_xy[snap_idx] = engine.u0[:, :, sz]
        slices_xz[snap_idx] = engine.u0[:, sy, :]
        slices_yz[snap_idx] = engine.u0[sx, :, :]
        snap_idx += 1

    engine.u0, engine.u1 = engine.u1, engine.u0
    engine.vh0, engine.vh1 = engine.vh1, engine.vh0

    if (n + 1) % max(1, Nt // 5) == 0:
        elapsed = _time.perf_counter() - t0
        print(f"  step {n+1}/{Nt} ({elapsed:.0f}s)")

elapsed = _time.perf_counter() - t0
print(f"FDTD done: {elapsed:.0f}s, {snap_idx} snapshots collected")

# ---- Select best frames ----
# Early: wavefront expanding (first ~20 dense snapshots)
# Mid: first reflections (~30-60ms)
# Late: reverberant decay
Ts = engine.Ts
frame_picks = [0, 2, 5, 10, 20, 40, 60, 80]  # from dense early
# Add some late frames
for target_ms in [100, 200, 500, 1000, 2000, 3000]:
    target_step = int(target_ms * 1e-3 / Ts)
    best = min(range(len(snap_steps)), key=lambda i: abs(snap_steps[i] - target_step))
    if best not in frame_picks:
        frame_picks.append(best)
frame_picks = sorted(set(i for i in frame_picks if i < snap_idx))

print(f"Rendering {len(frame_picks)} frames...")

for fi, si in enumerate(frame_picks):
    step_n = snap_steps[si]
    t_ms = step_n * Ts * 1000

    xy = slices_xy[si]
    xz = slices_xz[si]
    yz = slices_yz[si]

    # Per-frame auto-range (shows structure at all times)
    vmax_frame = max(np.max(np.abs(xy)), np.max(np.abs(xz)),
                     np.max(np.abs(yz)), 1e-10)

    fig = plt.figure(figsize=(22, 7), facecolor='#080808')
    gs = GridSpec(1, 4, width_ratios=[1.2, 1.2, 0.8, 0.05],
                  wspace=0.15, figure=fig)
    fig.patch.set_facecolor('#080808')

    # XY
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(xy.T, origin='lower', cmap=neon_cmap,
                      extent=[xv[0], xv[-1], yv[0], yv[-1]],
                      vmin=-vmax_frame, vmax=vmax_frame,
                      aspect='equal', interpolation='bilinear')
    ax0.contour(xv, yv, bn_mask_3d[:, :, sz].T.astype(float),
                levels=[0.5], colors='#00FF88', linewidths=1.0, alpha=0.7)
    ax0.plot(xv[sx], yv[sy], '*', color='#FF3333', markersize=16,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax0.plot(xv[rx], yv[ry], '^', color='#33FF33', markersize=13,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax0.set_xlabel('x (m)', color='#999999')
    ax0.set_ylabel('y (m)', color='#999999')
    ax0.set_title('XY plane (top-down)', color='#00CCFF', fontsize=11)
    ax0.tick_params(colors='#555555')
    for spine in ax0.spines.values():
        spine.set_color('#333333')

    # XZ
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(xz.T, origin='lower', cmap=neon_cmap,
               extent=[xv[0], xv[-1], zv[0], zv[-1]],
               vmin=-vmax_frame, vmax=vmax_frame,
               aspect='equal', interpolation='bilinear')
    ax1.contour(xv, zv, bn_mask_3d[:, sy, :].T.astype(float),
                levels=[0.5], colors='#00FF88', linewidths=1.0, alpha=0.7)
    ax1.plot(xv[sx], zv[sz], '*', color='#FF3333', markersize=16,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax1.plot(xv[rx], zv[rz], '^', color='#33FF33', markersize=13,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax1.set_xlabel('x (m)', color='#999999')
    ax1.set_ylabel('z (m)', color='#999999')
    ax1.set_title('XZ plane (side)', color='#00CCFF', fontsize=11)
    ax1.tick_params(colors='#555555')
    for spine in ax1.spines.values():
        spine.set_color('#333333')

    # YZ
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(yz.T, origin='lower', cmap=neon_cmap,
               extent=[yv[0], yv[-1], zv[0], zv[-1]],
               vmin=-vmax_frame, vmax=vmax_frame,
               aspect='equal', interpolation='bilinear')
    ax2.contour(yv, zv, bn_mask_3d[sx, :, :].T.astype(float),
                levels=[0.5], colors='#00FF88', linewidths=1.0, alpha=0.7)
    ax2.plot(yv[sy], zv[sz], '*', color='#FF3333', markersize=16,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax2.plot(yv[ry], zv[rz], '^', color='#33FF33', markersize=13,
             markeredgecolor='white', markeredgewidth=0.8, zorder=10)
    ax2.set_xlabel('y (m)', color='#999999')
    ax2.set_ylabel('z (m)', color='#999999')
    ax2.set_title('YZ plane (front)', color='#00CCFF', fontsize=11)
    ax2.tick_params(colors='#555555')
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    # Colorbar
    cax = fig.add_subplot(gs[3])
    cb = plt.colorbar(im0, cax=cax)
    cb.set_label('Pressure', color='#999999', fontsize=10)
    cb.ax.tick_params(colors='#555555')
    cb.outline.set_color('#333333')

    # Title
    fig.suptitle(
        f'PPFFDTD  |  BRAS S09 Seminar Room  |  t = {t_ms:.1f} ms  |  '
        f'peak = {vmax_frame:.2e}',
        color='#FF00FF', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.01, 0.01,
             'Red star = Source  |  Green triangle = Receiver  |  '
             'Green outline = Room boundary',
             color='#666666', fontsize=8)

    path = OUT_DIR / f'neon_{fi:03d}_t{int(t_ms):05d}ms.png'
    fig.savefig(path, dpi=150, facecolor='#080808',
                bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  [{fi+1}/{len(frame_picks)}] t={t_ms:.1f}ms, peak={vmax_frame:.2e}")

# GIF
try:
    from PIL import Image
    frames = [Image.open(OUT_DIR / f'neon_{fi:03d}_t{int(snap_steps[si]*Ts*1000):05d}ms.png')
              for fi, si in enumerate(frame_picks)]
    gif = OUT_DIR / 'ppffdtd_neon.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=350, loop=0)
    print(f"\nGIF saved: {gif}")
except ImportError:
    pass

print(f"Done! Frames at {OUT_DIR}/")
