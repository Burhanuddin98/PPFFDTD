"""
3D FDTD visualization: black bg, white wireframe, transparent neon waves.

Only high-pressure regions glow. Near-zero pressure = fully transparent.
Uses the actual CHORAS MeasurementRoom non-rectangular geometry.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import h5py
import time as _time

sys.path.insert(0, 'pffdtd/python')
plt.style.use('dark_background')

# Neon colormap
neon_cmap = LinearSegmentedColormap.from_list('neon', [
    (0.00, '#00EEFF'),
    (0.30, '#003366'),
    (0.50, '#000000'),
    (0.70, '#660033'),
    (1.00, '#FF00FF'),
])

DATA_DIR = Path('common/pffdtd_data')
OUT_DIR = Path('vis3d')
OUT_DIR.mkdir(exist_ok=True)

# ---- Room geometry (actual vertices from .geo file) ----
# Non-rectangular floor plan
floor_pts = np.array([
    [0.0, 5.1],   # P1
    [6.21, 4.0],  # P2
    [5.52, 0.0],  # P3
    [0.0, 0.0],   # P4
])
z_floor, z_ceil = 0.0, 3.3

# ---- Load grid ----
with h5py.File(DATA_DIR / 'vox_out.h5', 'r') as f:
    Nx, Ny, Nz = int(f['Nx'][()]), int(f['Ny'][()]), int(f['Nz'][()])
    xv, yv, zv = f['xv'][()], f['yv'][()], f['zv'][()]

with h5py.File(DATA_DIR / 'comms_out.h5', 'r') as f:
    in_ixyz = f['in_ixyz'][...]
    out_ixyz = f['out_ixyz'][...]

with h5py.File(DATA_DIR / 'sim_consts.h5', 'r') as f:
    Ts = float(f['Ts'][()])

NyNz = Ny * Nz
src = in_ixyz[0]
sx, sy, sz = src // NyNz, (src % NyNz) // Nz, src % Nz
rec = out_ixyz[0]
rx, ry, rz = rec // NyNz, (rec % NyNz) // Nz, rec % Nz

bn_mask = np.zeros(Nx*Ny*Nz, dtype=bool)
with h5py.File(DATA_DIR / 'vox_out.h5', 'r') as f:
    bn_mask[f['bn_ixyz'][...]] = True
bn_mask_3d = bn_mask.reshape(Nx, Ny, Nz)

print(f"Grid: {Nx}x{Ny}x{Nz}, Source: ({xv[sx]:.1f},{yv[sy]:.1f},{zv[sz]:.1f})")

# ---- Run FDTD ----
print("Running FDTD...")
from fdtd.sim_fdtd import SimEngine
from fdtd.sim_fdtd import (nb_flip_halos, nb_stencil_air_cart,
                            nb_stencil_bn_cart, nb_leapfrog_update,
                            nb_update_bnl_fd, nb_update_abc, nb_save_bn)

engine = SimEngine(str(DATA_DIR), energy_on=False, nthreads=12)
engine.load_h5_data()
engine.setup_mask()
engine.allocate_mem()
engine.set_coeffs()
engine.checks()
Nt = engine.Nt

snap_steps = list(range(2, 200, 2)) + list(range(200, 800, 10)) + list(range(800, Nt, 100))
snap_steps = sorted(set(s for s in snap_steps if s < Nt))
snap_set = set(snap_steps)

slices_xy, slices_xz, slices_yz, snap_times = [], [], [], []
t0 = _time.perf_counter()

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

    if n in snap_set:
        slices_xy.append(engine.u0[:, :, sz].copy())
        slices_xz.append(engine.u0[:, sy, :].copy())
        slices_yz.append(engine.u0[sx, :, :].copy())
        snap_times.append(n * Ts * 1000)

    engine.u0, engine.u1 = engine.u1, engine.u0
    engine.vh0, engine.vh1 = engine.vh1, engine.vh0

    if (n + 1) % (Nt // 5) == 0:
        print(f"  step {n+1}/{Nt} ({_time.perf_counter()-t0:.0f}s)")

print(f"FDTD: {_time.perf_counter()-t0:.0f}s, {len(slices_xy)} snapshots")

# ---- Pick frames ----
# Global vmax from ~10ms (after spike, wavefront is dominant feature)
# Early spike saturates, wavefront rings glow bright, late field fades to black
si_ref = min(30, len(slices_xy)-1)  # ~30th dense snapshot = ~6ms
global_vmax = max(np.max(np.abs(slices_xy[si_ref])),
                  np.max(np.abs(slices_xz[si_ref])),
                  np.max(np.abs(slices_yz[si_ref])), 1e-10)
# Clamp: don't let it be too small (ensures late field fades out properly)
global_vmax = max(global_vmax, 0.01)
print(f"Global vmax: {global_vmax:.4e} (fixed for all frames)")

n_frames = 45
picks = np.unique(np.linspace(0, len(slices_xy)-1, n_frames).astype(int))
print(f"Rendering {len(picks)} frames...")


def draw_room_wireframe(ax):
    """Draw the actual non-rectangular room wireframe in white."""
    # Bottom edges
    for i in range(len(floor_pts)):
        j = (i + 1) % len(floor_pts)
        ax.plot3D([floor_pts[i,0], floor_pts[j,0]],
                  [floor_pts[i,1], floor_pts[j,1]],
                  [z_floor, z_floor], color='white', lw=1.0, alpha=0.7)
    # Top edges
    for i in range(len(floor_pts)):
        j = (i + 1) % len(floor_pts)
        ax.plot3D([floor_pts[i,0], floor_pts[j,0]],
                  [floor_pts[i,1], floor_pts[j,1]],
                  [z_ceil, z_ceil], color='white', lw=1.0, alpha=0.7)
    # Verticals
    for pt in floor_pts:
        ax.plot3D([pt[0], pt[0]], [pt[1], pt[1]],
                  [z_floor, z_ceil], color='white', lw=1.0, alpha=0.7)


def pressure_to_rgba(data, vmax, threshold=0.02):
    """Convert pressure to RGBA. Below threshold = fully transparent."""
    norm = data / vmax  # [-1, 1]
    colors = neon_cmap((norm + 1) / 2)

    # Alpha: zero below threshold, ramps up with sqrt for brightness
    intensity = np.abs(norm)
    alpha = np.where(intensity < threshold, 0.0,
                     np.sqrt((intensity - threshold) / (1 - threshold)))
    alpha = np.clip(alpha, 0.0, 0.9)
    colors[:, :, 3] = alpha
    return colors


for fi, si in enumerate(picks):
    xy = slices_xy[si]
    xz = slices_xz[si]
    yz = slices_yz[si]
    t_ms = snap_times[si]

    vmax = global_vmax  # fixed — field fades to transparent as energy decays

    fig = plt.figure(figsize=(11, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    draw_room_wireframe(ax)

    # XY horizontal slice
    X, Y = np.meshgrid(xv, yv, indexing='ij')
    Z_h = np.full_like(X, zv[sz])
    ax.plot_surface(X, Y, Z_h, facecolors=pressure_to_rgba(xy, vmax),
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # XZ vertical slice
    X2, Z2 = np.meshgrid(xv, zv, indexing='ij')
    Y_v = np.full_like(X2, yv[sy])
    ax.plot_surface(X2, Y_v, Z2, facecolors=pressure_to_rgba(xz, vmax),
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # YZ vertical slice
    Y3, Z3 = np.meshgrid(yv, zv, indexing='ij')
    X_v = np.full_like(Y3, xv[sx])
    ax.plot_surface(X_v, Y3, Z3, facecolors=pressure_to_rgba(yz, vmax),
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # Source (red star)
    ax.scatter([xv[sx]], [yv[sy]], [zv[sz]], color='#FF2222', s=150,
               marker='*', edgecolors='white', linewidth=0.5,
               zorder=10, depthshade=False)
    # Receiver (green triangle)
    ax.scatter([xv[rx]], [yv[ry]], [zv[rz]], color='#22FF22', s=100,
               marker='^', edgecolors='white', linewidth=0.5,
               zorder=10, depthshade=False)

    # Clean axes
    pad = 0.5
    ax.set_xlim(-pad, 6.5 + pad)
    ax.set_ylim(-pad, 5.5 + pad)
    ax.set_zlim(-0.2, 3.6)
    ax.set_xlabel('x', color='#444444', fontsize=7, labelpad=-4)
    ax.set_ylabel('y', color='#444444', fontsize=7, labelpad=-4)
    ax.set_zlabel('z', color='#444444', fontsize=7, labelpad=-4)
    ax.tick_params(colors='#222222', labelsize=5, pad=-2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1a1a1a')
    ax.yaxis.pane.set_edgecolor('#1a1a1a')
    ax.zaxis.pane.set_edgecolor('#1a1a1a')
    ax.grid(False)

    # Slow rotation
    ax.view_init(elev=22, azim=-52 + fi * 0.7)

    # Title
    fig.text(0.5, 0.96, f'PPFFDTD    t = {t_ms:.1f} ms',
             color='white', fontsize=15, fontweight='bold',
             ha='center', fontfamily='Segoe UI')

    path = OUT_DIR / f'frame_{fi:03d}.png'
    fig.savefig(path, dpi=130, facecolor='black',
                bbox_inches='tight', pad_inches=0.05)
    plt.close()

    if (fi + 1) % 10 == 0 or fi == 0:
        print(f"  [{fi+1}/{len(picks)}] t={t_ms:.1f}ms, peak={vmax:.2e}")

# GIF
try:
    from PIL import Image
    frames = [Image.open(OUT_DIR / f'frame_{fi:03d}.png') for fi in range(len(picks))]
    gif = OUT_DIR / 'ppffdtd_3d.gif'
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=150, loop=0, optimize=True)
    print(f"\nGIF: {gif} ({os.path.getsize(gif)/1e6:.1f} MB)")
except ImportError:
    pass

print("Done!")
