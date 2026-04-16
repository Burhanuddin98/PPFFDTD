"""
3D FDTD visualization: black background, white wireframe, neon waves.

Uses matplotlib 3D to render three translucent pressure slices
inside a white wireframe room with source/receiver markers.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import h5py
import time as _time

sys.path.insert(0, 'pffdtd/python')

plt.style.use('dark_background')

# Neon colormap: cyan → transparent → magenta
neon_cmap = LinearSegmentedColormap.from_list('neon_wave', [
    (0.00, '#00DDFF'),
    (0.35, '#001133'),
    (0.50, '#00000000'),
    (0.65, '#330011'),
    (1.00, '#FF00DD'),
])

DATA_DIR = Path('common/pffdtd_data')
OUT_DIR = Path('vis3d')
OUT_DIR.mkdir(exist_ok=True)

# ---- Load grid info ----
with h5py.File(DATA_DIR / 'vox_out.h5', 'r') as f:
    Nx, Ny, Nz = int(f['Nx'][()]), int(f['Ny'][()]), int(f['Nz'][()])
    xv, yv, zv = f['xv'][()], f['yv'][()], f['zv'][()]
    bn_ixyz = f['bn_ixyz'][...]

with h5py.File(DATA_DIR / 'comms_out.h5', 'r') as f:
    in_ixyz = f['in_ixyz'][...]
    out_ixyz = f['out_ixyz'][...]

with h5py.File(DATA_DIR / 'sim_consts.h5', 'r') as f:
    Ts = float(f['Ts'][()])
    h_grid = float(f['h'][()])

NyNz = Ny * Nz
src = in_ixyz[0]
sx, sy, sz = src // NyNz, (src % NyNz) // Nz, src % Nz
rec = out_ixyz[0]
rx, ry, rz = rec // NyNz, (rec % NyNz) // Nz, rec % Nz

bn_mask = np.zeros(Nx * Ny * Nz, dtype=bool)
bn_mask[bn_ixyz] = True
bn_mask_3d = bn_mask.reshape(Nx, Ny, Nz)

# Room bounds (from boundary outline)
x0, x1 = xv[2], xv[-3]
y0, y1 = yv[2], yv[-3]
z0, z1 = zv[2], zv[-3]

print(f"Grid: {Nx}x{Ny}x{Nz}")
print(f"Room: [{x0:.1f},{x1:.1f}] x [{y0:.1f},{y1:.1f}] x [{z0:.1f},{z1:.1f}]")

# ---- Run FDTD with slice collection ----
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

# Snapshot schedule: dense early, sparse late
snap_steps = list(range(2, 300, 3))
snap_steps += list(range(300, 1000, 20))
snap_steps += list(range(1000, Nt, 150))
snap_steps = sorted(set(s for s in snap_steps if s < Nt))
snap_set = set(snap_steps)

slices_xy = []
slices_xz = []
slices_yz = []
snap_times = []

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

# ---- Render 3D frames ----
# Pick ~30 frames for GIF
n_frames = min(40, len(slices_xy))
picks = np.unique(np.linspace(0, len(slices_xy)-1, n_frames).astype(int))

print(f"Rendering {len(picks)} 3D frames...")


def draw_wireframe(ax):
    """Draw white wireframe room edges."""
    verts = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ]
    for i, j in edges:
        ax.plot3D(*zip(verts[i], verts[j]), color='white', linewidth=0.7, alpha=0.5)


for fi, si in enumerate(picks):
    xy = slices_xy[si]
    xz = slices_xz[si]
    yz = slices_yz[si]
    t_ms = snap_times[si]

    vmax = max(np.max(np.abs(xy)), np.max(np.abs(xz)),
               np.max(np.abs(yz)), 1e-10)

    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_facecolor('black')

    # Wireframe room
    draw_wireframe(ax)

    # XY slice (horizontal at source height)
    X_xy, Y_xy = np.meshgrid(xv, yv, indexing='ij')
    Z_xy = np.full_like(X_xy, zv[sz])
    colors_xy = neon_cmap((xy / vmax + 1) / 2)
    colors_xy[:, :, 3] = np.clip(np.abs(xy) / vmax * 2, 0.05, 0.8)
    ax.plot_surface(X_xy, Y_xy, Z_xy, facecolors=colors_xy,
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # XZ slice (vertical at source Y)
    X_xz, Z_xz = np.meshgrid(xv, zv, indexing='ij')
    Y_xz = np.full_like(X_xz, yv[sy])
    colors_xz = neon_cmap((xz / vmax + 1) / 2)
    colors_xz[:, :, 3] = np.clip(np.abs(xz) / vmax * 2, 0.05, 0.8)
    ax.plot_surface(X_xz, Y_xz, Z_xz, facecolors=colors_xz,
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # YZ slice (vertical at source X)
    Y_yz, Z_yz = np.meshgrid(yv, zv, indexing='ij')
    X_yz = np.full_like(Y_yz, xv[sx])
    colors_yz = neon_cmap((yz / vmax + 1) / 2)
    colors_yz[:, :, 3] = np.clip(np.abs(yz) / vmax * 2, 0.05, 0.8)
    ax.plot_surface(X_yz, Y_yz, Z_yz, facecolors=colors_yz,
                    shade=False, rstride=1, cstride=1, antialiased=False)

    # Source marker
    ax.scatter([xv[sx]], [yv[sy]], [zv[sz]], color='#FF2222',
               s=120, marker='*', edgecolors='white', linewidth=0.5,
               zorder=10, depthshade=False)
    # Receiver marker
    ax.scatter([xv[rx]], [yv[ry]], [zv[rz]], color='#22FF22',
               s=80, marker='^', edgecolors='white', linewidth=0.5,
               zorder=10, depthshade=False)

    # Styling
    ax.set_xlim(x0 - 0.3, x1 + 0.3)
    ax.set_ylim(y0 - 0.3, y1 + 0.3)
    ax.set_zlim(z0 - 0.1, z1 + 0.3)
    ax.set_xlabel('x (m)', color='#555555', fontsize=8, labelpad=-2)
    ax.set_ylabel('y (m)', color='#555555', fontsize=8, labelpad=-2)
    ax.set_zlabel('z (m)', color='#555555', fontsize=8, labelpad=-2)
    ax.tick_params(colors='#333333', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#222222')
    ax.yaxis.pane.set_edgecolor('#222222')
    ax.zaxis.pane.set_edgecolor('#222222')
    ax.grid(False)

    # Camera angle (slow rotation)
    elev = 25
    azim = -55 + fi * 0.8
    ax.view_init(elev=elev, azim=azim)

    # Title
    fig.text(0.5, 0.95, f'PPFFDTD   t = {t_ms:.1f} ms',
             color='white', fontsize=16, fontweight='bold',
             ha='center', va='top', fontfamily='monospace')
    fig.text(0.5, 0.03, 'BRAS S09 Seminar Room',
             color='#666666', fontsize=10, ha='center',
             fontfamily='monospace')

    path = OUT_DIR / f'frame_{fi:03d}.png'
    fig.savefig(path, dpi=120, facecolor='black',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

    if (fi + 1) % 10 == 0 or fi == 0:
        print(f"  [{fi+1}/{len(picks)}] t={t_ms:.1f}ms")

# GIF
try:
    from PIL import Image
    frames = [Image.open(OUT_DIR / f'frame_{fi:03d}.png') for fi in range(len(picks))]
    gif_path = OUT_DIR / 'ppffdtd_3d.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=180, loop=0, optimize=True)
    print(f"\nGIF: {gif_path} ({os.path.getsize(gif_path)/1e6:.1f} MB)")
except ImportError:
    print("Install Pillow for GIF")

print("Done!")
