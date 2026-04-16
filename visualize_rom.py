"""
ROM visualization dashboard: dark mode, neon theme.

Panel 1: T30 heatmap (floor vs wall absorption scale)
Panel 2: ROM vs FDTD IR comparison at 5 unseen test points
Panel 3: GP uncertainty map (where ROM is confident vs not)
Panel 4: Material sensitivity bars
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


def plot_rom_dashboard(rom, test_results, save_path='vis/rom_dashboard.png'):
    """Generate ROM dashboard from trained ROM + test results.

    Parameters
    ----------
    rom : NonIntrusiveROM (trained)
    test_results : list of dicts with keys:
        scales, ir_fdtd, ir_rom, t30_fdtd, t30_rom, corr, unc
    """
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'Segoe UI'
    plt.rcParams['font.size'] = 10

    # Neon colormaps
    t30_cmap = LinearSegmentedColormap.from_list('t30_neon', [
        (0.0, '#00FFCC'),
        (0.3, '#0066AA'),
        (0.5, '#4400AA'),
        (0.7, '#AA0066'),
        (1.0, '#FF0066'),
    ])
    unc_cmap = LinearSegmentedColormap.from_list('unc_neon', [
        (0.0, '#001122'),
        (0.3, '#003366'),
        (0.6, '#FF6600'),
        (1.0, '#FF0000'),
    ])

    fig = plt.figure(figsize=(20, 12), facecolor='#0a0a0a')
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ---- Panel 1: T30 heatmap (floor vs wall) ----
    ax1 = fig.add_subplot(gs[0, 0])

    n_grid = 30
    floor_range = np.linspace(0.3, 3.0, n_grid)
    wall_range = np.linspace(0.3, 3.0, n_grid)
    t30_grid = np.zeros((n_grid, n_grid))
    unc_grid = np.zeros((n_grid, n_grid))

    for i, fs in enumerate(floor_range):
        for j, ws in enumerate(wall_range):
            scales = np.array([fs, 1.0, ws])  # floor, ceiling=1.0, walls
            m = rom.evaluate_metrics(scales)
            t30_grid[i, j] = m.get('T30', (np.nan, 0))[0]
            unc_grid[i, j] = m.get('T30', (0, 0))[1]

    im1 = ax1.imshow(t30_grid.T, origin='lower', cmap=t30_cmap, aspect='auto',
                     extent=[0.3, 3.0, 0.3, 3.0], interpolation='bilinear')
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cb1.set_label('T30 (s)', color='#aaaaaa')
    cb1.ax.tick_params(colors='#666666')

    # Mark training points
    for p in rom.training_params:
        ax1.plot(p[0], p[2], 'o', color='white', markersize=3, alpha=0.4)

    # Mark test points
    for r in test_results:
        ax1.plot(r['scales'][0], r['scales'][2], 's', color='#00FF00',
                 markersize=7, markeredgecolor='white', markeredgewidth=0.5)

    ax1.set_xlabel('Floor absorption scale', color='#aaaaaa')
    ax1.set_ylabel('Wall absorption scale', color='#aaaaaa')
    ax1.set_title('T30 Parameter Space', color='#00CCFF', fontsize=13,
                  fontweight='bold')
    ax1.tick_params(colors='#666666')

    # ---- Panel 2: ROM vs FDTD IR overlay ----
    ax2 = fig.add_subplot(gs[0, 1])

    colors_test = ['#00FFCC', '#FF00FF', '#FFAA00', '#00AAFF', '#FF4444']
    for i, r in enumerate(test_results[:5]):
        t = np.arange(len(r['ir_fdtd'])) / 48000 * 1000  # ms
        t_r = np.arange(len(r['ir_rom'])) / 48000 * 1000
        n = min(len(t), 4800)  # show first 100ms

        ax2.plot(t[:n], r['ir_fdtd'][:n], color=colors_test[i],
                 alpha=0.7, linewidth=0.5)
        ax2.plot(t_r[:n], r['ir_rom'][:n], color=colors_test[i],
                 alpha=0.3, linewidth=1.5, linestyle='--')

    ax2.set_xlabel('Time (ms)', color='#aaaaaa')
    ax2.set_ylabel('Pressure', color='#aaaaaa')
    ax2.set_title('ROM vs FDTD (solid=FDTD, dashed=ROM)', color='#00CCFF',
                  fontsize=13, fontweight='bold')
    ax2.tick_params(colors='#666666')
    ax2.set_xlim(0, 100)

    # ---- Panel 3: Uncertainty map ----
    ax3 = fig.add_subplot(gs[1, 0])

    im3 = ax3.imshow(unc_grid.T, origin='lower', cmap=unc_cmap, aspect='auto',
                     extent=[0.3, 3.0, 0.3, 3.0], interpolation='bilinear')
    cb3 = plt.colorbar(im3, ax=ax3, shrink=0.85, pad=0.02)
    cb3.set_label('T30 uncertainty (s)', color='#aaaaaa')
    cb3.ax.tick_params(colors='#666666')

    for p in rom.training_params:
        ax3.plot(p[0], p[2], 'o', color='white', markersize=3, alpha=0.4)

    ax3.set_xlabel('Floor absorption scale', color='#aaaaaa')
    ax3.set_ylabel('Wall absorption scale', color='#aaaaaa')
    ax3.set_title('GP Uncertainty (low = confident)', color='#00CCFF',
                  fontsize=13, fontweight='bold')
    ax3.tick_params(colors='#666666')

    # ---- Panel 4: Test point results table ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    headers = ['Case', 'Floor', 'Ceil', 'Wall', 'T30 FDTD', 'T30 ROM', 'Err%', 'Corr']
    col_w = [0.06, 0.09, 0.09, 0.09, 0.14, 0.14, 0.1, 0.1]

    y = 0.92
    for j, h in enumerate(headers):
        x = sum(col_w[:j]) + col_w[j] / 2
        ax4.text(x, y, h, color='#00CCFF', fontsize=11, fontweight='bold',
                 ha='center', va='center', transform=ax4.transAxes)

    for i, r in enumerate(test_results):
        y = 0.82 - i * 0.12
        vals = [
            f'{i+1}',
            f'{r["scales"][0]:.2f}',
            f'{r["scales"][1]:.2f}',
            f'{r["scales"][2]:.2f}',
            f'{r["t30_fdtd"]:.3f}',
            f'{r["t30_rom"]:.3f}',
            f'{r["err"]:.1f}',
            f'{r["corr"]:.4f}',
        ]
        for j, v in enumerate(vals):
            x = sum(col_w[:j]) + col_w[j] / 2
            color = '#cccccc'
            if j == 6:  # error column
                err_val = float(v)
                color = '#00FF00' if err_val < 5 else '#FFAA00' if err_val < 10 else '#FF4444'
            ax4.text(x, y, v, color=color, fontsize=10,
                     ha='center', va='center', transform=ax4.transAxes)

    # Summary row
    y = 0.82 - len(test_results) * 0.12 - 0.08
    mean_err = np.mean([r['err'] for r in test_results])
    mean_corr = np.mean([r['corr'] for r in test_results])
    ax4.text(0.5, y, f'Mean error: {mean_err:.1f}%   |   Mean correlation: {mean_corr:.4f}',
             color='#FF00FF', fontsize=12, fontweight='bold',
             ha='center', transform=ax4.transAxes)

    ax4.set_title('Unseen Test Points', color='#00CCFF', fontsize=13,
                  fontweight='bold')

    # Main title
    fig.suptitle('PPFFDTD   Non-Intrusive ROM   Performance Dashboard',
                 color='white', fontsize=16, fontweight='bold', y=0.97)

    fig.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'Dashboard saved: {save_path}')
