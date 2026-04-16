"""Train ROM, validate on unseen points, generate dashboard."""
import sys, os, json, numpy as np, time
sys.path.insert(0, 'pffdtd/python')
sys.path.insert(0, 'ppffdtd')
from rom import NonIntrusiveROM, compute_metrics, postprocess_ir
from visualize_rom import plot_rom_dashboard

data_dir = 'common/pffdtd_data'
pffdtd_dir = 'pffdtd/python'

with open('common/exampleInput_PFFDTD.json') as f:
    baseline_alphas = json.load(f)['absorption_coefficients']

rom = NonIntrusiveROM(data_dir, pffdtd_dir)

# ---- Train ----
print('=' * 60)
print('TRAINING ROM (33 Smolyak cases)')
print('=' * 60)
rom.train(baseline_alphas, dim=3, level=2, use_gpu=False)
rom.save(data_dir + '/rom_v2.npz')

# ---- LOO ----
print('\n' + '=' * 60)
print('LEAVE-ONE-OUT VALIDATION')
print('=' * 60)
rom.validate_loo()

# ---- 5 unseen test points ----
print('\n' + '=' * 60)
print('UNSEEN POINT VALIDATION (5 fresh FDTD runs)')
print('=' * 60)

np.random.seed(42)
test_results = []

for i in range(5):
    scales = np.exp(np.random.uniform(np.log(0.4), np.log(2.5), 3))
    full_scales = np.array([scales[0], scales[1], scales[2], scales[2], scales[2], scales[2]])

    print(f'\n  Case {i+1}: floor={scales[0]:.2f}, ceil={scales[1]:.2f}, wall={scales[2]:.2f}')

    # FDTD
    DEF = rom._alpha_to_DEF(full_scales, baseline_alphas)
    ir_raw = rom._run_single_fdtd(DEF, use_gpu=False)
    ir_fdtd = postprocess_ir(ir_raw, rom.fs_native, rom.fmax_grid, 48000)

    # ROM
    ir_rom, unc = rom.evaluate(scales)

    # Compare
    n = min(len(ir_fdtd), len(ir_rom))
    corr = np.corrcoef(ir_fdtd[:n], ir_rom[:n])[0, 1]
    m_fdtd = compute_metrics(ir_fdtd, 48000)
    m_rom = compute_metrics(ir_rom[:n], 48000)
    err = abs(m_rom['T30'] - m_fdtd['T30']) / m_fdtd['T30'] * 100

    print(f'    T30: FDTD={m_fdtd["T30"]:.3f}s  ROM={m_rom["T30"]:.3f}s  err={err:.1f}%  corr={corr:.4f}')

    test_results.append({
        'scales': scales,
        'ir_fdtd': ir_fdtd[:n],
        'ir_rom': ir_rom[:n],
        't30_fdtd': m_fdtd['T30'],
        't30_rom': m_rom['T30'],
        'err': err,
        'corr': corr,
        'unc': unc,
    })

# ---- Summary ----
errs = [r['err'] for r in test_results]
corrs = [r['corr'] for r in test_results]
print(f'\n  Mean T30 error: {np.mean(errs):.1f}%')
print(f'  Mean correlation: {np.mean(corrs):.4f}')

# ---- Dashboard ----
print('\n' + '=' * 60)
print('GENERATING DASHBOARD')
print('=' * 60)
os.makedirs('vis', exist_ok=True)
plot_rom_dashboard(rom, test_results, 'vis/rom_dashboard.png')

print('\nDone!')
