"""ISO 3382 per-octave-band metrics extraction for CHORAS output.

Computes T30, T20, EDT, C80, D50, TS, and SPL_t0 per octave band from a
broadband impulse response. Bands above 0.9 * fmax_grid return None (JSON null).

Used by PFFDTDInterface.py to populate the CHORAS schema's
results[0].responses[i].parameters dict, which expects per-band arrays.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _bandpass(ir, fs, fc, octave_fraction=1.0, order=4):
    """4th-order zero-phase Butterworth band-pass at center fc (Hz)."""
    factor = 2 ** (octave_fraction / 2)
    f_lo = fc / factor
    f_hi = fc * factor
    nyq = 0.5 * fs
    f_lo = max(f_lo, 1.0)
    f_hi = min(f_hi, 0.95 * nyq)
    if f_hi <= f_lo:
        return np.zeros_like(ir)
    sos = butter(order, [f_lo, f_hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, ir)


def _edc_db(ir):
    """Schroeder backward-integrated energy decay curve in dB."""
    e2 = ir.astype(np.float64) ** 2
    edc = np.cumsum(e2[::-1])[::-1]
    edc /= (edc[0] + 1e-30)
    return 10.0 * np.log10(edc + 1e-30)


def _t_from_edc(edc_db, fs, db_lo, db_hi):
    """Reverberation time extrapolated from EDC slope between db_lo..db_hi (negative dB).

    Returns the time (s) for a 60 dB drop given the local slope, or None on bad fit.
    """
    # Find the index where EDC first drops below db_lo and db_hi.
    # np.argmax on a boolean array returns 0 when no element is True, so we
    # explicitly require at least one True occurrence.
    below_lo = edc_db < db_lo
    below_hi = edc_db < db_hi
    if not below_lo.any() or not below_hi.any():
        return None
    i_lo = int(np.argmax(below_lo))
    i_hi = int(np.argmax(below_hi))
    if i_hi <= i_lo + 5:
        return None
    t = np.arange(i_lo, i_hi) / fs
    y = edc_db[i_lo:i_hi]
    if not np.all(np.isfinite(y)):
        return None
    slope, _ = np.polyfit(t, y, 1)
    if slope >= 0 or not np.isfinite(slope):
        return None
    return float(-60.0 / slope)


def _t30(ir, fs):
    edc = _edc_db(ir)
    return _t_from_edc(edc, fs, -5.0, -35.0)


def _t20(ir, fs):
    edc = _edc_db(ir)
    return _t_from_edc(edc, fs, -5.0, -25.0)


def _edt(ir, fs):
    edc = _edc_db(ir)
    return _t_from_edc(edc, fs, 0.0, -10.0)


def _c80(ir, fs):
    """Clarity index C80 (dB): early/late energy ratio, split at 80 ms."""
    n80 = int(0.080 * fs)
    if n80 >= len(ir):
        return None
    e2 = ir.astype(np.float64) ** 2
    early = float(np.sum(e2[:n80]))
    late = float(np.sum(e2[n80:]))
    if late <= 0 or early <= 0:
        return None
    return float(10.0 * np.log10(early / late))


def _d50(ir, fs):
    """Definition D50: ratio of energy in first 50 ms to total energy (0..1)."""
    n50 = int(0.050 * fs)
    if n50 >= len(ir):
        return None
    e2 = ir.astype(np.float64) ** 2
    early = float(np.sum(e2[:n50]))
    total = float(np.sum(e2))
    if total <= 0:
        return None
    return float(early / total)


def _ts(ir, fs):
    """Center time TS (s): first moment of squared IR."""
    e2 = ir.astype(np.float64) ** 2
    total = float(np.sum(e2))
    if total <= 0:
        return None
    t = np.arange(len(ir)) / fs
    return float(np.sum(t * e2) / total)


def _spl_t0(ir):
    """Sound pressure level at t0 (dB re 20 uPa, peak)."""
    p_peak = float(np.max(np.abs(ir)))
    if p_peak <= 0:
        return None
    p_ref = 20e-6
    return float(20.0 * np.log10(p_peak / p_ref))


def compute_per_band_metrics(ir, fs, frequencies, fmax_grid):
    """Compute per-band ISO 3382 metrics for the CHORAS output schema.

    Args:
        ir: 1-D numpy array, the impulse response (after post-processing)
        fs: sample rate of ir (Hz)
        frequencies: list of octave-band centers from CHORAS JSON (e.g. [125,250,500,1000,2000])
        fmax_grid: PFFDTD grid f_max (= c0 / (2*h)). Bands above 0.9*fmax_grid return None.

    Returns:
        dict matching the CHORAS responses[i].parameters schema:
            {edt, t20, t30, c80, d50, ts, spl_t0_freq}
        Each value is a list of length len(frequencies) with None for ungated bands.
    """
    ir = np.asarray(ir, dtype=np.float64)
    f_cutoff = 0.9 * float(fmax_grid)

    keys = ("edt", "t20", "t30", "c80", "d50", "ts", "spl_t0_freq")
    out = {k: [] for k in keys}

    for fc in frequencies:
        if fc > f_cutoff:
            for k in keys:
                out[k].append(None)
            continue

        ir_band = _bandpass(ir, fs, fc)

        out["edt"].append(_edt(ir_band, fs))
        out["t20"].append(_t20(ir_band, fs))
        out["t30"].append(_t30(ir_band, fs))
        out["c80"].append(_c80(ir_band, fs))
        out["d50"].append(_d50(ir_band, fs))
        out["ts"].append(_ts(ir_band, fs))
        out["spl_t0_freq"].append(_spl_t0(ir_band))

    return out
