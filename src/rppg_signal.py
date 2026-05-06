"""rPPG signal helpers: BVP correlation and FFT-based heart rate."""

import numpy as np


def hr_from_bvp(signal, fs: float = 30.0,
                lo_hz: float = 0.7, hi_hz: float = 3.0) -> float:
    """FFT-based heart rate estimate (bpm). Returns 0 for flat/short signals."""
    s = np.asarray(signal, dtype=np.float64)
    s = s - s.mean()
    if s.std() < 1e-8 or len(s) < 8:
        return 0.0
    n = len(s)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    ps = np.abs(np.fft.rfft(s)) ** 2
    valid = (freqs >= lo_hz) & (freqs <= hi_hz)
    if valid.sum() == 0:
        return 0.0
    idx = int(np.argmax(ps[valid]))
    return float(freqs[valid][idx] * 60.0)


def bvp_agreement(b_orig, b_deid) -> float:
    s1 = np.asarray(b_orig, dtype=np.float64)
    s2 = np.asarray(b_deid, dtype=np.float64)
    if s1.std() < 1e-8 or s2.std() < 1e-8:
        return 0.0
    rho = float(np.corrcoef(s1, s2)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    return max(0.0, rho)


def hr_agreement(b_orig, b_deid, fs: float = 30.0,
                 tau_HR: float = 20.0) -> float:
    r_o = hr_from_bvp(b_orig, fs=fs)
    r_d = hr_from_bvp(b_deid, fs=fs)
    return float(max(0.0, 1.0 - abs(r_o - r_d) / tau_HR))
