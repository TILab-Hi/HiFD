# $DST/tests/test_rppg_signal.py
import numpy as np
import pytest
from src.rppg_signal import hr_from_bvp, bvp_agreement, hr_agreement


def test_hr_from_bvp_synthetic():
    fs = 30.0
    n = 300
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz = 72 bpm
    hr = hr_from_bvp(sig, fs=fs)
    assert abs(hr - 72.0) < 5.0


def test_hr_from_bvp_flat():
    assert hr_from_bvp(np.zeros(100), fs=30.0) == 0.0


def test_bvp_agreement_identity():
    rng = np.random.default_rng(0)
    s = rng.normal(size=128)
    assert bvp_agreement(s, s) == pytest.approx(1.0)


def test_hr_agreement_identical_signals():
    rng = np.random.default_rng(0)
    s = rng.normal(size=300)
    assert hr_agreement(s, s) == pytest.approx(1.0)
