# $DST/tests/test_agreements.py
import numpy as np
import pytest
from src.agreements import (
    s_age, s_categorical, s_landmark, s_gaze, s_bvp, s_hr,
)


def test_s_age_identity():
    assert s_age(30.0, 30.0) == 1.0


def test_s_age_monotone():
    assert s_age(30.0, 35.0) < 1.0
    assert s_age(30.0, 35.0) > s_age(30.0, 50.0)


def test_s_age_bounded():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a, b = rng.uniform(0, 100, size=2)
        s = s_age(a, b)
        assert 0.0 <= s <= 1.0


def test_s_categorical_identity():
    p = np.array([0.7, 0.2, 0.1])
    assert s_categorical(p, p) == pytest.approx(1.0)


def test_s_categorical_orthogonal():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert s_categorical(p, q) == pytest.approx(0.0)


def test_s_landmark_identity():
    L = np.array([[0.0, 0.0], [1.0, 1.0]])
    eye_l = np.array([0.0, 0.0])
    eye_r = np.array([1.0, 0.0])
    assert s_landmark(L, L, eye_l, eye_r) == 1.0


def test_s_landmark_zero_iod_returns_zero():
    L = np.array([[0.0, 0.0]])
    eye = np.array([0.5, 0.5])
    assert s_landmark(L, L, eye, eye) == 0.0


def test_s_gaze_identity():
    g = np.array([1.0, 0.0, 0.0])
    assert s_gaze(g, g) == pytest.approx(1.0, abs=1e-4)


def test_s_gaze_opposite():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    # 180° / 90° cap → score should be 0
    assert s_gaze(a, b) == 0.0


def test_s_bvp_identity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    assert s_bvp(x, x) == pytest.approx(1.0)


def test_s_bvp_flat_returns_zero():
    flat = np.zeros(50)
    other = np.arange(50, dtype=float)
    assert s_bvp(flat, other) == 0.0


def test_s_hr_identity():
    assert s_hr(72.0, 72.0) == 1.0


def test_s_hr_clamped():
    assert s_hr(60.0, 200.0) == 0.0
