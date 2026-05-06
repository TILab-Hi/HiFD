# $DST/tests/test_privacy.py
import numpy as np
import pytest
from src.privacy import P_single, P_bar


def test_P_single_identity_is_zero():
    e = np.array([1.0, 0.0, 0.0])
    assert P_single(e, e) == pytest.approx(0.0, abs=1e-6)


def test_P_single_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert P_single(a, b) == pytest.approx(0.5)


def test_P_single_opposite_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert P_single(a, b) == pytest.approx(1.0)


def test_P_bar_average():
    scores = {
        "arcface": np.array([0.1, 0.5]),
        "cosface": np.array([0.3, 0.7]),
    }
    result = P_bar(scores)
    np.testing.assert_allclose(result, [0.2, 0.6])
