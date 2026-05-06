# $DST/tests/test_utility_levels.py
import pytest
from src.utility_levels import U1, U2, U3


def test_U1_uniform():
    assert U1(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)


def test_U2_with_micro_exp():
    assert U2(0.6, 0.4) == pytest.approx(0.5)


def test_U2_without_micro_exp():
    assert U2(0.6, None) == pytest.approx(0.6)


def test_U3_per_estimator_average():
    bvp = [0.8, 0.6]
    hr = [0.4, 0.2]
    # combined per est: (0.8+0.4)/2=0.6, (0.6+0.2)/2=0.4 → mean 0.5
    assert U3(bvp, hr) == pytest.approx(0.5)
