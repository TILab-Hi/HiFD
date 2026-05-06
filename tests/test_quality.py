# $DST/tests/test_quality.py
import pytest
from src.quality import Q_score


def test_Q_score_perfect():
    # LPIPS=0 (identical), NIQE=0 (perfect quality) → Q=1
    assert Q_score(0.0, 0.0) == pytest.approx(1.0)


def test_Q_score_worst():
    # LPIPS=1, NIQE>=tau_N → Q=0
    assert Q_score(1.0, 5.0) == 0.0


def test_Q_score_balanced_alpha():
    # alpha=0.5: 50/50 split between LPIPS and NIQE terms
    q = Q_score(0.0, 5.0, tau_N=5.0, alpha=0.5)
    assert q == pytest.approx(0.5)
