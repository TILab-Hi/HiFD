# $DST/tests/test_composite.py
import pytest
from src.composite import hifd, compute_all_profiles


WEIGHTS_BAL = {"P": 0.2, "Q": 0.2, "U1": 0.2, "U2": 0.2, "U3": 0.2}


def test_hifd_all_ones():
    assert hifd(1.0, 1.0, 1.0, 1.0, 1.0, WEIGHTS_BAL) == pytest.approx(1.0)


def test_hifd_strict_min_pull_down():
    """A near-zero component should pull the harmonic mean to near-zero."""
    s = hifd(1.0, 1.0, 1.0, 1.0, 1e-3, WEIGHTS_BAL)
    assert s < 0.01


def test_hifd_handles_missing_level():
    """U3=None should drop the term and renormalize remaining weights."""
    s_l3 = hifd(0.5, 0.5, 0.5, 0.5, 0.5, WEIGHTS_BAL)
    s_l2 = hifd(0.5, 0.5, 0.5, 0.5, None, WEIGHTS_BAL)
    # With all-equal scores the harmonic mean equals the score regardless of
    # which subset of weights is renormalized.
    assert s_l3 == pytest.approx(s_l2)


def test_compute_all_profiles_returns_dict():
    profiles = {
        "PrivacyFirst": {"P": 0.5, "Q": 0.125, "U1": 0.125, "U2": 0.125, "U3": 0.125},
        "Balanced":     WEIGHTS_BAL,
    }
    out = compute_all_profiles(0.7, 0.7, 0.7, 0.7, 0.7, profiles)
    assert set(out.keys()) == {"PrivacyFirst", "Balanced"}
    for v in out.values():
        assert 0.0 <= v <= 1.0
