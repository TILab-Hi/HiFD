"""Per-task agreement functions for the HiFD metric.

Each function returns a scalar in [0, 1] for a single paired sample.
Batch wrappers return NumPy arrays over all paired samples.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 4.1 Age
# ---------------------------------------------------------------------------

def s_age(a_orig: float, a_deid: float, Delta_age: float = 100.0) -> float:
    """Age agreement score."""
    return max(0.0, 1.0 - abs(a_orig - a_deid) / Delta_age)


def batch_s_age(
    a_orig: np.ndarray, a_deid: np.ndarray, Delta_age: float = 100.0
) -> np.ndarray:
    """Vectorized age agreement."""
    return np.maximum(0.0, 1.0 - np.abs(a_orig - a_deid) / Delta_age)


# ---------------------------------------------------------------------------
# 4.2 Categorical (gender / ethnicity / macro_exp / micro_exp)
# ---------------------------------------------------------------------------

def s_categorical(p_orig: np.ndarray, p_deid: np.ndarray) -> float:
    """Categorical agreement via total variation distance."""
    assert p_orig.shape == p_deid.shape, (
        f"Shape mismatch: {p_orig.shape} vs {p_deid.shape}"
    )
    tv = 0.5 * np.sum(np.abs(p_orig - p_deid))
    return float(1.0 - tv)


def batch_s_categorical(
    p_orig: np.ndarray, p_deid: np.ndarray
) -> np.ndarray:
    """Vectorized categorical agreement. Arrays shape (N, C)."""
    tv = 0.5 * np.sum(np.abs(p_orig - p_deid), axis=1)
    return 1.0 - tv


# ---------------------------------------------------------------------------
# 4.3 Landmark
# ---------------------------------------------------------------------------

def s_landmark(
    L_orig: np.ndarray,
    L_deid: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    tau_NME: float = 0.10,
) -> float:
    """Landmark agreement via NME normalized by inter-ocular distance."""
    d_io = np.linalg.norm(left_eye - right_eye)
    if d_io < 1e-3:
        return 0.0
    nme = np.mean(np.linalg.norm(L_orig - L_deid, axis=1)) / d_io
    return float(max(0.0, 1.0 - nme / tau_NME))


# ---------------------------------------------------------------------------
# 4.4 Gaze
# ---------------------------------------------------------------------------

def s_gaze(
    g_orig: np.ndarray,
    g_deid: np.ndarray,
    theta_max_deg: float = 90.0,
) -> float:
    """Gaze agreement via angular distance."""
    g_o = g_orig / (np.linalg.norm(g_orig) + 1e-12)
    g_d = g_deid / (np.linalg.norm(g_deid) + 1e-12)
    cos_sim = float(np.clip(np.dot(g_o, g_d), -1.0, 1.0))
    theta = np.degrees(np.arccos(cos_sim))
    return float(max(0.0, 1.0 - theta / theta_max_deg))


# ---------------------------------------------------------------------------
# 4.5 rPPG - BVP waveform
# ---------------------------------------------------------------------------

def s_bvp(b_orig: np.ndarray, b_deid: np.ndarray) -> float:
    """BVP waveform agreement via Pearson correlation."""
    T = min(len(b_orig), len(b_deid))
    b_o, b_d = b_orig[:T], b_deid[:T]
    if np.std(b_o) < 1e-9 or np.std(b_d) < 1e-9:
        return 0.0
    rho = float(np.corrcoef(b_o, b_d)[0, 1])
    return max(0.0, rho)


# ---------------------------------------------------------------------------
# 4.6 rPPG - Heart Rate
# ---------------------------------------------------------------------------

def s_hr(r_orig: float, r_deid: float, tau_HR: float = 20.0) -> float:
    """Heart rate agreement."""
    return max(0.0, 1.0 - abs(r_orig - r_deid) / tau_HR)


def batch_s_hr(
    r_orig: np.ndarray, r_deid: np.ndarray, tau_HR: float = 20.0
) -> np.ndarray:
    """Vectorized heart rate agreement."""
    return np.maximum(0.0, 1.0 - np.abs(r_orig - r_deid) / tau_HR)
