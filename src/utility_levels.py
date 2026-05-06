"""Utility level aggregation for the HiFD metric.

Computes per-sample level scores U1, U2, U3 and per-method aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 5.1 Per-sample level scores
# ---------------------------------------------------------------------------

def U1(
    s_age: float,
    s_gender: float,
    s_eth: float,
    s_macro: float,
    s_lmk: float,
) -> float:
    """Level-1 utility: macro facial cues (image-level)."""
    return float(np.mean([s_age, s_gender, s_eth, s_macro, s_lmk]))


def U2(s_gaze: float, s_me: Optional[float] = None) -> float:
    """Level-2 utility: fine-grained cues (gaze, micro-expression)."""
    if s_me is not None:
        return 0.5 * (s_gaze + s_me)
    return s_gaze


def U3(bvp_scores: List[float], hr_scores: List[float]) -> float:
    """
    Level-3 utility: physiological signals.

    Args:
        bvp_scores: BVP agreement scores per rPPG estimator.
        hr_scores: HR agreement scores per rPPG estimator.

    Returns:
        Mean of per-estimator combined scores.
    """
    per_est = 0.5 * (np.asarray(bvp_scores) + np.asarray(hr_scores))
    return float(np.mean(per_est))


# ---------------------------------------------------------------------------
# Vectorized versions for batch computation
# ---------------------------------------------------------------------------

def batch_U1(
    s_age: np.ndarray,
    s_gender: np.ndarray,
    s_eth: np.ndarray,
    s_macro: np.ndarray,
    s_lmk: np.ndarray,
) -> np.ndarray:
    """Vectorized Level-1 utility over N samples."""
    stacked = np.stack([s_age, s_gender, s_eth, s_macro, s_lmk], axis=0)
    return np.mean(stacked, axis=0)


def batch_U2(
    s_gaze: np.ndarray, s_me: Optional[np.ndarray] = None
) -> np.ndarray:
    """Vectorized Level-2 utility over N samples."""
    if s_me is not None:
        return 0.5 * (s_gaze + s_me)
    return s_gaze.copy()


def batch_U3(
    bvp_scores_per_est: List[np.ndarray],
    hr_scores_per_est: List[np.ndarray],
) -> np.ndarray:
    """
    Vectorized Level-3 utility over N samples.

    Args:
        bvp_scores_per_est: list of (N,) arrays, one per rPPG estimator.
        hr_scores_per_est: list of (N,) arrays, one per rPPG estimator.

    Returns:
        (N,) array of U3 scores.
    """
    combined = []
    for bvp, hr in zip(bvp_scores_per_est, hr_scores_per_est):
        combined.append(0.5 * (bvp + hr))
    stacked = np.stack(combined, axis=0)
    return np.mean(stacked, axis=0)


# ---------------------------------------------------------------------------
# 5.2 Per-method aggregation
# ---------------------------------------------------------------------------

def aggregate_method_scores(
    per_sample: Dict[str, np.ndarray],
    method_name: str,
) -> pd.DataFrame:
    """
    Aggregate per-sample scores into a method-level summary DataFrame.

    Args:
        per_sample: dict mapping sub_score name -> (N,) array of per-sample values.
                    Keys should include: 'age', 'gender', 'ethnicity', 'macro_exp',
                    'landmark', 'gaze', 'micro_exp', 'bvp', 'hr', 'P', 'Q',
                    'U1', 'U2', 'U3'.
        method_name: name of the de-identification method.

    Returns:
        DataFrame with columns: method, dimension, sub_score, value, n_samples, std.
    """
    dim_map = {
        "age": "L1", "gender": "L1", "ethnicity": "L1",
        "macro_exp": "L1", "landmark": "L1",
        "gaze": "L2", "micro_exp": "L2",
        "bvp": "L3", "hr": "L3",
        "P": "Privacy", "Q": "Quality",
        "U1": "L1", "U2": "L2", "U3": "L3",
    }

    rows = []
    for score_name, arr in per_sample.items():
        rows.append({
            "method": method_name,
            "dimension": dim_map.get(score_name, "Other"),
            "sub_score": score_name,
            "value": float(np.mean(arr)),
            "n_samples": len(arr),
            "std": float(np.std(arr)),
        })

    return pd.DataFrame(rows)
