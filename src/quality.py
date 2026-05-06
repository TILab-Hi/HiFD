"""Quality score computation for the HiFD metric.

Combines LPIPS (perceptual similarity) and NIQE (no-reference quality)
into a single quality score Q in [0, 1].
"""

import numpy as np
from typing import Optional


def Q_score(
    lpips_val: float,
    niqe_val: float,
    tau_N: float = 5.0,
    alpha: float = 0.5,
) -> float:
    """
    Compute quality score from LPIPS and NIQE values.

    Q = alpha * (1 - LPIPS) + (1 - alpha) * NIQE*
    where NIQE* = max(0, min(1, (tau_N - NIQE) / tau_N))

    Args:
        lpips_val: LPIPS distance (0=identical, ~1=very different).
        niqe_val: NIQE score (lower = better quality).
        tau_N: NIQE cap value.
        alpha: Balance between LPIPS and NIQE terms.

    Returns:
        Quality score in [0, 1]. Higher = better quality.
    """
    lpips_term = 1.0 - min(max(lpips_val, 0.0), 1.0)
    niqe_star = max(0.0, min(1.0, (tau_N - niqe_val) / tau_N))
    return alpha * lpips_term + (1.0 - alpha) * niqe_star


def batch_Q_score(
    lpips_vals: np.ndarray,
    niqe_vals: np.ndarray,
    tau_N: float = 5.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Vectorized quality score computation.

    Args:
        lpips_vals: (N,) array of LPIPS distances.
        niqe_vals: (N,) array of NIQE scores.
        tau_N: NIQE cap value.
        alpha: Balance between LPIPS and NIQE terms.

    Returns:
        (N,) array of quality scores.
    """
    lpips_term = 1.0 - np.clip(lpips_vals, 0.0, 1.0)
    niqe_star = np.clip((tau_N - niqe_vals) / tau_N, 0.0, 1.0)
    return alpha * lpips_term + (1.0 - alpha) * niqe_star
