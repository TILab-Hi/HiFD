"""Composite HiFD score computation and application profiles.

Implements the weighted harmonic mean that combines privacy, quality,
and multi-level utility into a single score.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def hifd(
    P_bar: Optional[float],
    Q: Optional[float],
    U1: Optional[float],
    U2: Optional[float],
    U3: Optional[float],
    weights: Dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the composite HiFD score as a weighted harmonic mean.

    Components with value None are excluded and weights are auto-renormalized.

    Args:
        P_bar: Ensemble privacy score (mean across backbones, then samples).
        Q: Quality score.
        U1: Level-1 utility (macro-cues).
        U2: Level-2 utility (fine-grained).
        U3: Level-3 utility (physiological).
        weights: Dict with keys 'P', 'Q', 'U1', 'U2', 'U3' -> weight values.
        epsilon: Division guard to prevent division by zero.

    Returns:
        Composite HiFD score in [0, 1].
    """
    components = {"P": P_bar, "Q": Q, "U1": U1, "U2": U2, "U3": U3}
    active = {k: v for k, v in components.items() if v is not None}

    if not active:
        return 0.0

    w_active = {k: weights[k] for k in active}

    # Auto-renormalize weights to sum to 1
    w_sum = sum(w_active.values())
    if w_sum <= 0:
        return 0.0
    w_active = {k: v / w_sum for k, v in w_active.items()}

    num = sum(w_active.values())
    den = sum(w_active[k] / (active[k] + epsilon) for k in active)
    return num / den


def load_profiles(profiles_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load application profiles from YAML.

    Returns:
        Dict mapping profile_name -> weight dict.
    """
    with open(profiles_path, "r") as f:
        raw = yaml.safe_load(f)

    profiles = {}
    for name, weights in raw.items():
        profiles[name] = weights
    return profiles


def compute_all_profiles(
    P_bar: Optional[float],
    Q: Optional[float],
    U1: Optional[float],
    U2: Optional[float],
    U3: Optional[float],
    profiles: Dict[str, Dict[str, float]],
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute HiFD under all application profiles.

    Returns:
        Dict mapping profile_name -> composite score.
    """
    results = {}
    for name, weights in profiles.items():
        results[name] = hifd(P_bar, Q, U1, U2, U3, weights, epsilon)
    return results
