"""Privacy score computation for the HiFD metric.

P_single computes per-backbone privacy from face embedding cosine similarity.
P_bar computes the ensemble average across multiple recognition backbones.
"""

import numpy as np
from typing import Dict, List, Optional


def P_single(emb_orig: np.ndarray, emb_deid: np.ndarray) -> float:
    """
    Privacy score for a single recognition backbone and single sample.

    Converts cosine similarity to a privacy measure in [0, 1]:
      P = 0.5 * (1 - cos_sim)

    Args:
        emb_orig: L2-normalized face embedding from original image.
        emb_deid: L2-normalized face embedding from de-identified image.

    Returns:
        Privacy score in [0, 1]. Higher = more private.
    """
    e_o = emb_orig / (np.linalg.norm(emb_orig) + 1e-12)
    e_d = emb_deid / (np.linalg.norm(emb_deid) + 1e-12)
    cos_sim = float(np.clip(np.dot(e_o, e_d), -1.0, 1.0))
    return 0.5 * (1.0 - cos_sim)


def batch_P_single(
    emb_orig: np.ndarray, emb_deid: np.ndarray
) -> np.ndarray:
    """
    Vectorized privacy score for a single backbone.

    Args:
        emb_orig: (N, D) array of original embeddings.
        emb_deid: (N, D) array of de-identified embeddings.

    Returns:
        (N,) array of privacy scores.
    """
    # L2 normalize
    norm_o = np.linalg.norm(emb_orig, axis=1, keepdims=True) + 1e-12
    norm_d = np.linalg.norm(emb_deid, axis=1, keepdims=True) + 1e-12
    e_o = emb_orig / norm_o
    e_d = emb_deid / norm_d
    cos_sim = np.clip(np.sum(e_o * e_d, axis=1), -1.0, 1.0)
    return 0.5 * (1.0 - cos_sim)


def P_bar(
    backbone_scores: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Ensemble privacy score averaged across backbones.

    Args:
        backbone_scores: dict mapping backbone name -> (N,) array of per-sample
                         privacy scores from that backbone.

    Returns:
        (N,) array of ensemble privacy scores.
    """
    arrays = list(backbone_scores.values())
    stacked = np.stack(arrays, axis=0)  # (M, N)
    return np.mean(stacked, axis=0)     # (N,)


def compute_privacy_for_method(
    orig_embeddings: Dict[str, Dict[str, np.ndarray]],
    deid_embeddings: Dict[str, Dict[str, np.ndarray]],
    sample_ids: List[str],
    backbone_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-backbone privacy scores for a method across all samples.

    Args:
        orig_embeddings: {backbone_name: {sample_id: embedding_array}}
        deid_embeddings: {backbone_name: {sample_id: embedding_array}}
        sample_ids: list of sample IDs to evaluate.
        backbone_names: subset of backbones to use (default: all available).

    Returns:
        Dict mapping backbone_name -> (N,) array of per-sample privacy scores.
    """
    if backbone_names is None:
        backbone_names = sorted(orig_embeddings.keys())

    results = {}
    for bname in backbone_names:
        orig_embs = orig_embeddings[bname]
        deid_embs = deid_embeddings[bname]
        scores = np.array([
            P_single(orig_embs[sid], deid_embs[sid])
            for sid in sample_ids
        ])
        results[bname] = scores

    return results
