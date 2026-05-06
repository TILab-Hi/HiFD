"""
POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition
"""

from .model import POSTER, load_poster_model, EXPRESSION_LABELS_7CLASS

__all__ = ['POSTER', 'load_poster_model', 'EXPRESSION_LABELS_7CLASS']
