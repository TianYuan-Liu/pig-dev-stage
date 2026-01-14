"""
Feature engineering module for pig developmental stage classification.

Provides feature selection methods for high-dimensional gene expression data.
"""

from .fast_feature_selection import FastFeatureSelector

__all__ = [
    'FastFeatureSelector'
]

