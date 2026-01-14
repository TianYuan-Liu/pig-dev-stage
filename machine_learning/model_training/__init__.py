"""
Model training module for pig developmental stage classification.

Provides ordinal classification models for developmental stage prediction.
"""

from .models import OrdinalLightGBM

__all__ = [
    'OrdinalLightGBM'
]

