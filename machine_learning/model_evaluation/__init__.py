"""
Model evaluation module for pig developmental stage classification.

Provides evaluation metrics for model assessment.
"""

from .evaluation import (
    MetricCalculator,
    calculate_ordinal_metrics,
    calculate_near_miss_accuracy,
    calculate_all_metrics,
    bootstrap_metrics
)

__all__ = [
    'MetricCalculator',
    'calculate_ordinal_metrics',
    'calculate_near_miss_accuracy',
    'calculate_all_metrics',
    'bootstrap_metrics',
]
