"""
Model evaluation module for pig developmental stage classification.

Provides evaluation metrics and statistical tests for model assessment.
"""

from .evaluation import (
    MetricCalculator,
    calculate_ordinal_metrics,
    calculate_near_miss_accuracy,
    calculate_all_metrics,
    bootstrap_metrics
)
from .calibration import (
    ProbabilityCalibrator,
    ConformalPredictor,
    calculate_ece
)
from .statistical_tests import (
    PermutationTest,
    MultipleTestingCorrection,
    BootstrapSignificanceTest,
    McNemarTest
)

__all__ = [
    # Evaluation
    'MetricCalculator',
    'calculate_ordinal_metrics',
    'calculate_near_miss_accuracy',
    'calculate_all_metrics',
    'bootstrap_metrics',
    # Calibration
    'ProbabilityCalibrator',
    'ConformalPredictor',
    'calculate_ece',
    # Statistical tests
    'PermutationTest',
    'MultipleTestingCorrection',
    'BootstrapSignificanceTest',
    'McNemarTest'
]

