"""
Feature Selection Validation Analysis Package

This package contains tools for validating whether LightGBM feature selection
captures real biological signal or is overfitting to noise.

Modules:
    - feature_stability: Multi-seed stability analysis
    - expression_correlation: Expression-stage correlation analysis
    - alternative_selection: Comparison with other selection methods
    - run_all_analyses: Master script to run all analyses
"""

from .feature_stability import run_stability_analysis
from .expression_correlation import run_correlation_analysis
from .alternative_selection import run_method_comparison
from .run_all_analyses import run_all_analyses

__all__ = [
    'run_stability_analysis',
    'run_correlation_analysis',
    'run_method_comparison',
    'run_all_analyses'
]
