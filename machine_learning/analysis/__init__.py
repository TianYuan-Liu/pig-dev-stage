"""
Feature Selection Validation Analysis Package

This package contains tools for validating whether LightGBM feature selection
captures real biological signal or is overfitting to noise.

Modules:
    - seed_robustness: Multi-seed performance + feature stability analysis
    - feature_stability: Backward-compatible wrapper for seed_robustness
    - expression_correlation: Expression-stage correlation analysis
    - run_all_analyses: Master script to run all analyses
"""

from .feature_stability import run_stability_analysis
from .seed_robustness import run_seed_robustness
from .expression_correlation import run_correlation_analysis
from .run_all_analyses import run_all_analyses
from .within_project_validation import run_within_project_validation

__all__ = [
    'run_stability_analysis',
    'run_seed_robustness',
    'run_correlation_analysis',
    'run_all_analyses',
    'run_within_project_validation',
]
