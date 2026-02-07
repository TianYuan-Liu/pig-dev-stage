"""
Model training module for pig developmental stage classification.

Provides ordinal classification models and Optuna-based hyperparameter tuning.
"""

from .models import OrdinalLightGBM
from .hyperparameter_tuning import (
    run_tuning,
    create_study,
    create_objective,
    sample_params,
    compute_param_stability,
    SEARCH_SPACE,
    FIXED_PARAMS,
)

__all__ = [
    'OrdinalLightGBM',
    'run_tuning',
    'create_study',
    'create_objective',
    'sample_params',
    'compute_param_stability',
    'SEARCH_SPACE',
    'FIXED_PARAMS',
]

