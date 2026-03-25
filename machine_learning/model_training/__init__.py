"""
Model training module for pig developmental stage classification.

Provides ordinal classification models and Optuna-based hyperparameter tuning.
"""

from .models import OrdinalLightGBM


def __getattr__(name):
    """Lazy-import hyperparameter tuning symbols (requires optuna)."""
    _tuning_names = {
        'run_tuning', 'create_study', 'create_objective',
        'sample_params', 'compute_param_stability',
        'SEARCH_SPACE', 'FIXED_PARAMS',
    }
    if name in _tuning_names:
        from . import hyperparameter_tuning as _ht
        return getattr(_ht, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

