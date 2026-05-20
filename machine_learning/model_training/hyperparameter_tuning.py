"""
Hyperparameter tuning via Optuna for ordinal LightGBM.

Provides search space definitions, objective functions for inner CV,
and study creation utilities used by the nested CV loop in run_pipeline.py.
"""

import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter tuning. "
        "Install it with: pip install optuna"
    )

from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.model_training.model_factory import (
    get_model_class, MODEL_FIXED_PARAMS, get_param_space,
)
from machine_learning.utils.helpers import to_samples_x_genes_df

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "num_leaves": {"type": "int", "low": 15, "high": 127},      # folds selected 29–107
    "max_depth": {"type": "int", "low": 3, "high": 8},           # was 12; all folds ≤ 7
    "min_data_in_leaf": {"type": "int", "low": 20, "high": 50},  # was 5; all folds ≥ 36
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},  # floor raised from 50
    "feature_fraction": {"type": "float", "low": 0.4, "high": 1.0},
    "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},  # was 0.5; all folds ≥ 0.76
    "lambda_l1": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "lambda_l2": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

# Fixed params (not tuned)
FIXED_PARAMS = {
    "class_weight": "balanced",
    "monotone_constraints": True,
}


def sample_params(trial: "optuna.Trial", n_samples: Optional[int] = None) -> Dict[str, Any]:
    """Sample hyperparameters from the search space using an Optuna trial.

    When *n_samples* is provided the ``min_data_in_leaf`` range is clamped so
    that the upper bound never exceeds half the smallest inner-fold training
    set (ensuring LightGBM can always create at least one split).
    """
    params = {}
    for name, spec in SEARCH_SPACE.items():
        low, high = spec["low"], spec["high"]

        # Adapt min_data_in_leaf for small datasets so the tree can
        # always make at least one split in the inner-fold training set.
        if name == "min_data_in_leaf" and n_samples is not None:
            max_leaf = max(5, n_samples // 4)
            if max_leaf < high:
                low = max(5, max_leaf // 4)
                high = max_leaf

        if spec["type"] == "int":
            kwargs = {"name": name, "low": low, "high": high}
            if "step" in spec:
                kwargs["step"] = spec["step"]
            params[name] = trial.suggest_int(**kwargs)
        elif spec["type"] == "float":
            kwargs = {"name": name, "low": low, "high": high}
            if spec.get("log"):
                kwargs["log"] = True
            params[name] = trial.suggest_float(**kwargs)
    return params


# ---------------------------------------------------------------------------
# Helper: determine inner CV folds
# ---------------------------------------------------------------------------

def _determine_n_inner_folds(y, requested_k: int = 3) -> Tuple[int, str]:
    """Determine the number of inner CV folds based on class distribution."""
    min_samples = int(pd.Series(y).value_counts().min())
    if min_samples >= requested_k:
        return requested_k, f"{requested_k}-fold"
    elif min_samples >= 2:
        return min_samples, f"{min_samples}-fold (reduced from {requested_k})"
    else:
        return len(y), "leave-one-out"


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def create_objective(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: np.ndarray,
    n_inner_folds: int = 3,
    seed: int = 42,
    model_type: str = "ordinal_lgb",
) -> Callable:
    """
    Factory returning an Optuna objective that runs inner CV.

    Args:
        X_raw: Expression matrix (genes x samples) for the outer-fold training set.
        y: Labels for the outer-fold training set.
        sample_ids: Sample IDs for the outer-fold training set.
        gene_names: Gene names (row index of X_raw).
        n_inner_folds: Requested number of inner CV folds.
        seed: Random seed for reproducibility.

    Returns:
        An objective function compatible with ``study.optimize()``.
    """
    n_folds, _ = _determine_n_inner_folds(y, requested_k=n_inner_folds)

    X_T = X_raw.T  # samples x genes

    if n_folds == len(y):
        inner_cv = LeaveOneOut()
    else:
        inner_cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )

    n_samples = len(y)

    if model_type != "ordinal_lgb":
        custom_space = get_param_space(model_type)
        custom_fixed = MODEL_FIXED_PARAMS.get(model_type, {})
        model_cls = get_model_class(model_type)
    else:
        custom_space = None
        model_cls = OrdinalLightGBM

    def objective(trial: "optuna.Trial") -> float:
        if custom_space is not None:
            params = custom_space(trial)
        else:
            params = sample_params(trial, n_samples=n_samples)

        fold_scores: List[float] = []
        for fold_i, (train_idx, val_idx) in enumerate(inner_cv.split(X_T, y)):
            X_train_raw = X_raw.iloc[:, train_idx]
            X_val_raw = X_raw.iloc[:, val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            train_ids = sample_ids[train_idx]
            val_ids = sample_ids[val_idx]

            preprocessor = ExpressionPreprocessor()
            X_train_proc = preprocessor.fit_transform(X_train_raw)
            X_val_proc = preprocessor.transform(X_val_raw)

            X_train_df = to_samples_x_genes_df(
                X_train_proc, train_ids, preprocessor, gene_names
            )
            X_val_df = to_samples_x_genes_df(
                X_val_proc, val_ids, preprocessor, gene_names
            )

            fixed = custom_fixed if custom_space is not None else FIXED_PARAMS
            model = model_cls(**params, **fixed, seed=seed + fold_i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_df, y_train)

            y_pred = model.predict(X_val_df)
            ba = balanced_accuracy_score(y_val, y_pred)
            fold_scores.append(ba)

            # Report intermediate value for pruning
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------

def create_study(study_name: str, seed: int = 42) -> "optuna.Study":
    """
    Create an Optuna study with TPE sampler and median pruner.

    Args:
        study_name: Name for the study (e.g. "tissue_fold_1").
        seed: Random seed for the TPE sampler.

    Returns:
        An Optuna study configured for maximisation.
    """
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    return study


# ---------------------------------------------------------------------------
# Convenience: run a full tuning study and return results
# ---------------------------------------------------------------------------

def run_tuning(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: np.ndarray,
    study_name: str = "tuning",
    n_trials: int = 25,
    n_inner_folds: int = 3,
    seed: int = 42,
    timeout: int = 600,
    model_type: str = "ordinal_lgb",
) -> Dict[str, Any]:
    """
    Run a complete Optuna tuning study.

    Returns a dict with keys:
        best_params, best_score, n_trials_completed, n_trials_pruned,
        tuning_duration_s
    """
    study = create_study(study_name=study_name, seed=seed)
    objective = create_objective(
        X_raw, y, sample_ids, gene_names,
        n_inner_folds=n_inner_folds, seed=seed, model_type=model_type,
    )

    t0 = time.time()
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout,
        show_progress_bar=False,
    )
    duration = time.time() - t0

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "n_trials_completed": n_completed,
        "n_trials_pruned": n_pruned,
        "tuning_duration_s": round(duration, 1),
    }


def compute_param_stability(best_params_per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute mean/std of each tuned parameter across outer folds.

    Args:
        best_params_per_fold: List of best_params dicts, one per fold.

    Returns:
        Dict mapping param name -> {mean, std, per_fold}.
    """
    if not best_params_per_fold:
        return {}

    all_keys = list(best_params_per_fold[0].keys())
    stability = {}
    for key in all_keys:
        values = [fp[key] for fp in best_params_per_fold]
        stability[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": [float(v) for v in values],
        }
    return stability
