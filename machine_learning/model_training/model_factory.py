"""Factory for swappable ML models in the ordinal-stage classification pipeline.

Every model implementation in `models.py` exposes the same API:
    fit(X, y), predict(X), predict_proba(X), get_feature_importance(importance_type='gain'),
    classes_, label_encoder_, feature_names_

so they can drop into the existing nested-CV pipeline interchangeably.
"""

from __future__ import annotations

from typing import Dict, Type


def get_model_class(model_type: str) -> Type:
    """Return the model class for a given short name."""
    from machine_learning.model_training.models import (
        OrdinalLightGBM,
        MulticlassLightGBM,
        OrdinalRandomForest,
        OrdinalElasticNetLR,
        OrdinalRidge,
        OrdinalRidgeVarFilter,
        OrdinalMLP,
    )

    registry: Dict[str, Type] = {
        "ordinal_lgb":     OrdinalLightGBM,
        "multiclass_lgb":  MulticlassLightGBM,
        "random_forest":   OrdinalRandomForest,
        "elastic_net_lr":  OrdinalElasticNetLR,
        "ridge_continuous": OrdinalRidge,
        "ridge_var5k":     OrdinalRidgeVarFilter,
        "mlp":             OrdinalMLP,
    }
    if model_type not in registry:
        raise ValueError(
            f"Unknown model_type={model_type!r}. "
            f"Available: {sorted(registry)}"
        )
    return registry[model_type]


# Per-model FIXED_PARAMS (passed to constructor alongside tuned `best_params`).
# Only OrdinalLightGBM needs the LightGBM-specific fixed params; the others
# carry no fixed extras (their defaults are applied inside __init__).
MODEL_FIXED_PARAMS: Dict[str, Dict] = {
    "ordinal_lgb":      {"class_weight": "balanced", "monotone_constraints": True},
    "multiclass_lgb":   {"class_weight": "balanced"},
    "random_forest":    {"class_weight": "balanced"},
    "elastic_net_lr":   {"class_weight": "balanced"},
    "ridge_continuous": {},
    "ridge_var5k":      {"n_top_var": 5000},
    "mlp":              {},
}


# Per-model Optuna param spaces (used by hyperparameter_tuning.py). When the
# user passes `--n-trials 1` the survey uses each model's defaults only.
def get_param_space(model_type: str):
    """Return a callable trial->params for the given model_type."""
    if model_type == "ordinal_lgb":
        def space(trial):
            return {
                "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
                "max_depth":       trial.suggest_int("max_depth", 3, 8),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=50),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "lambda_l1":       trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
                "lambda_l2":       trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            }
        return space

    if model_type == "multiclass_lgb":
        def space(trial):
            return {
                "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "lambda_l2":       trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            }
        return space

    if model_type == "random_forest":
        def space(trial):
            return {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=200),
                "max_depth":        trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1]),
            }
        return space

    if model_type == "elastic_net_lr":
        def space(trial):
            return {
                "C":         trial.suggest_float("C", 1e-3, 10, log=True),
                "l1_ratio":  trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter":  500,
            }
        return space

    if model_type == "ridge_continuous":
        def space(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True),
            }
        return space

    if model_type == "ridge_var5k":
        def space(trial):
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True),
            }
        return space

    if model_type == "mlp":
        def space(trial):
            return {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
                "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "n_epochs":    trial.suggest_int("n_epochs", 20, 100, step=20),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
            }
        return space

    raise ValueError(f"No param space defined for model_type={model_type!r}")
