"""
Machine learning models for developmental stage classification.
LightGBM-based ordinal classifier with class balancing.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class OrdinalLightGBM(BaseEstimator, ClassifierMixin):
    """LightGBM with ordinal reduction (CORAL/CORN approach)."""

    def __init__(
        self,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.9,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        class_weight: Union[str, Dict] = 'balanced',
        monotone_constraints: bool = True,
        seed: int = 42
    ):
        """
        Initialize ordinal LightGBM.

        Args:
            num_leaves: Maximum number of leaves
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            min_data_in_leaf: Minimum data in leaf
            feature_fraction: Feature fraction for bagging
            bagging_fraction: Data fraction for bagging
            lambda_l1: L1 regularization
            lambda_l2: L2 regularization
            class_weight: Class weight strategy
            monotone_constraints: Apply monotonic constraints
            seed: Random seed
        """
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.class_weight = class_weight
        self.monotone_constraints = monotone_constraints
        self.seed = seed

        self.models_ = []
        self.classes_ = None
        self.n_classes_ = 0
        self.label_encoder_ = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit ordinal LightGBM using binary reduction.

        Args:
            X: Feature matrix
            y: Ordinal target variable

        Returns:
            self
        """
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)
        self.feature_names_ = X_df.columns.tolist()

        # Fit binary models for each threshold
        self.models_ = []
        for k in range(self.n_classes_ - 1):
            # Create binary labels
            y_binary = (y_encoded > k).astype(int)

            # Compute class weights (correct formula: neg_count / pos_count)
            if self.class_weight == 'balanced':
                n_pos = np.sum(y_binary)
                n_neg = len(y_binary) - n_pos
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            else:
                scale_pos_weight = 1.0

            # LightGBM parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'bagging_freq': 5,
                'lambda_l1': self.lambda_l1,
                'lambda_l2': self.lambda_l2,
                'min_data_in_leaf': self.min_data_in_leaf,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.seed + k,
                'verbosity': -1
            }

            # Split data for early stopping validation (80/20 split)
            n_train = int(0.8 * len(y_binary))
            indices = np.random.RandomState(self.seed + k).permutation(len(y_binary))
            train_idx, val_idx = indices[:n_train], indices[n_train:]

            X_train_k = X_df.iloc[train_idx] if isinstance(X_df, pd.DataFrame) else X_df[train_idx]
            X_val_k = X_df.iloc[val_idx] if isinstance(X_df, pd.DataFrame) else X_df[val_idx]
            y_train_k, y_val_k = y_binary[train_idx], y_binary[val_idx]

            # Train model with validation set for early stopping
            train_data = lgb.Dataset(X_train_k, label=y_train_k)
            val_data = lgb.Dataset(X_val_k, label=y_val_k, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            self.models_.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)

        n_samples = X_df.shape[0]
        cumulative_probs = np.zeros((n_samples, self.n_classes_))

        # Get cumulative probabilities
        for k in range(self.n_classes_ - 1):
            prob_greater = self.models_[k].predict(X_df, num_iteration=self.models_[k].best_iteration)
            cumulative_probs[:, k + 1] = prob_greater

        # Convert to class probabilities
        class_probs = np.zeros((n_samples, self.n_classes_))
        class_probs[:, 0] = 1 - cumulative_probs[:, 1]
        for k in range(1, self.n_classes_ - 1):
            class_probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k + 1]
        class_probs[:, -1] = cumulative_probs[:, -1]

        # Apply monotonic post-processing if enabled
        if self.monotone_constraints:
            class_probs = self._apply_monotonic_constraints(class_probs)

        # Ensure valid probabilities
        class_probs = np.clip(class_probs, 0, 1)
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)

        return class_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ordinal classes.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        Aggregate feature importance across binary LightGBM models.

        Args:
            importance_type: LightGBM importance type (e.g., 'gain', 'split')

        Returns:
            Series of mean importance scores indexed by feature name.
        """
        if not self.models_:
            raise ValueError("Model not fitted")

        importance_frames = []
        for model in self.models_:
            scores = model.feature_importance(importance_type=importance_type)
            names = model.feature_name()
            importance_frames.append(pd.Series(scores, index=names))

        importance = pd.concat(importance_frames, axis=1).mean(axis=1)
        importance = importance.sort_values(ascending=False)
        return importance

    def _apply_monotonic_constraints(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply monotonic constraints to ensure ordinal consistency.

        Args:
            probs: Probability matrix

        Returns:
            Constrained probabilities
        """
        n_samples = probs.shape[0]

        for i in range(n_samples):
            # Find the mode
            mode_idx = np.argmax(probs[i])

            # Enforce unimodality: probabilities should decrease from mode
            for j in range(mode_idx - 1, -1, -1):
                if probs[i, j] > probs[i, j + 1]:
                    probs[i, j] = probs[i, j + 1]

            for j in range(mode_idx + 1, self.n_classes_):
                if probs[i, j] > probs[i, j - 1]:
                    probs[i, j] = probs[i, j - 1]

        return probs