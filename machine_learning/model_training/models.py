"""
Machine learning models for developmental stage classification.
Includes ordinal and binary classifiers with class balancing.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class OrdinalLogisticRegression(BaseEstimator, ClassifierMixin):
    """Cumulative link ordinal logistic regression with Elastic Net penalty."""

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        class_weight: Union[str, Dict] = 'balanced',
        seed: int = 42
    ):
        """
        Initialize ordinal logistic regression.

        Args:
            alpha: Regularization strength (inverse of C)
            l1_ratio: L1 ratio for Elastic Net
            max_iter: Maximum iterations
            class_weight: Class weight strategy
            seed: Random seed
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed

        self.models_ = []
        self.classes_ = None
        self.n_classes_ = 0
        self.label_encoder_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit ordinal regression using cumulative link approach.

        Args:
            X: Feature matrix
            y: Ordinal target variable

        Returns:
            self
        """
        # Encode labels to ensure ordinal structure
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Compute class weights
        if self.class_weight == 'balanced':
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(y_encoded), y=y_encoded
            )
            class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
        else:
            class_weight_dict = self.class_weight

        # Fit binary classifiers for each threshold
        self.models_ = []
        for k in range(self.n_classes_ - 1):
            # Create binary labels: 0 if y <= k, 1 if y > k
            y_binary = (y_encoded > k).astype(int)

            # Compute sample weights for this binary problem
            sample_weights = np.ones(len(y_binary))
            if class_weight_dict:
                for class_idx, weight in class_weight_dict.items():
                    if class_idx <= k:
                        sample_weights[y_encoded == class_idx] = weight
                    else:
                        sample_weights[y_encoded == class_idx] = weight

            # Fit binary classifier
            model = LogisticRegression(
                penalty='elasticnet' if self.l1_ratio < 1.0 else 'l1',
                solver='saga',
                C=1.0 / self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=self.seed + k
            )

            model.fit(X, y_binary, sample_weight=sample_weights)
            self.models_.append(model)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        cumulative_probs = np.zeros((n_samples, self.n_classes_))

        # Get cumulative probabilities
        for k in range(self.n_classes_ - 1):
            # Probability of being in class > k
            prob_greater = self.models_[k].predict_proba(X)[:, 1]
            cumulative_probs[:, k + 1] = prob_greater

        # Convert cumulative to class probabilities
        class_probs = np.zeros((n_samples, self.n_classes_))
        class_probs[:, 0] = 1 - cumulative_probs[:, 1]
        for k in range(1, self.n_classes_ - 1):
            class_probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k + 1]
        class_probs[:, -1] = cumulative_probs[:, -1]

        # Ensure probabilities are valid
        class_probs = np.clip(class_probs, 0, 1)
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)

        return class_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ordinal classes.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)


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

        # Fit binary models for each threshold
        self.models_ = []
        for k in range(self.n_classes_ - 1):
            # Create binary labels
            y_binary = (y_encoded > k).astype(int)

            # Compute class weights
            if self.class_weight == 'balanced':
                pos_weight = len(y_binary) / (2 * np.sum(y_binary))
                scale_pos_weight = pos_weight
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

            # Train model
            train_data = lgb.Dataset(X_df, label=y_binary)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
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


class StageClassifier:
    """Wrapper class for stage classification with automatic model selection."""

    def __init__(
        self,
        n_classes: int,
        model_type: str = 'auto',
        **model_params
    ):
        """
        Initialize stage classifier.

        Args:
            n_classes: Number of developmental stages
            model_type: Model type ('ordinal_lr', 'ordinal_lgb', 'binary_lr', 'auto')
            **model_params: Parameters passed to the model
        """
        self.n_classes = n_classes
        self.model_type = model_type
        self.model_params = model_params
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the appropriate model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            self
        """
        # Auto-select model type based on number of classes
        if self.model_type == 'auto':
            if self.n_classes == 2:
                self.model_type = 'binary_lr'
            elif self.n_classes >= 3:
                self.model_type = 'ordinal_lr'

        # Initialize model
        if self.model_type == 'ordinal_lr':
            self.model_ = OrdinalLogisticRegression(**self.model_params)
        elif self.model_type == 'ordinal_lgb':
            self.model_ = OrdinalLightGBM(**self.model_params)
        elif self.model_type == 'binary_lr':
            # Set default l1_ratio if not provided for elasticnet
            lr_params = self.model_params.copy()
            if 'penalty' not in lr_params or lr_params.get('penalty') == 'elasticnet':
                if 'l1_ratio' not in lr_params:
                    lr_params['l1_ratio'] = 0.5
            self.model_ = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                **lr_params
            )
        elif self.model_type == 'svm':
            self.model_ = SVC(
                probability=True,
                kernel='rbf',
                **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fit model
        self.model_.fit(X, y)
        logger.info(f"Fitted {self.model_type} for {self.n_classes}-class classification")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict:
        """Get model parameters."""
        params = {'n_classes': self.n_classes, 'model_type': self.model_type}
        params.update(self.model_params)
        return params

    def set_params(self, **params) -> 'StageClassifier':
        """Set model parameters."""
        for key, value in params.items():
            if key in ['n_classes', 'model_type']:
                setattr(self, key, value)
            else:
                self.model_params[key] = value
        return self