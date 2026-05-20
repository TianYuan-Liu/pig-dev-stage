"""
Machine learning models for developmental stage classification.
LightGBM-based ordinal classifier with class balancing.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

logger = logging.getLogger(__name__)


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

        # One consistent stratified validation split for all thresholds.
        # Note: when training the final model on ALL data, this 80/20 internal
        # split means 20% of samples are used only for early stopping, not for
        # gradient updates. This is a deliberate tradeoff: we accept slightly
        # less training data in exchange for automatic iteration selection.
        # Stratified split ensures each class is represented in both partitions.
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        train_idx, val_idx = next(sss.split(X_df, y_encoded))

        X_train_split = X_df.iloc[train_idx] if isinstance(X_df, pd.DataFrame) else X_df[train_idx]
        X_val_split = X_df.iloc[val_idx] if isinstance(X_df, pd.DataFrame) else X_df[val_idx]

        # Fit binary models for each threshold
        self.models_ = []
        for k in range(self.n_classes_ - 1):
            # Create binary labels
            y_binary = (y_encoded > k).astype(int)

            # Check if binary split is degenerate (too few of either class
            # for a meaningful 80/20 stratified split)
            n_pos = int(y_binary.sum())
            n_neg = len(y_binary) - n_pos
            degenerate = n_pos < 2 or n_neg < 2

            if degenerate:
                # Train on ALL data with fixed n_estimators, no early stopping
                if self.class_weight == 'balanced':
                    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
                else:
                    scale_pos_weight = 1.0

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

                logger.warning(
                    f"Threshold {k}: degenerate split (n_pos={n_pos}, n_neg={n_neg}). "
                    f"Training on all data with {self.n_estimators} iterations, no early stopping."
                )
                train_data = lgb.Dataset(X_df, label=y_binary)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.n_estimators,
                    callbacks=[lgb.log_evaluation(0)]
                )
            else:
                # Normal path: 80/20 split with early stopping
                if self.class_weight == 'balanced':
                    n_pos_train = np.sum(y_binary[train_idx])
                    n_neg_train = len(train_idx) - n_pos_train
                    scale_pos_weight = n_neg_train / n_pos_train if n_pos_train > 0 else 1.0
                else:
                    scale_pos_weight = 1.0

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

                y_train_k, y_val_k = y_binary[train_idx], y_binary[val_idx]

                train_data = lgb.Dataset(X_train_split, label=y_train_k)
                val_data = lgb.Dataset(X_val_split, label=y_val_k, reference=train_data)
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

        # Enforce monotonicity: P(Y>k) must be <= P(Y>k-1)
        for k in range(2, self.n_classes_):
            cumulative_probs[:, k] = np.minimum(cumulative_probs[:, k], cumulative_probs[:, k - 1])

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


# ============================================================================
# Additional model wrappers for the ML-framework autoresearch sweep
# All conform to the same interface as OrdinalLightGBM:
#   - fit(X, y) -> self
#   - predict(X) -> labels (original encoding)
#   - predict_proba(X) -> probability matrix
#   - get_feature_importance(importance_type='gain') -> pd.Series indexed by feature
#   - classes_, label_encoder_, feature_names_ attributes
# ============================================================================


class _OrdinalBase(BaseEstimator, ClassifierMixin):
    """Mixin handling the LabelEncoder + feature_name plumbing so each
    subclass only needs to implement `_fit_inner`, `_predict_proba_inner`
    and `_feature_importance_inner`."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _fit_setup(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X)
        self.feature_names_ = X_df.columns.tolist()
        return X_df, y_enc

    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred_enc = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_enc)


# ----------------------------------------------------------------------------

class MulticlassLightGBM(_OrdinalBase):
    """LightGBM multiclass (no Frank-Hall reduction). Same hyperparams as
    OrdinalLightGBM but with objective='multiclass'."""

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
        seed: int = 42,
    ):
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
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        X_df, y_enc = self._fit_setup(X, y)
        params = {
            "objective": "multiclass",
            "num_class": self.n_classes_,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "seed": self.seed,
            "verbose": -1,
        }
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y_enc)
        else:
            sw = None
        dtrain = lgb.Dataset(X_df.values, label=y_enc, weight=sw,
                             feature_name=self.feature_names_,
                             free_raw_data=False)
        self.model_ = lgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_.predict(X_df.values)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        names = self.model_.feature_name()
        scores = self.model_.feature_importance(importance_type=importance_type)
        s = pd.Series(scores, index=names).sort_values(ascending=False)
        return s


# ----------------------------------------------------------------------------

class OrdinalRandomForest(_OrdinalBase):
    """sklearn RandomForestClassifier; multiclass by default. Feature importance
    via the natural impurity-based scores."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 12,
        min_samples_leaf: int = 2,
        max_features="sqrt",
        class_weight: Union[str, Dict] = "balanced",
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        X_df, y_enc = self._fit_setup(X, y)
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(X_df.values, y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_.predict_proba(X_df.values)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)


# ----------------------------------------------------------------------------

class OrdinalElasticNetLR(_OrdinalBase):
    """sklearn LogisticRegression with elastic-net penalty (saga solver).
    Feature importance = mean absolute coefficient across classes."""

    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 500,
        class_weight: Union[str, Dict] = "balanced",
        seed: int = 42,
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        X_df, y_enc = self._fit_setup(X, y)
        # saga + elasticnet requires standardised features for stability;
        # we leave preprocessing to the caller's pipeline (already log-CPM).
        self.model_ = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(X_df.values, y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_.predict_proba(X_df.values)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        coefs = np.abs(self.model_.coef_).mean(axis=0)
        return pd.Series(coefs, index=self.feature_names_).sort_values(
            ascending=False)


# ----------------------------------------------------------------------------

class OrdinalRidge(_OrdinalBase):
    """sklearn Ridge regression on integer-coded stage. Predictions are
    converted to class probabilities via Gaussian distance softmax."""

    def __init__(
        self,
        alpha: float = 1.0,
        seed: int = 42,
        proba_sigma: float = 0.8,
    ):
        self.alpha = alpha
        self.seed = seed
        self.proba_sigma = proba_sigma
        self.model_ = None

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        X_df, y_enc = self._fit_setup(X, y)
        self.model_ = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model_.fit(X_df.values, y_enc.astype(float))
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_hat = self.model_.predict(X_df.values)
        # Gaussian distance softmax over class centres 0..n_classes-1
        ks = np.arange(self.n_classes_)
        dist = (y_hat[:, None] - ks[None, :]) ** 2
        logits = -dist / (2.0 * (self.proba_sigma ** 2))
        # softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return pd.Series(
            np.abs(self.model_.coef_), index=self.feature_names_,
        ).sort_values(ascending=False)


# ----------------------------------------------------------------------------

class OrdinalRidgeVarFilter(_OrdinalBase):
    """Ridge with an internal variance filter that keeps the top-N highest-
    variance features before fitting. Implemented inside the class so the
    rest of the pipeline (preprocessor, CV harness) remains unchanged."""

    def __init__(
        self,
        alpha: float = 1.0,
        n_top_var: int = 5000,
        seed: int = 42,
        proba_sigma: float = 0.8,
    ):
        self.alpha = alpha
        self.n_top_var = n_top_var
        self.seed = seed
        self.proba_sigma = proba_sigma
        self.model_ = None
        self._selected: Optional[List[str]] = None

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        X_df, y_enc = self._fit_setup(X, y)
        var = X_df.var(axis=0)
        n_keep = min(int(self.n_top_var), len(var))
        topN = var.sort_values(ascending=False).head(n_keep).index
        self._selected = list(topN)
        # Override feature_names_ so get_feature_importance returns only the
        # selected genes (matching the Ridge.coef_ shape).
        self.feature_names_ = self._selected
        X_sel = X_df[self._selected]
        self.model_ = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model_.fit(X_sel.values, y_enc.astype(float))
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_sel = X_df.reindex(columns=self._selected, fill_value=0.0)
        y_hat = self.model_.predict(X_sel.values)
        ks = np.arange(self.n_classes_)
        dist = (y_hat[:, None] - ks[None, :]) ** 2
        logits = -dist / (2.0 * (self.proba_sigma ** 2))
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return pd.Series(
            np.abs(self.model_.coef_), index=self._selected,
        ).sort_values(ascending=False)


# ----------------------------------------------------------------------------

class OrdinalMLP(_OrdinalBase):
    """PyTorch 3-layer MLP, multinomial output. Feature importance = mean
    absolute input-layer weight aggregated across output neurons."""

    def __init__(
        self,
        hidden_size: int = 128,
        lr: float = 1e-3,
        n_epochs: int = 40,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.seed = seed
        self.model_ = None
        self._device = "cpu"

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.n_classes_),
        ).to(self._device)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        X_df, y_enc = self._fit_setup(X, y)
        # Standardise input (per-feature z-score) for stable training
        self._mean = X_df.values.mean(axis=0)
        self._std = X_df.values.std(axis=0) + 1e-8
        X_std = (X_df.values - self._mean) / self._std
        # Class weights for balanced loss
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(self.n_classes_),
                                  y=y_enc)
        weight_tensor = torch.tensor(cw, dtype=torch.float32, device=self._device)
        self._build(X_std.shape[1])
        Xt = torch.tensor(X_std, dtype=torch.float32, device=self._device)
        yt = torch.tensor(y_enc, dtype=torch.long, device=self._device)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        self.model_.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
        import torch
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_std = (X_df.values - self._mean) / self._std
        with torch.no_grad():
            xt = torch.tensor(X_std, dtype=torch.float32, device=self._device)
            logits = self.model_(xt)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        # Use absolute input-layer weights aggregated across output neurons
        first_layer = self.model_[0]  # nn.Linear(n_features, hidden_size)
        W = first_layer.weight.detach().cpu().numpy()  # (hidden, in)
        imp = np.abs(W).mean(axis=0)
        return pd.Series(imp, index=self.feature_names_).sort_values(
            ascending=False)