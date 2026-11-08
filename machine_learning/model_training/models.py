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
# Additional model wrappers for the ML-framework benchmark sweep
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
        # CRASHER FIX: coerce to contiguous float32. The raw log-CPM matrix can
        # arrive as object/mixed dtype from the pipeline join, which trips
        # sklearn's tree builder ("Buffer dtype mismatch"). float32 is exact
        # enough for splits and halves memory on the 31.9k-feature matrix.
        self.model_.fit(np.ascontiguousarray(X_df.values, dtype=np.float32), y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_.predict_proba(
            np.ascontiguousarray(X_df.values, dtype=np.float32))

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
        max_iter: int = 200,
        tol: float = 1e-2,
        class_weight: Union[str, Dict] = "balanced",
        seed: int = 42,
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def fit(self, X, y):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        X_df, y_enc = self._fit_setup(X, y)
        # CRASHER/COST FIX: saga elastic-net does NOT converge in reasonable time
        # on 31,908 raw log-CPM features (the COST finding). We bound max_iter
        # (200) and relax tol (1e-2) so the fit terminates deterministically; the
        # residual ConvergenceWarning is suppressed because non-convergence here
        # is the documented behaviour, not an error. float32 + contiguous matrix.
        self.model_ = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(
                np.ascontiguousarray(X_df.values, dtype=np.float32), y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.model_.predict_proba(
            np.ascontiguousarray(X_df.values, dtype=np.float32))

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
        self._gene_std: Optional[pd.Series] = None

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        X_df, y_enc = self._fit_setup(X, y)
        self._gene_std = X_df.std(axis=0)
        self.model_ = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model_.fit(X_df.values, y_enc.astype(float))
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_hat = self.model_.predict(X_df.values)
        ks = np.arange(self.n_classes_)
        dist = (y_hat[:, None] - ks[None, :]) ** 2
        logits = -dist / (2.0 * (self.proba_sigma ** 2))
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return pd.Series(
            np.abs(self.model_.coef_), index=self.feature_names_,
        ).sort_values(ascending=False)


# ----------------------------------------------------------------------------

class OrdinalRidgeStdWeight(OrdinalRidge):
    """Identical Ridge fit and predictions as OrdinalRidge — only the
    feature importance is reweighted by per-gene std(x) of the training data.
    Equivalent to standardised regression coefficients: a gene's importance
    becomes the per-SD change in predicted stage produced by that gene.

    Internal pig BA is unchanged (fit/predict identical); only the top-K
    gene ranking shifts toward genes with larger expression range."""

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        coef = np.abs(self.model_.coef_)
        std = self._gene_std.reindex(self.feature_names_).fillna(0.0).values
        imp = coef * std
        return pd.Series(imp, index=self.feature_names_).sort_values(
            ascending=False)


# ----------------------------------------------------------------------------

class OrdinalPLS(_OrdinalBase):
    """sklearn PLSRegression on the integer-coded stage index.

    Mirrors OrdinalRidge: a continuous regression onto the ordinal stage index,
    converted to class probabilities via the same Gaussian-distance softmax.

    Differences from OrdinalRidge:
      * Per-gene z-score standardisation is done INSIDE the class, fit on the
        TRAIN matrix only (mean/scale stored), so the same fitted scaler is
        applied to the held-out / target matrix. PLS is sensitive to feature
        scaling, so this is required for stable latent components. (Note: PLS
        also internally centres X and y; explicit z-scoring puts every gene on
        a comparable scale before the SIMPLS/NIPALS decomposition.)
      * `n_components` (number of latent variables) is the tunable knob.
      * `get_feature_importance` returns the per-component-normalised VIP
        (Variable Importance in Projection), the canonical PLS gene-attribution.

    VIP_j = sqrt( p * sum_a (w_ja / ||w_a||)^2 * SSY_a / sum_a SSY_a )
      where w_a is the weight vector of component a (normalised to unit norm
      before squaring), and SSY_a is the response sum-of-squares explained by
      component a (= ||t_a||^2 * q_a^2 for a single response). With this
      normalisation, mean(VIP^2) == 1, so VIP > 1 flags above-average genes.
    """

    def __init__(
        self,
        n_components: int = 10,
        seed: int = 42,
        proba_sigma: float = 0.8,
    ):
        self.n_components = n_components
        self.seed = seed
        self.proba_sigma = proba_sigma
        self.model_ = None
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._vip: Optional[np.ndarray] = None

    def _standardize_fit(self, Xv: np.ndarray) -> np.ndarray:
        self._mean = Xv.mean(axis=0)
        # guard zero-variance genes (constant in this train fold)
        self._scale = Xv.std(axis=0)
        self._scale[self._scale == 0] = 1.0
        return (Xv - self._mean) / self._scale

    def _standardize_apply(self, Xv: np.ndarray) -> np.ndarray:
        return (Xv - self._mean) / self._scale

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        X_df, y_enc = self._fit_setup(X, y)
        Xv = np.ascontiguousarray(X_df.values, dtype=np.float64)
        Xz = self._standardize_fit(Xv)
        # cap n_components at the structurally valid maximum
        n_comp = int(min(self.n_components, Xz.shape[1], Xz.shape[0] - 1))
        n_comp = max(1, n_comp)
        self.model_ = PLSRegression(n_components=n_comp, scale=False)
        self.model_.fit(Xz, y_enc.astype(float))
        self._vip = self._compute_vip()
        return self

    def _compute_vip(self) -> np.ndarray:
        """Per-component-normalised VIP. See class docstring for the formula.

        Uses the fitted PLS attributes:
          x_weights_  : W  (p x A)   raw weight vectors per component
          x_scores_   : T  (n x A)   latent X scores
          y_loadings_ : Q  (k x A)   y loadings (k=1 here)
        SSY_a = ||t_a||^2 * q_a^2  is the response variance explained by comp a.
        """
        W = np.asarray(self.model_.x_weights_)          # (p, A)
        T = np.asarray(self.model_.x_scores_)           # (n, A)
        Q = np.asarray(self.model_.y_loadings_)         # (k, A); k=1
        p = W.shape[0]
        # SSY per component (single response -> q is scalar per component)
        q = Q.ravel()                                   # (A,)
        ssy = (T ** 2).sum(axis=0) * (q ** 2)           # (A,)
        ssy_total = ssy.sum()
        if ssy_total <= 0:
            return np.ones(p, dtype=float)
        # normalise each component's weight vector to unit norm BEFORE squaring
        w_norm = np.linalg.norm(W, axis=0)              # (A,)
        w_norm[w_norm == 0] = 1.0
        W_unit = W / w_norm[None, :]                    # (p, A)
        weighted = (W_unit ** 2) * (ssy / ssy_total)[None, :]   # (p, A)
        vip = np.sqrt(p * weighted.sum(axis=1))         # (p,)
        return vip

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xv = np.ascontiguousarray(X_df.values, dtype=np.float64)
        Xz = self._standardize_apply(Xv)
        y_hat = self.model_.predict(Xz).ravel()
        ks = np.arange(self.n_classes_)
        dist = (y_hat[:, None] - ks[None, :]) ** 2
        logits = -dist / (2.0 * (self.proba_sigma ** 2))
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return pd.Series(
            self._vip, index=self.feature_names_,
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


# ============================================================================
# DISCOVERED model families for the ML/DL benchmark model sweep.
# Each conforms to the same interface as OrdinalLightGBM (fit/predict/
# predict_proba/get_feature_importance + classes_/label_encoder_/feature_names_).
# Defensive against the known crashers: float dtype coercion for tree/instance
# models, and bounded iterations for linear solvers on ~31.9k features.
# ============================================================================


def _as_float32(X_df: pd.DataFrame) -> np.ndarray:
    """Contiguous float32 view — avoids the sklearn-tree dtype crash on the
    log-CPM matrix and roughly halves memory vs float64."""
    return np.ascontiguousarray(X_df.values, dtype=np.float32)


def _uniform_importance(feature_names) -> pd.Series:
    """Importance fallback for models without a native attribution (e.g. KNN,
    NB). Returns a flat series so downstream ranking code never crashes."""
    return pd.Series(1.0, index=feature_names).sort_values(ascending=False)


class _SklearnClassifierWrapper(_OrdinalBase):
    """Generic adapter: subclasses build `self.model_` in `_make_estimator()`.

    Handles label plumbing, float32 coercion, predict_proba (with a softmax
    fallback for estimators that only expose decision_function), and a
    pluggable feature-importance source.
    """

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y):
        X_df, y_enc = self._fit_setup(X, y)
        self._X_dtype = _as_float32(X_df)
        self.model_ = self._make_estimator()
        self.model_.fit(self._X_dtype, y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Xv = _as_float32(X_df)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(Xv)
        # decision_function -> softmax fallback (LinearSVC etc.)
        df = self.model_.decision_function(Xv)
        if df.ndim == 1:  # binary
            df = np.column_stack([-df, df])
        df = df - df.max(axis=1, keepdims=True)
        ex = np.exp(df)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        m = self.model_
        if hasattr(m, "feature_importances_"):
            return pd.Series(m.feature_importances_, index=self.feature_names_
                             ).sort_values(ascending=False)
        if hasattr(m, "coef_"):
            coef = np.abs(np.atleast_2d(m.coef_)).mean(axis=0)
            return pd.Series(coef, index=self.feature_names_
                             ).sort_values(ascending=False)
        return _uniform_importance(self.feature_names_)


# ----------------------------------------------------------------------------

class OrdinalHistGB(_SklearnClassifierWrapper):
    """sklearn HistGradientBoostingClassifier — histogram-binned GBDT, fast on
    wide matrices. Multiclass softmax output."""

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 200,
                 max_leaf_nodes: int = 31, l2_regularization: float = 0.0,
                 max_depth: Optional[int] = None, class_weight="balanced",
                 seed: int = 42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization, max_depth=self.max_depth,
            class_weight=self.class_weight, random_state=self.seed)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        # HistGB has no native importances; flat fallback keeps interface stable.
        return _uniform_importance(self.feature_names_)


class OrdinalGradientBoosting(_SklearnClassifierWrapper):
    """sklearn GradientBoostingClassifier (exact, deviance loss)."""

    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 200,
                 max_depth: int = 3, subsample: float = 1.0, seed: int = 42):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            learning_rate=self.learning_rate, n_estimators=self.n_estimators,
            max_depth=self.max_depth, subsample=self.subsample,
            random_state=self.seed)


class OrdinalExtraTrees(_SklearnClassifierWrapper):
    """sklearn ExtraTreesClassifier — randomized-split forest."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 12,
                 min_samples_leaf: int = 2, max_features="sqrt",
                 class_weight="balanced", seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, class_weight=self.class_weight,
            random_state=self.seed, n_jobs=-1)


class OrdinalLinearSVC(_SklearnClassifierWrapper):
    """sklearn LinearSVC (dual hinge). No predict_proba -> softmax over
    decision_function (handled in the base wrapper)."""

    def __init__(self, C: float = 1.0, max_iter: int = 5000,
                 class_weight="balanced", seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.svm import LinearSVC
        # dual='auto' picks the efficient solver for n_features >> n_samples.
        return LinearSVC(C=self.C, max_iter=self.max_iter,
                         class_weight=self.class_weight, dual="auto",
                         random_state=self.seed)


class OrdinalKNN(_SklearnClassifierWrapper):
    """sklearn KNeighborsClassifier. Importance is a flat fallback."""

    def __init__(self, n_neighbors: int = 15, weights: str = "distance",
                 p: int = 2, seed: int = 42):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                    weights=self.weights, p=self.p, n_jobs=-1)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return _uniform_importance(self.feature_names_)


class OrdinalLDA(_SklearnClassifierWrapper):
    """sklearn LinearDiscriminantAnalysis (lsqr + shrinkage for p >> n)."""

    def __init__(self, shrinkage="auto", seed: int = 42):
        self.shrinkage = shrinkage
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)


class OrdinalGaussianNB(_SklearnClassifierWrapper):
    """sklearn GaussianNB — fast diagonal-Gaussian baseline."""

    def __init__(self, var_smoothing: float = 1e-9, seed: int = 42):
        self.var_smoothing = var_smoothing
        self.seed = seed
        self.model_ = None

    def _make_estimator(self):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=self.var_smoothing)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        return _uniform_importance(self.feature_names_)


# ----------------------------------------------------------------------------
# Dimensionality-reduction-first pipelines (PCA -> {ridge, logistic}).
# PCA is fit on the (float32) training matrix; the downstream estimator sees a
# compact ~k-dim representation. This is the classic "denoise then classify"
# strategy for p >> n genomics data.
# ----------------------------------------------------------------------------

class OrdinalPCARidge(_OrdinalBase):
    """PCA -> Ridge regression on the ordinal index (same Gaussian-softmax
    probability head as OrdinalRidge)."""

    def __init__(self, n_components: int = 50, alpha: float = 1.0,
                 proba_sigma: float = 0.8, seed: int = 42):
        self.n_components = n_components
        self.alpha = alpha
        self.proba_sigma = proba_sigma
        self.seed = seed
        self.model_ = None
        self._pca = None

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        X_df, y_enc = self._fit_setup(X, y)
        Xv = _as_float32(X_df)
        k = min(self.n_components, min(Xv.shape) - 1)
        self._pca = PCA(n_components=k, random_state=self.seed)
        Z = self._pca.fit_transform(Xv)
        self.model_ = Ridge(alpha=self.alpha, random_state=self.seed)
        self.model_.fit(Z, y_enc.astype(float))
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Z = self._pca.transform(_as_float32(X_df))
        y_hat = self.model_.predict(Z)
        ks = np.arange(self.n_classes_)
        dist = (y_hat[:, None] - ks[None, :]) ** 2
        logits = -dist / (2.0 * (self.proba_sigma ** 2))
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        # Back-project ridge coefs through PCA loadings to gene space.
        comp_coef = self.model_.coef_                  # (k,)
        loadings = self._pca.components_               # (k, p)
        gene_imp = np.abs(comp_coef @ loadings)        # (p,)
        return pd.Series(gene_imp, index=self.feature_names_
                         ).sort_values(ascending=False)


class OrdinalPCALogistic(_OrdinalBase):
    """PCA -> multinomial LogisticRegression (lbfgs)."""

    def __init__(self, n_components: int = 50, C: float = 1.0,
                 class_weight="balanced", max_iter: int = 1000, seed: int = 42):
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.seed = seed
        self.model_ = None
        self._pca = None

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        X_df, y_enc = self._fit_setup(X, y)
        Xv = _as_float32(X_df)
        k = min(self.n_components, min(Xv.shape) - 1)
        self._pca = PCA(n_components=k, random_state=self.seed)
        Z = self._pca.fit_transform(Xv)
        self.model_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter,
            solver="lbfgs", random_state=self.seed)
        self.model_.fit(Z, y_enc)
        return self

    def predict_proba(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        Z = self._pca.transform(_as_float32(X_df))
        return self.model_.predict_proba(Z)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        comp_coef = np.abs(self.model_.coef_).mean(axis=0)   # (k,)
        loadings = self._pca.components_                      # (k, p)
        gene_imp = np.abs(comp_coef @ loadings)
        return pd.Series(gene_imp, index=self.feature_names_
                         ).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Soft-voting ensemble of the strongest linear/tree seeds. Averages calibrated
# class probabilities -> robust, low-variance combination.
# ----------------------------------------------------------------------------

class OrdinalSoftVote(_OrdinalBase):
    """Soft-voting ensemble: mean predict_proba over OrdinalRidge +
    OrdinalHistGB + OrdinalLDA. Captures linear-ordinal + nonlinear-tree +
    shrinkage-discriminant signal. Probability matrix is renormalised."""

    def __init__(self, alpha: float = 10.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self._members = None

    def fit(self, X, y):
        X_df, y_enc = self._fit_setup(X, y)
        self._members = [
            OrdinalRidge(alpha=self.alpha, seed=self.seed),
            OrdinalHistGB(seed=self.seed),
            OrdinalLDA(seed=self.seed),
        ]
        for m in self._members:
            m.fit(X_df, y)   # members re-encode identical labels -> aligned classes_
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self._members]
        P = np.mean(probs, axis=0)
        return P / P.sum(axis=1, keepdims=True)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        # Use the ridge member's gene attribution (interpretable, dense).
        return self._members[0].get_feature_importance(importance_type)