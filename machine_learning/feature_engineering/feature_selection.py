"""
Feature selection module with stability selection across CV folds.
Implements variance filtering and Elastic Net-based selection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class StableFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection with stability across CV folds."""

    def __init__(
        self,
        variance_threshold_percentile: float = 20,
        elastic_net_alpha: float = 0.5,
        elastic_net_l1_ratio: float = 0.5,
        stability_threshold: float = 0.6,
        max_features: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize feature selector.

        Args:
            variance_threshold_percentile: Remove genes below this variance percentile
            elastic_net_alpha: Regularization strength for Elastic Net
            elastic_net_l1_ratio: L1 ratio for Elastic Net (0=Ridge, 1=Lasso)
            stability_threshold: Minimum frequency across folds for stability
            max_features: Maximum number of features to select
            seed: Random seed
        """
        self.variance_threshold_percentile = variance_threshold_percentile
        self.elastic_net_alpha = elastic_net_alpha
        self.elastic_net_l1_ratio = elastic_net_l1_ratio
        self.stability_threshold = stability_threshold
        self.max_features = max_features
        self.seed = seed

        self.variance_selector_ = None
        self.elastic_net_ = None
        self.selected_features_ = None
        self.feature_importance_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit feature selector.

        Args:
            X: Feature matrix (samples x features)
            y: Target variable

        Returns:
            self
        """
        np.random.seed(self.seed)

        # Step 1: Variance thresholding
        if self.variance_threshold_percentile > 0:
            # Convert to numpy array for consistent variance calculation (ddof=0)
            X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            variances = np.var(X_array, axis=0)
            threshold = np.percentile(variances, self.variance_threshold_percentile)
            self.variance_selector_ = VarianceThreshold(threshold=threshold)

            X_var_filtered = self.variance_selector_.fit_transform(X_array)

            # Convert back to DataFrame
            if isinstance(X, pd.DataFrame):
                X_var_filtered = pd.DataFrame(
                    X_var_filtered,
                    index=X.index,
                    columns=X.columns[self.variance_selector_.get_support()]
                )
            logger.info(f"Variance filtering: {X.shape[1]} -> {X_var_filtered.shape[1]} features")
        else:
            X_var_filtered = X
            self.variance_selector_ = None

        # Step 2: Elastic Net-based selection
        # Convert target to numeric if needed
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y.values

        # Fit Elastic Net logistic regression
        self.elastic_net_ = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=1.0 / self.elastic_net_alpha,
            l1_ratio=self.elastic_net_l1_ratio,
            max_iter=1000,
            random_state=self.seed
        )

        # Handle multiclass case
        # Note: multi_class parameter deprecated in sklearn 1.5+, will auto-select multinomial
        if len(np.unique(y_encoded)) > 2:
            pass  # LogisticRegression automatically handles multiclass

        self.elastic_net_.fit(X_var_filtered, y_encoded)

        # Get feature importance (coefficients)
        if len(self.elastic_net_.coef_.shape) == 1:
            # Binary classification
            coef_abs = np.abs(self.elastic_net_.coef_)
        else:
            # Multiclass - average absolute coefficients across classes
            coef_abs = np.mean(np.abs(self.elastic_net_.coef_), axis=0)

        # Store feature importance
        if isinstance(X_var_filtered, pd.DataFrame):
            self.feature_importance_ = pd.Series(
                coef_abs,
                index=X_var_filtered.columns
            ).sort_values(ascending=False)
        else:
            # For numpy arrays, just store the coefficients
            sorted_indices = np.argsort(coef_abs)[::-1]
            self.feature_importance_ = pd.Series(
                coef_abs[sorted_indices],
                index=sorted_indices
            )

        # Select features with non-zero coefficients
        if isinstance(X_var_filtered, pd.DataFrame):
            self.selected_features_ = self.feature_importance_[
                self.feature_importance_ > 0
            ].index.tolist()

            # Limit to max_features if specified
            if self.max_features and len(self.selected_features_) > self.max_features:
                self.selected_features_ = self.selected_features_[:self.max_features]

            self.feature_names_ = self.selected_features_
        else:
            # For numpy arrays, use boolean mask
            self.selected_features_ = coef_abs > 0

            # Limit to max_features if specified
            if self.max_features:
                n_selected = np.sum(self.selected_features_)
                if n_selected > self.max_features:
                    # Select top max_features by coefficient magnitude
                    top_indices = np.argsort(coef_abs)[-self.max_features:]
                    self.selected_features_ = np.zeros(len(coef_abs), dtype=bool)
                    self.selected_features_[top_indices] = True

            self.feature_names_ = None

        # Log the number of selected features
        if isinstance(self.selected_features_, list):
            n_selected = len(self.selected_features_)
        else:
            n_selected = np.sum(self.selected_features_)
        logger.info(f"Elastic Net selected {n_selected} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted")

        # Apply variance filtering first if used
        if self.variance_selector_ is not None:
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            X_var_filtered = self.variance_selector_.transform(X_array)

            if isinstance(X, pd.DataFrame):
                X_var_filtered = pd.DataFrame(
                    X_var_filtered,
                    index=X.index,
                    columns=X.columns[self.variance_selector_.get_support()]
                )
        else:
            X_var_filtered = X

        # Select features based on type
        if isinstance(X_var_filtered, pd.DataFrame):
            # DataFrame with column names
            if isinstance(self.selected_features_, list):
                available_features = [f for f in self.selected_features_ if f in X_var_filtered.columns]
                return X_var_filtered[available_features]
            else:
                # Boolean mask
                return X_var_filtered.iloc[:, self.selected_features_]
        else:
            # Numpy array with boolean mask
            if isinstance(self.selected_features_, np.ndarray) and self.selected_features_.dtype == bool:
                return X_var_filtered[:, self.selected_features_]
            else:
                # If selected_features_ is a list of indices, convert to boolean
                mask = np.zeros(X_var_filtered.shape[1], dtype=bool)
                if isinstance(self.selected_features_, list):
                    for idx in self.selected_features_:
                        if isinstance(idx, int) and idx < X_var_filtered.shape[1]:
                            mask[idx] = True
                return X_var_filtered[:, mask]


class StabilitySelector:
    """Track feature stability across multiple CV folds."""

    def __init__(
        self,
        stability_threshold: float = 0.6,
        min_frequency: int = 3
    ):
        """
        Initialize stability selector.

        Args:
            stability_threshold: Minimum fraction of folds for stability
            min_frequency: Minimum absolute frequency across folds
        """
        self.stability_threshold = stability_threshold
        self.min_frequency = min_frequency
        self.feature_counts_ = {}
        self.fold_features_ = []
        self.n_folds_ = 0

    def add_fold_features(
        self,
        features: List[str],
        importance_scores: Optional[Dict[str, float]] = None
    ):
        """
        Add features selected in one CV fold.

        Args:
            features: List of selected feature names
            importance_scores: Optional importance scores for features
        """
        self.fold_features_.append(features)
        self.n_folds_ += 1

        for feature in features:
            if feature not in self.feature_counts_:
                self.feature_counts_[feature] = {
                    'count': 0,
                    'importance_scores': []
                }

            self.feature_counts_[feature]['count'] += 1

            if importance_scores and feature in importance_scores:
                self.feature_counts_[feature]['importance_scores'].append(
                    importance_scores[feature]
                )

    def get_stable_features(
        self,
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Get features that appear stably across folds.

        Args:
            top_n: Return only top N most stable features

        Returns:
            List of stable feature names
        """
        if self.n_folds_ == 0:
            return []

        stable_features = []
        for feature, info in self.feature_counts_.items():
            frequency = info['count'] / self.n_folds_

            if frequency >= self.stability_threshold and info['count'] >= self.min_frequency:
                stable_features.append((feature, frequency, info['count']))

        # Sort by frequency (stability)
        stable_features.sort(key=lambda x: (x[1], x[2]), reverse=True)

        feature_names = [f[0] for f in stable_features]

        if top_n:
            return feature_names[:top_n]
        return feature_names

    def get_stability_report(self) -> pd.DataFrame:
        """
        Get detailed stability report for all features.

        Returns:
            DataFrame with stability statistics
        """
        if self.n_folds_ == 0:
            return pd.DataFrame()

        report_data = []
        for feature, info in self.feature_counts_.items():
            frequency = info['count'] / self.n_folds_
            mean_importance = (
                np.mean(info['importance_scores'])
                if info['importance_scores'] else 0
            )

            report_data.append({
                'Feature': feature,
                'Fold_Count': info['count'],
                'Fold_Frequency': frequency,
                'Mean_Importance': mean_importance,
                'Is_Stable': frequency >= self.stability_threshold
            })

        report = pd.DataFrame(report_data)
        report = report.sort_values(['Is_Stable', 'Fold_Frequency', 'Mean_Importance'],
                                  ascending=[False, False, False])

        return report

    def get_fold_overlap_matrix(self) -> pd.DataFrame:
        """
        Get matrix showing feature overlap between folds.

        Returns:
            DataFrame with fold overlap statistics
        """
        if self.n_folds_ < 2:
            return pd.DataFrame()

        overlap_matrix = np.zeros((self.n_folds_, self.n_folds_))

        for i in range(self.n_folds_):
            for j in range(self.n_folds_):
                if i == j:
                    overlap_matrix[i, j] = len(self.fold_features_[i])
                else:
                    set_i = set(self.fold_features_[i])
                    set_j = set(self.fold_features_[j])
                    overlap = len(set_i.intersection(set_j))
                    overlap_matrix[i, j] = overlap

        return pd.DataFrame(
            overlap_matrix,
            index=[f'Fold_{i}' for i in range(self.n_folds_)],
            columns=[f'Fold_{i}' for i in range(self.n_folds_)]
        )


class MultiStageFeatureSelector:
    """Feature selection for multi-stage developmental classification."""

    def __init__(
        self,
        base_selector: StableFeatureSelector,
        n_classes: int
    ):
        """
        Initialize multi-stage selector.

        Args:
            base_selector: Base feature selector
            n_classes: Number of developmental stages
        """
        self.base_selector = base_selector
        self.n_classes = n_classes
        self.stage_specific_features_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit selector with stage-specific feature selection.

        Args:
            X: Feature matrix
            y: Stage labels

        Returns:
            self
        """
        # Overall feature selection
        self.base_selector.fit(X, y)

        # Stage-specific feature selection (one-vs-rest)
        if self.n_classes > 2:
            for stage in np.unique(y):
                # Create binary labels
                y_binary = (y == stage).astype(int)

                # Fit selector for this stage
                stage_selector = StableFeatureSelector(
                    variance_threshold_percentile=0,  # Already filtered
                    elastic_net_alpha=self.base_selector.elastic_net_alpha,
                    elastic_net_l1_ratio=self.base_selector.elastic_net_l1_ratio,
                    seed=self.base_selector.seed
                )

                # Use already selected features
                X_selected = self.base_selector.transform(X)
                stage_selector.fit(X_selected, y_binary)

                self.stage_specific_features_[stage] = stage_selector.selected_features_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed matrix
        """
        return self.base_selector.transform(X)

    def get_stage_markers(self, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Get top marker genes for each stage.

        Args:
            top_n: Number of top markers per stage

        Returns:
            Dictionary mapping stages to marker genes
        """
        markers = {}

        for stage, features in self.stage_specific_features_.items():
            # Get importance scores for this stage
            if hasattr(self, f'stage_importance_{stage}'):
                importance = getattr(self, f'stage_importance_{stage}')
                top_features = importance.nlargest(top_n).index.tolist()
            else:
                top_features = features[:top_n] if len(features) > top_n else features

            markers[stage] = top_features

        return markers
