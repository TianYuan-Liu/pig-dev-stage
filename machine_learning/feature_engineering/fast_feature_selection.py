"""
Fast feature selection module using univariate methods.
Much faster than Elastic Net for high-dimensional data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    chi2,
    VarianceThreshold
)
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class FastFeatureSelector(BaseEstimator, TransformerMixin):
    """Fast feature selection using univariate statistical tests."""

    def __init__(
        self,
        method: str = 'mutual_info',  # 'f_classif', 'mutual_info', or 'chi2'
        variance_threshold_percentile: float = 20,
        max_features: Optional[int] = 2000,
        n_jobs: int = -1,
        seed: int = 42
    ):
        """
        Initialize fast feature selector.

        Args:
            method: Feature ranking method ('f_classif', 'mutual_info', 'chi2')
            variance_threshold_percentile: Remove genes below this variance percentile
            max_features: Maximum number of features to select
            n_jobs: Number of parallel jobs (-1 for all cores)
            seed: Random seed
        """
        self.method = method
        self.variance_threshold_percentile = variance_threshold_percentile
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.seed = seed

        self.variance_selector_ = None
        self.univariate_selector_ = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit feature selector using fast univariate methods.

        Args:
            X: Feature matrix (samples x features)
            y: Target variable

        Returns:
            self
        """
        np.random.seed(self.seed)

        # Step 1: Fast variance thresholding
        if self.variance_threshold_percentile > 0:
            logger.info(f"Applying variance threshold (percentile={self.variance_threshold_percentile})")
            X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

            # Calculate variances efficiently
            variances = np.var(X_array, axis=0)
            threshold = np.percentile(variances, self.variance_threshold_percentile)

            self.variance_selector_ = VarianceThreshold(threshold=threshold)
            X_var_filtered = self.variance_selector_.fit_transform(X_array)

            # Keep track of feature names
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

        # Step 2: Fast univariate feature selection
        # Convert target to numeric if needed
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y.values if hasattr(y, 'values') else y

        # Select scoring function based on method
        if self.method == 'f_classif':
            score_func = f_classif
            logger.info("Using ANOVA F-statistic for feature selection")
        elif self.method == 'mutual_info':
            # Mutual information with parallel processing
            score_func = lambda X, y: mutual_info_classif(
                X, y, random_state=self.seed, n_neighbors=3
            )
            logger.info("Using Mutual Information for feature selection")
        elif self.method == 'chi2':
            # Ensure non-negative values for chi2
            if isinstance(X_var_filtered, pd.DataFrame):
                X_var_filtered = X_var_filtered.clip(lower=0)
            else:
                X_var_filtered = np.clip(X_var_filtered, 0, None)
            score_func = chi2
            logger.info("Using Chi-squared test for feature selection")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Determine number of features to select
        n_features_to_select = min(
            self.max_features or X_var_filtered.shape[1],
            X_var_filtered.shape[1]
        )

        # Apply univariate feature selection
        self.univariate_selector_ = SelectKBest(
            score_func=score_func,
            k=n_features_to_select
        )

        # Fit the selector
        logger.info(f"Selecting top {n_features_to_select} features...")
        if isinstance(X_var_filtered, pd.DataFrame):
            X_selected = self.univariate_selector_.fit_transform(
                X_var_filtered.values, y_encoded
            )
        else:
            X_selected = self.univariate_selector_.fit_transform(
                X_var_filtered, y_encoded
            )

        # Store feature scores and names
        self.feature_scores_ = self.univariate_selector_.scores_

        if isinstance(X_var_filtered, pd.DataFrame):
            # Get selected feature names
            selected_mask = self.univariate_selector_.get_support()
            self.selected_features_ = X_var_filtered.columns[selected_mask].tolist()
            self.feature_names_ = self.selected_features_

            # Create series of scores for selected features
            self.feature_scores_ = pd.Series(
                self.univariate_selector_.scores_[selected_mask],
                index=self.selected_features_
            ).sort_values(ascending=False)
        else:
            self.selected_features_ = self.univariate_selector_.get_support()
            self.feature_names_ = None

        logger.info(f"Selected {len(self.selected_features_)} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix with selected features
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

        # Apply univariate selection
        if isinstance(X_var_filtered, pd.DataFrame):
            X_selected = self.univariate_selector_.transform(X_var_filtered.values)

            # Return as DataFrame with proper column names
            return pd.DataFrame(
                X_selected,
                index=X_var_filtered.index,
                columns=self.selected_features_
            )
        else:
            return self.univariate_selector_.transform(X_var_filtered)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Transformed feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            Series of feature scores
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector not fitted")
        return self.feature_scores_


class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Hybrid approach: Fast univariate filtering followed by
    more sophisticated selection on reduced set.
    """

    def __init__(
        self,
        initial_features: int = 5000,
        final_features: int = 2000,
        method: str = 'mutual_info',
        use_elastic_net: bool = False,
        n_jobs: int = -1,
        seed: int = 42
    ):
        """
        Initialize hybrid selector.

        Args:
            initial_features: Features to keep after first pass
            final_features: Final number of features
            method: Method for initial selection
            use_elastic_net: Whether to use Elastic Net for final selection
            n_jobs: Number of parallel jobs
            seed: Random seed
        """
        self.initial_features = initial_features
        self.final_features = final_features
        self.method = method
        self.use_elastic_net = use_elastic_net
        self.n_jobs = n_jobs
        self.seed = seed

        self.fast_selector_ = None
        self.final_selector_ = None
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit hybrid selector.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            self
        """
        # First pass: Fast univariate selection
        self.fast_selector_ = FastFeatureSelector(
            method=self.method,
            max_features=self.initial_features,
            variance_threshold_percentile=20,
            n_jobs=self.n_jobs,
            seed=self.seed
        )

        X_reduced = self.fast_selector_.fit_transform(X, y)

        # Second pass: More sophisticated selection on reduced set
        if self.use_elastic_net and self.initial_features > self.final_features:
            # Import the original StableFeatureSelector for final selection
            from machine_learning.feature_engineering.feature_selection import StableFeatureSelector

            self.final_selector_ = StableFeatureSelector(
                max_features=self.final_features,
                seed=self.seed
            )
            X_final = self.final_selector_.fit_transform(X_reduced, y)
            self.selected_features_ = self.final_selector_.selected_features_
        else:
            # Just use another round of fast selection
            self.final_selector_ = FastFeatureSelector(
                method=self.method,
                max_features=self.final_features,
                variance_threshold_percentile=0,  # Already filtered
                n_jobs=self.n_jobs,
                seed=self.seed
            )
            X_final = self.final_selector_.fit_transform(X_reduced, y)
            self.selected_features_ = self.final_selector_.selected_features_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        X_reduced = self.fast_selector_.transform(X)
        return self.final_selector_.transform(X_reduced)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Transformed feature matrix
        """
        self.fit(X, y)
        return self.transform(X)