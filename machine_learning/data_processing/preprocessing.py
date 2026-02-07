"""
Preprocessing pipeline for RNA-seq expression data.
Includes transformation, normalization, and covariate adjustment.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class ExpressionPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline for gene expression data."""

    def __init__(
        self,
        log_transform: bool = None,
        standardize: bool = None,
        min_variance_percentile: float = None,
        remove_covariates: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize preprocessor.

        Parameters default to values from config.yaml. Explicit arguments override config.

        Args:
            log_transform: Apply log2(TPM+1) transformation
            standardize: Apply z-score normalization per gene
            min_variance_percentile: Remove genes below this variance percentile
            remove_covariates: List of covariates to regress out
            seed: Random seed
        """
        # Load defaults from config, fall back to sensible hardcoded defaults
        try:
            from machine_learning.utils.config_loader import get_config
            config = get_config()
            preproc_config = config.get('preprocessing', default={})
            if not isinstance(preproc_config, dict):
                preproc_config = {}
        except (ImportError, Exception):
            preproc_config = {}

        self.log_transform = log_transform if log_transform is not None else preproc_config.get('log_transform', True)
        self.standardize = standardize if standardize is not None else preproc_config.get('standardize', False)
        self.min_variance_percentile = min_variance_percentile if min_variance_percentile is not None else preproc_config.get('min_variance_percentile', 0)
        self.remove_covariates = remove_covariates if remove_covariates is not None else []
        self.seed = seed

        self.scaler_ = None
        self.variance_threshold_ = None
        self.selected_genes_ = None
        self.covariate_models_ = {}
        self.gene_means_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None, sample_metadata: Optional[pd.DataFrame] = None):
        """
        Fit preprocessing parameters on training data.

        Args:
            X: Expression matrix (genes x samples)
            y: Target variable (unused)
            sample_metadata: Sample metadata for covariate regression

        Returns:
            self
        """
        X_processed = X.copy()

        # Log transformation
        if self.log_transform:
            if (X_processed < 0).any().any():
                raise ValueError(
                    f"Cannot apply log2(x+1) transform: {(X_processed < 0).sum().sum()} "
                    f"negative values detected. Check upstream data or disable log_transform."
                )
            X_processed = np.log2(X_processed + 1)
            logger.info("Applied log2(TPM+1) transformation")

        # Variance filtering
        if self.min_variance_percentile > 0:
            gene_variances = X_processed.var(axis=1)
            self.variance_threshold_ = np.percentile(
                gene_variances, self.min_variance_percentile
            )
            self.selected_genes_ = gene_variances >= self.variance_threshold_
            X_processed = X_processed[self.selected_genes_]
            logger.info(
                f"Selected {self.selected_genes_.sum()}/{len(gene_variances)} genes "
                f"above variance percentile {self.min_variance_percentile}"
            )
        else:
            self.selected_genes_ = pd.Series(True, index=X.index)

        self.feature_names_ = X_processed.index.tolist()

        # Covariate regression
        if self.remove_covariates and sample_metadata is not None:
            X_processed = self._fit_covariate_regression(
                X_processed, sample_metadata
            )

        # Standardization (z-score)
        if self.standardize:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            # Transpose for sklearn (samples x features)
            X_transposed = X_processed.T
            self.scaler_.fit(X_transposed)
            logger.info("Fitted z-score standardization")

        return self

    def transform(self, X: pd.DataFrame, sample_metadata: Optional[pd.DataFrame] = None):
        """
        Transform expression data using fitted parameters.

        Args:
            X: Expression matrix (genes x samples)
            sample_metadata: Sample metadata for covariate regression

        Returns:
            Preprocessed expression matrix
        """
        X_processed = X.copy()

        # Log transformation
        if self.log_transform:
            if (X_processed < 0).any().any():
                raise ValueError(
                    f"Cannot apply log2(x+1) transform: {(X_processed < 0).sum().sum()} "
                    f"negative values detected. Check upstream data or disable log_transform."
                )
            X_processed = np.log2(X_processed + 1)

        # Apply gene selection
        if self.selected_genes_ is not None:
            # Handle case where test data has different genes
            common_genes = X_processed.index.intersection(self.selected_genes_.index)
            selected = self.selected_genes_.loc[common_genes]
            X_processed = X_processed.loc[selected[selected].index]

        # Covariate regression
        if self.remove_covariates and sample_metadata is not None:
            X_processed = self._transform_covariate_regression(
                X_processed, sample_metadata
            )

        # Standardization
        if self.standardize and self.scaler_ is not None:
            X_transposed = X_processed.T
            X_scaled = self.scaler_.transform(X_transposed)
            X_processed = pd.DataFrame(
                X_scaled.T,
                index=X_processed.index,
                columns=X_processed.columns
            )

        return X_processed

    def fit_transform(
        self,
        X: pd.DataFrame,
        y=None,
        sample_metadata: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit and transform expression data in one step.

        Args:
            X: Expression matrix (genes x samples)
            y: Target variable (unused)
            sample_metadata: Sample metadata for covariate regression

        Returns:
            Preprocessed expression matrix
        """
        self.fit(X, y, sample_metadata)
        return self.transform(X, sample_metadata)

    def _fit_covariate_regression(
        self,
        X: pd.DataFrame,
        sample_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fit covariate regression models and remove effects.

        Args:
            X: Expression matrix
            sample_metadata: Sample metadata

        Returns:
            Residualized expression matrix
        """
        # Align samples
        common_samples = X.columns.intersection(sample_metadata.index)
        X_aligned = X[common_samples]
        metadata_aligned = sample_metadata.loc[common_samples]

        # Prepare covariate matrix
        covariate_matrix = self._prepare_covariates(metadata_aligned)

        if covariate_matrix.shape[1] == 0:
            logger.warning("No valid covariates found for regression")
            return X

        # Fit regression for each gene
        residuals = pd.DataFrame(
            index=X_aligned.index,
            columns=X_aligned.columns,
            dtype=float
        )

        for gene in X_aligned.index:
            expression = X_aligned.loc[gene].values
            model = LinearRegression()
            model.fit(covariate_matrix, expression)
            self.covariate_models_[gene] = model
            residuals.loc[gene] = expression - model.predict(covariate_matrix)

        logger.info(f"Removed effects of {covariate_matrix.shape[1]} covariates")

        # Add back mean for interpretability
        gene_means = X_aligned.mean(axis=1)
        self.gene_means_ = gene_means
        residuals = residuals.add(gene_means, axis=0)

        return residuals

    def _transform_covariate_regression(
        self,
        X: pd.DataFrame,
        sample_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply fitted covariate regression to new data.

        Args:
            X: Expression matrix
            sample_metadata: Sample metadata

        Returns:
            Residualized expression matrix
        """
        if not self.covariate_models_:
            return X

        # Align samples
        common_samples = X.columns.intersection(sample_metadata.index)
        X_aligned = X[common_samples]
        metadata_aligned = sample_metadata.loc[common_samples]

        # Prepare covariate matrix
        covariate_matrix = self._prepare_covariates(metadata_aligned)

        # Apply regression for each gene
        residuals = pd.DataFrame(
            index=X_aligned.index,
            columns=X_aligned.columns,
            dtype=float
        )

        for gene in X_aligned.index:
            if gene in self.covariate_models_:
                expression = X_aligned.loc[gene].values
                model = self.covariate_models_[gene]
                residuals.loc[gene] = expression - model.predict(covariate_matrix)
                # Add back training mean (from fit, not test data)
                residuals.loc[gene] += self.gene_means_[gene]
            else:
                # Gene not in training set, keep original values
                residuals.loc[gene] = X_aligned.loc[gene]

        return residuals

    def _prepare_covariates(self, metadata: pd.DataFrame) -> np.ndarray:
        """
        Prepare covariate matrix from metadata.

        Args:
            metadata: Sample metadata

        Returns:
            Covariate matrix for regression
        """
        covariate_list = []

        for covar in self.remove_covariates:
            if covar not in metadata.columns:
                logger.warning(f"Covariate {covar} not found in metadata")
                continue

            values = metadata[covar]

            # Handle categorical variables
            if values.dtype == 'object' or isinstance(values.dtype, pd.CategoricalDtype):
                # One-hot encode with handling for unknown categories
                dummies = pd.get_dummies(values, prefix=covar, drop_first=True)
                covariate_list.append(dummies.values)
            else:
                # Numeric covariate
                # Fill missing values with median
                median_val = values.median()
                values_filled = values.fillna(median_val)
                covariate_list.append(values_filled.values.reshape(-1, 1))

        if covariate_list:
            return np.hstack(covariate_list)
        else:
            return np.empty((len(metadata), 0))


