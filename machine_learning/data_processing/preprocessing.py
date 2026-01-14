"""
Preprocessing pipeline for RNA-seq expression data.
Includes transformation, normalization, and covariate adjustment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class ExpressionPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline for gene expression data."""

    def __init__(
        self,
        log_transform: bool = True,
        standardize: bool = True,
        min_variance_percentile: float = 20,
        remove_covariates: Optional[List[str]] = None,
        seed: int = 42
    ):
        """
        Initialize preprocessor.

        Args:
            log_transform: Apply log2(TPM+1) transformation
            standardize: Apply z-score normalization per gene
            min_variance_percentile: Remove genes below this variance percentile
            remove_covariates: List of covariates to regress out
            seed: Random seed
        """
        self.log_transform = log_transform
        self.standardize = standardize
        self.min_variance_percentile = min_variance_percentile
        self.remove_covariates = remove_covariates if remove_covariates is not None else []
        self.seed = seed

        self.scaler_ = None
        self.variance_threshold_ = None
        self.selected_genes_ = None
        self.covariate_models_ = {}
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
        np.random.seed(self.seed)

        X_processed = X.copy()

        # Log transformation
        if self.log_transform:
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
                # Add back training mean
                residuals.loc[gene] += X_aligned.loc[gene].mean()
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
            if values.dtype == 'object' or pd.api.types.is_categorical_dtype(values):
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


class SexEncoder:
    """Encode sex labels with explicit Unknown level."""

    def __init__(self):
        """Initialize encoder."""
        self.classes_ = ['Female', 'Male', 'Unknown', 'Other']
        self.n_classes_ = len(self.classes_)

    def fit(self, X: pd.Series):
        """
        Fit encoder (no-op for predefined classes).

        Args:
            X: Sex labels

        Returns:
            self
        """
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        """
        Transform sex labels to one-hot encoding.

        Args:
            X: Sex labels

        Returns:
            One-hot encoded DataFrame
        """
        X_filled = X.fillna('Unknown')

        # Map variations to standard labels
        mapping = {
            'F': 'Female',
            'M': 'Male',
            'female': 'Female',
            'male': 'Male',
            'Other/pooled': 'Other'
        }

        X_mapped = X_filled.replace(mapping)

        # Handle any remaining non-standard values
        X_mapped[~X_mapped.isin(self.classes_)] = 'Other'

        # One-hot encode
        encoded = pd.get_dummies(X_mapped, prefix='Sex')

        # Ensure all expected columns exist
        for class_name in self.classes_:
            col_name = f'Sex_{class_name}'
            if col_name not in encoded.columns:
                encoded[col_name] = 0

        return encoded[['Sex_Female', 'Sex_Male', 'Sex_Unknown', 'Sex_Other']]

    def fit_transform(self, X: pd.Series) -> pd.DataFrame:
        """
        Fit and transform sex labels.

        Args:
            X: Sex labels

        Returns:
            One-hot encoded DataFrame
        """
        return self.fit(X).transform(X)


def prepare_data_for_modeling(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    target_column: str = 'Stage',
    stage_mapping: Optional[Dict[str, str]] = None,
    preprocessor: Optional[ExpressionPreprocessor] = None,
    fit_preprocessor: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare data for machine learning modeling.

    Args:
        expression: Expression matrix (genes x samples)
        metadata: Sample metadata
        target_column: Name of target column in metadata
        stage_mapping: Optional mapping to merge stages
        preprocessor: Preprocessor instance
        fit_preprocessor: Whether to fit preprocessor

    Returns:
        Tuple of (preprocessed_expression, target_labels, processed_metadata)
    """
    # Align samples
    common_samples = expression.columns.intersection(metadata.index)
    X = expression[common_samples]
    metadata_aligned = metadata.loc[common_samples]

    # Get target variable
    if target_column not in metadata_aligned.columns:
        raise ValueError(f"Target column {target_column} not found in metadata")

    y = metadata_aligned[target_column].copy()

    # Apply stage mapping if provided
    if stage_mapping:
        y = y.map(stage_mapping).fillna(y)

    # Remove samples with missing target
    valid_mask = y.notna()
    X = X.loc[:, valid_mask]
    y = y[valid_mask]
    metadata_aligned = metadata_aligned[valid_mask]

    # Preprocess expression
    if preprocessor is None:
        preprocessor = ExpressionPreprocessor()

    if fit_preprocessor:
        X_processed = preprocessor.fit_transform(X, sample_metadata=metadata_aligned)
    else:
        X_processed = preprocessor.transform(X, sample_metadata=metadata_aligned)

    # Encode sex if present
    if 'Sex' in metadata_aligned.columns:
        sex_encoder = SexEncoder()
        sex_encoded = sex_encoder.fit_transform(metadata_aligned['Sex'])
        metadata_aligned = pd.concat([metadata_aligned, sex_encoded], axis=1)

    return X_processed, y, metadata_aligned