"""
Cross-validation framework with proper train/validation/test splits.
Implements nested CV for hyperparameter tuning with no data leakage.

This module consolidates cross-validation functionality from:
- Basic nested CV with group support
- GridSearch-based nested CV for hyperparameter tuning
- Repeated nested CV for robust estimates
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, train_test_split, GridSearchCV
)
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class NestedCrossValidator:
    """Nested cross-validation with proper data leakage control."""

    def __init__(
        self,
        n_outer_splits: int = 5,
        n_inner_splits: int = 3,
        stratify: bool = True,
        group_column: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize nested CV.

        Args:
            n_outer_splits: Number of outer CV folds
            n_inner_splits: Number of inner CV folds for hyperparameter tuning
            stratify: Whether to use stratified splits
            group_column: Column name for group-based splitting (e.g., batch)
            seed: Random seed
        """
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.stratify = stratify
        self.group_column = group_column
        self.seed = seed

        self.outer_cv = None
        self.inner_cv = None
        self.fold_indices = {}

    def setup_cv_splitters(
        self,
        n_samples: int,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ):
        """
        Set up cross-validation splitters.

        Args:
            n_samples: Number of samples
            y: Target variable for stratification
            groups: Group labels for GroupKFold
        """
        # Outer CV
        if groups is not None and self.group_column:
            self.outer_cv = GroupKFold(n_splits=self.n_outer_splits)
            logger.info(f"Using GroupKFold for outer CV with {self.n_outer_splits} splits")
        elif self.stratify and y is not None:
            self.outer_cv = StratifiedKFold(
                n_splits=self.n_outer_splits,
                shuffle=True,
                random_state=self.seed
            )
            logger.info(f"Using StratifiedKFold for outer CV with {self.n_outer_splits} splits")
        else:
            from sklearn.model_selection import KFold
            self.outer_cv = KFold(
                n_splits=self.n_outer_splits,
                shuffle=True,
                random_state=self.seed
            )
            logger.info(f"Using KFold for outer CV with {self.n_outer_splits} splits")

        # Inner CV (always stratified if possible)
        if self.stratify and y is not None:
            self.inner_cv = StratifiedKFold(
                n_splits=self.n_inner_splits,
                shuffle=True,
                random_state=self.seed + 1
            )
        else:
            from sklearn.model_selection import KFold
            self.inner_cv = KFold(
                n_splits=self.n_inner_splits,
                shuffle=True,
                random_state=self.seed + 1
            )

    def split_outer(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Generate outer CV splits.

        Args:
            X: Feature matrix
            y: Target variable
            metadata: Sample metadata (for groups)

        Yields:
            Tuple of (fold_idx, train_idx, test_idx)
        """
        groups = None
        if self.group_column and metadata is not None:
            if self.group_column in metadata.columns:
                groups = metadata[self.group_column].values

        # Set up CV splitters if not already done
        if self.outer_cv is None:
            self.setup_cv_splitters(len(y), y.values, groups)

        # Generate splits
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.outer_cv.split(X, y, groups)
        ):
            # Store fold indices
            self.fold_indices[fold_idx] = {
                'train': train_idx,
                'test': test_idx
            }

            yield fold_idx, train_idx, test_idx

    def split_inner(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        metadata_train: Optional[pd.DataFrame] = None
    ):
        """
        Generate inner CV splits for hyperparameter tuning.

        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            metadata_train: Training metadata

        Yields:
            Tuple of (inner_fold_idx, inner_train_idx, inner_val_idx)
        """
        groups = None
        if self.group_column and metadata_train is not None:
            if self.group_column in metadata_train.columns:
                groups = metadata_train[self.group_column].values

        # Generate inner splits
        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
            self.inner_cv.split(X_train, y_train, groups)
        ):
            yield inner_fold_idx, inner_train_idx, inner_val_idx


class ModelEvaluator:
    """Evaluate models with proper cross-validation."""

    def __init__(
        self,
        cv_splitter: NestedCrossValidator,
        preprocessor: Optional[BaseEstimator] = None,
        feature_selector: Optional[BaseEstimator] = None
    ):
        """
        Initialize evaluator.

        Args:
            cv_splitter: Cross-validation splitter
            preprocessor: Data preprocessor
            feature_selector: Feature selection method
        """
        self.cv_splitter = cv_splitter
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.results = []

    def evaluate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Optional[pd.DataFrame] = None,
        scoring: Union[str, List[str]] = 'balanced_accuracy',
        return_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model using nested cross-validation.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            metadata: Sample metadata
            scoring: Scoring metric(s)
            return_predictions: Whether to return predictions

        Returns:
            Dictionary of results
        """
        from sklearn.metrics import balanced_accuracy_score, f1_score

        if isinstance(scoring, str):
            scoring = [scoring]

        fold_results = []
        all_predictions = []
        all_true = []
        all_probabilities = []

        # Outer CV loop
        for fold_idx, train_idx, test_idx in self.cv_splitter.split_outer(X, y, metadata):
            logger.info(f"Processing outer fold {fold_idx + 1}/{self.cv_splitter.n_outer_splits}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            metadata_train = metadata.iloc[train_idx] if metadata is not None else None
            metadata_test = metadata.iloc[test_idx] if metadata is not None else None

            # Preprocessing (fit on train, apply to test)
            if self.preprocessor:
                preprocessor_clone = clone(self.preprocessor)
                # Need to transpose back to genes x samples for preprocessor
                X_train_T = X_train.T if hasattr(X_train, 'T') else X_train
                X_test_T = X_test.T if hasattr(X_test, 'T') else X_test

                X_train_T = preprocessor_clone.fit_transform(
                    X_train_T, y_train, sample_metadata=metadata_train
                )
                X_test_T = preprocessor_clone.transform(
                    X_test_T, sample_metadata=metadata_test
                )

                # Transpose back to samples x genes
                X_train = X_train_T.T if hasattr(X_train_T, 'T') else X_train_T
                X_test = X_test_T.T if hasattr(X_test_T, 'T') else X_test_T

            # Feature selection (fit on train, apply to test)
            if self.feature_selector:
                selector_clone = clone(self.feature_selector)
                selector_clone.fit(X_train, y_train)
                X_train = selector_clone.transform(X_train)
                X_test = selector_clone.transform(X_test)

            # Train model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            # Predictions
            y_pred = model_clone.predict(X_test)

            # Probabilities (if available)
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test)
                all_probabilities.extend(y_prob)
            else:
                y_prob = None

            # Calculate metrics
            fold_metrics = {}
            for metric in scoring:
                if metric == 'balanced_accuracy':
                    score = balanced_accuracy_score(y_test, y_pred)
                elif metric == 'f1_macro':
                    score = f1_score(y_test, y_pred, average='macro')
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                fold_metrics[metric] = score

            fold_results.append({
                'fold': fold_idx,
                'metrics': fold_metrics,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

            if return_predictions:
                all_predictions.extend(y_pred)
                all_true.extend(y_test)

            logger.info(f"Fold {fold_idx} metrics: {fold_metrics}")

        # Aggregate results
        results = {
            'fold_results': fold_results,
            'mean_scores': {},
            'std_scores': {},
            'cv_scores': {}
        }

        for metric in scoring:
            scores = [f['metrics'][metric] for f in fold_results]
            results['mean_scores'][metric] = np.mean(scores)
            results['std_scores'][metric] = np.std(scores)
            results['cv_scores'][metric] = scores

        if return_predictions:
            results['predictions'] = all_predictions
            results['true_labels'] = all_true
            if all_probabilities:
                results['probabilities'] = all_probabilities

        return results

    def hyperparameter_search(
        self,
        model_class: type,
        param_grid: Dict[str, List],
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Optional[pd.DataFrame] = None,
        scoring: str = 'balanced_accuracy'
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search using nested CV.

        Args:
            model_class: Model class to instantiate
            param_grid: Hyperparameter grid
            X: Feature matrix
            y: Target variable
            metadata: Sample metadata
            scoring: Scoring metric

        Returns:
            Best parameters and results
        """
        from itertools import product
        from sklearn.metrics import balanced_accuracy_score

        best_params = {}
        best_scores = []

        # Outer CV loop
        for fold_idx, train_idx, test_idx in self.cv_splitter.split_outer(X, y, metadata):
            logger.info(f"Hyperparameter search for outer fold {fold_idx + 1}")

            # Split outer data
            X_train_outer = X.iloc[train_idx]
            y_train_outer = y.iloc[train_idx]
            metadata_train_outer = metadata.iloc[train_idx] if metadata is not None else None

            # Preprocess outer train
            if self.preprocessor:
                preprocessor_clone = clone(self.preprocessor)
                X_train_outer = preprocessor_clone.fit_transform(
                    X_train_outer, y_train_outer, sample_metadata=metadata_train_outer
                )

            # Grid search on inner CV
            param_combinations = list(product(*[param_grid[k] for k in param_grid]))
            param_names = list(param_grid.keys())

            best_inner_score = -np.inf
            best_inner_params = None

            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                inner_scores = []

                # Inner CV loop
                for inner_idx, inner_train_idx, inner_val_idx in self.cv_splitter.split_inner(
                    X_train_outer, y_train_outer, metadata_train_outer
                ):
                    # Split inner data
                    X_inner_train = X_train_outer.iloc[inner_train_idx]
                    X_inner_val = X_train_outer.iloc[inner_val_idx]
                    y_inner_train = y_train_outer.iloc[inner_train_idx]
                    y_inner_val = y_train_outer.iloc[inner_val_idx]

                    # Feature selection on inner train
                    if self.feature_selector:
                        selector_clone = clone(self.feature_selector)
                        X_inner_train = selector_clone.fit_transform(X_inner_train, y_inner_train)
                        X_inner_val = selector_clone.transform(X_inner_val)

                    # Train model with current parameters
                    model = model_class(**param_dict)
                    model.fit(X_inner_train, y_inner_train)
                    y_pred = model.predict(X_inner_val)

                    # Score
                    if scoring == 'balanced_accuracy':
                        score = balanced_accuracy_score(y_inner_val, y_pred)
                    else:
                        raise ValueError(f"Unknown scoring: {scoring}")

                    inner_scores.append(score)

                # Average inner CV score
                mean_inner_score = np.mean(inner_scores)

                if mean_inner_score > best_inner_score:
                    best_inner_score = mean_inner_score
                    best_inner_params = param_dict

            best_params[f'fold_{fold_idx}'] = best_inner_params
            best_scores.append(best_inner_score)
            logger.info(f"Fold {fold_idx} best params: {best_inner_params}, score: {best_inner_score:.4f}")

        # Find most common best parameters across folds
        from collections import Counter
        param_counts = Counter(str(p) for p in best_params.values())
        most_common_params = eval(param_counts.most_common(1)[0][0])

        return {
            'best_params': most_common_params,
            'fold_best_params': best_params,
            'fold_best_scores': best_scores,
            'mean_best_score': np.mean(best_scores),
            'std_best_score': np.std(best_scores)
        }


class GridSearchNestedCV:
    """
    Nested cross-validation with GridSearchCV for hyperparameter tuning.
    
    This class provides:
    - Outer loop: Model evaluation (unbiased performance estimate)
    - Inner loop: Hyperparameter tuning via GridSearchCV
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, List],
        outer_cv: int = 5,
        inner_cv: int = 3,
        scoring: str = 'balanced_accuracy',
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 1
    ):
        """
        Initialize nested cross-validator with GridSearch.

        Args:
            model: Base model to evaluate
            param_grid: Hyperparameter grid for tuning
            outer_cv: Number of outer CV folds (for evaluation)
            inner_cv: Number of inner CV folds (for tuning)
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            seed: Random seed
            verbose: Verbosity level
        """
        self.model = model
        self.param_grid = param_grid
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.cv_results_ = {}
        self.best_params_ = []
        self.test_scores_ = []
        self.predictions_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform nested cross-validation.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting nested CV: {self.outer_cv} outer folds, {self.inner_cv} inner folds")

        # Create stratified k-fold splitters
        outer_cv_splitter = StratifiedKFold(
            n_splits=self.outer_cv,
            shuffle=True,
            random_state=self.seed
        )
        inner_cv_splitter = StratifiedKFold(
            n_splits=self.inner_cv,
            shuffle=True,
            random_state=self.seed + 1
        )

        # Storage for results
        outer_scores = []
        outer_predictions = []
        outer_probabilities = []
        best_params_list = []
        fold_models = []

        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
            if self.verbose > 0:
                logger.info(f"Outer fold {fold_idx + 1}/{self.outer_cv}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=clone(self.model),
                param_grid=self.param_grid,
                cv=inner_cv_splitter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit=True,
                verbose=0
            )

            # Fit on training set
            grid_search.fit(X_train, y_train)
            best_params_list.append(grid_search.best_params_)

            if self.verbose > 1:
                logger.info(f"  Best params for fold {fold_idx + 1}: {grid_search.best_params_}")

            # Evaluate on test set
            y_pred = grid_search.predict(X_test)

            # Get probabilities if available
            if hasattr(grid_search.best_estimator_, 'predict_proba'):
                y_prob = grid_search.best_estimator_.predict_proba(X_test)
                outer_probabilities.append((test_idx, y_prob))
            else:
                y_prob = None

            # Calculate multiple metrics
            fold_metrics = self._calculate_metrics(y_test, y_pred, y_prob)
            outer_scores.append(fold_metrics)

            # Store predictions
            outer_predictions.append((test_idx, y_pred))

            # Store the best model
            fold_models.append(grid_search.best_estimator_)

            if self.verbose > 0:
                logger.info(f"  Fold {fold_idx + 1} {self.scoring}: {fold_metrics[self.scoring]:.3f}")

        # Aggregate results
        self.test_scores_ = outer_scores
        self.best_params_ = best_params_list
        self.fold_models_ = fold_models

        # Calculate summary statistics
        metrics_summary = self._summarize_metrics(outer_scores)

        # Create full predictions array
        self.predictions_ = self._reconstruct_predictions(outer_predictions, len(y))

        if outer_probabilities:
            self.probabilities_ = self._reconstruct_predictions(outer_probabilities, len(y))

        # Final results
        results = {
            'metrics_summary': metrics_summary,
            'fold_metrics': outer_scores,
            'best_params': best_params_list,
            'predictions': self.predictions_
        }

        if self.verbose > 0:
            logger.info(f"Nested CV complete: {self.scoring} = "
                       f"{metrics_summary[self.scoring]['mean']:.3f} "
                       f"(+/- {metrics_summary[self.scoring]['std']:.3f})")

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate multiple evaluation metrics."""
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Add AUROC if probabilities available
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    metrics['auroc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro')
            except Exception:
                pass

        return metrics

    def _summarize_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Calculate summary statistics for all metrics."""
        summary = {}
        metric_names = fold_metrics[0].keys()

        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        return summary

    def _reconstruct_predictions(
        self,
        fold_predictions: List[Tuple],
        n_samples: int
    ) -> np.ndarray:
        """Reconstruct full prediction array from fold predictions."""
        first_pred = fold_predictions[0][1]

        if isinstance(first_pred, np.ndarray) and first_pred.ndim > 1:
            n_classes = first_pred.shape[1]
            full_array = np.zeros((n_samples, n_classes))
        else:
            full_array = np.zeros(n_samples)

        for indices, preds in fold_predictions:
            full_array[indices] = preds

        return full_array

    def get_confidence_intervals(
        self,
        metric: str = 'balanced_accuracy',
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence intervals for a metric.

        Args:
            metric: Metric name
            confidence: Confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        if not self.test_scores_:
            raise ValueError("No results available. Run fit first.")

        values = [score[metric] for score in self.test_scores_]

        mean = np.mean(values)
        std = np.std(values)
        n = len(values)

        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * std / np.sqrt(n)

        return (mean - margin, mean + margin)


class RepeatedNestedCV:
    """
    Repeated nested cross-validation for even more robust estimates.
    Runs nested CV multiple times with different random seeds.
    """

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, List],
        n_repeats: int = 5,
        outer_cv: int = 5,
        inner_cv: int = 3,
        scoring: str = 'balanced_accuracy',
        n_jobs: int = -1,
        base_seed: int = 42,
        verbose: int = 1
    ):
        """
        Initialize repeated nested CV.

        Args:
            model: Base model
            param_grid: Hyperparameter grid
            n_repeats: Number of repetitions
            outer_cv: Outer CV folds
            inner_cv: Inner CV folds
            scoring: Scoring metric
            n_jobs: Parallel jobs
            base_seed: Base random seed
            verbose: Verbosity level
        """
        self.model = model
        self.param_grid = param_grid
        self.n_repeats = n_repeats
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.base_seed = base_seed
        self.verbose = verbose

        self.repeat_results_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform repeated nested cross-validation.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Aggregated results across all repetitions
        """
        logger.info(f"Starting repeated nested CV: {self.n_repeats} repeats")

        all_results = []

        for repeat in range(self.n_repeats):
            if self.verbose > 0:
                logger.info(f"Repeat {repeat + 1}/{self.n_repeats}")

            nested_cv = GridSearchNestedCV(
                model=self.model,
                param_grid=self.param_grid,
                outer_cv=self.outer_cv,
                inner_cv=self.inner_cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                seed=self.base_seed + repeat * 100,
                verbose=self.verbose - 1
            )

            results = nested_cv.fit(X, y)
            all_results.append(results)

        self.repeat_results_ = all_results

        aggregated = self._aggregate_results(all_results)

        if self.verbose > 0:
            mean_score = aggregated['overall_metrics'][self.scoring]['mean']
            std_score = aggregated['overall_metrics'][self.scoring]['std']
            logger.info(f"Repeated nested CV complete: {self.scoring} = "
                       f"{mean_score:.3f} (+/- {std_score:.3f})")

        return aggregated

    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all repetitions."""
        all_fold_scores = []
        for repeat_result in all_results:
            all_fold_scores.extend(repeat_result['fold_metrics'])

        overall_metrics = {}
        metric_names = all_fold_scores[0].keys()

        for metric in metric_names:
            values = [score[metric] for score in all_fold_scores]
            overall_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_folds': len(values)
            }

        return {
            'overall_metrics': overall_metrics,
            'repeat_results': self.repeat_results_
        }