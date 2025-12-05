"""
Nested cross-validation for robust model evaluation and hyperparameter tuning.
Implements proper train/validation/test splits for publication-ready results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class NestedCrossValidator:
    """
    Nested cross-validation with proper separation of:
    - Outer loop: Model evaluation (unbiased performance estimate)
    - Inner loop: Hyperparameter tuning
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
        Initialize nested cross-validator.

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

        if hasattr(self, 'predict_proba'):
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
                    # Binary classification
                    metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                    metrics['auroc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro')
            except:
                pass

        return metrics

    def _summarize_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Calculate summary statistics for all metrics."""
        summary = {}

        # Get all metric names
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
        # Check first element to determine if it's predictions or probabilities
        first_pred = fold_predictions[0][1]

        if isinstance(first_pred, np.ndarray) and first_pred.ndim > 1:
            # Probabilities
            n_classes = first_pred.shape[1]
            full_array = np.zeros((n_samples, n_classes))
        else:
            # Predictions
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

        # Calculate confidence interval
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)

        # Use t-distribution for small samples
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

            # Create nested CV with different seed
            nested_cv = NestedCrossValidator(
                model=self.model,
                param_grid=self.param_grid,
                outer_cv=self.outer_cv,
                inner_cv=self.inner_cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                seed=self.base_seed + repeat * 100,
                verbose=self.verbose - 1
            )

            # Run nested CV
            results = nested_cv.fit(X, y)
            all_results.append(results)

        self.repeat_results_ = all_results

        # Aggregate results across all repeats
        aggregated = self._aggregate_results(all_results)

        if self.verbose > 0:
            mean_score = aggregated['overall_metrics'][self.scoring]['mean']
            std_score = aggregated['overall_metrics'][self.scoring]['std']
            logger.info(f"Repeated nested CV complete: {self.scoring} = "
                       f"{mean_score:.3f} (+/- {std_score:.3f})")

        return aggregated

    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all repetitions."""
        # Collect all fold scores across all repeats
        all_fold_scores = []
        for repeat_result in all_results:
            all_fold_scores.extend(repeat_result['fold_metrics'])

        # Calculate overall statistics
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