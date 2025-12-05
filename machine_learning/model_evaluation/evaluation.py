"""
Evaluation metrics module for developmental stage classification.
Includes ordinal-specific metrics and bootstrap confidence intervals.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, mean_absolute_error, accuracy_score,
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def calculate_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Calculate metrics specific to ordinal classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Ordered list of class labels

    Returns:
        Dictionary of metric values
    """
    metrics = {}

    # Encode to numeric if needed
    if labels is not None:
        le = LabelEncoder()
        le.fit(labels)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred

    # Mean Absolute Error (in stage units)
    metrics['mae'] = mean_absolute_error(y_true_encoded, y_pred_encoded)

    # Quadratic-weighted Cohen's kappa
    metrics['quadratic_kappa'] = cohen_kappa_score(
        y_true_encoded, y_pred_encoded, weights='quadratic'
    )

    # Linear-weighted Cohen's kappa
    metrics['linear_kappa'] = cohen_kappa_score(
        y_true_encoded, y_pred_encoded, weights='linear'
    )

    # Spearman correlation
    metrics['spearman_r'], metrics['spearman_p'] = stats.spearmanr(
        y_true_encoded, y_pred_encoded
    )

    # Kendall's tau
    metrics['kendall_tau'], metrics['kendall_p'] = stats.kendalltau(
        y_true_encoded, y_pred_encoded
    )

    return metrics


def calculate_near_miss_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 1,
    labels: Optional[List] = None
) -> float:
    """
    Calculate accuracy allowing for near-miss predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        tolerance: Number of adjacent stages to allow
        labels: Ordered list of class labels

    Returns:
        Near-miss accuracy
    """
    # Encode to numeric
    if labels is not None:
        le = LabelEncoder()
        le.fit(labels)
        y_true_encoded = le.transform(y_true)
        y_pred_encoded = le.transform(y_pred)
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred

    # Calculate absolute difference
    diff = np.abs(y_true_encoded - y_pred_encoded)

    # Count predictions within tolerance
    correct = np.sum(diff <= tolerance)

    return correct / len(y_true)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List] = None,
    is_ordinal: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate comprehensive set of metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        labels: Class labels
        is_ordinal: Whether to calculate ordinal metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Additional comprehensive metrics
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Sample size information
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique_classes.tolist(), class_counts.tolist()))
    metrics['n_samples'] = len(y_true)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics['per_class_precision'] = {
        label: report[str(label)]['precision'] for label in labels if str(label) in report
    }
    metrics['per_class_recall'] = {
        label: report[str(label)]['recall'] for label in labels if str(label) in report
    }
    metrics['per_class_f1'] = {
        label: report[str(label)]['f1-score'] for label in labels if str(label) in report
    }

    # Add AUROC and AUPRC if probabilities are provided
    if y_prob is not None:
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            # Use probabilities for positive class
            y_prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            metrics['auroc'] = roc_auc_score(y_true, y_prob_positive)
            metrics['auprc'] = average_precision_score(y_true, y_prob_positive)
        else:
            # Multiclass classification
            # One-vs-rest AUROC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=labels if labels is not None else np.unique(y_true))

            # Calculate macro and weighted AUROC
            try:
                metrics['auroc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro')
                metrics['auroc_weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted')
            except ValueError as e:
                logger.warning(f"Could not calculate multiclass AUROC: {e}")
                metrics['auroc_macro'] = np.nan
                metrics['auroc_weighted'] = np.nan

            # Per-class AUROC
            metrics['per_class_auroc'] = {}
            for i, label in enumerate(labels if labels is not None else np.unique(y_true)):
                try:
                    metrics['per_class_auroc'][label] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                except:
                    metrics['per_class_auroc'][label] = np.nan

    # Ordinal metrics
    if is_ordinal and len(np.unique(y_true)) > 2:
        ordinal_metrics = calculate_ordinal_metrics(y_true, y_pred, labels)
        metrics.update(ordinal_metrics)

        # Near-miss accuracy
        metrics['accuracy_pm1'] = calculate_near_miss_accuracy(
            y_true, y_pred, tolerance=1, labels=labels
        )

    return metrics


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals for metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Function to calculate metric
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level
        seed: Random seed

    Returns:
        Dictionary with mean, std, and CI bounds
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, lower_percentile),
        'ci_upper': np.percentile(bootstrap_scores, upper_percentile)
    }


class MetricCalculator:
    """Comprehensive metric calculation for stage classification."""

    def __init__(
        self,
        is_ordinal: bool = True,
        calculate_bootstrap: bool = True,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42
    ):
        """
        Initialize metric calculator.

        Args:
            is_ordinal: Whether to calculate ordinal metrics
            calculate_bootstrap: Whether to calculate bootstrap CIs
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level for CIs
            seed: Random seed
        """
        self.is_ordinal = is_ordinal
        self.calculate_bootstrap = calculate_bootstrap
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.seed = seed

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List] = None
    ) -> Dict:
        """
        Calculate all metrics with optional bootstrap CIs.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            labels: Class labels

        Returns:
            Dictionary of metrics
        """
        # Calculate basic metrics
        metrics = calculate_all_metrics(
            y_true, y_pred, y_prob, labels, self.is_ordinal
        )

        # Add bootstrap confidence intervals
        if self.calculate_bootstrap:
            bootstrap_results = {}

            # Bootstrap balanced accuracy
            ba_bootstrap = bootstrap_metrics(
                y_true, y_pred,
                balanced_accuracy_score,
                self.n_bootstraps,
                self.confidence_level,
                self.seed
            )
            bootstrap_results['balanced_accuracy'] = ba_bootstrap

            # Bootstrap F1 macro
            f1_bootstrap = bootstrap_metrics(
                y_true, y_pred,
                lambda y_t, y_p: f1_score(y_t, y_p, average='macro', zero_division=0),
                self.n_bootstraps,
                self.confidence_level,
                self.seed
            )
            bootstrap_results['f1_macro'] = f1_bootstrap

            # Bootstrap ordinal metrics if applicable
            if self.is_ordinal and len(np.unique(y_true)) > 2:
                # Bootstrap MAE
                if labels is not None:
                    le = LabelEncoder()
                    le.fit(labels)
                    y_true_encoded = le.transform(y_true)
                    y_pred_encoded = le.transform(y_pred)
                else:
                    y_true_encoded = y_true
                    y_pred_encoded = y_pred

                mae_bootstrap = bootstrap_metrics(
                    y_true_encoded, y_pred_encoded,
                    mean_absolute_error,
                    self.n_bootstraps,
                    self.confidence_level,
                    self.seed
                )
                bootstrap_results['mae'] = mae_bootstrap

            metrics['bootstrap'] = bootstrap_results

        return metrics

    def format_results(self, metrics: Dict) -> pd.DataFrame:
        """
        Format metrics as a DataFrame for easy viewing.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted DataFrame
        """
        rows = []

        # Basic metrics
        for key in ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']:
            if key in metrics:
                row = {'Metric': key, 'Value': metrics[key]}

                # Add bootstrap CI if available
                if 'bootstrap' in metrics and key in metrics['bootstrap']:
                    boot = metrics['bootstrap'][key]
                    row['Mean (BS)'] = boot['mean']
                    row['95% CI'] = f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"

                rows.append(row)

        # Ordinal metrics
        if self.is_ordinal:
            for key in ['mae', 'quadratic_kappa', 'spearman_r', 'accuracy_pm1']:
                if key in metrics:
                    row = {'Metric': key, 'Value': metrics[key]}

                    if 'bootstrap' in metrics and key in metrics['bootstrap']:
                        boot = metrics['bootstrap'][key]
                        row['Mean (BS)'] = boot['mean']
                        row['95% CI'] = f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"

                    rows.append(row)

        return pd.DataFrame(rows)