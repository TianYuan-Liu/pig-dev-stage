"""
Probability calibration and conformal prediction for confidence estimation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrate predicted probabilities for better reliability."""

    def __init__(
        self,
        method: str = 'isotonic',
        cv: int = 3,
        ensemble: bool = False
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('isotonic' or 'platt')
            cv: Number of CV folds for calibration
            ensemble: Whether to use ensemble calibration
        """
        self.method = method
        self.cv = cv
        self.ensemble = ensemble
        self.calibrators_ = {}
        self.classes_ = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Fit calibration mapping.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            self
        """
        # Get unique classes
        self.classes_ = np.unique(y_true)
        n_classes = len(self.classes_)

        # Binary or multiclass calibration
        if n_classes == 2:
            # Binary calibration
            if self.method == 'isotonic':
                self.calibrators_[0] = IsotonicRegression(out_of_bounds='clip')
                self.calibrators_[0].fit(y_prob[:, 1], y_true == self.classes_[1])
            else:  # Platt
                from sklearn.linear_model import LogisticRegression
                self.calibrators_[0] = LogisticRegression()
                self.calibrators_[0].fit(
                    y_prob[:, 1].reshape(-1, 1),
                    y_true == self.classes_[1]
                )
        else:
            # Multiclass: calibrate each class separately
            for i, class_label in enumerate(self.classes_):
                y_binary = (y_true == class_label).astype(int)
                class_probs = y_prob[:, i]

                if self.method == 'isotonic':
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(class_probs, y_binary)
                else:  # Platt
                    from sklearn.linear_model import LogisticRegression
                    calibrator = LogisticRegression()
                    calibrator.fit(class_probs.reshape(-1, 1), y_binary)

                self.calibrators_[i] = calibrator

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        n_samples = y_prob.shape[0]
        n_classes = y_prob.shape[1]

        if n_classes == 2:
            # Binary calibration
            if self.method == 'isotonic':
                calibrated_pos = self.calibrators_[0].transform(y_prob[:, 1])
            else:
                calibrated_pos = self.calibrators_[0].predict_proba(
                    y_prob[:, 1].reshape(-1, 1)
                )[:, 1]

            calibrated = np.zeros((n_samples, 2))
            calibrated[:, 1] = calibrated_pos
            calibrated[:, 0] = 1 - calibrated_pos
        else:
            # Multiclass calibration
            calibrated = np.zeros_like(y_prob)

            for i in range(n_classes):
                if i in self.calibrators_:
                    if self.method == 'isotonic':
                        calibrated[:, i] = self.calibrators_[i].transform(y_prob[:, i])
                    else:
                        calibrated[:, i] = self.calibrators_[i].predict_proba(
                            y_prob[:, i].reshape(-1, 1)
                        )[:, 1]
                else:
                    calibrated[:, i] = y_prob[:, i]

            # Normalize to ensure sum to 1
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return calibrated


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification."""

    def __init__(
        self,
        confidence_level: float = 0.9,
        method: str = 'lac',
        mondrian: bool = False
    ):
        """
        Initialize conformal predictor.

        Args:
            confidence_level: Target coverage level (1 - alpha)
            method: Nonconformity method ('lac', 'aps', 'raps')
            mondrian: Whether to use Mondrian (class-conditional) CP
        """
        self.confidence_level = confidence_level
        self.method = method
        self.mondrian = mondrian
        self.alpha = 1 - confidence_level

        self.calibration_scores_ = None
        self.quantile_ = None
        self.classes_ = None

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_prob_cal: np.ndarray
    ):
        """
        Calibrate conformal predictor on calibration set.

        Args:
            y_cal: True labels for calibration
            y_prob_cal: Predicted probabilities for calibration

        Returns:
            self
        """
        n_cal = len(y_cal)
        self.classes_ = np.arange(y_prob_cal.shape[1])

        # Compute nonconformity scores
        if self.method == 'lac':  # Least Ambiguous set-valued Classifier
            # Score = 1 - p(true class)
            scores = np.zeros(n_cal)
            for i in range(n_cal):
                true_class = y_cal[i]
                if isinstance(true_class, str):
                    # Find class index
                    true_idx = np.where(self.classes_ == true_class)[0][0]
                else:
                    true_idx = int(true_class)
                scores[i] = 1 - y_prob_cal[i, true_idx]

        elif self.method == 'aps':  # Adaptive Prediction Sets
            # Score = sum of probabilities until true class
            scores = np.zeros(n_cal)
            for i in range(n_cal):
                sorted_idx = np.argsort(-y_prob_cal[i])  # Sort descending
                true_class = y_cal[i]
                if isinstance(true_class, str):
                    true_idx = np.where(self.classes_ == true_class)[0][0]
                else:
                    true_idx = int(true_class)

                # Find position of true class in sorted list
                true_pos = np.where(sorted_idx == true_idx)[0][0]
                scores[i] = np.sum(y_prob_cal[i, sorted_idx[:true_pos + 1]])

        else:  # Default to LAC
            scores = np.zeros(n_cal)
            for i in range(n_cal):
                true_class = int(y_cal[i]) if not isinstance(y_cal[i], str) else 0
                scores[i] = 1 - y_prob_cal[i, true_class]

        self.calibration_scores_ = scores

        # Compute quantile
        n_cal = len(scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = np.clip(q_level, 0, 1)
        self.quantile_ = np.quantile(scores, q_level)

        logger.info(f"Conformal calibration: quantile={self.quantile_:.3f} at level {q_level:.3f}")

        return self

    def predict(self, y_prob: np.ndarray) -> List[List[int]]:
        """
        Produce prediction sets.

        Args:
            y_prob: Predicted probabilities

        Returns:
            List of prediction sets (list of class indices)
        """
        if self.quantile_ is None:
            raise ValueError("Conformal predictor not calibrated")

        n_samples = y_prob.shape[0]
        n_classes = y_prob.shape[1]
        prediction_sets = []

        for i in range(n_samples):
            if self.method == 'lac':
                # Include classes with score <= quantile
                scores = 1 - y_prob[i]
                pred_set = [j for j in range(n_classes) if scores[j] <= self.quantile_]

            elif self.method == 'aps':
                # Include top classes until cumsum > quantile
                sorted_idx = np.argsort(-y_prob[i])
                cumsum = 0
                pred_set = []
                for j in sorted_idx:
                    cumsum += y_prob[i, j]
                    pred_set.append(j)
                    if cumsum > self.quantile_:
                        break
            else:
                # Default LAC
                scores = 1 - y_prob[i]
                pred_set = [j for j in range(n_classes) if scores[j] <= self.quantile_]

            # Ensure at least one prediction
            if len(pred_set) == 0:
                pred_set = [np.argmax(y_prob[i])]

            prediction_sets.append(pred_set)

        return prediction_sets

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        prediction_sets: List[List[int]]
    ) -> Dict[str, float]:
        """
        Evaluate coverage and efficiency of prediction sets.

        Args:
            y_true: True labels
            prediction_sets: Predicted sets

        Returns:
            Dictionary of metrics
        """
        n_samples = len(y_true)

        # Coverage: fraction where true label is in prediction set
        coverage_count = 0
        set_sizes = []

        for i in range(n_samples):
            true_class = int(y_true[i]) if not isinstance(y_true[i], str) else 0
            if true_class in prediction_sets[i]:
                coverage_count += 1
            set_sizes.append(len(prediction_sets[i]))

        marginal_coverage = coverage_count / n_samples
        mean_set_size = np.mean(set_sizes)

        # Conditional coverage by class
        class_coverage = {}
        for class_label in np.unique(y_true):
            class_mask = y_true == class_label
            class_cov = sum(
                1 for i in range(n_samples)
                if class_mask[i] and int(y_true[i]) in prediction_sets[i]
            ) / sum(class_mask)
            class_coverage[class_label] = class_cov

        return {
            'marginal_coverage': marginal_coverage,
            'mean_set_size': mean_set_size,
            'median_set_size': np.median(set_sizes),
            'class_coverage': class_coverage,
            'singleton_fraction': sum(1 for s in set_sizes if s == 1) / n_samples,
            'empty_fraction': sum(1 for s in set_sizes if s == 0) / n_samples
        }


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error.

    Args:
        y_true: True labels (binary)
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece