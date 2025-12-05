"""
Statistical testing module for validating model performance.
Includes permutation testing and significance calculations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    permutation_test_score
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer
)
from joblib import Parallel, delayed
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class PermutationTest:
    """
    Permutation testing to establish statistical significance.
    Tests whether model performance is better than chance.
    """

    def __init__(
        self,
        model: BaseEstimator,
        n_permutations: int = 1000,
        scoring: str = 'balanced_accuracy',
        cv: int = 5,
        n_jobs: int = -1,
        seed: int = 42,
        verbose: int = 1
    ):
        """
        Initialize permutation tester.

        Args:
            model: Model to test
            n_permutations: Number of permutations
            scoring: Scoring metric
            cv: Number of CV folds
            n_jobs: Parallel jobs
            seed: Random seed
            verbose: Verbosity level
        """
        self.model = model
        self.n_permutations = n_permutations
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.observed_score_ = None
        self.permutation_scores_ = None
        self.p_value_ = None

    def test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        return_distribution: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform permutation test.

        Args:
            X: Feature matrix
            y: Target variable
            return_distribution: Whether to return full null distribution

        Returns:
            Dictionary with test results
        """
        logger.info(f"Starting permutation test with {self.n_permutations} permutations")

        # Set random seed
        np.random.seed(self.seed)

        # Get the scoring function
        if self.scoring == 'balanced_accuracy':
            scorer = make_scorer(balanced_accuracy_score)
        elif self.scoring == 'f1_macro':
            scorer = make_scorer(f1_score, average='macro', zero_division=0)
        elif self.scoring == 'auroc':
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorer = self.scoring

        # Run permutation test
        score, perm_scores, pvalue = permutation_test_score(
            estimator=clone(self.model),
            X=X,
            y=y,
            scoring=scorer,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed),
            n_permutations=self.n_permutations,
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbose=self.verbose
        )

        self.observed_score_ = score
        self.permutation_scores_ = perm_scores
        self.p_value_ = pvalue

        # Calculate confidence interval for null distribution
        null_ci = np.percentile(perm_scores, [2.5, 97.5])

        # Effect size (Cohen's d)
        effect_size = (score - np.mean(perm_scores)) / np.std(perm_scores)

        results = {
            'observed_score': score,
            'p_value': pvalue,
            'null_mean': np.mean(perm_scores),
            'null_std': np.std(perm_scores),
            'null_ci_lower': null_ci[0],
            'null_ci_upper': null_ci[1],
            'effect_size': effect_size,
            'significant': pvalue < 0.05
        }

        if return_distribution:
            results['permutation_scores'] = perm_scores

        if self.verbose > 0:
            logger.info(f"Permutation test complete:")
            logger.info(f"  Observed {self.scoring}: {score:.3f}")
            logger.info(f"  Null distribution: {np.mean(perm_scores):.3f} "
                       f"(+/- {np.std(perm_scores):.3f})")
            logger.info(f"  P-value: {pvalue:.4f}")
            logger.info(f"  Effect size: {effect_size:.2f}")

        return results

    def plot_null_distribution(self) -> Optional['matplotlib.figure.Figure']:
        """
        Plot the null distribution with observed score.

        Returns:
            Matplotlib figure
        """
        if self.permutation_scores_ is None:
            raise ValueError("No results available. Run test first.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot null distribution
            sns.histplot(
                self.permutation_scores_,
                kde=True,
                ax=ax,
                label=f'Null distribution (n={len(self.permutation_scores_)})'
            )

            # Add observed score
            ax.axvline(
                self.observed_score_,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Observed score: {self.observed_score_:.3f}'
            )

            # Add significance threshold (95th percentile)
            threshold = np.percentile(self.permutation_scores_, 95)
            ax.axvline(
                threshold,
                color='green',
                linestyle=':',
                linewidth=1,
                label=f'95% threshold: {threshold:.3f}'
            )

            ax.set_xlabel(self.scoring.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Permutation Test: p-value = {self.p_value_:.4f}')
            ax.legend()

            plt.tight_layout()
            return fig

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None


class MultipleTestingCorrection:
    """
    Handle multiple testing corrections for multiple comparisons.
    """

    @staticmethod
    def bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bonferroni correction.

        Args:
            p_values: Array of p-values
            alpha: Significance level

        Returns:
            (adjusted_p_values, significant_mask)
        """
        n_tests = len(p_values)
        adjusted_p = np.minimum(p_values * n_tests, 1.0)
        significant = adjusted_p < alpha
        return adjusted_p, significant

    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR correction.

        Args:
            p_values: Array of p-values
            alpha: Significance level

        Returns:
            (adjusted_p_values, significant_mask)
        """
        from statsmodels.stats.multitest import multipletests

        rejected, adjusted_p, _, _ = multipletests(
            p_values,
            alpha=alpha,
            method='fdr_bh'
        )

        return adjusted_p, rejected


class BootstrapSignificanceTest:
    """
    Bootstrap-based significance testing for model comparison.
    """

    def __init__(
        self,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42
    ):
        """
        Initialize bootstrap tester.

        Args:
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level
            seed: Random seed
        """
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.seed = seed

    def compare_models(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        metric_func: Callable = balanced_accuracy_score
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Compare two models using bootstrap.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            metric_func: Metric function

        Returns:
            Dictionary with comparison results
        """
        np.random.seed(self.seed)
        n_samples = len(y_true)

        # Calculate observed difference
        score1 = metric_func(y_true, y_pred1)
        score2 = metric_func(y_true, y_pred2)
        observed_diff = score1 - score2

        # Bootstrap distribution of differences
        bootstrap_diffs = []

        for _ in range(self.n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)

            y_true_boot = y_true[indices]
            y_pred1_boot = y_pred1[indices]
            y_pred2_boot = y_pred2[indices]

            # Calculate difference
            score1_boot = metric_func(y_true_boot, y_pred1_boot)
            score2_boot = metric_func(y_true_boot, y_pred2_boot)
            diff_boot = score1_boot - score2_boot

            bootstrap_diffs.append(diff_boot)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs - np.mean(bootstrap_diffs)) >= np.abs(observed_diff))

        # Confidence interval for difference
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return {
            'model1_score': score1,
            'model2_score': score2,
            'observed_difference': observed_diff,
            'p_value': p_value,
            'ci_difference': (ci_lower, ci_upper),
            'significant': p_value < 0.05,
            'bootstrap_mean_diff': np.mean(bootstrap_diffs),
            'bootstrap_std_diff': np.std(bootstrap_diffs)
        }


class McNemarTest:
    """
    McNemar's test for comparing two classifiers on the same dataset.
    """

    @staticmethod
    def test(
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        correction: bool = True
    ) -> Dict[str, float]:
        """
        Perform McNemar's test.

        Args:
            y_true: True labels
            y_pred1: Predictions from classifier 1
            y_pred2: Predictions from classifier 2
            correction: Apply continuity correction

        Returns:
            Dictionary with test results
        """
        # Create contingency table
        correct1 = y_pred1 == y_true
        correct2 = y_pred2 == y_true

        # Count disagreements
        n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 right
        n10 = np.sum(correct1 & ~correct2)  # Model 1 right, Model 2 wrong

        # McNemar's statistic
        if correction and (n01 + n10) > 0:
            # With continuity correction
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        elif n01 + n10 > 0:
            # Without correction
            statistic = (n01 - n10) ** 2 / (n01 + n10)
        else:
            statistic = 0

        # P-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            'n01': n01,
            'n10': n10,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }