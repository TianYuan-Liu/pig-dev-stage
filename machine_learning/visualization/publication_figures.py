"""
Publication-ready visualization module for model results.
Creates high-quality figures suitable for scientific publications.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class PublicationFigures:
    """Generate publication-quality figures for ML results."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = 'whitegrid',
        palette: str = 'Set2',
        figure_format: str = 'pdf'
    ):
        """
        Initialize figure generator.

        Args:
            output_dir: Directory to save figures
            style: Seaborn style
            palette: Color palette
            figure_format: Output format (pdf, png, svg)
        """
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.style = style
        self.palette = palette
        self.figure_format = figure_format

        sns.set_style(style)
        sns.set_palette(palette)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        normalize: bool = True,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create publication-quality confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize (show percentages)
            title: Figure title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            fmt = '.1f'
            cbar_label = 'Percentage (%)'
        else:
            cm_normalized = cm
            fmt = 'd'
            cbar_label = 'Count'

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': cbar_label},
            ax=ax
        )

        # Set labels
        if labels:
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')

        if title:
            ax.set_title(title, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save if requested
        if save_name:
            save_path = self.output_dir / f"{save_name}.{self.figure_format}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        labels: Optional[List] = None,
        title: str = "ROC Curves",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curves with AUC values.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            labels: Class labels
            title: Figure title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Binary classification
        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                linewidth=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})'
            )

        # Multiclass classification
        else:
            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)

            # Plot ROC for each class
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)

                label = labels[i] if labels else f'Class {class_label}'
                ax.plot(
                    fpr, tpr,
                    linewidth=2,
                    label=f'{label} (AUC = {roc_auc:.3f})'
                )

            # Add micro-average
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(),
                y_prob.ravel()
            )
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            ax.plot(
                fpr_micro, tpr_micro,
                linewidth=2,
                linestyle='--',
                color='red',
                label=f'Micro-average (AUC = {roc_auc_micro:.3f})'
            )

        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.{self.figure_format}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")

        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        labels: Optional[List] = None,
        title: str = "Precision-Recall Curves",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curves with average precision.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            labels: Class labels
            title: Figure title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Binary classification
        if y_prob.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])

            ax.plot(
                recall, precision,
                linewidth=2,
                label=f'PR curve (AP = {avg_precision:.3f})'
            )

        # Multiclass classification
        else:
            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)

            for i, class_label in enumerate(classes):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i],
                    y_prob[:, i]
                )
                avg_precision = average_precision_score(
                    y_true_bin[:, i],
                    y_prob[:, i]
                )

                label = labels[i] if labels else f'Class {class_label}'
                ax.plot(
                    recall, precision,
                    linewidth=2,
                    label=f'{label} (AP = {avg_precision:.3f})'
                )

        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='lower left', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.{self.figure_format}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curves to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_scores: pd.Series,
        top_n: int = 30,
        title: str = "Top Features by Importance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance scores.

        Args:
            feature_scores: Series of feature scores
            top_n: Number of top features to show
            title: Figure title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        # Get top features
        top_features = feature_scores.nlargest(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features.values, color=sns.color_palette(self.palette)[0])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(top_features.values):
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.{self.figure_format}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")

        return fig

    def plot_cross_validation_performance(
        self,
        cv_scores: Dict[str, List],
        metrics: List[str] = ['balanced_accuracy', 'f1_macro'],
        title: str = "Cross-Validation Performance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cross-validation performance distributions.

        Args:
            cv_scores: Dictionary of metric scores across folds
            metrics: Metrics to plot
            title: Figure title
            save_name: Filename to save

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))

        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            if metric in cv_scores:
                scores = cv_scores[metric]

                # Violin plot
                parts = axes[idx].violinplot(
                    [scores],
                    positions=[0],
                    showmeans=True,
                    showextrema=True
                )

                # Customize colors
                for pc in parts['bodies']:
                    pc.set_facecolor(sns.color_palette(self.palette)[idx])
                    pc.set_alpha(0.7)

                # Add scatter points
                axes[idx].scatter(
                    np.zeros(len(scores)),
                    scores,
                    alpha=0.5,
                    s=30,
                    color='black'
                )

                # Statistics
                mean_val = np.mean(scores)
                std_val = np.std(scores)

                axes[idx].set_xticks([])
                axes[idx].set_ylabel('Score', fontweight='bold')
                axes[idx].set_title(
                    f'{metric.replace("_", " ").title()}\n'
                    f'Mean: {mean_val:.3f} ± {std_val:.3f}',
                    fontweight='bold'
                )
                axes[idx].grid(True, axis='y', alpha=0.3)
                axes[idx].set_ylim([0, 1.05])

        fig.suptitle(title, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_name:
            save_path = self.output_dir / f"{save_name}.{self.figure_format}"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved CV performance to {save_path}")

        return fig

    def create_performance_summary_table(
        self,
        results_dict: Dict[str, Dict],
        metrics: List[str] = ['balanced_accuracy', 'auroc', 'f1_macro', 'matthews_corrcoef'],
        save_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a summary table of performance metrics.

        Args:
            results_dict: Dictionary of tissue -> metrics
            metrics: Metrics to include
            save_name: Filename to save

        Returns:
            DataFrame with summary
        """
        rows = []

        for tissue, tissue_results in results_dict.items():
            row = {'Tissue': tissue}

            for metric in metrics:
                if metric in tissue_results:
                    value = tissue_results[metric]

                    # Format with confidence interval if available
                    if f'{metric}_ci' in tissue_results:
                        ci = tissue_results[f'{metric}_ci']
                        row[metric] = f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                    else:
                        row[metric] = f"{value:.3f}"
                else:
                    row[metric] = "N/A"

            # Add sample size if available
            if 'n_samples' in tissue_results:
                row['N'] = tissue_results['n_samples']

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by tissue name
        df = df.sort_values('Tissue')

        if save_name:
            # Save as CSV
            csv_path = self.output_dir / f"{save_name}.csv"
            df.to_csv(csv_path, index=False)

            # Save as LaTeX
            latex_path = self.output_dir / f"{save_name}.tex"
            df.to_latex(latex_path, index=False, escape=False)

            logger.info(f"Saved performance table to {csv_path} and {latex_path}")

        return df