"""
Identification of conserved developmental markers across tissues.
Finds genes with consistent developmental patterns while respecting tissue-specific biology.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ConservedMarkerIdentifier:
    """
    Identifies genes with conserved developmental patterns across tissues.

    This respects the biological reality that:
    1. Some genes show consistent developmental patterns across tissues (conserved)
    2. Other genes are tissue-specific in their developmental expression
    3. Both types are biologically important
    """

    def __init__(
        self,
        n_conserved: int = 100,
        n_tissue_specific: int = 50,
        consistency_threshold: float = 0.7,
        min_tissues: int = 3,
        random_state: int = 42
    ):
        """
        Initialize conserved marker identifier.

        Args:
            n_conserved: Number of conserved markers to identify
            n_tissue_specific: Number of tissue-specific markers per tissue
            consistency_threshold: Minimum correlation for conserved pattern
            min_tissues: Minimum tissues showing pattern for conserved status
            random_state: Random seed
        """
        self.n_conserved = n_conserved
        self.n_tissue_specific = n_tissue_specific
        self.consistency_threshold = consistency_threshold
        self.min_tissues = min_tissues
        self.random_state = random_state

        # Will be populated during analysis
        self.conserved_markers = None
        self.tissue_specific_markers = {}
        self.marker_scores = None
        self.tissue_patterns = {}

    def identify_markers(
        self,
        tissue_data: Dict[str, Dict],
        save_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Identify conserved and tissue-specific developmental markers.

        Args:
            tissue_data: Dict with tissue names as keys, each containing:
                - 'expression': Gene expression matrix (genes x samples)
                - 'metadata': Sample metadata with 'Stage' column
            save_path: Optional path to save results

        Returns:
            Dictionary with marker categories and gene indices
        """
        logger.info("Identifying conserved developmental markers...")

        # Extract common genes across tissues
        common_genes = self._find_common_genes(tissue_data)
        logger.info(f"Found {len(common_genes)} common genes across tissues")

        # Calculate developmental scores for each gene in each tissue
        tissue_scores = {}
        tissue_patterns = {}

        for tissue, data in tissue_data.items():
            expr = data['expression']
            stages = data['metadata']['Stage'].values

            # Filter to common genes
            gene_mask = np.isin(np.arange(expr.shape[0]), common_genes)
            expr_common = expr[gene_mask]

            # Calculate developmental association scores
            scores, patterns = self._calculate_developmental_scores(
                expr_common, stages
            )
            tissue_scores[tissue] = scores
            tissue_patterns[tissue] = patterns

        self.tissue_patterns = tissue_patterns

        # Identify conserved markers
        self.conserved_markers = self._identify_conserved(
            tissue_scores, tissue_patterns
        )

        # Identify tissue-specific markers
        for tissue in tissue_data.keys():
            self.tissue_specific_markers[tissue] = self._identify_tissue_specific(
                tissue_scores, tissue
            )

        # Create comprehensive marker dictionary
        markers = {
            'conserved': self.conserved_markers,
            'common_genes': common_genes,
            **{f'tissue_specific_{t}': idx
               for t, idx in self.tissue_specific_markers.items()}
        }

        # Save results if requested
        if save_path:
            self._save_markers(markers, tissue_scores, save_path)

        logger.info(f"Identified {len(self.conserved_markers)} conserved markers")
        logger.info(f"Identified tissue-specific markers for {len(self.tissue_specific_markers)} tissues")

        return markers

    def _find_common_genes(self, tissue_data: Dict) -> np.ndarray:
        """Find genes present in all tissues."""
        # For simplicity, assume all tissues have the same genes
        # In reality, you'd need to match by gene names
        first_tissue = list(tissue_data.keys())[0]
        n_genes = tissue_data[first_tissue]['expression'].shape[0]

        # Find minimum number of genes across tissues
        for tissue, data in tissue_data.items():
            n_genes = min(n_genes, data['expression'].shape[0])

        return np.arange(n_genes)

    def _calculate_developmental_scores(
        self,
        expression: np.ndarray,
        stages: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate how strongly each gene is associated with developmental stages.

        Returns:
            - scores: Association strength for each gene
            - patterns: Expression pattern across stages for each gene
        """
        # Encode stages numerically if needed
        from sklearn.preprocessing import LabelEncoder
        if stages.dtype == object:
            le = LabelEncoder()
            stages_encoded = le.fit_transform(stages)
            stage_order = le.classes_
        else:
            stages_encoded = stages
            stage_order = np.unique(stages)

        n_genes = expression.shape[0]
        scores = np.zeros(n_genes)
        patterns = np.zeros((n_genes, len(stage_order)))

        # Calculate F-statistic for each gene
        for i in range(n_genes):
            gene_expr = expression[i, :]

            # Skip genes with no variation
            if np.std(gene_expr) == 0:
                continue

            # F-statistic for association with stages
            f_stat, p_val = stats.f_oneway(
                *[gene_expr[stages_encoded == s] for s in range(len(stage_order))]
            )

            # Store score (higher is stronger association)
            scores[i] = f_stat if not np.isnan(f_stat) else 0

            # Calculate mean expression pattern across stages
            for j, stage in enumerate(range(len(stage_order))):
                stage_mask = stages_encoded == stage
                if np.any(stage_mask):
                    patterns[i, j] = np.mean(gene_expr[stage_mask])

        return scores, patterns

    def _identify_conserved(
        self,
        tissue_scores: Dict[str, np.ndarray],
        tissue_patterns: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Identify genes with conserved developmental patterns across tissues.
        """
        # Convert to arrays for easier manipulation
        tissues = list(tissue_scores.keys())
        n_genes = len(tissue_scores[tissues[0]])

        # Calculate pattern consistency across tissues
        consistency_scores = np.zeros(n_genes)

        for i in range(n_genes):
            # Get patterns for this gene across tissues
            gene_patterns = []
            gene_scores = []

            for tissue in tissues:
                if i < len(tissue_patterns[tissue]):
                    gene_patterns.append(tissue_patterns[tissue][i])
                    gene_scores.append(tissue_scores[tissue][i])

            if len(gene_patterns) < self.min_tissues:
                continue

            # Calculate pairwise correlations between tissue patterns
            correlations = []
            for j in range(len(gene_patterns)):
                for k in range(j + 1, len(gene_patterns)):
                    if np.std(gene_patterns[j]) > 0 and np.std(gene_patterns[k]) > 0:
                        corr = np.corrcoef(gene_patterns[j], gene_patterns[k])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

            # Consistency score: mean correlation weighted by expression strength
            if correlations:
                mean_correlation = np.mean(correlations)
                mean_score = np.mean(gene_scores)
                consistency_scores[i] = mean_correlation * mean_score

        # Select top conserved genes
        n_select = min(self.n_conserved, np.sum(consistency_scores > 0))
        conserved_indices = np.argsort(consistency_scores)[-n_select:]

        # Filter by consistency threshold
        conserved_filtered = []
        for idx in conserved_indices:
            if consistency_scores[idx] >= self.consistency_threshold:
                conserved_filtered.append(idx)

        return np.array(conserved_filtered)

    def _identify_tissue_specific(
        self,
        tissue_scores: Dict[str, np.ndarray],
        target_tissue: str
    ) -> np.ndarray:
        """
        Identify genes specific to a particular tissue's development.
        """
        # Get scores for target tissue
        target_scores = tissue_scores[target_tissue].copy()

        # Calculate specificity: high in target, low in others
        specificity_scores = np.zeros_like(target_scores)

        for i in range(len(target_scores)):
            target_score = target_scores[i]

            # Get scores in other tissues
            other_scores = []
            for tissue, scores in tissue_scores.items():
                if tissue != target_tissue and i < len(scores):
                    other_scores.append(scores[i])

            if other_scores:
                # Specificity: target score / mean of others
                mean_other = np.mean(other_scores)
                if mean_other > 0:
                    specificity_scores[i] = target_score / mean_other
                else:
                    specificity_scores[i] = target_score

        # Select top tissue-specific genes
        n_select = min(self.n_tissue_specific, np.sum(specificity_scores > 1))
        specific_indices = np.argsort(specificity_scores)[-n_select:]

        return specific_indices

    def plot_marker_patterns(
        self,
        tissue_data: Dict,
        marker_type: str = 'conserved',
        n_genes: int = 10,
        save_path: Optional[Path] = None
    ):
        """
        Visualize expression patterns of identified markers.
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        if marker_type == 'conserved':
            markers = self.conserved_markers[:n_genes]
            title_prefix = "Conserved"
        else:
            tissue = marker_type.replace('tissue_specific_', '')
            markers = self.tissue_specific_markers.get(tissue, [])[:n_genes]
            title_prefix = f"{tissue}-specific"

        for idx, marker_idx in enumerate(markers):
            if idx >= 10:
                break

            ax = axes[idx]

            # Plot pattern across tissues
            for tissue, patterns in self.tissue_patterns.items():
                if marker_idx < len(patterns):
                    pattern = patterns[marker_idx]
                    ax.plot(pattern, label=tissue, marker='o')

            ax.set_title(f"{title_prefix} Gene {marker_idx}")
            ax.set_xlabel("Developmental Stage")
            ax.set_ylabel("Expression")
            ax.legend(fontsize=6)

        plt.suptitle(f"{title_prefix} Developmental Markers", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def create_marker_heatmap(
        self,
        tissue_data: Dict,
        save_path: Optional[Path] = None
    ):
        """
        Create heatmap showing marker expression across tissues and stages.
        """
        # Prepare data for heatmap
        all_markers = np.concatenate([
            self.conserved_markers[:20],
            *[markers[:10] for markers in self.tissue_specific_markers.values()]
        ])

        # Create matrix
        tissues = list(tissue_data.keys())
        data_matrix = []
        labels = []

        for tissue in tissues:
            patterns = self.tissue_patterns[tissue]
            for marker_idx in all_markers:
                if marker_idx < len(patterns):
                    data_matrix.append(patterns[marker_idx])
                    labels.append(f"{tissue}_Gene{marker_idx}")

        data_matrix = np.array(data_matrix)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            data_matrix,
            xticklabels=[f"Stage_{i}" for i in range(data_matrix.shape[1])],
            yticklabels=labels,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Expression Level'}
        )
        plt.title("Developmental Marker Expression Patterns")
        plt.xlabel("Developmental Stage")
        plt.ylabel("Tissue / Gene")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def _save_markers(
        self,
        markers: Dict,
        tissue_scores: Dict,
        save_path: Path
    ):
        """Save identified markers to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        markers_json = {}
        for key, value in markers.items():
            if isinstance(value, np.ndarray):
                markers_json[key] = value.tolist()
            else:
                markers_json[key] = value

        # Add scores
        scores_json = {}
        for tissue, scores in tissue_scores.items():
            scores_json[tissue] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'max': float(np.max(scores))
            }

        output = {
            'markers': markers_json,
            'scores_summary': scores_json,
            'parameters': {
                'n_conserved': self.n_conserved,
                'n_tissue_specific': self.n_tissue_specific,
                'consistency_threshold': self.consistency_threshold,
                'min_tissues': self.min_tissues
            }
        }

        import json
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Markers saved to {save_path}")