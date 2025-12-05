"""
Stage granularity selection module.
Implements adaptive merging to select optimal stage classification scheme per tissue.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StageGranularitySelector:
    """Select optimal stage classification granularity based on sample counts."""

    def __init__(
        self,
        min_samples_4class: int = 40,
        min_samples_3class: int = 30,
        min_samples_2class: int = 25,
        min_total_2class: int = 60
    ):
        """
        Initialize stage selector.

        Args:
            min_samples_4class: Minimum samples per class for 4-class scheme
            min_samples_3class: Minimum samples per class for 3-class scheme
            min_samples_2class: Minimum samples per class for 2-class scheme
            min_total_2class: Minimum total samples for 2-class scheme
        """
        self.min_samples_4class = min_samples_4class
        self.min_samples_3class = min_samples_3class
        self.min_samples_2class = min_samples_2class
        self.min_total_2class = min_total_2class

        self.stage_order = ['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal', 'Adult']

        # Define stage merging schemes
        self.schemes = {
            '5-class': {
                'n_classes': 5,
                'mapping': {
                    'Infant': 'Infant',
                    'Early childhood': 'Early childhood',
                    'Pre-pubertal': 'Pre-pubertal',
                    'Post-pubertal': 'Post-pubertal',
                    'Adult': 'Adult'
                },
                'labels': ['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal', 'Adult']
            },
            '4-class': {
                'n_classes': 4,
                'mapping': {
                    'Infant': 'Infant',
                    'Early childhood': 'Early childhood',
                    'Pre-pubertal': 'Pre-pubertal',
                    'Post-pubertal': 'Post-pubertal/Adult',
                    'Adult': 'Post-pubertal/Adult'
                },
                'labels': ['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal/Adult']
            },
            '3-class': {
                'n_classes': 3,
                'mapping': {
                    'Infant': 'Early (0-59d)',
                    'Early childhood': 'Early (0-59d)',
                    'Pre-pubertal': 'Pre-pubertal (60-149d)',
                    'Post-pubertal': 'Late (≥150d)',
                    'Adult': 'Late (≥150d)'
                },
                'labels': ['Early (0-59d)', 'Pre-pubertal (60-149d)', 'Late (≥150d)']
            },
            '2-class': {
                'n_classes': 2,
                'mapping': {
                    'Infant': 'Pre-pubertal (<150d)',
                    'Early childhood': 'Pre-pubertal (<150d)',
                    'Pre-pubertal': 'Pre-pubertal (<150d)',
                    'Post-pubertal': 'Post-pubertal (≥150d)',
                    'Adult': 'Post-pubertal (≥150d)'
                },
                'labels': ['Pre-pubertal (<150d)', 'Post-pubertal (≥150d)']
            }
        }

    def select_scheme(
        self,
        stage_counts: pd.Series,
        tissue_name: str = ""
    ) -> Tuple[str, Dict]:
        """
        Select optimal classification scheme for a tissue.

        Args:
            stage_counts: Series with stage names as index and counts as values
            tissue_name: Name of tissue (for logging)

        Returns:
            Tuple of (scheme_name, scheme_info_dict)
        """
        # Ensure we have counts for all expected stages
        counts = np.zeros(5)
        for i, stage in enumerate(self.stage_order):
            if stage in stage_counts.index:
                counts[i] = stage_counts[stage]

        total_with_age = counts.sum()

        # Check 4-class eligibility
        merged_4class = counts.copy()
        merged_4class[3] += merged_4class[4]  # Merge Adult into Post-pubertal
        merged_4class = merged_4class[:4]

        if all(merged_4class >= self.min_samples_4class):
            logger.info(f"{tissue_name}: Selected 4-class scheme")
            return '4-class', self.schemes['4-class']

        # Check 3-class eligibility
        merged_3class = np.array([
            counts[0] + counts[1],  # Infant + Early childhood
            counts[2],               # Pre-pubertal
            counts[3] + counts[4]    # Post-pubertal + Adult
        ])

        if all(merged_3class >= self.min_samples_3class):
            logger.info(f"{tissue_name}: Selected 3-class scheme")
            return '3-class', self.schemes['3-class']

        # Check 2-class eligibility
        merged_2class = np.array([
            counts[:3].sum(),  # Pre-pubertal (<150d)
            counts[3:].sum()   # Post-pubertal (≥150d)
        ])

        if all(merged_2class >= self.min_samples_2class) and \
           total_with_age >= self.min_total_2class:
            logger.info(f"{tissue_name}: Selected 2-class scheme")
            return '2-class', self.schemes['2-class']

        # Not eligible for any scheme
        logger.warning(
            f"{tissue_name}: Not eligible for any classification scheme. "
            f"Counts: {dict(zip(self.stage_order, counts))}"
        )
        return None, None

    def evaluate_all_tissues(
        self,
        tissue_stage_counts: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Evaluate all tissues and select optimal schemes.

        Args:
            tissue_stage_counts: DataFrame with tissues as rows, stages as columns

        Returns:
            Dictionary mapping tissue names to scheme information
        """
        results = {}

        for tissue in tissue_stage_counts.index:
            if tissue == 'Total' or tissue == 'Age_Missing':
                continue

            # Get stage counts for this tissue
            stage_counts = tissue_stage_counts.loc[tissue, self.stage_order]

            # Select optimal scheme
            scheme_name, scheme_info = self.select_scheme(stage_counts, tissue)

            if scheme_name:
                results[tissue] = {
                    'scheme': scheme_name,
                    'n_classes': scheme_info['n_classes'],
                    'mapping': scheme_info['mapping'],
                    'labels': scheme_info['labels'],
                    'original_counts': stage_counts.to_dict(),
                    'merged_counts': self._get_merged_counts(stage_counts, scheme_info['mapping'])
                }
            else:
                results[tissue] = {
                    'scheme': None,
                    'n_classes': 0,
                    'eligible': False,
                    'original_counts': stage_counts.to_dict()
                }

        return results

    def _get_merged_counts(
        self,
        stage_counts: pd.Series,
        mapping: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Get counts after merging stages according to mapping.

        Args:
            stage_counts: Original stage counts
            mapping: Stage mapping dictionary

        Returns:
            Dictionary of merged class counts
        """
        merged = {}
        for original_stage, mapped_stage in mapping.items():
            if original_stage in stage_counts.index:
                count = stage_counts[original_stage]
                if mapped_stage in merged:
                    merged[mapped_stage] += count
                else:
                    merged[mapped_stage] = count

        return merged

    def get_eligible_tissues(
        self,
        tissue_stage_counts: pd.DataFrame,
        min_scheme: Optional[str] = None
    ) -> List[str]:
        """
        Get list of tissues eligible for classification.

        Args:
            tissue_stage_counts: DataFrame with tissue stage counts
            min_scheme: Minimum scheme required ('2-class', '3-class', or '4-class')

        Returns:
            List of eligible tissue names
        """
        results = self.evaluate_all_tissues(tissue_stage_counts)

        eligible = []
        for tissue, info in results.items():
            if info['scheme'] is not None:
                if min_scheme is None:
                    eligible.append(tissue)
                elif min_scheme == '2-class':
                    eligible.append(tissue)
                elif min_scheme == '3-class' and info['scheme'] in ['3-class', '4-class']:
                    eligible.append(tissue)
                elif min_scheme == '4-class' and info['scheme'] == '4-class':
                    eligible.append(tissue)

        return eligible

    def prepare_labels(
        self,
        stages: pd.Series,
        scheme_name: str
    ) -> pd.Series:
        """
        Convert original stage labels according to selected scheme.

        Args:
            stages: Original stage labels
            scheme_name: Name of classification scheme

        Returns:
            Mapped stage labels
        """
        if scheme_name not in self.schemes:
            raise ValueError(f"Unknown scheme: {scheme_name}")

        scheme = self.schemes[scheme_name]
        mapped = stages.map(scheme['mapping'])

        # Convert to ordered categorical
        mapped = pd.Categorical(
            mapped,
            categories=scheme['labels'],
            ordered=True
        )

        return mapped

    def get_summary_table(
        self,
        tissue_stage_counts: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create summary table of tissue eligibility.

        Args:
            tissue_stage_counts: DataFrame with tissue stage counts

        Returns:
            Summary DataFrame
        """
        results = self.evaluate_all_tissues(tissue_stage_counts)

        summary_data = []
        for tissue, info in results.items():
            row = {
                'Tissue': tissue,
                'Scheme': info['scheme'] or 'Ineligible',
                'N_Classes': info['n_classes'],
                'Total_Samples': sum(info['original_counts'].values())
            }

            # Add original stage counts
            for stage in self.stage_order:
                row[f'{stage}_Count'] = info['original_counts'].get(stage, 0)

            summary_data.append(row)

        summary = pd.DataFrame(summary_data)

        # Sort by scheme (4-class first) and then by total samples
        scheme_order = {'4-class': 0, '3-class': 1, '2-class': 2, 'Ineligible': 3}
        summary['_scheme_order'] = summary['Scheme'].map(scheme_order)
        summary = summary.sort_values(['_scheme_order', 'Total_Samples'], ascending=[True, False])
        summary = summary.drop('_scheme_order', axis=1)

        return summary