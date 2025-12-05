"""
Data loader module for pig developmental stage classification.
Handles loading TPM matrices and metadata from pigGTEx resources.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and process pigGTEx RNA-seq data with metadata."""

    STAGE_MAPPING = {
        'Infant': (0, 20),
        'Early childhood': (21, 59),
        'Pre-pubertal': (60, 149),
        'Post-pubertal': (150, 365),
        'Adult': (366, float('inf'))
    }

    STAGE_ORDER = ['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal', 'Adult']

    def __init__(self, data_dir: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing pigGTEx expression files
            metadata_path: Path to metadata file (Excel, CSV, or JSON)
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.metadata = None
        self.expression_data = {}

        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")

    def age_to_stage(self, age_days: float) -> str:
        """
        Map age in days to developmental stage.

        Args:
            age_days: Age in days

        Returns:
            Stage label
        """
        if pd.isna(age_days):
            return None

        for stage, (min_age, max_age) in self.STAGE_MAPPING.items():
            if min_age <= age_days <= max_age:
                return stage

        return 'Adult'  # Default for ages > 365

    def load_metadata(self, metadata_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load metadata from file.

        Args:
            metadata_file: Optional path to metadata file

        Returns:
            DataFrame with sample metadata
        """
        if metadata_file:
            self.metadata_path = Path(metadata_file)

        if not self.metadata_path:
            logger.warning("No metadata file specified, creating minimal metadata from filenames")
            return self._create_minimal_metadata()

        suffix = self.metadata_path.suffix.lower()

        if suffix == '.xlsx':
            self.metadata = pd.read_excel(self.metadata_path)
        elif suffix == '.csv':
            self.metadata = pd.read_csv(self.metadata_path)
        elif suffix == '.tsv' or suffix == '.txt':
            self.metadata = pd.read_csv(self.metadata_path, sep='\t')
        elif suffix == '.json':
            with open(self.metadata_path) as f:
                data = json.load(f)
            self.metadata = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported metadata file format: {suffix}")

        # Standardize column names
        self.metadata.columns = [col.strip().replace(' ', '_') for col in self.metadata.columns]

        # Set Sample_ID as index if present
        if 'Sample_ID' in self.metadata.columns:
            self.metadata = self.metadata.set_index('Sample_ID')

        # Add stage column if age is present
        if 'Age' in self.metadata.columns:
            self.metadata['Stage'] = self.metadata['Age'].apply(self.age_to_stage)
            self.metadata['Stage'] = pd.Categorical(
                self.metadata['Stage'],
                categories=self.STAGE_ORDER,
                ordered=True
            )

        # Encode sex with explicit Unknown level
        if 'Sex' in self.metadata.columns:
            self.metadata['Sex'] = self.metadata['Sex'].fillna('Unknown')
            self.metadata['Sex'] = pd.Categorical(self.metadata['Sex'])

        logger.info(f"Loaded metadata for {len(self.metadata)} samples")
        return self.metadata

    def _create_minimal_metadata(self) -> pd.DataFrame:
        """Create minimal metadata from available expression files."""
        files = list(self.data_dir.glob("*.expr_tpm.txt.gz"))

        metadata = []
        for file in files:
            tissue = file.stem.replace('.expr_tpm.txt', '')
            metadata.append({'Tissue': tissue, 'File': file.name})

        self.metadata = pd.DataFrame(metadata)
        return self.metadata

    def load_expression(
        self,
        tissue: str,
        min_tpm: float = 0.1,
        min_detection_rate: float = 0.1,
        protein_coding_only: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load expression data for a specific tissue.

        Args:
            tissue: Tissue name
            min_tpm: Minimum TPM threshold for detection
            min_detection_rate: Minimum fraction of samples with expression
            protein_coding_only: Whether to filter to protein-coding genes

        Returns:
            Tuple of (expression_matrix, sample_metadata)
        """
        # Find expression file (handle spaces in tissue names)
        expr_files = list(self.data_dir.glob(f"{tissue}*.expr_tpm.txt.gz"))
        if not expr_files:
            # Try with underscores instead of spaces
            tissue_underscore = tissue.replace(' ', '_')
            expr_files = list(self.data_dir.glob(f"{tissue_underscore}*.expr_tpm.txt.gz"))

        if not expr_files:
            raise ValueError(f"No expression file found for tissue: {tissue} or {tissue.replace(' ', '_')}")

        expr_file = expr_files[0]
        logger.info(f"Loading expression data from {expr_file}")

        # Load expression matrix
        with gzip.open(expr_file, 'rt') as f:
            expr_df = pd.read_csv(f, sep='\t', index_col=0)

        logger.info(f"Loaded expression matrix: {expr_df.shape}")

        # Filter genes by detection rate
        detection_rate = (expr_df > min_tpm).mean(axis=1)
        keep_genes = detection_rate >= min_detection_rate

        expr_df = expr_df.loc[keep_genes]
        logger.info(f"After filtering by detection rate: {expr_df.shape}")

        # Get sample metadata for this tissue
        if self.metadata is not None and 'Tissue' in self.metadata.columns:
            tissue_mask = self.metadata['Tissue'] == tissue
            tissue_metadata = self.metadata[tissue_mask].copy()

            # Align expression and metadata by matching sample IDs
            if len(tissue_metadata) > 0:
                # Find common samples between expression and metadata
                common_samples = [s for s in expr_df.columns if s in tissue_metadata.index]

                if len(common_samples) == 0:
                    # If no exact matches, try to match by position if counts are the same
                    logger.info(f"Sample ID formats differ between expression and metadata for {tissue}")
                    logger.info(f"Expression IDs example: {list(expr_df.columns[:3])}")
                    logger.info(f"Metadata IDs example: {list(tissue_metadata.index[:3])}")

                    if len(tissue_metadata) == len(expr_df.columns):
                        logger.info(f"Sample counts match ({len(tissue_metadata)}), using positional alignment")
                        tissue_metadata.index = expr_df.columns
                        common_samples = expr_df.columns.tolist()
                    else:
                        # Use min of both as a fallback
                        n_samples = min(len(tissue_metadata), len(expr_df.columns))
                        logger.info(f"Sample count mismatch: {len(expr_df.columns)} expression vs {len(tissue_metadata)} metadata")
                        logger.info(f"Using first {n_samples} samples from both datasets via positional alignment")
                        tissue_metadata = tissue_metadata.iloc[:n_samples]
                        tissue_metadata.index = expr_df.columns[:n_samples]
                        common_samples = expr_df.columns[:n_samples].tolist()

                # Filter both to common samples
                expr_df = expr_df[common_samples]
                tissue_metadata = tissue_metadata.loc[common_samples]

                logger.info(f"Aligned {len(common_samples)} samples for {tissue}")
        else:
            # Create minimal metadata
            tissue_metadata = pd.DataFrame(
                index=expr_df.columns,
                data={'Tissue': tissue, 'Sample_ID': expr_df.columns}
            )

        self.expression_data[tissue] = expr_df

        return expr_df, tissue_metadata

    def load_all_tissues(
        self,
        tissues: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load expression data for multiple tissues.

        Args:
            tissues: List of tissue names (None = load all)
            **kwargs: Arguments passed to load_expression

        Returns:
            Dictionary mapping tissue names to (expression, metadata) tuples
        """
        if tissues is None:
            # Find all available tissues
            expr_files = list(self.data_dir.glob("*.expr_tpm.txt.gz"))
            tissues = [f.stem.replace('.expr_tpm.txt', '') for f in expr_files]

        results = {}
        for tissue in tissues:
            try:
                results[tissue] = self.load_expression(tissue, **kwargs)
                logger.info(f"Successfully loaded {tissue}")
            except Exception as e:
                logger.error(f"Failed to load {tissue}: {e}")

        return results

    def get_tissue_stage_counts(self) -> pd.DataFrame:
        """
        Get sample counts by tissue and stage.

        Returns:
            DataFrame with tissues as rows and stages as columns
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded")

        if 'Tissue' not in self.metadata.columns or 'Stage' not in self.metadata.columns:
            raise ValueError("Metadata must contain 'Tissue' and 'Stage' columns")

        counts = pd.crosstab(self.metadata['Tissue'], self.metadata['Stage'])
        counts = counts.reindex(columns=self.STAGE_ORDER, fill_value=0)

        # Add total column
        counts['Total'] = counts.sum(axis=1)

        # Add age missing count
        tissue_totals = self.metadata.groupby('Tissue').size()
        tissue_with_age = self.metadata[self.metadata['Stage'].notna()].groupby('Tissue').size()
        counts['Age_Missing'] = tissue_totals - tissue_with_age.reindex(tissue_totals.index, fill_value=0)

        return counts.sort_values('Total', ascending=False)

    def get_eligible_tissues(
        self,
        min_samples_4class: int = 40,
        min_samples_3class: int = 30,
        min_samples_2class: int = 25,
        min_total_2class: int = 60
    ) -> Dict[str, Dict]:
        """
        Determine eligible tissues and their stage schemes based on sample counts.

        Args:
            min_samples_4class: Minimum samples per class for 4-class
            min_samples_3class: Minimum samples per class for 3-class
            min_samples_2class: Minimum samples per class for 2-class
            min_total_2class: Minimum total samples for 2-class

        Returns:
            Dictionary with tissue eligibility information
        """
        counts = self.get_tissue_stage_counts()
        eligible = {}

        for tissue in counts.index:
            tissue_counts = counts.loc[tissue, self.STAGE_ORDER].values
            total_with_age = tissue_counts.sum()

            tissue_info = {
                'total_samples': int(counts.loc[tissue, 'Total']),
                'samples_with_age': int(total_with_age),
                'age_missing': int(counts.loc[tissue, 'Age_Missing']),
                'stage_counts': dict(zip(self.STAGE_ORDER, tissue_counts.astype(int))),
                'eligible': False,
                'stage_scheme': None,
                'n_classes': 0
            }

            # Check 4-class eligibility (merge Adult into Post-pubertal)
            merged_4class = tissue_counts.copy()
            merged_4class[3] += merged_4class[4]  # Merge Adult into Post-pubertal
            merged_4class = merged_4class[:4]

            if all(merged_4class >= min_samples_4class):
                tissue_info['eligible'] = True
                tissue_info['stage_scheme'] = '4-class'
                tissue_info['n_classes'] = 4
                tissue_info['class_mapping'] = {
                    'Infant': 'Infant',
                    'Early childhood': 'Early childhood',
                    'Pre-pubertal': 'Pre-pubertal',
                    'Post-pubertal': 'Post-pubertal/Adult',
                    'Adult': 'Post-pubertal/Adult'
                }
            # Check 3-class eligibility
            elif tissue_counts[0] + tissue_counts[1] >= min_samples_3class and \
                 tissue_counts[2] >= min_samples_3class and \
                 tissue_counts[3] + tissue_counts[4] >= min_samples_3class:
                tissue_info['eligible'] = True
                tissue_info['stage_scheme'] = '3-class'
                tissue_info['n_classes'] = 3
                tissue_info['class_mapping'] = {
                    'Infant': 'Early (0-59d)',
                    'Early childhood': 'Early (0-59d)',
                    'Pre-pubertal': 'Pre-pubertal (60-149d)',
                    'Post-pubertal': 'Late (150+d)',
                    'Adult': 'Late (150+d)'
                }
            # Check 2-class eligibility
            elif tissue_counts[:3].sum() >= min_samples_2class and \
                 tissue_counts[3:].sum() >= min_samples_2class and \
                 total_with_age >= min_total_2class:
                tissue_info['eligible'] = True
                tissue_info['stage_scheme'] = '2-class'
                tissue_info['n_classes'] = 2
                tissue_info['class_mapping'] = {
                    'Infant': 'Pre-pubertal (<150d)',
                    'Early childhood': 'Pre-pubertal (<150d)',
                    'Pre-pubertal': 'Pre-pubertal (<150d)',
                    'Post-pubertal': 'Post-pubertal (≥150d)',
                    'Adult': 'Post-pubertal (≥150d)'
                }

            eligible[tissue] = tissue_info

        return eligible