"""
Data loader module for pig developmental stage classification.
Handles loading TPM matrices and metadata from pigGTEx resources.
"""

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_default_stage_mapping():
    """Get default stage mapping (used when config unavailable)."""
    return {
        'Infant': (0, 20),
        'Early childhood': (21, 59),
        'Pre-pubertal': (60, 149),
        'Post-pubertal': (150, 365),
        'Adult': (366, float('inf'))
    }


def _get_default_age_conversion():
    """Get default age unit conversion factors (used when config unavailable)."""
    return {
        'days': 1,
        'weeks': 7,
        'months': 30,
        'years': 365
    }


def _load_stage_mapping_from_config():
    """Load stage mapping from config file."""
    try:
        from machine_learning.utils.config_loader import get_config
        config = get_config()
        stage_config = config.stage_mapping

        # Convert config format to tuple format
        mapping = {}
        for stage, bounds in stage_config.items():
            # Stage names in config now use proper formatting directly
            stage_name = stage
            min_days = bounds.get('min_days', 0)
            max_days = bounds.get('max_days')
            if max_days is None:
                max_days = float('inf')
            mapping[stage_name] = (min_days, max_days)

        return mapping
    except (ImportError, Exception):
        return _get_default_stage_mapping()


def _load_age_conversion_from_config():
    """Load age conversion factors from config file."""
    try:
        from machine_learning.utils.config_loader import get_config
        config = get_config()
        return config.get('age_conversion', default=_get_default_age_conversion())
    except (ImportError, Exception):
        return _get_default_age_conversion()


class DataLoader:
    """Load and process pigGTEx RNA-seq data with metadata."""

    # Class-level defaults (can be overridden by config)
    STAGE_MAPPING = _get_default_stage_mapping()
    STAGE_ORDER = ['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal', 'Adult']
    AGE_CONVERSION = _get_default_age_conversion()

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

        # Load stage mapping and age conversion from config
        self.stage_mapping = _load_stage_mapping_from_config()
        self.age_conversion = _load_age_conversion_from_config()

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

        for stage, (min_age, max_age) in self.stage_mapping.items():
            if min_age <= age_days <= max_age:
                return stage

        return 'Adult'  # Default for ages beyond defined ranges

    def parse_age_string(self, age_str: str) -> Optional[float]:
        """
        Parse age string to days.

        Handles formats: "0 day", "6 months", "1 year", "4 weeks", etc.
        Conversion factors are loaded from config.yaml.

        Args:
            age_str: Age string from metadata

        Returns:
            Age in days, or None if unparseable
        """
        if pd.isna(age_str) or str(age_str).strip().lower() == 'unknown':
            return None

        age_str = str(age_str).lower().strip()

        match = re.match(r'(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)', age_str)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        # Use config-based conversion factors
        if 'day' in unit:
            return value * self.age_conversion.get('days', 1)
        elif 'week' in unit:
            return value * self.age_conversion.get('weeks', 7)
        elif 'month' in unit:
            return value * self.age_conversion.get('months', 30)
        elif 'year' in unit:
            return value * self.age_conversion.get('years', 365)

        return None

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

        # Standardize column names (replace spaces with underscores)
        self.metadata.columns = [col.strip().replace(' ', '_') for col in self.metadata.columns]

        # Map columns from new PigGTEx format to expected format
        column_mapping = {
            'BioSample': 'Sample_ID',
            'Tissue_class': 'Tissue',
            'Main_categories': 'Tissue_Main',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in self.metadata.columns and new_col not in self.metadata.columns:
                self.metadata[new_col] = self.metadata[old_col]
                logger.debug(f"Mapped column {old_col} -> {new_col}")

        # Set Sample_ID as index if present
        if 'Sample_ID' in self.metadata.columns:
            self.metadata = self.metadata.set_index('Sample_ID')

        # Parse age strings to days and create Stage column
        if 'Age' in self.metadata.columns:
            # Check if Age column contains strings (new format) or numbers (old format)
            sample_age = self.metadata['Age'].dropna().iloc[0] if len(self.metadata['Age'].dropna()) > 0 else None
            if sample_age is not None and isinstance(sample_age, str):
                # New format: parse age strings like "0 day", "6 months"
                self.metadata['Age_Days'] = self.metadata['Age'].apply(self.parse_age_string)
                self.metadata['Stage'] = self.metadata['Age_Days'].apply(self.age_to_stage)
                logger.info(f"Parsed age strings to days for {self.metadata['Age_Days'].notna().sum()} samples")
            else:
                # Old format: Age is already numeric days
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

        # Derive TechBatch composite batch variable
        from machine_learning.data_processing.batch_variables import create_tech_batch
        self.metadata['TechBatch'] = create_tech_batch(self.metadata)

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load expression data for a specific tissue.

        Args:
            tissue: Tissue name

        Returns:
            Tuple of (expression_matrix, sample_metadata)
        """
        # Find expression file (exact match to avoid e.g. "Blood" matching "Blood_vessel")
        expr_files = list(self.data_dir.glob(f"{tissue}.expr_tpm.txt.gz"))
        if not expr_files:
            # Try with underscores instead of spaces
            tissue_underscore = tissue.replace(' ', '_')
            expr_files = list(self.data_dir.glob(f"{tissue_underscore}.expr_tpm.txt.gz"))

        if not expr_files:
            raise ValueError(f"No expression file found for tissue: {tissue} or {tissue.replace(' ', '_')}")

        expr_file = expr_files[0]
        logger.info(f"Loading expression data from {expr_file}")

        # Load expression matrix
        with gzip.open(expr_file, 'rt') as f:
            expr_df = pd.read_csv(f, sep='\t', index_col=0)

        logger.info(f"Loaded expression matrix: {expr_df.shape}")

        # Get sample metadata for this tissue
        if self.metadata is not None and 'Tissue' in self.metadata.columns:
            tissue_mask = (
                (self.metadata['Tissue'] == tissue) &
                (self.metadata['Tissue_Main'] == tissue) &
                (self.metadata['Sub_categories'] == tissue)
            )
            tissue_metadata = self.metadata[tissue_mask].copy()

            # Align expression and metadata by matching sample IDs
            if len(tissue_metadata) > 0:
                # Find common samples between expression and metadata
                common_samples = [s for s in expr_df.columns if s in tissue_metadata.index]

                if len(common_samples) == 0:
                    # No matching sample IDs - this is a data integrity error
                    logger.error(f"No matching sample IDs between expression and metadata for {tissue}")
                    logger.error(f"Expression IDs example: {list(expr_df.columns[:3])}")
                    logger.error(f"Metadata IDs example: {list(tissue_metadata.index[:3])}")
                    raise ValueError(
                        f"No matching sample IDs between expression data and metadata for {tissue}. "
                        f"Expression has {len(expr_df.columns)} samples, metadata has {len(tissue_metadata)} samples. "
                        f"Please ensure sample IDs match between datasets."
                    )

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