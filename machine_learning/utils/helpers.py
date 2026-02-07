"""Shared helper functions used across the ML pipeline and analysis scripts."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def to_samples_x_genes_df(X_proc, sample_ids, preprocessor_obj, fallback_genes):
    """Convert processed expression data (genes x samples) to samples-by-genes DataFrame."""
    if isinstance(X_proc, pd.DataFrame):
        result = X_proc.T.copy()
        result.index = sample_ids
    else:
        X_array = np.asarray(X_proc)
        if X_array.shape[0] == len(fallback_genes) or X_array.shape[0] != len(sample_ids):
            X_array = X_array.T
        proc_genes = (
            np.asarray(getattr(preprocessor_obj, 'feature_names_', None))
            if getattr(preprocessor_obj, 'feature_names_', None)
            else np.asarray(fallback_genes)
        )
        result = pd.DataFrame(X_array, index=sample_ids, columns=proc_genes)
    return result


def load_and_prepare_tissue(tissue_name: str):
    """
    Load data and apply stage selection for a tissue.

    Returns:
        (X_raw, y, sample_ids, gene_names, scheme_name) or None on failure.
        X_raw is genes x samples, y is integer-encoded stage labels.
    """
    from machine_learning.data_processing.data_loader import DataLoader
    from machine_learning.data_processing.stage_selection import StageGranularitySelector

    data_loader = DataLoader(
        data_dir=PROJECT_ROOT / "data/pigGTEx",
        metadata_path=PROJECT_ROOT / "data/PigGTEx_v0.MetaTable.xlsx"
    )
    data_loader.load_metadata()
    expr_data, metadata = data_loader.load_expression(tissue_name)

    X = expr_data
    y = metadata['Stage'].values
    sample_ids = metadata.index.values
    gene_names = expr_data.index.values

    stage_selector = StageGranularitySelector()
    stage_counts = pd.Series(y).value_counts()
    scheme_name, scheme = stage_selector.select_scheme(stage_counts, tissue_name)

    if scheme is None:
        return None

    if 'mapping' in scheme:
        labels = scheme.get('labels', [])
        stage_mapping = scheme['mapping']
        y_mapped = [stage_mapping.get(s, s) for s in y]
        label_to_int = {label: i for i, label in enumerate(labels)}
        y = np.array([label_to_int.get(label, -1) for label in y_mapped])

        valid_mask = y != -1
        X = X.loc[:, valid_mask]
        y = y[valid_mask]
        sample_ids = sample_ids[valid_mask]
        gene_names = X.index.values

    return X, y, sample_ids, gene_names, scheme_name
