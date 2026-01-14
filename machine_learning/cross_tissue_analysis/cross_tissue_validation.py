"""
Cross-tissue validation framework for assessing model generalization.
Implements leave-one-tissue-out and cross-tissue transfer learning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

from machine_learning.data_processing.stage_selection import StageGranularitySelector

logger = logging.getLogger(__name__)


class CrossTissueValidator:
    """Framework for cross-tissue validation and generalization analysis."""

    def __init__(
        self,
        model,
        preprocessor,
        min_common_genes: int = 1000,
        seed: int = 42,
        stage_selector: Optional[StageGranularitySelector] = None,
        common_scheme: str = 'auto'
    ):
        """
        Initialize cross-tissue validator.

        Args:
            model: Base model for classification
            preprocessor: Data preprocessor
            min_common_genes: Minimum common genes required
            seed: Random seed
            stage_selector: StageGranularitySelector for label harmonization
            common_scheme: Stage scheme for cross-tissue validation.
                          'auto' = determine lowest common denominator
                          '2-class', '3-class', '4-class' = use specific scheme
        """
        self.model = model
        self.preprocessor = preprocessor
        self.min_common_genes = min_common_genes
        self.seed = seed
        self.stage_selector = stage_selector or StageGranularitySelector()
        self.common_scheme = common_scheme

        self.results = {}
        self.common_genes = None
        self.tissue_models = {}

    def _harmonize_labels(self, stages: pd.Series, scheme: str) -> pd.Series:
        """
        Map raw stage labels to common scheme.

        Args:
            stages: Series with raw stage labels (e.g., 'Infant', 'Pre-pubertal')
            scheme: Target scheme ('2-class', '3-class', '4-class')

        Returns:
            Series with harmonized labels
        """
        return self.stage_selector.prepare_labels(stages, scheme)

    def _determine_common_scheme(
        self,
        tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Auto-determine common scheme from tissue data.

        Args:
            tissue_data: Dict of tissue_name -> (expression_df, metadata_df)

        Returns:
            (common_scheme, eligible_tissues, excluded_tissues)
        """
        # Extract metadata for each tissue
        tissue_metadata = {name: meta for name, (_, meta) in tissue_data.items()}
        return self.stage_selector.determine_common_scheme(tissue_metadata)

    def leave_one_tissue_out(
        self,
        tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        target_tissue: str,
        common_scheme: Optional[str] = None
    ) -> Dict:
        """
        Train on all tissues except one, test on held-out tissue.

        Args:
            tissue_data: Dictionary of tissue -> (expression, metadata) tuples
            target_tissue: Tissue to hold out for testing
            common_scheme: Override common scheme (None uses self.common_scheme)

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"LOTO validation: Testing on {target_tissue}")

        # === Determine common scheme ===
        effective_scheme = common_scheme or self.common_scheme
        excluded_tissues = []

        if effective_scheme == 'auto':
            effective_scheme, eligible, excluded_tissues = self._determine_common_scheme(tissue_data)
            if effective_scheme is None:
                raise ValueError("No common scheme available for any tissues")
            logger.info(f"Auto-selected common scheme: {effective_scheme}")
            if excluded_tissues:
                logger.warning(f"Excluding {len(excluded_tissues)} tissues: {excluded_tissues}")
                # Remove excluded tissues from data
                tissue_data = {k: v for k, v in tissue_data.items() if k not in excluded_tissues}

        # Separate training and test tissues
        train_tissues = {k: v for k, v in tissue_data.items() if k != target_tissue}
        test_data = tissue_data[target_tissue]

        if not train_tissues:
            raise ValueError("No training tissues available")

        # Find common genes across training tissues only (fix data leakage)
        common_genes = self._find_common_genes(list(train_tissues.values()))

        if len(common_genes) < self.min_common_genes:
            logger.warning(f"Only {len(common_genes)} common genes found")

        # Combine training data from all tissues
        X_train_combined = []
        y_train_combined = []
        tissue_labels_train = []

        for tissue_name, (expr, metadata) in train_tissues.items():
            # Filter to common genes
            expr_filtered = expr.loc[expr.index.intersection(common_genes)]

            # Remove samples with missing stages
            valid_idx = metadata['Stage'].notna()
            # Convert boolean series to array to avoid alignment issues
            valid_idx_array = valid_idx.values
            expr_valid = expr_filtered.iloc[:, valid_idx_array]
            metadata_valid = metadata[valid_idx_array]

            # Transpose to samples x genes
            X_tissue = expr_valid.T

            # === Harmonize labels to common scheme ===
            y_tissue_raw = metadata_valid['Stage']
            y_tissue = self._harmonize_labels(y_tissue_raw, effective_scheme)

            X_train_combined.append(X_tissue)
            y_train_combined.extend(y_tissue.tolist())
            tissue_labels_train.extend([tissue_name] * len(y_tissue))

        # Concatenate all training data
        X_train = pd.concat(X_train_combined, axis=0)
        y_train = pd.Series(y_train_combined)

        # Prepare test data
        X_test_expr, test_metadata = test_data
        X_test_expr = X_test_expr.loc[X_test_expr.index.intersection(common_genes)]
        valid_test = test_metadata['Stage'].notna()
        valid_test_array = valid_test.values
        X_test = X_test_expr.iloc[:, valid_test_array].T

        # === Harmonize test labels to common scheme ===
        y_test_raw = test_metadata[valid_test_array]['Stage']
        y_test = self._harmonize_labels(y_test_raw, effective_scheme)

        # Align gene order
        common_genes_ordered = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_genes_ordered]
        X_test = X_test[common_genes_ordered]

        # Preprocess if needed
        if self.preprocessor:
            # Fit on combined training data
            X_train_proc = self.preprocessor.fit_transform(X_train.T).T
            X_test_proc = self.preprocessor.transform(X_test.T).T
        else:
            X_train_proc = X_train
            X_test_proc = X_test

        # Train model
        model_clone = clone(self.model)
        model_clone.fit(X_train_proc, y_train)

        # Predict on test tissue
        y_pred = model_clone.predict(X_test_proc)

        # Calculate metrics
        metrics = {
            'target_tissue': target_tissue,
            'common_scheme': effective_scheme,
            'n_train_tissues': len(train_tissues),
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
            'n_common_genes': len(common_genes_ordered),
            'excluded_tissues': excluded_tissues,
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }

        # Store results
        self.results[f'LOTO_{target_tissue}'] = metrics
        logger.info(f"LOTO {target_tissue}: Acc={metrics['balanced_accuracy']:.3f}")

        return metrics

    def cross_tissue_transfer(
        self,
        source_data: Tuple[pd.DataFrame, pd.DataFrame],
        target_data: Tuple[pd.DataFrame, pd.DataFrame],
        source_name: str = "source",
        target_name: str = "target",
        common_scheme: Optional[str] = None
    ) -> Dict:
        """
        Train on one tissue, test on another with harmonized labels.

        Args:
            source_data: (expression, metadata) for source tissue
            target_data: (expression, metadata) for target tissue
            source_name: Name of source tissue
            target_name: Name of target tissue
            common_scheme: Override common scheme (None uses self.common_scheme)

        Returns:
            Performance metrics
        """
        logger.info(f"Transfer: {source_name} -> {target_name}")

        # === Determine common scheme if auto ===
        effective_scheme = common_scheme or self.common_scheme
        if effective_scheme == 'auto':
            tissue_metadata = {
                source_name: source_data[1],
                target_name: target_data[1]
            }
            effective_scheme, eligible, excluded = self.stage_selector.determine_common_scheme(tissue_metadata)
            if effective_scheme is None:
                raise ValueError(f"No common scheme for {source_name} and {target_name}")
            logger.info(f"Using common scheme: {effective_scheme}")

        # Find common genes
        source_expr, source_meta = source_data
        target_expr, target_meta = target_data

        common_genes = source_expr.index.intersection(target_expr.index)

        if len(common_genes) < self.min_common_genes:
            logger.warning(f"Only {len(common_genes)} common genes")

        # Prepare source data
        source_expr_filtered = source_expr.loc[common_genes]
        valid_source = source_meta['Stage'].notna()
        valid_source_array = valid_source.values
        X_source = source_expr_filtered.iloc[:, valid_source_array].T

        # === Harmonize source labels ===
        y_source_raw = source_meta[valid_source_array]['Stage']
        y_source = self._harmonize_labels(y_source_raw, effective_scheme)

        # Prepare target data
        target_expr_filtered = target_expr.loc[common_genes]
        valid_target = target_meta['Stage'].notna()
        valid_target_array = valid_target.values
        X_target = target_expr_filtered.iloc[:, valid_target_array].T

        # === Harmonize target labels ===
        y_target_raw = target_meta[valid_target_array]['Stage']
        y_target = self._harmonize_labels(y_target_raw, effective_scheme)

        # Preprocess
        if self.preprocessor:
            preprocessor_clone = clone(self.preprocessor)
            X_source_proc = preprocessor_clone.fit_transform(X_source.T).T
            X_target_proc = preprocessor_clone.transform(X_target.T).T
        else:
            X_source_proc = X_source
            X_target_proc = X_target

        # Train on source
        model_clone = clone(self.model)
        model_clone.fit(X_source_proc, y_source)

        # Test on target
        y_pred = model_clone.predict(X_target_proc)

        # Metrics
        metrics = {
            'source': source_name,
            'target': target_name,
            'common_scheme': effective_scheme,
            'n_source_samples': len(y_source),
            'n_target_samples': len(y_target),
            'n_common_genes': len(common_genes),
            'balanced_accuracy': balanced_accuracy_score(y_target, y_pred),
            'f1_macro': f1_score(y_target, y_pred, average='macro', zero_division=0)
        }

        key = f'transfer_{source_name}_to_{target_name}'
        self.results[key] = metrics

        return metrics

    def within_tissue_baseline(
        self,
        tissue_data: Tuple[pd.DataFrame, pd.DataFrame],
        tissue_name: str,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train and test within same tissue (baseline).

        Args:
            tissue_data: (expression, metadata) for tissue
            tissue_name: Name of tissue
            test_size: Fraction for testing

        Returns:
            Performance metrics
        """
        from sklearn.model_selection import train_test_split

        expr, metadata = tissue_data

        # Prepare data
        valid_idx = metadata['Stage'].notna()
        valid_idx_array = valid_idx.values
        X = expr.iloc[:, valid_idx_array].T
        y = metadata[valid_idx_array]['Stage']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.seed
        )

        # Preprocess
        if self.preprocessor:
            preprocessor_clone = clone(self.preprocessor)
            X_train_proc = preprocessor_clone.fit_transform(X_train.T).T
            X_test_proc = preprocessor_clone.transform(X_test.T).T
        else:
            X_train_proc = X_train
            X_test_proc = X_test

        # Train and test
        model_clone = clone(self.model)
        model_clone.fit(X_train_proc, y_train)
        y_pred = model_clone.predict(X_test_proc)

        # Metrics
        metrics = {
            'tissue': tissue_name,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }

        self.results[f'within_{tissue_name}'] = metrics
        logger.info(f"Within-tissue {tissue_name}: Acc={metrics['balanced_accuracy']:.3f}")

        return metrics

    def compute_tissue_similarity(
        self,
        tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        method: str = 'correlation'
    ) -> pd.DataFrame:
        """
        Compute pairwise tissue similarity matrix.

        Args:
            tissue_data: Dictionary of tissue data
            method: Similarity method ('correlation', 'euclidean')

        Returns:
            Similarity matrix
        """
        # Find common genes
        all_data = list(tissue_data.values())
        common_genes = self._find_common_genes(all_data)

        # Compute mean expression per tissue
        tissue_means = {}
        for tissue_name, (expr, metadata) in tissue_data.items():
            expr_filtered = expr.loc[expr.index.intersection(common_genes)]
            tissue_means[tissue_name] = expr_filtered.mean(axis=1)

        # Create matrix
        mean_df = pd.DataFrame(tissue_means)

        if method == 'correlation':
            similarity = mean_df.corr()
        elif method == 'euclidean':
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(mean_df.T, metric='euclidean')
            similarity = pd.DataFrame(
                1 / (1 + squareform(distances)),
                index=mean_df.columns,
                columns=mean_df.columns
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        return similarity

    def create_tissue_embedding(
        self,
        tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        method: str = 'umap',
        n_components: int = 2,
        max_samples_per_tissue: int = 100
    ) -> pd.DataFrame:
        """
        Create low-dimensional embedding of all tissues.

        Args:
            tissue_data: Dictionary of tissue data
            method: Embedding method ('pca', 'umap')
            n_components: Number of components
            max_samples_per_tissue: Max samples to use per tissue

        Returns:
            DataFrame with embedding coordinates
        """
        # Find common genes
        common_genes = self._find_common_genes(list(tissue_data.values()))

        # Collect samples from all tissues
        all_samples = []
        sample_info = []

        for tissue_name, (expr, metadata) in tissue_data.items():
            # Filter to common genes
            expr_filtered = expr.loc[expr.index.intersection(common_genes)]

            # Get valid samples
            valid_idx = metadata['Stage'].notna()
            valid_idx_array = valid_idx.values
            expr_valid = expr_filtered.iloc[:, valid_idx_array]
            metadata_valid = metadata[valid_idx_array]

            # Subsample if needed
            n_samples = min(max_samples_per_tissue, expr_valid.shape[1])
            if n_samples < expr_valid.shape[1]:
                sample_idx = np.random.choice(
                    expr_valid.shape[1], n_samples, replace=False
                )
                expr_valid = expr_valid.iloc[:, sample_idx]
                metadata_valid = metadata_valid.iloc[sample_idx]

            # Add to collection
            for i in range(expr_valid.shape[1]):
                all_samples.append(expr_valid.iloc[:, i].values)
                sample_info.append({
                    'tissue': tissue_name,
                    'stage': metadata_valid.iloc[i]['Stage'],
                    'sample_id': expr_valid.columns[i]
                })

        # Create matrix
        X = np.array(all_samples)

        # Standardize
        X = StandardScaler().fit_transform(X)

        # Embed
        if method == 'pca':
            embedder = PCA(n_components=n_components, random_state=self.seed)
        elif method == 'umap':
            embedder = umap.UMAP(
                n_components=n_components,
                random_state=self.seed,
                n_neighbors=15,
                min_dist=0.1
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        embedding = embedder.fit_transform(X)

        # Create DataFrame
        embed_df = pd.DataFrame(embedding, columns=[f'Dim{i+1}' for i in range(n_components)])
        info_df = pd.DataFrame(sample_info)
        result = pd.concat([info_df, embed_df], axis=1)

        return result

    def _find_common_genes(
        self,
        data_list: List[Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> pd.Index:
        """Find genes common to all datasets."""
        if not data_list:
            return pd.Index([])

        # Start with first dataset's genes
        common = data_list[0][0].index

        # Intersect with others
        for expr, _ in data_list[1:]:
            common = common.intersection(expr.index)

        return common

    def generate_transfer_matrix(
        self,
        tissue_names: List[str]
    ) -> pd.DataFrame:
        """
        Generate transfer learning performance matrix.

        Args:
            tissue_names: List of tissue names

        Returns:
            Matrix with transfer performance
        """
        matrix = pd.DataFrame(
            np.nan,
            index=tissue_names,
            columns=tissue_names
        )

        for source in tissue_names:
            for target in tissue_names:
                if source == target:
                    # Within-tissue performance
                    key = f'within_{source}'
                    if key in self.results:
                        matrix.loc[source, target] = self.results[key]['balanced_accuracy']
                else:
                    # Cross-tissue transfer
                    key = f'transfer_{source}_to_{target}'
                    if key in self.results:
                        matrix.loc[source, target] = self.results[key]['balanced_accuracy']

        return matrix

    def calculate_generalization_gap(self) -> pd.DataFrame:
        """
        Calculate difference between within-tissue and cross-tissue performance.

        Returns:
            DataFrame with generalization gaps
        """
        gaps = []

        for key, result in self.results.items():
            if key.startswith('transfer_'):
                source = result['source']
                target = result['target']

                # Get within-tissue baseline for target
                within_key = f'within_{target}'
                if within_key in self.results:
                    within_acc = self.results[within_key]['balanced_accuracy']
                    transfer_acc = result['balanced_accuracy']
                    gap = within_acc - transfer_acc

                    gaps.append({
                        'source': source,
                        'target': target,
                        'within_accuracy': within_acc,
                        'transfer_accuracy': transfer_acc,
                        'generalization_gap': gap,
                        'relative_performance': transfer_acc / within_acc
                    })

        return pd.DataFrame(gaps)