#!/usr/bin/env python3
"""
Main pipeline script with enhanced logging capabilities.
This demonstrates the refactored Python ML pipeline with comprehensive logging.
"""

import sys
import json
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import logging configuration
from machine_learning.utils.logging_config import (
    setup_logging, PerformanceLogger, LogContext, log_execution_time,
    create_module_logger
)

# Import progress tracking
from machine_learning.utils.progress_tracker import (
    PipelineProgressTracker, PhaseProgressContext
)

# Import from new structure
from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.model_evaluation.evaluation import MetricCalculator
from machine_learning.cross_tissue_analysis.cross_tissue_validation import CrossTissueValidator
from machine_learning.utils.config_loader import get_config

# Setup main logger (will be reconfigured based on args)
logger = logging.getLogger(__name__)
perf_logger = None
progress_tracker = None


def _to_serializable(value):
    """Recursively convert numpy/pandas objects to JSON-friendly types."""
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_serializable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Series):
        return _to_serializable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient='list')
    return value


def log_data_statistics(data: pd.DataFrame, name: str, logger):
    """Log detailed statistics about a dataset."""
    stats = {
        'shape': data.shape,
        'memory_usage_mb': float(data.memory_usage().sum() / 1024 / 1024),
        'dtypes': {str(k): int(v) for k, v in data.dtypes.value_counts().to_dict().items()},
        'null_counts': int(data.isnull().sum().sum()),
        'zero_rate': float((data == 0).sum().sum() / data.size) if data.size > 0 else 0
    }

    logger.debug(f"{name} statistics: {json.dumps(stats, default=str)}")
    return stats


def validate_data_quality(data: pd.DataFrame, metadata: pd.DataFrame, tissue: str, logger):
    """Validate data quality and log any issues."""
    issues = []

    # Check for missing values
    if data.isnull().any().any():
        null_genes = data.isnull().sum(axis=1)
        null_genes = null_genes[null_genes > 0]
        issues.append(f"Found {len(null_genes)} genes with missing values")
        logger.warning(f"{tissue}: {len(null_genes)} genes have missing expression values")

    # Check for constant genes
    const_genes = data.var(axis=1) == 0
    if const_genes.any():
        issues.append(f"Found {const_genes.sum()} genes with zero variance")
        logger.warning(f"{tissue}: {const_genes.sum()} genes have constant expression")

    # Check for outlier samples
    sample_means = data.mean(axis=0)
    outliers = np.abs(sample_means - sample_means.mean()) > 3 * sample_means.std()
    if outliers.any():
        outlier_samples = data.columns[outliers]
        issues.append(f"Found {len(outlier_samples)} potential outlier samples")
        logger.warning(f"{tissue}: {len(outlier_samples)} samples may be outliers: {outlier_samples[:5].tolist()}...")

    # Check metadata alignment
    if len(data.columns) != len(metadata):
        issues.append(f"Metadata mismatch: {len(data.columns)} samples vs {len(metadata)} metadata entries")
        logger.error(f"{tissue}: Sample count mismatch between expression and metadata")

    # Log summary
    if issues:
        logger.info(f"{tissue} data quality issues: {'; '.join(issues)}")
    else:
        logger.info(f"{tissue}: Data quality check passed")

    return issues


def get_lightgbm_params() -> dict:
    """
    Get LightGBM hyperparameters from configuration file.

    Returns:
        Dictionary of LightGBM parameters
    """
    config = get_config()
    return config.lightgbm_params.copy()


def create_lightgbm_model(seed: int = None) -> OrdinalLightGBM:
    """
    Create an OrdinalLightGBM model with parameters from config.

    Args:
        seed: Random seed for reproducibility. If None, uses config value.

    Returns:
        Configured OrdinalLightGBM model
    """
    config = get_config()
    params = get_lightgbm_params()

    if seed is None:
        seed = config.get("splitting", "seed", default=42)

    return OrdinalLightGBM(**params, seed=seed)


def validate_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 42,
    n_folds: int = 3,
    tissue_logger=None
) -> Dict[str, float]:
    """
    Run stratified k-fold CV on training data to validate hyperparameters.

    Args:
        X_train: Training features (samples x genes)
        y_train: Training labels
        seed: Random seed
        n_folds: Number of CV folds
        tissue_logger: Logger instance

    Returns:
        Dictionary with CV balanced accuracy mean and std
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    log = tissue_logger or logger

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        fold_model = create_lightgbm_model(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_tr, y_tr)

        y_val_pred = fold_model.predict(X_val)
        ba = balanced_accuracy_score(y_val, y_val_pred)
        fold_scores.append(ba)
        log.debug(f"  CV fold {fold_i + 1}/{n_folds}: balanced_accuracy={ba:.3f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    log.info(
        f"Hyperparameter validation CV ({n_folds}-fold): "
        f"balanced_accuracy={cv_mean:.3f} +/- {cv_std:.3f}"
    )

    return {'cv_balanced_accuracy_mean': cv_mean, 'cv_balanced_accuracy_std': cv_std}


@log_execution_time(logger)
def run_single_tissue_pipeline(
    tissue_name: str,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    tracker: Optional[PipelineProgressTracker] = None
) -> Optional[Dict[str, Any]]:
    """
    Run the ML pipeline for a single tissue with detailed logging.

    Args:
        tissue_name: Name of the tissue to process
        output_dir: Directory to save results
        config: Optional configuration parameters

    Returns:
        Dictionary with results or None if processing failed
    """
    config = config or {}
    tissue_logger = create_module_logger(f"tissue.{tissue_name}", logger)

    # Start tissue tracking
    if tracker:
        tracker.start_tissue(tissue_name)

    with LogContext(tissue_logger, f"Processing {tissue_name}"):
        try:
            # Track performance
            if perf_logger:
                perf_logger.start_timer(f"{tissue_name}_total")
                perf_logger.log_memory_usage(f"{tissue_name}_start")

            # ===== 1. DATA LOADING PHASE =====
            phase_name = "Data Loading"
            with LogContext(tissue_logger, phase_name, level=logging.INFO):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_loading")

                # Get paths from configuration
                pipeline_config = get_config()
                data_dir = PROJECT_ROOT / pipeline_config.paths.get("data_dir", "data/pigGTEx")
                metadata_file = PROJECT_ROOT / pipeline_config.paths.get("metadata_file", "data/PigGTEx_v0.MetaTable.xlsx")

                data_loader = DataLoader(
                    data_dir=data_dir,
                    metadata_path=metadata_file
                )
                data_loader.load_metadata()

                tissue_logger.debug(f"Loaded metadata with {len(data_loader.metadata)} entries")

                # Check if tissue data exists (handle spaces in tissue names)
                expr_pattern = pipeline_config.paths.get(
                    "expression_pattern",
                    "data/pigGTEx/{tissue_name}.expr_tpm.txt.gz"
                )
                tissue_file = PROJECT_ROOT / expr_pattern.format(tissue_name=tissue_name)
                if not tissue_file.exists():
                    # Try with underscores instead of spaces
                    tissue_file_alt = PROJECT_ROOT / expr_pattern.format(
                        tissue_name=tissue_name.replace(' ', '_')
                    )
                    if tissue_file_alt.exists():
                        tissue_file = tissue_file_alt
                        tissue_logger.info(f"Using file with underscores: {tissue_file_alt.name}")
                    else:
                        tissue_logger.warning(f"No expression file found at {tissue_file} or {tissue_file_alt}")
                        return None

                tissue_logger.info(f"Loading expression data from {tissue_file}")
                expr_data, metadata = data_loader.load_expression(tissue_name)

                # Log data statistics
                log_data_statistics(expr_data, f"{tissue_name}_expression", tissue_logger)

                # Extract features
                X = expr_data
                y = metadata['Stage'].values
                sample_ids = metadata.index.values
                gene_names = expr_data.index.values

                tissue_logger.info(
                    f"Loaded {tissue_name}: {X.shape[0]} genes x {X.shape[1]} samples, "
                    f"Stage distribution: {pd.Series(y).value_counts().to_dict()}"
                )

                # Validate data quality
                validate_data_quality(X, metadata, tissue_name, tissue_logger)

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_loading")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 2. STAGE SELECTION PHASE =====
            phase_name = "Stage Selection"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_stage_selection")

                stage_selector = StageGranularitySelector()
                stage_counts = pd.Series(y).value_counts()

                tissue_logger.debug(f"Stage counts: {stage_counts.to_dict()}")

                # Check minimum samples per class
                min_samples_required = config.get('min_samples_per_class', 10)
                small_classes = stage_counts[stage_counts < min_samples_required]
                if len(small_classes) > 0:
                    tissue_logger.warning(
                        f"Found classes with insufficient samples (<{min_samples_required}): "
                        f"{small_classes.to_dict()}"
                    )

                scheme_name, scheme = stage_selector.select_scheme(stage_counts, tissue_name)

                if scheme is None:
                    tissue_logger.warning(f"{tissue_name} not eligible for any classification scheme")
                    return None

                labels = scheme.get('labels', scheme.get('stages', []))
                tissue_logger.info(
                    f"Selected scheme '{scheme_name}' with {scheme.get('n_classes', len(labels))} classes: "
                    f"{labels}"
                )

                # Map original stage labels to selected scheme
                if 'mapping' in scheme:
                    y_original = y.copy()
                    stage_mapping = scheme['mapping']
                    y_mapped = []

                    for stage in y:
                        if stage in stage_mapping:
                            y_mapped.append(stage_mapping[stage])
                        else:
                            tissue_logger.warning(f"Stage '{stage}' not found in mapping, keeping original")
                            y_mapped.append(stage)

                    y = np.array(y_mapped)

                    # Convert to ordered categorical based on scheme labels
                    unique_labels = labels if isinstance(labels, list) else list(labels)
                    label_to_int = {label: i for i, label in enumerate(unique_labels)}
                    y = np.array([label_to_int.get(label, -1) for label in y])

                    # Filter out samples with unknown labels (-1)
                    valid_mask = y != -1
                    if not valid_mask.all():
                        n_unknown = (~valid_mask).sum()
                        tissue_logger.warning(
                            f"Removing {n_unknown} samples with stage labels that don't match "
                            f"the selected scheme. These would have been assigned unknown class (-1)."
                        )
                        X = X.loc[:, valid_mask]
                        y = y[valid_mask]
                        sample_ids = sample_ids[valid_mask]
                        metadata = metadata.loc[valid_mask]
                    else:
                        tissue_logger.info("All stage labels match selected scheme - no unknown samples")

                    tissue_logger.info(
                        f"Remapped stages: {pd.Series(y_original).value_counts().to_dict()} -> "
                        f"{pd.Series(y_mapped).value_counts().to_dict()}"
                    )
                    tissue_logger.info(
                        f"Final sample count after filtering: {y.shape[0]} samples with "
                        f"distribution: {pd.Series(y).value_counts().sort_index().to_dict()}"
                    )

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_stage_selection")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 3. TRAIN/TEST SPLIT (BEFORE PREPROCESSING TO AVOID DATA LEAKAGE) =====
            phase_name = "Train/Test Split"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)

                from sklearn.model_selection import train_test_split
                split_ratio = config.get('train_ratio', 0.7)
                random_state = config.get('seed', 42)
                min_samples_per_class = pd.Series(y).value_counts().min()

                # Split raw data BEFORE any preprocessing
                # X is (genes x samples), need to transpose for split then transpose back
                X_T = X.T  # samples x genes
                if min_samples_per_class >= 2:
                    X_train_raw_T, X_test_raw_T, y_train, y_test, train_ids, test_ids = train_test_split(
                        X_T, y, sample_ids, test_size=1 - split_ratio,
                        stratify=y, random_state=random_state
                    )
                    tissue_logger.info("Using stratified train/test split")
                else:
                    X_train_raw_T, X_test_raw_T, y_train, y_test, train_ids, test_ids = train_test_split(
                        X_T, y, sample_ids, test_size=1 - split_ratio,
                        random_state=random_state
                    )
                    tissue_logger.warning("Not enough samples for stratification, using random split")

                # Transpose back to genes x samples format
                X_train_raw = X_train_raw_T.T
                X_test_raw = X_test_raw_T.T

                tissue_logger.info(
                    f"Train/test split: {len(y_train)}/{len(y_test)} samples "
                    f"({split_ratio:.0%}/{1-split_ratio:.0%})"
                )
                tissue_logger.debug(
                    f"Train label distribution: {pd.Series(y_train).value_counts().to_dict()}"
                )
                tissue_logger.debug(
                    f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}"
                )

                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 4. PREPROCESSING (FIT ON TRAIN ONLY) =====
            phase_name = "Preprocessing"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_preprocessing")

                preprocessor = ExpressionPreprocessor()

                # Log preprocessing parameters
                variance_threshold = getattr(preprocessor, 'variance_threshold', 'default')
                tissue_logger.debug(f"Preprocessing with variance_threshold={variance_threshold}")

                # FIT on training data ONLY, then transform both train and test
                X_train_processed = preprocessor.fit_transform(X_train_raw)
                X_test_processed = preprocessor.transform(X_test_raw)

                tissue_logger.info(
                    f"Preprocessor fitted on training data only ({X_train_raw.shape[1]} samples)"
                )

                # Helper function to convert processed data to DataFrame
                def to_samples_x_genes_df(X_proc, sample_ids_subset, preprocessor_obj, fallback_genes):
                    if isinstance(X_proc, pd.DataFrame):
                        result = X_proc.T.copy()
                        result.index = sample_ids_subset
                    else:
                        X_array = np.asarray(X_proc)
                        if X_array.shape[0] == len(fallback_genes) or X_array.shape[0] != len(sample_ids_subset):
                            X_array = X_array.T
                        proc_genes = (
                            np.asarray(getattr(preprocessor_obj, 'feature_names_', None))
                            if getattr(preprocessor_obj, 'feature_names_', None)
                            else np.asarray(fallback_genes)
                        )
                        result = pd.DataFrame(X_array, index=sample_ids_subset, columns=proc_genes)
                    return result

                X_train_full = to_samples_x_genes_df(X_train_processed, train_ids, preprocessor, gene_names)
                X_test_full = to_samples_x_genes_df(X_test_processed, test_ids, preprocessor, gene_names)

                if X_train_full.shape[0] != len(y_train):
                    raise ValueError(
                        f"Preprocessed train row count {X_train_full.shape[0]} does not match labels {len(y_train)}"
                    )
                if X_test_full.shape[0] != len(y_test):
                    raise ValueError(
                        f"Preprocessed test row count {X_test_full.shape[0]} does not match labels {len(y_test)}"
                    )

                tissue_logger.info(
                    f"Preprocessing complete: train {X_train_raw.shape} -> {X_train_full.shape}, "
                    f"test {X_test_raw.shape} -> {X_test_full.shape} "
                    f"({X_train_raw.shape[0] - X_train_full.shape[1]} genes removed)"
                )

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_preprocessing")
                    perf_logger.log_memory_usage(f"{tissue_name}_after_preprocessing")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 5. HYPERPARAMETER VALIDATION (OPTIONAL) =====
            cv_results = None
            pipeline_config_obj = get_config()
            hp_validation_enabled = config.get(
                'hyperparameter_validation',
                getattr(pipeline_config_obj, 'hyperparameter_validation', False)
            )
            if hp_validation_enabled:
                phase_name = "Hyperparameter Validation"
                with LogContext(tissue_logger, phase_name):
                    if tracker:
                        tracker.start_phase(phase_name)
                    cv_results = validate_hyperparameters(
                        X_train_full, y_train,
                        seed=config.get('seed', 42),
                        tissue_logger=tissue_logger
                    )
                    if tracker:
                        tracker.end_phase(phase_name)

            # ===== 6. MODEL TRAINING PHASE =====
            phase_name = "Model Training"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_training")

                model = create_lightgbm_model(seed=config.get('seed', 42))
                X_train = X_train_full
                X_test = X_test_full
                tissue_logger.debug(
                    f"Using OrdinalLightGBM for classification on all "
                    f"{X_train.shape[1]} preprocessed features"
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    model.fit(X_train, y_train)

                    if w:
                        for warning in w:
                            tissue_logger.warning(f"Model training warning: {warning.message}")

                # Extract feature importance from the trained model
                importance = model.get_feature_importance()
                top_genes = importance.head(1000).index.to_numpy()
                tissue_logger.info(
                    f"Top 10 genes by importance: {importance.head(10).index.tolist()}"
                )

                # Make predictions on test set
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                # Make predictions on training set (for overfitting detection)
                y_train_pred = model.predict(X_train)
                y_train_prob = model.predict_proba(X_train)

                # Log prediction distribution
                pred_dist = pd.Series(y_pred).value_counts()
                tissue_logger.debug(f"Prediction distribution: {pred_dist.to_dict()}")

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_training")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 7. EVALUATION PHASE =====
            phase_name = "Evaluation"
            with LogContext(tissue_logger, "Model Evaluation"):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_evaluation")

                evaluator = MetricCalculator()
                label_series = pd.Series(y).dropna()
                labels = label_series.unique().tolist()

                # Test set metrics
                metrics = evaluator.calculate(y_test, y_pred, y_prob=y_prob, labels=labels)

                # Training set metrics (for overfitting detection)
                train_metrics = evaluator.calculate(
                    y_train, y_train_pred, y_prob=y_train_prob, labels=labels
                )

                # Log detailed metrics with train-test comparison
                tissue_logger.info(
                    f"Test metrics - "
                    f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}, "
                    f"F1 Macro: {metrics.get('f1_macro', 0):.3f}, "
                    f"F1 Weighted: {metrics.get('f1_weighted', 0):.3f}"
                )
                tissue_logger.info(
                    f"Train metrics - "
                    f"Balanced Accuracy: {train_metrics['balanced_accuracy']:.3f}, "
                    f"F1 Macro: {train_metrics.get('f1_macro', 0):.3f}, "
                    f"F1 Weighted: {train_metrics.get('f1_weighted', 0):.3f}"
                )

                # Log train-test gap (overfitting indicator)
                ba_gap = train_metrics['balanced_accuracy'] - metrics['balanced_accuracy']
                f1_gap = train_metrics.get('f1_macro', 0) - metrics.get('f1_macro', 0)
                tissue_logger.info(
                    f"Train-test gap - "
                    f"Balanced Accuracy: {ba_gap:+.3f}, "
                    f"F1 Macro: {f1_gap:+.3f}"
                )
                if ba_gap > 0.15:
                    tissue_logger.warning(
                        f"Possible overfitting: train-test balanced accuracy gap is {ba_gap:.3f}"
                    )

                # Log confusion matrix if available
                if 'confusion_matrix' in metrics:
                    tissue_logger.debug(f"Confusion matrix:\n{metrics['confusion_matrix']}")

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_evaluation")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 7. SAVE RESULTS =====
            phase_name = "Saving Results"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_saving")
                    tissue_perf_time = perf_logger.metrics.get(f"{tissue_name}_total", 0)
                    tissue_perf_memory = perf_logger.log_memory_usage(f"{tissue_name}_end")

                results = {
                    'tissue': tissue_name,
                    'scheme': scheme_name,
                    'n_samples': len(y),
                    'n_genes_initial': X.shape[0],
                    'n_genes_preprocessed': X_train_full.shape[1],
                    'n_features_used': X_train.shape[1],
                    'metrics': metrics,
                    'train_metrics': train_metrics,
                    'top_genes': top_genes[:1000].tolist(),
                    'feature_importance': importance.head(1000).to_dict()
                }

                # Add CV results if hyperparameter validation was run
                if cv_results is not None:
                    results['cv_validation'] = cv_results

                # Add performance metrics if available
                if perf_logger:
                    results['performance'] = {
                        'total_time_seconds': tissue_perf_time,
                        'final_memory_mb': tissue_perf_memory,
                        'stage_timings': {
                            k: v for k, v in perf_logger.metrics.items()
                            if tissue_name in k
                        }
                    }

                output_file = output_dir / f"{tissue_name}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(_to_serializable(results), f, indent=2)

                tissue_logger.info(f"Results saved to {output_file}")

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_saving")
                if tracker:
                    tracker.end_phase(phase_name)

            # Mark tissue as completed
            if tracker:
                tracker.complete_tissue(tissue_name, success=True)

            return results

        except Exception as e:
            tissue_logger.error(f"Pipeline failed for {tissue_name}: {str(e)}")
            tissue_logger.debug(f"Traceback: {traceback.format_exc()}")

            # Mark tissue as failed
            if tracker:
                tracker.complete_tissue(tissue_name, success=False)

            return None


@log_execution_time(logger)
def run_cross_tissue_validation(
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    tracker: Optional[PipelineProgressTracker] = None
) -> Optional[Dict[str, Any]]:
    """
    Run cross-tissue validation with detailed logging.

    Args:
        output_dir: Directory to save results
        config: Optional configuration parameters

    Returns:
        Dictionary with validation results
    """
    config = config or {}
    cv_logger = create_module_logger("cross_tissue", logger)

    # Start cross-validation tracking
    if tracker:
        tracker.start_cross_validation()

    with LogContext(cv_logger, "Cross-Tissue Validation"):
        try:
            if perf_logger:
                perf_logger.start_timer("cross_tissue_total")

            # Load eligible tissues - check multiple possible paths
            possible_paths = [
                PROJECT_ROOT / "results/results/eligible_tissues.json",
                PROJECT_ROOT / "artifacts/results/eligible_tissues.json",
                PROJECT_ROOT / "machine_learning/model_outputs/eligible_tissues.json"
            ]

            eligible_path = None
            for path in possible_paths:
                if path.exists():
                    eligible_path = path
                    break

            if not eligible_path:
                cv_logger.warning(
                    "Eligible tissue list not found in any expected location; skipping cross-tissue validation"
                )
                return None

            with open(eligible_path) as f:
                eligible_data = json.load(f)

            # Get unique tissues, excluding duplicates and ineligible ones
            all_tissues = eligible_data.get('all_eligible', eligible_data.get('available', []))
            ineligible = eligible_data.get('ineligible', [])

            # Remove duplicates and normalize tissue names
            unique_tissues = []
            seen = set()
            for tissue in all_tissues:
                if tissue not in ineligible:
                    normalized = tissue.replace('_', ' ').lower()
                    if normalized not in seen:
                        unique_tissues.append(tissue)
                        seen.add(normalized)

            tissues = unique_tissues

            cv_logger.info(f"Processing {len(tissues)} tissues for cross-validation: {tissues}")

            # Initialize components
            data_loader = DataLoader(
                data_dir=PROJECT_ROOT / "data/pigGTEx",
                metadata_path=PROJECT_ROOT / "data/PigGTEx_v0.MetaTable.xlsx"
            )
            data_loader.load_metadata()

            model = OrdinalLightGBM(
                num_leaves=31,
                max_depth=6,
                learning_rate=0.05,
                n_estimators=200,
                min_data_in_leaf=10,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                lambda_l1=0.1,
                lambda_l2=0.1,
                class_weight='balanced',
                seed=config.get('seed', 42)
            )
            preprocessor = ExpressionPreprocessor()

            # Initialize stage selector for label harmonization
            stage_selector = StageGranularitySelector()

            # Initialize validator with label harmonization
            validator = CrossTissueValidator(
                model=model,
                preprocessor=preprocessor,
                stage_selector=stage_selector,
                common_scheme='auto'  # Auto-select lowest common denominator
            )

            # Load data for each tissue with progress bar
            tissue_data = {}
            cv_logger.info("Loading tissue data...")

            # Use custom progress bar if tracker available, else use tqdm
            if tracker:
                progress_bar = tracker.create_progress_bar(len(tissues), "Loading tissues", "tissue")
            else:
                progress_bar = tqdm(tissues, desc="Loading tissues")

            for tissue in (tissues if tracker else progress_bar):
                if tracker:
                    progress_bar.update(1)
                try:
                    if perf_logger:
                        perf_logger.start_timer(f"load_{tissue}")

                    expr_data, metadata = data_loader.load_expression(tissue)
                    # Store as tuple for CrossTissueValidator compatibility
                    tissue_data[tissue] = (expr_data, metadata)

                    cv_logger.debug(
                        f"Loaded {tissue}: {expr_data.shape}, "
                        f"stages: {metadata['Stage'].value_counts().to_dict()}"
                    )

                    if perf_logger:
                        perf_logger.end_timer(f"load_{tissue}")

                except Exception as e:
                    cv_logger.warning(f"Failed to load {tissue}: {e}")
                    continue

            if tracker and hasattr(progress_bar, 'close'):
                progress_bar.close()

            cv_logger.info(f"Successfully loaded {len(tissue_data)} tissues")
            if perf_logger:
                perf_logger.log_memory_usage("after_loading_all_tissues")

            # Update memory usage for tracker
            if tracker and perf_logger:
                import psutil
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                tracker.update_memory(mem_mb)

            # Run leave-one-tissue-out validation
            cv_logger.info("Running leave-one-tissue-out validation...")
            if perf_logger:
                perf_logger.start_timer("leave_one_out")

            try:
                # Run leave-one-out for each tissue
                loto_results = {}
                within_tissue_results = {}

                cv_logger.info(f"Running LOTO validation for {len(tissue_data)} tissues...")
                for target_tissue in tissue_data.keys():
                    cv_logger.debug(f"Processing LOTO for {target_tissue}...")

                    # Run leave-one-tissue-out validation
                    loto_result = validator.leave_one_tissue_out(tissue_data, target_tissue)
                    loto_results[target_tissue] = loto_result

                    # Also run within-tissue baseline for comparison
                    within_result = validator.within_tissue_baseline(
                        tissue_data[target_tissue],
                        target_tissue,
                        test_size=0.2
                    )
                    within_tissue_results[target_tissue] = within_result

                # Compile results
                results = {
                    'leave_one_out': loto_results,
                    'within_tissue': within_tissue_results,
                    'tissues_processed': list(tissue_data.keys()),
                    'status': 'success'
                }

                cv_logger.info(f"LOTO validation completed for {len(loto_results)} tissues")

            except Exception as e:
                cv_logger.error(f"Leave-one-out validation failed: {e}")
                # Return partial results if available
                results = {
                    'error': str(e),
                    'tissues_loaded': list(tissue_data.keys()),
                    'status': 'partial_failure'
                }

            if perf_logger:
                perf_logger.end_timer("leave_one_out")
                perf_logger.end_timer("cross_tissue_total")

                # Add performance metrics to results
                results['performance'] = {
                    'total_time': perf_logger.metrics.get('cross_tissue_total', 0),
                    'tissue_loading_times': {
                        tissue: perf_logger.metrics.get(f"load_{tissue}", 0)
                        for tissue in tissues
                    },
                    'validation_time': perf_logger.metrics.get('leave_one_out', 0)
                }

            # Save results
            output_file = output_dir / "cross_tissue_validation_new.json"
            with open(output_file, 'w') as f:
                json.dump(_to_serializable(results), f, indent=2)

            cv_logger.info(f"Cross-tissue validation complete. Results saved to {output_file}")

            # Log summary
            if 'within_tissue' in results:
                cv_logger.info("Within-Tissue Performance Summary:")
                for tissue, metrics in results['within_tissue'].items():
                    acc = metrics.get('balanced_accuracy', 0)
                    cv_logger.info(f"  {tissue:20} : {acc:.3f}")

            if 'leave_one_out' in results:
                cv_logger.info("Leave-One-Out Performance Summary:")
                for tissue, metrics in results['leave_one_out'].items():
                    acc = metrics.get('balanced_accuracy', 0)
                    cv_logger.info(f"  {tissue:20} : {acc:.3f}")

            return results

        except Exception as e:
            cv_logger.error(f"Cross-tissue validation failed: {str(e)}")
            cv_logger.debug(f"Traceback: {traceback.format_exc()}")
            return None


def get_eligible_tissues(use_logger=True):
    """Get list of all eligible tissues from the saved results."""
    # Try multiple possible locations for eligible tissues file
    possible_paths = [
        PROJECT_ROOT / "results/results/eligible_tissues.json",
        PROJECT_ROOT / "artifacts/results/eligible_tissues.json",
        PROJECT_ROOT / "machine_learning/model_outputs/eligible_tissues.json"
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Get unique tissues from all_eligible
                all_eligible = data.get('all_eligible', data.get('available', []))
                # Remove duplicates (e.g., Small_intestine vs Small intestine)
                unique_tissues = []
                seen = set()
                for tissue in all_eligible:
                    normalized = tissue.replace('_', ' ').lower()
                    if normalized not in seen:
                        unique_tissues.append(tissue)
                        seen.add(normalized)
                return unique_tissues

    # If no eligible tissues file found, return a default set
    if use_logger and 'logger' in globals():
        logger.warning("No eligible tissues file found. Using default tissue list.")
    else:
        print("Warning: No eligible tissues file found. Using default tissue list.")
    return ['Muscle', 'Brain', 'Liver', 'Blood', 'Lung']


def main(args=None):
    """Main pipeline execution with enhanced logging."""
    global logger, perf_logger, progress_tracker

    # Parse arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Run ML pipeline with enhanced logging")
        parser.add_argument(
            '--tissues',
            nargs='+',
            default=None,  # Changed from ['Muscle'] to None
            help='Tissues to process (default: all eligible tissues)'
        )
        parser.add_argument(
            '--all-tissues',
            action='store_true',
            help='Process all eligible tissues (same as default behavior)'
        )
        parser.add_argument(
            '--max-features',
            type=int,
            default=2000,
            help='Maximum number of features to select (default: 2000)'
        )
        parser.add_argument(
            '--max-iter',
            type=int,
            default=5000,
            help='Maximum iterations for model training (default: 5000)'
        )
        parser.add_argument(
            '--train-ratio',
            type=float,
            default=0.7,
            help='Training data ratio (default: 0.7)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed (default: 42)'
        )
        parser.add_argument(
            '--skip-cv',
            action='store_true',
            help='Skip cross-tissue validation'
        )
        parser.add_argument(
            '--hyperparameter-validation',
            action='store_true',
            help='Run 3-fold stratified CV to validate hyperparameters before final training'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        parser.add_argument(
            '--simple-logging',
            action='store_true',
            help='Use simple logging without enhanced features'
        )
        parser.add_argument(
            '--no-progress',
            action='store_true',
            help='Disable progress tracking'
        )
        parser.add_argument(
            '--use-rich',
            action='store_true',
            help='Use rich library for enhanced progress display'
        )

        args = parser.parse_args()

    # If no tissues specified or --all-tissues flag used, use all eligible tissues
    if args.tissues is None or args.all_tissues:
        # This needs to happen before logger setup to avoid circular dependency
        args.tissues = get_eligible_tissues(use_logger=False)

    # Setup logging based on arguments
    if args.simple_logging:
        # Use simple logging configuration
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        perf_logger = None
    else:
        # Use enhanced logging configuration
        logger = setup_logging(
            name="ml_pipeline",
            level=args.log_level,
            log_dir=PROJECT_ROOT / "logs",
            console=True,
            file=True,
            json_file=True,
            colored=True
        )
        perf_logger = PerformanceLogger(logger)

    # Initialize progress tracker
    if not args.no_progress:
        progress_tracker = PipelineProgressTracker(
            total_tissues=len(args.tissues) if args.tissues else 0,
            include_cross_validation=not args.skip_cv,
            history_file=PROJECT_ROOT / "machine_learning/timing_history.json",
            use_rich=args.use_rich
        )
    else:
        progress_tracker = None

    # Log startup information
    logger.info("="*80)
    logger.info("MACHINE LEARNING PIPELINE - NEW STRUCTURE")
    logger.info("="*80)

    # Log system information if using enhanced logging
    if perf_logger:
        perf_logger.log_system_info()

    # Log configuration
    config = {
        'max_features': args.max_features,
        'max_iter': args.max_iter,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'tissues': args.tissues,
        'skip_cross_validation': args.skip_cv,
        'hyperparameter_validation': args.hyperparameter_validation
    }
    logger.info(f"Pipeline configuration: {json.dumps(config, indent=2)}")

    # Create output directory
    output_dir = PROJECT_ROOT / "machine_learning/model_outputs"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Track overall performance
    if perf_logger:
        perf_logger.start_timer("pipeline_total")

    # Start progress tracking
    if progress_tracker:
        progress_tracker.start_pipeline()

    results = {}

    # Process individual tissues
    if args.tissues:
        logger.info(f"Processing {len(args.tissues)} tissues: {args.tissues}")

        # Show overall progress if not using rich
        if progress_tracker and not progress_tracker.use_rich:
            print(f"\nProcessing {len(args.tissues)} tissues...")
            print(f"Progress tracking enabled. ETA will improve after first few tissues.\n")

        for i, tissue in enumerate(args.tissues):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {tissue} [{i+1}/{len(args.tissues)}]")
            logger.info("="*60)

            tissue_results = run_single_tissue_pipeline(
                tissue, output_dir, config, tracker=progress_tracker
            )
            if tissue_results:
                results[tissue] = tissue_results
                logger.info(f"✓ {tissue} completed successfully")
            else:
                logger.error(f"✗ {tissue} processing failed")

    # Run cross-tissue validation
    if not args.skip_cv:
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-TISSUE VALIDATION")
        logger.info("="*60)

        cross_results = run_cross_tissue_validation(
            output_dir, config, tracker=progress_tracker
        )
        if cross_results:
            results['cross_tissue'] = cross_results
            logger.info("✓ Cross-tissue validation completed")
        else:
            logger.error("✗ Cross-tissue validation failed")

    # Final summary
    if perf_logger:
        total_time = perf_logger.end_timer("pipeline_total")
    else:
        total_time = 0

    # Complete progress tracking
    if progress_tracker:
        summary = progress_tracker.finish_pipeline()

    # Skip duplicate summary if progress tracker already showed it
    if not progress_tracker:
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        if perf_logger:
            logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Tissues processed: {len([k for k in results.keys() if k != 'cross_tissue'])}")
        logger.info(f"Results saved to: {output_dir}")

    # Save overall summary
    summary_file = output_dir / "pipeline_summary.json"
    summary = {
        'configuration': config,
        'tissues_processed': list(results.keys()),
        'success_rate': len(results) / (len(args.tissues) if args.tissues else 1)
    }

    if perf_logger:
        summary['execution_time'] = total_time
        summary['performance_metrics'] = perf_logger.get_summary()

    # Add progress tracking summary
    if progress_tracker:
        summary['progress_summary'] = progress_tracker.get_progress_summary()

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pipeline summary saved to: {summary_file}")
    logger.info("Pipeline execution completed")

    # Return appropriate results based on what was run
    tissue_results = {k: v for k, v in results.items() if k != 'cross_tissue'}
    cross_results = results.get('cross_tissue')

    # For compatibility with the original main() return format
    if len(tissue_results) == 1 and 'Muscle' in tissue_results:
        return tissue_results['Muscle'], cross_results
    else:
        return tissue_results, cross_results


if __name__ == "__main__":
    muscle_results, cross_results = main()