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
from machine_learning.feature_engineering.feature_selection import StableFeatureSelector
from machine_learning.feature_engineering.fast_feature_selection import FastFeatureSelector, HybridFeatureSelector
from machine_learning.model_training.models import (
    OrdinalLogisticRegression,
    StageClassifier
)
from sklearn.linear_model import LogisticRegression as BinaryLogisticRegression
from machine_learning.model_evaluation.evaluation import MetricCalculator
from machine_learning.cross_tissue_analysis.cross_tissue_validation import CrossTissueValidator

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

                data_loader = DataLoader(
                    data_dir=PROJECT_ROOT / "data/pigGTEx",
                    metadata_path=PROJECT_ROOT / "data/full_metadata.csv"
                )
                data_loader.load_metadata()

                tissue_logger.debug(f"Loaded metadata with {len(data_loader.metadata)} entries")

                # Check if tissue data exists (handle spaces in tissue names)
                tissue_file = PROJECT_ROOT / f"data/pigGTEx/{tissue_name}.expr_tpm.txt.gz"
                if not tissue_file.exists():
                    # Try with underscores instead of spaces
                    tissue_file_alt = PROJECT_ROOT / f"data/pigGTEx/{tissue_name.replace(' ', '_')}.expr_tpm.txt.gz"
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

                    tissue_logger.info(
                        f"Remapped stages: {pd.Series(y_original).value_counts().to_dict()} -> "
                        f"{pd.Series(y_mapped).value_counts().to_dict()}"
                    )

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_stage_selection")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 3. PREPROCESSING PHASE =====
            phase_name = "Preprocessing"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_preprocessing")

                preprocessor = ExpressionPreprocessor()

                # Log preprocessing parameters (check if attribute exists)
                variance_threshold = getattr(preprocessor, 'variance_threshold', 'default')
                tissue_logger.debug(f"Preprocessing with variance_threshold={variance_threshold}")

                X_processed = preprocessor.fit_transform(X)

                # Ensure we keep gene names aligned with the processed matrix
                if isinstance(X_processed, pd.DataFrame):
                    X_processed_df = X_processed.T.copy()
                    X_processed_df.index = sample_ids
                else:
                    X_processed_array = np.asarray(X_processed)
                    if X_processed_array.shape[0] == gene_names.shape[0]:
                        X_processed_array = X_processed_array.T

                    processed_gene_names = (
                        np.asarray(getattr(preprocessor, 'feature_names_', None))
                        if getattr(preprocessor, 'feature_names_', None)
                        else np.asarray(gene_names)
                    )

                    X_processed_df = pd.DataFrame(
                        X_processed_array,
                        index=sample_ids,
                        columns=processed_gene_names
                    )

                if X_processed_df.shape[0] != len(y):
                    raise ValueError(
                        "Preprocessed expression row count does not match target labels"
                    )

                # Log preprocessing results
                tissue_logger.info(
                    f"Preprocessing complete: {X.shape} -> {X_processed_df.shape} "
                    f"({X.shape[0] - X_processed_df.shape[1]} genes removed)"
                )

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_preprocessing")
                    perf_logger.log_memory_usage(f"{tissue_name}_after_preprocessing")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 4. FEATURE SELECTION PHASE =====
            phase_name = "Feature Selection"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_feature_selection")

                max_features = min(config.get('max_features', 2000), X_processed_df.shape[1])

                # Use fast feature selector for much better performance
                feature_selector = FastFeatureSelector(
                    method='mutual_info',  # Use mutual information for fast feature ranking
                    max_features=max_features,
                    variance_threshold_percentile=20,
                    n_jobs=-1,  # Use all cores for parallel processing
                    seed=config.get('seed', 42)
                )

                tissue_logger.debug(
                    f"Feature selection parameters: max_features={max_features}, "
                    f"method=mutual_info (fast univariate selection)"
                )

                # Convert y to pandas Series for feature selector
                y_series = pd.Series(y, index=X_processed_df.index)
                X_selected_df = feature_selector.fit_transform(X_processed_df, y_series)

                if not isinstance(X_selected_df, pd.DataFrame):
                    X_selected_df = pd.DataFrame(
                        X_selected_df,
                        index=X_processed_df.index,
                        columns=feature_selector.selected_features_
                    )

                selected_genes = X_selected_df.columns.to_numpy()
                X_selected = X_selected_df.to_numpy()

                if X_selected.shape[1] != len(selected_genes):
                    raise ValueError(
                        "Selected feature matrix column count does not match "
                        "selected gene names length"
                    )

                tissue_logger.info(
                    f"Selected {len(selected_genes)}/{X_processed_df.shape[1]} features, "
                    f"Top 10 genes: {selected_genes[:10].tolist()}"
                )

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_feature_selection")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 5. MODEL TRAINING PHASE =====
            phase_name = "Model Training"
            with LogContext(tissue_logger, phase_name):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_training")

                # Select model based on scheme
                if '2-class' in scheme_name:
                    model = BinaryLogisticRegression(
                        max_iter=config.get('max_iter', 5000),  # Increased for convergence
                        random_state=config.get('seed', 42),
                        class_weight='balanced',  # Fix: Add class weight balancing
                        solver='liblinear' if X_selected.shape[0] < 1000 else 'lbfgs'  # Adaptive solver
                    )
                    tissue_logger.debug("Using binary logistic regression with balanced class weights")
                else:
                    model = OrdinalLogisticRegression(
                        max_iter=config.get('max_iter', 5000),  # Increased for convergence
                        seed=config.get('seed', 42)
                    )
                    tissue_logger.debug("Using ordinal logistic regression")

                # Train/test split with stratification
                from sklearn.model_selection import train_test_split
                split_ratio = config.get('train_ratio', 0.7)
                random_state = config.get('seed', 42)

                # Use stratified split if we have enough samples per class
                min_samples_per_class = pd.Series(y).value_counts().min()
                if min_samples_per_class >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=1-split_ratio,
                        stratify=y, random_state=random_state
                    )
                    tissue_logger.info(f"Using stratified train/test split")
                else:
                    # Fall back to random split without stratification
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=1-split_ratio,
                        random_state=random_state
                    )
                    tissue_logger.warning(f"Not enough samples for stratification, using random split")

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

                # Fit model with warning capture
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    model.fit(X_train, y_train)

                    if w:
                        for warning in w:
                            tissue_logger.warning(f"Model training warning: {warning.message}")

                # Make predictions
                y_pred = model.predict(X_test)

                # Get probability predictions for AUROC/AUPRC
                y_prob = model.predict_proba(X_test)

                # Log prediction distribution
                pred_dist = pd.Series(y_pred).value_counts()
                tissue_logger.debug(f"Prediction distribution: {pred_dist.to_dict()}")

                if perf_logger:
                    perf_logger.end_timer(f"{tissue_name}_training")
                if tracker:
                    tracker.end_phase(phase_name)

            # ===== 6. EVALUATION PHASE =====
            phase_name = "Evaluation"
            with LogContext(tissue_logger, "Model Evaluation"):
                if tracker:
                    tracker.start_phase(phase_name)
                if perf_logger:
                    perf_logger.start_timer(f"{tissue_name}_evaluation")

                evaluator = MetricCalculator()
                label_series = pd.Series(y).dropna()
                labels = label_series.unique().tolist()
                # Pass probability predictions for AUROC/AUPRC calculation
                metrics = evaluator.calculate(y_test, y_pred, y_prob=y_prob, labels=labels)

                # Log detailed metrics
                tissue_logger.info(
                    f"Performance metrics - "
                    f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}, "
                    f"F1 Macro: {metrics.get('f1_macro', 0):.3f}, "
                    f"F1 Weighted: {metrics.get('f1_weighted', 0):.3f}"
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
                    'n_genes_preprocessed': X_processed_df.shape[1],
                    'n_features_selected': len(selected_genes),
                    'metrics': metrics,
                    'top_genes': selected_genes[:50].tolist()
                }

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
                metadata_path=PROJECT_ROOT / "data/full_metadata.csv"
            )
            data_loader.load_metadata()

            model = BinaryLogisticRegression(
                max_iter=config.get('max_iter', 1000),
                random_state=config.get('seed', 42)
            )
            preprocessor = ExpressionPreprocessor()
            feature_selector = StableFeatureSelector(
                max_features=config.get('max_features', 2000)
            )

            # Initialize validator
            validator = CrossTissueValidator(
                model=model,
                preprocessor=preprocessor,
                feature_selector=feature_selector
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
        'skip_cross_validation': args.skip_cv
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