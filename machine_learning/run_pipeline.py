#!/usr/bin/env python3
"""Main pipeline script for pig developmental stage classification."""

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
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.utils.logging_config import (
    setup_logging, LogContext, log_execution_time, create_module_logger
)
from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.model_training.hyperparameter_tuning import (
    run_tuning, compute_param_stability, FIXED_PARAMS,
)
from machine_learning.model_evaluation.evaluation import MetricCalculator
from machine_learning.utils.config_loader import get_config
from machine_learning.utils.helpers import to_samples_x_genes_df

logger = logging.getLogger(__name__)


def _to_serializable(value):
    """Recursively convert numpy/pandas objects to JSON-friendly types."""
    if isinstance(value, dict):
        return {_to_serializable(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_serializable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        v = value.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
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

    if data.isnull().any().any():
        null_genes = data.isnull().sum(axis=1)
        null_genes = null_genes[null_genes > 0]
        issues.append(f"Found {len(null_genes)} genes with missing values")
        logger.warning(f"{tissue}: {len(null_genes)} genes have missing expression values")

    const_genes = data.var(axis=1) == 0
    if const_genes.any():
        issues.append(f"Found {const_genes.sum()} genes with zero variance")
        logger.warning(f"{tissue}: {const_genes.sum()} genes have constant expression")

    sample_means = data.mean(axis=0)
    outliers = np.abs(sample_means - sample_means.mean()) > 3 * sample_means.std()
    if outliers.any():
        outlier_samples = data.columns[outliers]
        issues.append(f"Found {len(outlier_samples)} potential outlier samples")
        logger.warning(f"{tissue}: {len(outlier_samples)} samples may be outliers: {outlier_samples[:5].tolist()}...")

    if len(data.columns) != len(metadata):
        issues.append(f"Metadata mismatch: {len(data.columns)} samples vs {len(metadata)} metadata entries")
        logger.error(f"{tissue}: Sample count mismatch between expression and metadata")

    if issues:
        logger.info(f"{tissue} data quality issues: {'; '.join(issues)}")
    else:
        logger.info(f"{tissue}: Data quality check passed")

    return issues



def determine_n_folds(y, default_k=5):
    """
    Determine the number of CV folds based on class distribution.

    Returns (n_folds, method_name):
        - default_k if min_samples_per_class >= default_k
        - min_samples_per_class if 3 <= min_samples_per_class < default_k
        - 'loo' if min_samples_per_class < 3
    """
    min_samples = int(pd.Series(y).value_counts().min())
    if min_samples >= default_k:
        return default_k, f"{default_k}-fold"
    elif min_samples >= 3:
        return min_samples, f"{min_samples}-fold (reduced from {default_k})"
    else:
        return len(y), "leave-one-out"


def aggregate_cv_metrics(fold_metrics_list):
    """
    Aggregate metric dicts from multiple CV folds.

    Returns dict of {metric_name: {mean, std, per_fold}} for each scalar numeric metric.
    """
    if not fold_metrics_list:
        return {}

    scalar_keys = []
    for key, val in fold_metrics_list[0].items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            scalar_keys.append(key)

    aggregated = {}
    for key in scalar_keys:
        values = []
        for fm in fold_metrics_list:
            v = fm.get(key)
            if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                values.append(float(v))
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'per_fold': values
            }
    return aggregated


@log_execution_time(logger)
def run_single_tissue_pipeline(
    tissue_name: str,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run the ML pipeline for a single tissue.

    Args:
        tissue_name: Name of the tissue to process
        output_dir: Directory to save results
        config: Optional configuration parameters

    Returns:
        Dictionary with results or None if processing failed
    """
    config = config or {}
    tissue_logger = create_module_logger(f"tissue.{tissue_name}", logger)

    with LogContext(tissue_logger, f"Processing {tissue_name}"):
        try:
            # ===== 1. DATA LOADING PHASE =====
            with LogContext(tissue_logger, "Data Loading", level=logging.INFO):
                pipeline_config = get_config()
                data_dir = PROJECT_ROOT / pipeline_config.paths.get("data_dir", "data/pigGTEx")
                metadata_file = PROJECT_ROOT / pipeline_config.paths.get("metadata_file", "data/PigGTEx_v0.MetaTable.xlsx")

                data_loader = DataLoader(data_dir=data_dir, metadata_path=metadata_file)
                data_loader.load_metadata()

                tissue_logger.debug(f"Loaded metadata with {len(data_loader.metadata)} entries")

                expr_pattern = pipeline_config.paths.get(
                    "expression_pattern",
                    "data/pigGTEx/{tissue_name}.expr_tpm.txt.gz"
                )
                tissue_file = PROJECT_ROOT / expr_pattern.format(tissue_name=tissue_name)
                if not tissue_file.exists():
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

                log_data_statistics(expr_data, f"{tissue_name}_expression", tissue_logger)

                X = expr_data
                y = metadata['Stage'].values
                sample_ids = metadata.index.values
                gene_names = expr_data.index.values

                tissue_logger.info(
                    f"Loaded {tissue_name}: {X.shape[0]} genes x {X.shape[1]} samples, "
                    f"Stage distribution: {pd.Series(y).value_counts().to_dict()}"
                )

                validate_data_quality(X, metadata, tissue_name, tissue_logger)

            # ===== 2. STAGE SELECTION PHASE =====
            with LogContext(tissue_logger, "Stage Selection"):
                stage_selector = StageGranularitySelector()
                stage_counts = pd.Series(y).value_counts()

                tissue_logger.debug(f"Stage counts: {stage_counts.to_dict()}")

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

                    unique_labels = labels if isinstance(labels, list) else list(labels)
                    label_to_int = {label: i for i, label in enumerate(unique_labels)}
                    y = np.array([label_to_int.get(label, -1) for label in y])

                    valid_mask = y != -1
                    if not valid_mask.all():
                        n_unknown = (~valid_mask).sum()
                        tissue_logger.warning(
                            f"Removing {n_unknown} samples with stage labels that don't match "
                            f"the selected scheme."
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

            # ===== 3. NESTED CROSS-VALIDATION EVALUATION =====
            with LogContext(tissue_logger, "Nested Cross-Validation Evaluation"):
                random_state = config.get('seed', 42)
                default_k = config.get('n_cv_folds', 5)
                n_trials = config.get('n_trials', 25)
                n_inner_folds = config.get('n_inner_folds', 3)
                timeout = config.get('timeout', 600)
                n_folds, eval_method = determine_n_folds(y, default_k=default_k)

                tissue_logger.info(
                    f"Nested CV strategy: {eval_method} (outer_folds={n_folds}, "
                    f"inner_folds={n_inner_folds}, n_trials={n_trials}, "
                    f"n_samples={len(y)}, min_per_class={int(pd.Series(y).value_counts().min())})"
                )

                if n_folds == len(y):
                    cv_splitter = LeaveOneOut()
                else:
                    cv_splitter = StratifiedKFold(
                        n_splits=n_folds, shuffle=True, random_state=random_state
                    )

                X_T = X.T  # samples x genes

                fold_test_metrics_list = []
                fold_train_metrics_list = []
                per_fold_details = []
                best_params_per_fold = []

                all_y_true = []
                all_y_pred = []
                all_y_prob = []
                all_sample_indices = []

                # No bootstrap for per-fold metrics (expensive, not needed per fold)
                fold_evaluator = MetricCalculator(calculate_bootstrap=False)

                for fold_i, (train_idx, test_idx) in enumerate(cv_splitter.split(X_T, y)):
                    tissue_logger.info(
                        f"Outer fold {fold_i + 1}/{n_folds}: "
                        f"train={len(train_idx)}, test={len(test_idx)}"
                    )

                    X_train_raw = X.iloc[:, train_idx]
                    X_test_raw = X.iloc[:, test_idx]
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    fold_train_ids = sample_ids[train_idx]
                    fold_test_ids = sample_ids[test_idx]

                    # --- Inner hyperparameter tuning ---
                    tissue_logger.info(
                        f"  Tuning hyperparameters ({n_trials} trials, "
                        f"{n_inner_folds} inner folds)..."
                    )
                    tuning_result = run_tuning(
                        X_raw=X_train_raw,
                        y=y_train,
                        sample_ids=fold_train_ids,
                        gene_names=gene_names,
                        study_name=f"{tissue_name}_fold{fold_i + 1}",
                        n_trials=n_trials,
                        n_inner_folds=n_inner_folds,
                        seed=random_state + fold_i,
                        timeout=timeout,
                    )
                    best_params = tuning_result["best_params"]
                    best_params_per_fold.append(best_params)

                    tissue_logger.info(
                        f"  Inner best score: {tuning_result['best_score']:.3f} "
                        f"({tuning_result['n_trials_completed']} completed, "
                        f"{tuning_result['n_trials_pruned']} pruned, "
                        f"{tuning_result['tuning_duration_s']}s)"
                    )

                    # --- Retrain on full outer train with tuned params ---
                    fold_preprocessor = ExpressionPreprocessor()
                    X_train_proc = fold_preprocessor.fit_transform(X_train_raw)
                    X_test_proc = fold_preprocessor.transform(X_test_raw)

                    X_train_df = to_samples_x_genes_df(
                        X_train_proc, fold_train_ids, fold_preprocessor, gene_names
                    )
                    X_test_df = to_samples_x_genes_df(
                        X_test_proc, fold_test_ids, fold_preprocessor, gene_names
                    )

                    fold_model = OrdinalLightGBM(
                        **best_params, **FIXED_PARAMS, seed=random_state + fold_i
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fold_model.fit(X_train_df, y_train)

                    fold_labels = list(fold_model.classes_)

                    y_pred = fold_model.predict(X_test_df)
                    y_prob = fold_model.predict_proba(X_test_df)

                    y_train_pred = fold_model.predict(X_train_df)
                    y_train_prob = fold_model.predict_proba(X_train_df)

                    fold_test_m = fold_evaluator.calculate(
                        y_test, y_pred, y_prob=y_prob, labels=fold_labels
                    )
                    fold_train_m = fold_evaluator.calculate(
                        y_train, y_train_pred, y_prob=y_train_prob, labels=fold_labels
                    )

                    fold_test_metrics_list.append(fold_test_m)
                    fold_train_metrics_list.append(fold_train_m)

                    tissue_logger.info(
                        f"  Fold {fold_i + 1} test balanced_accuracy="
                        f"{fold_test_m['balanced_accuracy']:.3f}, "
                        f"train balanced_accuracy={fold_train_m['balanced_accuracy']:.3f}"
                    )

                    per_fold_details.append({
                        'fold': fold_i + 1,
                        'n_train': len(train_idx),
                        'n_test': len(test_idx),
                        'best_params': best_params,
                        'inner_best_score': tuning_result['best_score'],
                        'test_balanced_accuracy': fold_test_m['balanced_accuracy'],
                        'train_balanced_accuracy': fold_train_m['balanced_accuracy'],
                        'n_trials_completed': tuning_result['n_trials_completed'],
                        'n_trials_pruned': tuning_result['n_trials_pruned'],
                        'tuning_duration_s': tuning_result['tuning_duration_s'],
                    })

                    all_y_true.append(y_test)
                    all_y_pred.append(y_pred)
                    all_y_prob.append(y_prob)
                    all_sample_indices.extend(test_idx.tolist())

                # Aggregate CV metrics
                agg_test_metrics = aggregate_cv_metrics(fold_test_metrics_list)
                agg_train_metrics = aggregate_cv_metrics(fold_train_metrics_list)

                assert sorted(all_sample_indices) == list(range(len(y))), (
                    f"CV sanity check failed: expected {len(y)} unique test indices, "
                    f"got {len(all_sample_indices)} (unique: {len(set(all_sample_indices))})"
                )

                # Pool all held-out predictions — bootstrap only on the final pooled result
                pooled_y_true = np.concatenate(all_y_true)
                pooled_y_pred = np.concatenate(all_y_pred)
                pooled_y_prob = np.concatenate(all_y_prob, axis=0)

                all_labels = sorted(set(pooled_y_true.tolist()))
                pooled_evaluator = MetricCalculator(calculate_bootstrap=True)
                metrics = pooled_evaluator.calculate(
                    pooled_y_true, pooled_y_pred,
                    y_prob=pooled_y_prob, labels=all_labels
                )

                param_stability = compute_param_stability(best_params_per_fold)

                tissue_logger.info(
                    f"CV pooled test metrics - "
                    f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}, "
                    f"F1 Macro: {metrics.get('f1_macro', 0):.3f}, "
                    f"F1 Weighted: {metrics.get('f1_weighted', 0):.3f}"
                )

                if 'balanced_accuracy' in agg_test_metrics:
                    ba = agg_test_metrics['balanced_accuracy']
                    tissue_logger.info(
                        f"CV aggregated test balanced_accuracy: "
                        f"{ba['mean']:.3f} +/- {ba['std']:.3f} "
                        f"(per fold: {[f'{v:.3f}' for v in ba['per_fold']]})"
                    )

            # ===== 4. FINAL MODEL TRAINING (ON ALL DATA) =====
            with LogContext(tissue_logger, "Final Model Training"):
                # Tune hyperparameters on all data via inner CV
                tissue_logger.info(
                    f"Tuning final model hyperparameters ({n_trials} trials)..."
                )
                final_tuning = run_tuning(
                    X_raw=X,
                    y=y,
                    sample_ids=sample_ids,
                    gene_names=gene_names,
                    study_name=f"{tissue_name}_final",
                    n_trials=n_trials,
                    n_inner_folds=n_inner_folds,
                    seed=random_state,
                    timeout=timeout,
                )
                final_model_params = final_tuning["best_params"]

                tissue_logger.info(
                    f"Final tuning best inner score: {final_tuning['best_score']:.3f} "
                    f"({final_tuning['n_trials_completed']} completed, "
                    f"{final_tuning['n_trials_pruned']} pruned, "
                    f"{final_tuning['tuning_duration_s']}s)"
                )

                final_preprocessor = ExpressionPreprocessor()
                X_all_processed = final_preprocessor.fit_transform(X)
                X_all_df = to_samples_x_genes_df(
                    X_all_processed, sample_ids, final_preprocessor, gene_names
                )

                tissue_logger.info(
                    f"Final model preprocessing: {X.shape} -> {X_all_df.shape} "
                    f"({X.shape[0] - X_all_df.shape[1]} genes removed)"
                )

                final_model = OrdinalLightGBM(
                    **final_model_params, **FIXED_PARAMS, seed=random_state
                )
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    final_model.fit(X_all_df, y)
                    if w:
                        for warning in w:
                            tissue_logger.warning(f"Final model training warning: {warning.message}")

                importance = final_model.get_feature_importance()
                top_genes = importance.head(1000).index.to_numpy()
                tissue_logger.info(
                    f"Top 10 genes by importance: {importance.head(10).index.tolist()}"
                )

                y_all_pred = final_model.predict(X_all_df)
                y_all_prob = final_model.predict_proba(X_all_df)
                train_evaluator = MetricCalculator(calculate_bootstrap=False)
                train_metrics = train_evaluator.calculate(
                    y, y_all_pred, y_prob=y_all_prob,
                    labels=list(final_model.classes_)
                )

                tissue_logger.info(
                    f"Final model (all data) - "
                    f"Balanced Accuracy: {train_metrics['balanced_accuracy']:.3f}, "
                    f"F1 Macro: {train_metrics.get('f1_macro', 0):.3f}"
                )

            # ===== 5. SAVE RESULTS =====
            with LogContext(tissue_logger, "Saving Results"):
                results = {
                    'tissue': tissue_name,
                    'scheme': scheme_name,
                    'n_samples': len(y),
                    'n_genes_initial': X.shape[0],
                    'n_genes_preprocessed': X_all_df.shape[1],
                    'n_features_used': X_all_df.shape[1],
                    'metrics': metrics,
                    'train_metrics': train_metrics,
                    'top_genes': top_genes[:1000].tolist(),
                    'feature_importance': importance.head(1000).to_dict(),
                    'cross_validation': {
                        'n_folds': n_folds,
                        'n_tuning_trials': n_trials,
                        'n_inner_folds': n_inner_folds,
                        'evaluation_method': eval_method,
                        'seed': random_state,
                        'aggregated_metrics': agg_test_metrics,
                        'aggregated_train_metrics': agg_train_metrics,
                        'per_fold_details': per_fold_details,
                        'param_stability': param_stability,
                    },
                    'final_model_params': final_model_params,
                }

                output_file = output_dir / f"{tissue_name}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(_to_serializable(results), f, indent=2)

                tissue_logger.info(f"Results saved to {output_file}")

            return results

        except Exception as e:
            tissue_logger.error(f"Pipeline failed for {tissue_name}: {str(e)}")
            tissue_logger.debug(f"Traceback: {traceback.format_exc()}")
            return None


def get_eligible_tissues(use_logger=True):
    """Get list of all eligible tissues from the saved results."""
    possible_paths = [
        PROJECT_ROOT / "results/results/eligible_tissues.json",
        PROJECT_ROOT / "artifacts/results/eligible_tissues.json",
        PROJECT_ROOT / "machine_learning/model_outputs/eligible_tissues.json"
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                all_eligible = data.get('all_eligible', data.get('available', []))
                unique_tissues = []
                seen = set()
                for tissue in all_eligible:
                    normalized = tissue.replace('_', ' ').lower()
                    if normalized not in seen:
                        unique_tissues.append(tissue)
                        seen.add(normalized)
                return unique_tissues

    if use_logger and 'logger' in globals():
        logger.warning("No eligible tissues file found. Using default tissue list.")
    else:
        print("Warning: No eligible tissues file found. Using default tissue list.")
    return ['Muscle', 'Brain', 'Liver', 'Blood', 'Lung']


def main(args=None):
    """Main pipeline execution."""
    global logger

    if args is None:
        parser = argparse.ArgumentParser(description="Run ML pipeline")
        parser.add_argument(
            '--tissues', nargs='+', default=None,
            help='Tissues to process (default: all eligible tissues)'
        )
        parser.add_argument(
            '--all-tissues', action='store_true',
            help='Process all eligible tissues (same as default behavior)'
        )
        parser.add_argument(
            '--n-cv-folds', type=int, default=5,
            help='Number of CV folds (auto-reduced for small datasets, default: 5)'
        )
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Random seed (default: 42)'
        )
        parser.add_argument(
            '--n-trials', type=int, default=25,
            help='Number of Optuna trials per outer fold for hyperparameter tuning (default: 25)'
        )
        parser.add_argument(
            '--n-inner-folds', type=int, default=3,
            help='Number of inner CV folds for hyperparameter tuning (default: 3)'
        )
        parser.add_argument(
            '--timeout', type=int, default=600,
            help='Per-study Optuna timeout in seconds (default: 600)'
        )
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )

        args = parser.parse_args()

    if args.tissues is None or args.all_tissues:
        args.tissues = get_eligible_tissues(use_logger=False)

    # Setup logging
    logger = setup_logging(name="ml_pipeline", level=args.log_level)

    # Log startup
    logger.info("=" * 80)
    logger.info("MACHINE LEARNING PIPELINE")
    logger.info("=" * 80)

    config = {
        'n_cv_folds': args.n_cv_folds,
        'seed': args.seed,
        'n_trials': getattr(args, 'n_trials', 25),
        'n_inner_folds': getattr(args, 'n_inner_folds', 3),
        'timeout': getattr(args, 'timeout', 600),
        'tissues': args.tissues,
    }
    logger.info(f"Pipeline configuration: {json.dumps(config, indent=2)}")

    output_dir = PROJECT_ROOT / "machine_learning/model_outputs"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    results = {}

    # Process individual tissues
    if args.tissues:
        logger.info(f"Processing {len(args.tissues)} tissues: {args.tissues}")

        for i, tissue in enumerate(args.tissues):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {tissue} [{i+1}/{len(args.tissues)}]")
            logger.info("=" * 60)

            tissue_results = run_single_tissue_pipeline(tissue, output_dir, config)
            if tissue_results:
                results[tissue] = tissue_results
                logger.info(f"  {tissue} completed successfully")
            else:
                logger.error(f"  {tissue} processing failed")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Tissues processed: {len(results)}")
    logger.info(f"Results saved to: {output_dir}")

    # Save overall summary
    summary_file = output_dir / "pipeline_summary.json"
    summary = {
        'configuration': config,
        'tissues_processed': list(results.keys()),
        'success_rate': len(results) / (len(args.tissues) if args.tissues else 1)
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pipeline summary saved to: {summary_file}")
    logger.info("Pipeline execution completed")

    # Return results
    if len(results) == 1 and 'Muscle' in results:
        return results['Muscle'], None
    else:
        return results, None


if __name__ == "__main__":
    muscle_results, cross_results = main()
