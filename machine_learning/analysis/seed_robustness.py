#!/usr/bin/env python3
"""
Seed Robustness Analysis for Developmental Stage Classification.

Combines feature stability analysis and multi-seed performance robustness
into a single script. For each tissue and seed:
  1. Split data, preprocess train-only, train model, evaluate
  2. Extract top feature genes

Then computes:
  - Per-seed balanced accuracy and F1 macro (robustness)
  - Pairwise Jaccard similarity of top genes (feature stability)
  - Gene frequency / core gene analysis
"""

import sys
import json
import warnings
import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.model_training.models import OrdinalLightGBM
from machine_learning.model_training.hyperparameter_tuning import run_tuning, FIXED_PARAMS
from machine_learning.utils.helpers import (
    jaccard_similarity, to_samples_x_genes_df, load_and_prepare_tissue,
)

warnings.filterwarnings('ignore')

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 9999, 54321, 100000]
DEFAULT_TISSUES = ['Muscle', 'Liver', 'Brain', 'Blood', 'Lung']
N_TOP_GENES = 50


def run_single_seed(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: np.ndarray,
    seed: int,
    n_features: int = N_TOP_GENES,
    train_ratio: float = 0.7,
    n_tuning_trials: int = 30,
    n_inner_folds: int = 3,
) -> Dict:
    """
    Run one full pipeline iteration with a given seed.

    Uses Optuna inner CV tuning on the train split to find best params,
    then evaluates on the held-out test split.

    Returns dict with balanced_accuracy, f1_macro, top_genes, best_params.
    """
    X_T = X_raw.T  # samples x genes
    min_per_class = pd.Series(y).value_counts().min()

    split_kwargs = dict(test_size=1 - train_ratio, random_state=seed)
    if min_per_class >= 2:
        split_kwargs['stratify'] = y

    X_train_T, X_test_T, y_train, y_test = train_test_split(X_T, y, **split_kwargs)

    X_train_raw = X_train_T.T
    X_test_raw = X_test_T.T

    train_ids = X_train_T.index if hasattr(X_train_T, 'index') else np.arange(len(y_train))
    test_ids = X_test_T.index if hasattr(X_test_T, 'index') else np.arange(len(y_test))

    # Inner CV tuning on train split
    tuning_result = run_tuning(
        X_raw=X_train_raw,
        y=y_train,
        sample_ids=np.asarray(train_ids),
        gene_names=gene_names,
        study_name=f"seed_{seed}",
        n_trials=n_tuning_trials,
        n_inner_folds=n_inner_folds,
        seed=seed,
    )
    best_params = tuning_result["best_params"]

    # Retrain on full train split with tuned params
    preprocessor = ExpressionPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    X_train_df = to_samples_x_genes_df(X_train_proc, train_ids, preprocessor, gene_names)
    X_test_df = to_samples_x_genes_df(X_test_proc, test_ids, preprocessor, gene_names)

    model = OrdinalLightGBM(**best_params, **FIXED_PARAMS, seed=seed)
    model.fit(X_train_df, y_train)

    y_pred = model.predict(X_test_df)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    importance = model.get_feature_importance()
    top_genes = importance.head(n_features).index.tolist()

    return {
        'balanced_accuracy': float(ba),
        'f1_macro': float(f1),
        'top_genes': top_genes,
        'best_params': best_params,
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def interpret_stability(mean_jaccard: float, n_stable: int, n_features: int) -> str:
    """Interpret stability results."""
    stable_ratio = n_stable / n_features
    if mean_jaccard >= 0.6 and stable_ratio >= 0.5:
        return "STRONG STABILITY - Features are robust, likely capturing real biological signal"
    elif mean_jaccard >= 0.4 and stable_ratio >= 0.3:
        return "MODERATE STABILITY - Some features are robust, some may be noise"
    elif mean_jaccard >= 0.2 and stable_ratio >= 0.1:
        return "WEAK STABILITY - Many features change between runs, potential overfitting concern"
    else:
        return "POOR STABILITY - Features are highly unstable, likely capturing noise/overfitting"


def run_tissue_analysis(
    tissue_name: str,
    seeds: List[int],
    n_features: int = N_TOP_GENES,
    n_tuning_trials: int = 30,
    n_inner_folds: int = 3,
) -> Optional[Dict]:
    """Run multi-seed robustness + feature stability for a single tissue."""
    print(f"\n{'='*60}")
    print(f"  {tissue_name}")
    print(f"{'='*60}")

    try:
        prepared = load_and_prepare_tissue(tissue_name)
    except Exception as e:
        print(f"  Error loading {tissue_name}: {e}")
        return None

    if prepared is None:
        print(f"  {tissue_name} not eligible for classification")
        return None

    X_raw, y, sample_ids, gene_names, scheme_name = prepared
    print(f"  Samples: {len(y)}, Genes: {X_raw.shape[0]}, Scheme: {scheme_name}")

    seed_results = {}
    for seed in tqdm(seeds, desc=f"  Seeds"):
        try:
            seed_results[seed] = run_single_seed(
                X_raw, y, sample_ids, gene_names, seed, n_features,
                n_tuning_trials=n_tuning_trials,
                n_inner_folds=n_inner_folds,
            )
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    if len(seed_results) < 2:
        print(f"  Not enough successful runs")
        return None

    # --- Performance robustness ---
    ba_values = [r['balanced_accuracy'] for r in seed_results.values()]
    f1_values = [r['f1_macro'] for r in seed_results.values()]

    # --- Feature stability ---
    seed_keys = list(seed_results.keys())
    jaccard_values = []
    for s1, s2 in combinations(seed_keys, 2):
        g1 = set(seed_results[s1]['top_genes'])
        g2 = set(seed_results[s2]['top_genes'])
        jaccard_values.append(jaccard_similarity(g1, g2))

    # Gene frequency analysis
    gene_counts = {}
    for r in seed_results.values():
        for gene in r['top_genes']:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    gene_frequency = pd.Series(gene_counts).sort_values(ascending=False)
    n_runs = len(seed_results)
    core_genes = gene_frequency[gene_frequency >= n_runs * 0.5].index.tolist()
    stable_genes = gene_frequency[gene_frequency >= n_runs * 0.8].index.tolist()

    result = {
        'tissue': tissue_name,
        'scheme': scheme_name,
        'n_samples': len(y),
        'n_seeds': n_runs,
        'n_features': n_features,
        # Performance
        'balanced_accuracy': {
            'mean': float(np.mean(ba_values)),
            'std': float(np.std(ba_values)),
            'min': float(np.min(ba_values)),
            'max': float(np.max(ba_values)),
            'per_seed': {str(s): r['balanced_accuracy'] for s, r in seed_results.items()},
        },
        'f1_macro': {
            'mean': float(np.mean(f1_values)),
            'std': float(np.std(f1_values)),
            'min': float(np.min(f1_values)),
            'max': float(np.max(f1_values)),
            'per_seed': {str(s): r['f1_macro'] for s, r in seed_results.items()},
        },
        # Feature stability
        'jaccard_similarity': {
            'mean': float(np.mean(jaccard_values)),
            'std': float(np.std(jaccard_values)),
            'min': float(np.min(jaccard_values)),
            'max': float(np.max(jaccard_values)),
            'all_values': jaccard_values,
        },
        'gene_frequency': gene_frequency.to_dict(),
        'core_genes_50pct': core_genes,
        'stable_genes_80pct': stable_genes,
        'n_core_genes': len(core_genes),
        'n_stable_genes': len(stable_genes),
        'top_genes_per_seed': {str(s): r['top_genes'] for s, r in seed_results.items()},
        'interpretation': interpret_stability(
            np.mean(jaccard_values), len(stable_genes), n_features
        ),
    }

    # Print summary
    print(f"\n  BA: {np.mean(ba_values):.3f} +/- {np.std(ba_values):.3f} "
          f"(range {np.min(ba_values):.3f}-{np.max(ba_values):.3f})")
    print(f"  F1: {np.mean(f1_values):.3f} +/- {np.std(f1_values):.3f}")
    print(f"  Feature Jaccard: {np.mean(jaccard_values):.3f} +/- {np.std(jaccard_values):.3f}")
    print(f"  Core genes (>50%): {len(core_genes)}/{n_features}")
    print(f"  Stable genes (>80%): {len(stable_genes)}/{n_features}")
    print(f"  {result['interpretation']}")

    return result


def run_seed_robustness(
    tissues: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_features: int = N_TOP_GENES,
    output_dir: Optional[Path] = None,
    n_tuning_trials: int = 30,
    n_inner_folds: int = 3,
) -> Dict:
    """Run full seed robustness analysis across multiple tissues."""
    if tissues is None:
        tissues = DEFAULT_TISSUES
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if output_dir is None:
        output_dir = PROJECT_ROOT / "machine_learning/analysis/results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SEED ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print(f"Tissues: {tissues}")
    print(f"Seeds:   {seeds}")
    print(f"Features per tissue: {n_features}")
    print(f"Tuning trials per seed: {n_tuning_trials}, inner folds: {n_inner_folds}")

    all_results = {}
    for tissue in tissues:
        result = run_tissue_analysis(
            tissue, seeds, n_features,
            n_tuning_trials=n_tuning_trials,
            n_inner_folds=n_inner_folds,
        )
        if result:
            all_results[tissue] = result

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    rows = []
    for tissue, r in all_results.items():
        ba = r['balanced_accuracy']
        f1 = r['f1_macro']
        rows.append({
            'Tissue': tissue,
            'Scheme': r['scheme'],
            'BA mean': f"{ba['mean']:.3f}",
            'BA std': f"{ba['std']:.3f}",
            'F1 mean': f"{f1['mean']:.3f}",
            'Jaccard': f"{r['jaccard_similarity']['mean']:.3f}",
            'Stable(80%)': r['n_stable_genes'],
        })

    if rows:
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))

        # Stability check
        for tissue, r in all_results.items():
            ba_std = r['balanced_accuracy']['std']
            status = "STABLE" if ba_std < 0.05 else "VARIABLE"
            print(f"  {tissue}: BA std={ba_std:.4f} -> {status}")

    # Overall feature stability conclusion
    if all_results:
        mean_jaccard_overall = np.mean(
            [r['jaccard_similarity']['mean'] for r in all_results.values()]
        )
        mean_stable_ratio = np.mean(
            [r['n_stable_genes'] / r['n_features'] for r in all_results.values()]
        )
        print(f"\nOverall Jaccard: {mean_jaccard_overall:.3f}")
        print(f"Overall stable gene ratio: {mean_stable_ratio:.1%}")

    # Save results
    output_file = output_dir / "seed_robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Seed robustness analysis (performance + feature stability)"
    )
    parser.add_argument(
        '--tissues', nargs='+', default=DEFAULT_TISSUES,
        help='Tissues to evaluate'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
        help='Random seeds'
    )
    parser.add_argument(
        '--n-features', type=int, default=N_TOP_GENES,
        help='Number of top features to compare'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(PROJECT_ROOT / "machine_learning/analysis/results"),
        help='Output directory'
    )
    parser.add_argument(
        '--n-tuning-trials', type=int, default=30,
        help='Optuna trials per seed for hyperparameter tuning (default: 30)'
    )
    parser.add_argument(
        '--n-inner-folds', type=int, default=3,
        help='Inner CV folds for tuning (default: 3)'
    )
    args = parser.parse_args()

    run_seed_robustness(
        tissues=args.tissues,
        seeds=args.seeds,
        n_features=args.n_features,
        output_dir=Path(args.output_dir),
        n_tuning_trials=args.n_tuning_trials,
        n_inner_folds=args.n_inner_folds,
    )


if __name__ == "__main__":
    main()
