#!/usr/bin/env python3
"""
Feature Stability Analysis for LightGBM Age Prediction Model

This script analyzes whether the same genes are consistently selected across
different random seeds, which helps determine if the model is overfitting.

Key metrics:
- Jaccard similarity across seeds (>60% suggests stable, biological signal)
- Feature frequency (how often each gene is selected across seeds)
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector
from machine_learning.model_training.models import OrdinalLightGBM

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def run_feature_selection_with_seed(
    X_processed: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_features: int = 50,
    train_ratio: float = 0.7
) -> Tuple[List[str], pd.Series]:
    """
    Run feature selection pipeline with a specific random seed.

    Returns:
        Tuple of (selected_genes, feature_importance)
    """
    from sklearn.model_selection import train_test_split

    # Check if stratification is possible
    min_samples_per_class = pd.Series(y).value_counts().min()

    if min_samples_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=1 - train_ratio,
            stratify=y, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=1 - train_ratio,
            random_state=seed
        )

    # Train LightGBM model for feature importance
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
        seed=seed
    )

    model.fit(X_train, y_train)
    importance = model.get_feature_importance()

    # Get top features
    selected_genes = importance.head(n_features).index.tolist()

    return selected_genes, importance


def analyze_tissue_stability(
    tissue_name: str,
    seeds: List[int],
    n_features: int = 50
) -> Dict:
    """
    Analyze feature selection stability for a single tissue across multiple seeds.

    Returns:
        Dictionary with stability metrics and selected genes per seed
    """
    print(f"\n{'='*60}")
    print(f"Analyzing stability for {tissue_name}")
    print(f"{'='*60}")

    # Load data
    data_loader = DataLoader(
        data_dir=PROJECT_ROOT / "data/pigGTEx",
        metadata_path=PROJECT_ROOT / "data/PigGTEx_v0.MetaTable.xlsx"
    )
    data_loader.load_metadata()

    try:
        expr_data, metadata = data_loader.load_expression(tissue_name)
    except Exception as e:
        print(f"  Error loading {tissue_name}: {e}")
        return None

    X = expr_data
    y = metadata['Stage'].values

    # Stage selection
    stage_selector = StageGranularitySelector()
    stage_counts = pd.Series(y).value_counts()
    scheme_name, scheme = stage_selector.select_scheme(stage_counts, tissue_name)

    if scheme is None:
        print(f"  {tissue_name} not eligible for classification")
        return None

    # Map stages
    if 'mapping' in scheme:
        labels = scheme.get('labels', scheme.get('stages', []))
        stage_mapping = scheme['mapping']
        y_mapped = [stage_mapping.get(stage, stage) for stage in y]
        label_to_int = {label: i for i, label in enumerate(labels)}
        y = np.array([label_to_int.get(label, -1) for label in y_mapped])

        # Filter invalid samples
        valid_mask = y != -1
        X = X.loc[:, valid_mask]
        y = y[valid_mask]
        metadata = metadata.loc[valid_mask]

    # Preprocess
    preprocessor = ExpressionPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Convert to DataFrame with proper orientation
    if isinstance(X_processed, pd.DataFrame):
        X_processed_df = X_processed.T.copy()
    else:
        X_processed_array = np.asarray(X_processed)
        if X_processed_array.shape[0] != len(y):
            X_processed_array = X_processed_array.T

        gene_names = np.asarray(getattr(preprocessor, 'feature_names_', None) or X.index.values)
        X_processed_df = pd.DataFrame(
            X_processed_array,
            index=metadata.index,
            columns=gene_names
        )

    print(f"  Samples: {len(y)}, Genes: {X_processed_df.shape[1]}, Scheme: {scheme_name}")

    # Run feature selection with different seeds
    all_selected_genes = {}
    all_importances = {}

    for seed in tqdm(seeds, desc=f"  Running {len(seeds)} seeds"):
        try:
            selected, importance = run_feature_selection_with_seed(
                X_processed_df, y, seed, n_features
            )
            all_selected_genes[seed] = selected
            all_importances[seed] = importance
        except Exception as e:
            print(f"  Error with seed {seed}: {e}")
            continue

    if len(all_selected_genes) < 2:
        print(f"  Not enough successful runs for stability analysis")
        return None

    # Calculate pairwise Jaccard similarity
    seed_pairs = list(combinations(all_selected_genes.keys(), 2))
    jaccard_values = []

    for s1, s2 in seed_pairs:
        genes1 = set(all_selected_genes[s1])
        genes2 = set(all_selected_genes[s2])
        jaccard_values.append(jaccard_similarity(genes1, genes2))

    # Calculate gene frequency across seeds
    gene_counts = {}
    for genes in all_selected_genes.values():
        for gene in genes:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    # Sort by frequency
    gene_frequency = pd.Series(gene_counts).sort_values(ascending=False)

    # Calculate core genes (appear in >50% of runs)
    n_runs = len(all_selected_genes)
    core_genes = gene_frequency[gene_frequency >= n_runs * 0.5].index.tolist()
    stable_genes = gene_frequency[gene_frequency >= n_runs * 0.8].index.tolist()

    results = {
        'tissue': tissue_name,
        'n_seeds': len(all_selected_genes),
        'n_features': n_features,
        'scheme': scheme_name,
        'n_samples': len(y),
        'jaccard_similarity': {
            'mean': float(np.mean(jaccard_values)),
            'std': float(np.std(jaccard_values)),
            'min': float(np.min(jaccard_values)),
            'max': float(np.max(jaccard_values)),
            'all_values': jaccard_values
        },
        'gene_frequency': gene_frequency.to_dict(),
        'core_genes_50pct': core_genes,
        'stable_genes_80pct': stable_genes,
        'n_core_genes': len(core_genes),
        'n_stable_genes': len(stable_genes),
        'selected_genes_per_seed': {str(k): v for k, v in all_selected_genes.items()},
        'interpretation': interpret_stability(np.mean(jaccard_values), len(stable_genes), n_features)
    }

    # Print summary
    print(f"\n  Results for {tissue_name}:")
    print(f"    Mean Jaccard similarity: {np.mean(jaccard_values):.3f} (+/- {np.std(jaccard_values):.3f})")
    print(f"    Core genes (>50%): {len(core_genes)}/{n_features}")
    print(f"    Stable genes (>80%): {len(stable_genes)}/{n_features}")
    print(f"    Interpretation: {results['interpretation']}")

    return results


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


def run_stability_analysis(
    tissues: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_features: int = 50,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run full stability analysis across multiple tissues.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

    if tissues is None:
        tissues = ['Muscle', 'Liver', 'Brain', 'Blood', 'Lung']

    if output_dir is None:
        output_dir = PROJECT_ROOT / "machine_learning/analysis/results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FEATURE STABILITY ANALYSIS")
    print("="*80)
    print(f"Tissues: {tissues}")
    print(f"Seeds: {seeds}")
    print(f"Features per tissue: {n_features}")

    all_results = {}

    for tissue in tissues:
        result = analyze_tissue_stability(tissue, seeds, n_features)
        if result:
            all_results[tissue] = result

    # Generate summary
    print("\n" + "="*80)
    print("STABILITY ANALYSIS SUMMARY")
    print("="*80)

    summary_data = []
    for tissue, result in all_results.items():
        summary_data.append({
            'Tissue': tissue,
            'Mean Jaccard': result['jaccard_similarity']['mean'],
            'Std': result['jaccard_similarity']['std'],
            'Stable Genes (80%)': result['n_stable_genes'],
            'Core Genes (50%)': result['n_core_genes'],
            'Interpretation': result['interpretation'].split(' - ')[0]
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Overall conclusion
    mean_jaccard_overall = np.mean([r['jaccard_similarity']['mean'] for r in all_results.values()])
    mean_stable_ratio = np.mean([r['n_stable_genes'] / r['n_features'] for r in all_results.values()])

    print(f"\n{'='*80}")
    print("OVERALL CONCLUSION")
    print(f"{'='*80}")
    print(f"Average Jaccard across tissues: {mean_jaccard_overall:.3f}")
    print(f"Average stable gene ratio: {mean_stable_ratio:.1%}")

    if mean_jaccard_overall >= 0.5:
        print("\nCONCLUSION: Features are STABLE across random seeds.")
        print("This suggests the model is capturing real biological signal, NOT overfitting.")
        print("Low cross-tissue overlap is likely a BIOLOGICAL phenomenon.")
    elif mean_jaccard_overall >= 0.3:
        print("\nCONCLUSION: Features show MODERATE stability.")
        print("Some features are robust, but there may be some noise.")
    else:
        print("\nCONCLUSION: Features are UNSTABLE.")
        print("This suggests potential overfitting - different runs select different genes.")
        print("Consider using more regularization or stricter feature selection.")

    # Save results
    output_file = output_dir / "feature_stability_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary_file = output_dir / "feature_stability_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Feature Stability Analysis")
    parser.add_argument(
        '--tissues', nargs='+',
        default=['Muscle', 'Liver', 'Brain', 'Blood', 'Lung'],
        help='Tissues to analyze'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[42, 123, 456, 789, 1024],
        help='Random seeds to use'
    )
    parser.add_argument(
        '--n-features', type=int, default=50,
        help='Number of top features to compare'
    )

    args = parser.parse_args()

    run_stability_analysis(
        tissues=args.tissues,
        seeds=args.seeds,
        n_features=args.n_features
    )


if __name__ == "__main__":
    main()
