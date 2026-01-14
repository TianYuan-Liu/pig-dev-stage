#!/usr/bin/env python3
"""
Alternative Feature Selection Comparison

This script compares LightGBM-based feature selection with alternative methods:
1. Variance-based selection (highest variance genes)
2. Correlation-based selection (genes most correlated with stage)
3. Mutual information (non-linear relationships)

High overlap between methods suggests the features capture real biological signal.
Low overlap suggests method-specific artifacts or noise.
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector

warnings.filterwarnings('ignore')


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def intersection_size(set1: set, set2: set) -> int:
    """Calculate intersection size between two sets."""
    return len(set1 & set2)


def load_lightgbm_genes(tissue: str, n_genes: int = 50) -> List[str]:
    """Load top genes from existing LightGBM model results."""
    results_dir = PROJECT_ROOT / "machine_learning/model_outputs"
    result_file = results_dir / f"{tissue}_results.json"

    if not result_file.exists():
        raise FileNotFoundError(f"Results file not found: {result_file}")

    with open(result_file) as f:
        results = json.load(f)

    return results.get('top_genes', [])[:n_genes]


def variance_based_selection(
    expr_data: pd.DataFrame,
    n_features: int = 50
) -> List[str]:
    """Select genes with highest variance across samples."""
    variances = expr_data.var(axis=1)
    return variances.nlargest(n_features).index.tolist()


def correlation_based_selection(
    expr_data: pd.DataFrame,
    stage_values: np.ndarray,
    n_features: int = 50
) -> Tuple[List[str], pd.Series]:
    """Select genes with highest absolute Spearman correlation with stage."""
    correlations = {}

    for gene in expr_data.index:
        expr_values = expr_data.loc[gene].values
        rho, _ = stats.spearmanr(expr_values, stage_values)
        correlations[gene] = abs(rho) if not np.isnan(rho) else 0

    corr_series = pd.Series(correlations).sort_values(ascending=False)
    return corr_series.head(n_features).index.tolist(), corr_series


def mutual_info_selection(
    expr_data: pd.DataFrame,
    stage_values: np.ndarray,
    n_features: int = 50
) -> Tuple[List[str], np.ndarray]:
    """Select genes with highest mutual information with stage."""
    # Transpose to samples x genes format
    X = expr_data.T.values

    # Calculate mutual information
    mi_scores = mutual_info_classif(X, stage_values, random_state=42)

    # Create Series for ranking
    mi_series = pd.Series(mi_scores, index=expr_data.index)
    top_genes = mi_series.nlargest(n_features).index.tolist()

    return top_genes, mi_series


def analyze_tissue_methods(
    tissue_name: str,
    n_features: int = 50
) -> Optional[Dict]:
    """
    Compare feature selection methods for a single tissue.
    """
    print(f"\n{'='*60}")
    print(f"Comparing methods for {tissue_name}")
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

    # Map stages to numeric values
    if 'mapping' in scheme:
        labels = scheme.get('labels', scheme.get('stages', []))
        stage_mapping = scheme['mapping']
        y_mapped = [stage_mapping.get(stage, stage) for stage in y]
        label_to_int = {label: i for i, label in enumerate(labels)}
        y_numeric = np.array([label_to_int.get(label, -1) for label in y_mapped])

        # Filter invalid samples
        valid_mask = y_numeric != -1
        X = X.loc[:, valid_mask]
        y_numeric = y_numeric[valid_mask]
    else:
        unique_stages = sorted(pd.Series(y).unique())
        label_to_int = {label: i for i, label in enumerate(unique_stages)}
        y_numeric = np.array([label_to_int[label] for label in y])

    print(f"  Samples: {len(y_numeric)}, Genes: {X.shape[0]}, Scheme: {scheme_name}")

    # Get genes from each method
    try:
        lightgbm_genes = set(load_lightgbm_genes(tissue_name, n_features))
        print(f"  LightGBM genes: {len(lightgbm_genes)}")
    except FileNotFoundError:
        print(f"  Warning: No LightGBM results for {tissue_name}")
        lightgbm_genes = set()

    variance_genes = set(variance_based_selection(X, n_features))
    print(f"  Variance-based genes: {len(variance_genes)}")

    correlation_genes, corr_scores = correlation_based_selection(X, y_numeric, n_features)
    correlation_genes = set(correlation_genes)
    print(f"  Correlation-based genes: {len(correlation_genes)}")

    mi_genes, mi_scores = mutual_info_selection(X, y_numeric, n_features)
    mi_genes = set(mi_genes)
    print(f"  Mutual info genes: {len(mi_genes)}")

    # Calculate pairwise overlaps
    methods = {
        'LightGBM': lightgbm_genes,
        'Variance': variance_genes,
        'Correlation': correlation_genes,
        'MutualInfo': mi_genes
    }

    overlap_matrix = {}
    jaccard_matrix = {}

    for m1 in methods:
        overlap_matrix[m1] = {}
        jaccard_matrix[m1] = {}
        for m2 in methods:
            overlap_matrix[m1][m2] = intersection_size(methods[m1], methods[m2])
            jaccard_matrix[m1][m2] = jaccard_similarity(methods[m1], methods[m2])

    # Calculate overlap with LightGBM
    lgbm_overlaps = {
        'Variance': intersection_size(lightgbm_genes, variance_genes),
        'Correlation': intersection_size(lightgbm_genes, correlation_genes),
        'MutualInfo': intersection_size(lightgbm_genes, mi_genes)
    }

    lgbm_jaccards = {
        'Variance': jaccard_similarity(lightgbm_genes, variance_genes),
        'Correlation': jaccard_similarity(lightgbm_genes, correlation_genes),
        'MutualInfo': jaccard_similarity(lightgbm_genes, mi_genes)
    }

    # Find genes selected by all methods
    common_genes = lightgbm_genes & variance_genes & correlation_genes & mi_genes
    # Genes selected by at least 3 methods
    gene_counts = {}
    for gene_set in methods.values():
        for gene in gene_set:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
    robust_genes = [g for g, c in gene_counts.items() if c >= 3]

    results = {
        'tissue': tissue_name,
        'n_features': n_features,
        'scheme': scheme_name,
        'n_samples': len(y_numeric),
        'methods': {
            'LightGBM': list(lightgbm_genes),
            'Variance': list(variance_genes),
            'Correlation': list(correlation_genes),
            'MutualInfo': list(mi_genes)
        },
        'overlap_with_lightgbm': lgbm_overlaps,
        'jaccard_with_lightgbm': lgbm_jaccards,
        'overlap_matrix': overlap_matrix,
        'jaccard_matrix': jaccard_matrix,
        'common_genes_all_methods': list(common_genes),
        'n_common_all_methods': len(common_genes),
        'robust_genes_3plus': robust_genes,
        'n_robust_genes': len(robust_genes),
        'interpretation': interpret_method_comparison(lgbm_jaccards, len(common_genes), n_features)
    }

    # Print summary
    print(f"\n  Overlap with LightGBM (out of {n_features}):")
    for method, overlap in lgbm_overlaps.items():
        jaccard = lgbm_jaccards[method]
        print(f"    {method}: {overlap} genes ({overlap/n_features*100:.1f}%), Jaccard: {jaccard:.3f}")

    print(f"\n  Common genes across ALL methods: {len(common_genes)}")
    print(f"  Robust genes (>=3 methods): {len(robust_genes)}")
    print(f"\n  Interpretation: {results['interpretation']}")

    return results


def interpret_method_comparison(
    lgbm_jaccards: Dict[str, float],
    n_common: int,
    n_features: int
) -> str:
    """Interpret method comparison results."""
    avg_jaccard = np.mean(list(lgbm_jaccards.values()))
    corr_jaccard = lgbm_jaccards.get('Correlation', 0)

    if avg_jaccard >= 0.4 or corr_jaccard >= 0.5:
        return "STRONG AGREEMENT - LightGBM selects similar genes as other methods. Real signal."
    elif avg_jaccard >= 0.25 or corr_jaccard >= 0.35:
        return "MODERATE AGREEMENT - Reasonable overlap with other methods."
    elif avg_jaccard >= 0.15:
        return "WEAK AGREEMENT - LightGBM selects somewhat different genes."
    else:
        return "POOR AGREEMENT - LightGBM selects very different genes. May indicate method-specific artifacts."


def run_method_comparison(
    tissues: Optional[List[str]] = None,
    n_features: int = 50,
    output_dir: Optional[Path] = None
) -> Dict:
    """Run method comparison across multiple tissues."""
    if tissues is None:
        tissues = ['Muscle', 'Liver', 'Brain', 'Blood', 'Lung']

    if output_dir is None:
        output_dir = PROJECT_ROOT / "machine_learning/analysis/results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ALTERNATIVE FEATURE SELECTION COMPARISON")
    print("="*80)
    print(f"Tissues: {tissues}")
    print(f"Features per tissue: {n_features}")
    print("\nMethods compared:")
    print("  1. LightGBM importance (current method)")
    print("  2. Variance-based (highest variance genes)")
    print("  3. Correlation-based (Spearman with stage)")
    print("  4. Mutual Information (non-linear relationships)")

    all_results = {}

    for tissue in tissues:
        result = analyze_tissue_methods(tissue, n_features)
        if result:
            all_results[tissue] = result

    # Generate summary
    print("\n" + "="*80)
    print("METHOD COMPARISON SUMMARY")
    print("="*80)

    summary_data = []
    for tissue, result in all_results.items():
        lgbm_j = result['jaccard_with_lightgbm']
        summary_data.append({
            'Tissue': tissue,
            'LightGBM-Var': f"{lgbm_j['Variance']:.2f}",
            'LightGBM-Corr': f"{lgbm_j['Correlation']:.2f}",
            'LightGBM-MI': f"{lgbm_j['MutualInfo']:.2f}",
            'Common (all)': result['n_common_all_methods'],
            'Robust (3+)': result['n_robust_genes'],
            'Status': result['interpretation'].split(' - ')[0]
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Overall conclusion
    avg_corr_jaccard = np.mean([r['jaccard_with_lightgbm']['Correlation'] for r in all_results.values()])
    avg_mi_jaccard = np.mean([r['jaccard_with_lightgbm']['MutualInfo'] for r in all_results.values()])
    total_robust = sum(r['n_robust_genes'] for r in all_results.values())

    print(f"\n{'='*80}")
    print("OVERALL CONCLUSION")
    print(f"{'='*80}")
    print(f"Average Jaccard (LightGBM vs Correlation): {avg_corr_jaccard:.3f}")
    print(f"Average Jaccard (LightGBM vs MutualInfo): {avg_mi_jaccard:.3f}")
    print(f"Total robust genes (>=3 methods agree): {total_robust}")

    if avg_corr_jaccard >= 0.35:
        print("\nCONCLUSION: LightGBM selection AGREES well with correlation-based methods.")
        print("This suggests the features capture real age-correlated expression changes.")
        print("Low cross-tissue overlap is likely BIOLOGICAL, not a method artifact.")
    elif avg_corr_jaccard >= 0.20:
        print("\nCONCLUSION: MODERATE agreement between methods.")
        print("LightGBM captures some correlation-based signal plus additional patterns.")
    else:
        print("\nCONCLUSION: WEAK agreement between methods.")
        print("LightGBM may be capturing complex interactions not seen by simpler methods.")
        print("This is not necessarily bad - tree-based methods capture non-linear patterns.")

    # Save results
    output_file = output_dir / "method_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary_file = output_dir / "method_comparison_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Alternative Feature Selection Comparison")
    parser.add_argument(
        '--tissues', nargs='+',
        default=['Muscle', 'Liver', 'Brain', 'Blood', 'Lung'],
        help='Tissues to analyze'
    )
    parser.add_argument(
        '--n-features', type=int, default=50,
        help='Number of top features to compare'
    )

    args = parser.parse_args()

    run_method_comparison(
        tissues=args.tissues,
        n_features=args.n_features
    )


if __name__ == "__main__":
    main()
