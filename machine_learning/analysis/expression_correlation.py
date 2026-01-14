#!/usr/bin/env python3
"""
Expression-Stage Correlation Analysis

This script validates that top genes selected by LightGBM show meaningful
correlation with developmental stages. If genes are capturing real biological
signal, they should show clear expression trends across stages.

Key metrics:
- Spearman correlation between gene expression and developmental stage
- Percentage of genes with significant correlation (|r| > 0.3)
- Expression pattern monotonicity
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
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.data_processing.preprocessing import ExpressionPreprocessor
from machine_learning.data_processing.stage_selection import StageGranularitySelector

warnings.filterwarnings('ignore')


def load_top_genes(tissue: str, results_dir: Optional[Path] = None) -> List[str]:
    """Load top genes from existing model results."""
    if results_dir is None:
        results_dir = PROJECT_ROOT / "machine_learning/model_outputs"

    result_file = results_dir / f"{tissue}_results.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Results file not found: {result_file}")

    with open(result_file) as f:
        results = json.load(f)

    return results.get('top_genes', [])


def calculate_gene_correlations(
    expr_data: pd.DataFrame,
    stage_values: np.ndarray,
    genes: List[str]
) -> pd.DataFrame:
    """
    Calculate Spearman correlation between gene expression and stage.

    Args:
        expr_data: Gene expression DataFrame (genes x samples)
        stage_values: Numeric stage labels
        genes: List of genes to analyze

    Returns:
        DataFrame with correlation statistics for each gene
    """
    results = []

    for gene in genes:
        if gene not in expr_data.index:
            continue

        expr_values = expr_data.loc[gene].values

        # Calculate Spearman correlation
        rho, pval = stats.spearmanr(expr_values, stage_values)

        # Calculate mean expression per stage
        stage_means = {}
        unique_stages = np.unique(stage_values)
        for stage in unique_stages:
            mask = stage_values == stage
            stage_means[int(stage)] = float(np.mean(expr_values[mask]))

        # Check monotonicity
        means_sorted = [stage_means[s] for s in sorted(stage_means.keys())]
        is_increasing = all(means_sorted[i] <= means_sorted[i+1] for i in range(len(means_sorted)-1))
        is_decreasing = all(means_sorted[i] >= means_sorted[i+1] for i in range(len(means_sorted)-1))
        is_monotonic = is_increasing or is_decreasing

        results.append({
            'gene': gene,
            'spearman_rho': rho,
            'p_value': pval,
            'abs_rho': abs(rho),
            'significant': pval < 0.05 and abs(rho) >= 0.3,
            'is_monotonic': is_monotonic,
            'direction': 'increasing' if rho > 0 else 'decreasing',
            'stage_means': stage_means
        })

    return pd.DataFrame(results)


def analyze_tissue_correlations(
    tissue_name: str,
    n_genes: int = 50
) -> Optional[Dict]:
    """
    Analyze expression-stage correlations for a single tissue.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing correlations for {tissue_name}")
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

    # Stage selection (to get numeric labels)
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
        # If no mapping, use original labels as numeric
        unique_stages = sorted(pd.Series(y).unique())
        label_to_int = {label: i for i, label in enumerate(unique_stages)}
        y_numeric = np.array([label_to_int[label] for label in y])

    print(f"  Samples: {len(y_numeric)}, Scheme: {scheme_name}")
    print(f"  Stage distribution: {pd.Series(y_numeric).value_counts().sort_index().to_dict()}")

    # Load top genes from model results
    try:
        top_genes = load_top_genes(tissue_name)[:n_genes]
        print(f"  Loaded {len(top_genes)} top genes from model results")
    except FileNotFoundError:
        print(f"  Warning: No model results found for {tissue_name}, skipping")
        return None

    # Calculate correlations
    corr_df = calculate_gene_correlations(X, y_numeric, top_genes)

    if len(corr_df) == 0:
        print(f"  No genes found in expression data")
        return None

    # Summary statistics
    n_significant = (corr_df['significant']).sum()
    n_monotonic = (corr_df['is_monotonic']).sum()
    mean_abs_rho = corr_df['abs_rho'].mean()

    # Genes with strong correlation (|r| > 0.5)
    strong_corr = corr_df[corr_df['abs_rho'] >= 0.5]

    results = {
        'tissue': tissue_name,
        'scheme': scheme_name,
        'n_samples': len(y_numeric),
        'n_genes_analyzed': len(corr_df),
        'n_significant': int(n_significant),
        'pct_significant': float(n_significant / len(corr_df) * 100),
        'n_monotonic': int(n_monotonic),
        'pct_monotonic': float(n_monotonic / len(corr_df) * 100),
        'mean_abs_correlation': float(mean_abs_rho),
        'n_strong_correlation': len(strong_corr),
        'gene_correlations': corr_df.to_dict('records'),
        'stage_labels': {int(v): k for k, v in label_to_int.items()},
        'interpretation': interpret_correlation_results(
            n_significant / len(corr_df),
            n_monotonic / len(corr_df),
            mean_abs_rho
        )
    }

    # Print summary
    print(f"\n  Results for {tissue_name}:")
    print(f"    Significant correlations (|r|>0.3, p<0.05): {n_significant}/{len(corr_df)} ({results['pct_significant']:.1f}%)")
    print(f"    Monotonic genes: {n_monotonic}/{len(corr_df)} ({results['pct_monotonic']:.1f}%)")
    print(f"    Mean |correlation|: {mean_abs_rho:.3f}")
    print(f"    Strong correlations (|r|>0.5): {len(strong_corr)}")
    print(f"    Interpretation: {results['interpretation']}")

    # Show top correlating genes
    top_corr = corr_df.nlargest(5, 'abs_rho')
    print(f"\n    Top 5 correlating genes:")
    for _, row in top_corr.iterrows():
        print(f"      {row['gene']}: r={row['spearman_rho']:.3f} (p={row['p_value']:.2e})")

    return results


def interpret_correlation_results(
    pct_significant: float,
    pct_monotonic: float,
    mean_abs_rho: float
) -> str:
    """Interpret correlation analysis results."""
    if pct_significant >= 0.8 and mean_abs_rho >= 0.4:
        return "STRONG SIGNAL - Top genes show clear expression-stage relationships. NOT overfitting."
    elif pct_significant >= 0.6 and mean_abs_rho >= 0.3:
        return "MODERATE SIGNAL - Most genes show meaningful correlations with stage."
    elif pct_significant >= 0.4:
        return "WEAK SIGNAL - Some genes correlate with stage, but many do not."
    else:
        return "POOR SIGNAL - Most genes show no correlation with stage. POTENTIAL OVERFITTING."


def run_correlation_analysis(
    tissues: Optional[List[str]] = None,
    n_genes: int = 50,
    output_dir: Optional[Path] = None
) -> Dict:
    """Run full correlation analysis across multiple tissues."""
    if tissues is None:
        tissues = ['Muscle', 'Liver', 'Brain', 'Blood', 'Lung']

    if output_dir is None:
        output_dir = PROJECT_ROOT / "machine_learning/analysis/results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPRESSION-STAGE CORRELATION ANALYSIS")
    print("="*80)
    print(f"Tissues: {tissues}")
    print(f"Genes per tissue: {n_genes}")

    all_results = {}

    for tissue in tissues:
        result = analyze_tissue_correlations(tissue, n_genes)
        if result:
            all_results[tissue] = result

    # Generate summary
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)

    summary_data = []
    for tissue, result in all_results.items():
        summary_data.append({
            'Tissue': tissue,
            'Pct Significant': f"{result['pct_significant']:.1f}%",
            'Pct Monotonic': f"{result['pct_monotonic']:.1f}%",
            'Mean |r|': f"{result['mean_abs_correlation']:.3f}",
            'Strong (>0.5)': result['n_strong_correlation'],
            'Interpretation': result['interpretation'].split(' - ')[0]
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Overall conclusion
    mean_pct_significant = np.mean([r['pct_significant'] for r in all_results.values()])
    mean_abs_corr = np.mean([r['mean_abs_correlation'] for r in all_results.values()])

    print(f"\n{'='*80}")
    print("OVERALL CONCLUSION")
    print(f"{'='*80}")
    print(f"Average % significant correlations: {mean_pct_significant:.1f}%")
    print(f"Average |correlation|: {mean_abs_corr:.3f}")

    if mean_pct_significant >= 70:
        print("\nCONCLUSION: Top genes show STRONG correlation with developmental stage.")
        print("This confirms the model is capturing real age-related gene expression changes.")
        print("Low cross-tissue overlap is likely a BIOLOGICAL phenomenon, NOT overfitting.")
    elif mean_pct_significant >= 50:
        print("\nCONCLUSION: Top genes show MODERATE correlation with developmental stage.")
        print("Most genes are related to development, but some may be noise.")
    else:
        print("\nCONCLUSION: Top genes show WEAK correlation with developmental stage.")
        print("This raises concerns about potential overfitting.")

    # Save results
    output_file = output_dir / "expression_correlation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary_file = output_dir / "expression_correlation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Expression-Stage Correlation Analysis")
    parser.add_argument(
        '--tissues', nargs='+',
        default=['Muscle', 'Liver', 'Brain', 'Blood', 'Lung'],
        help='Tissues to analyze'
    )
    parser.add_argument(
        '--n-genes', type=int, default=50,
        help='Number of top genes to analyze'
    )

    args = parser.parse_args()

    run_correlation_analysis(
        tissues=args.tissues,
        n_genes=args.n_genes
    )


if __name__ == "__main__":
    main()
