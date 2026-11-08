#!/usr/bin/env python3
"""
Cross-species sensitivity analysis for Figure S1.

Performs bootstrap analysis and threshold sensitivity testing
for the pig-human muscle developmental correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_expression_stats():
    """Load the cross-species expression statistics."""
    stats_file = PROJECT_ROOT / "paper/figures/output/stats/fig4_expression_stats.csv"
    if not stats_file.exists():
        raise FileNotFoundError(f"Expression stats not found: {stats_file}")
    
    df = pd.read_csv(stats_file)
    return df

def bootstrap_correlation(x, y, n_bootstrap=1000, random_state=42):
    """
    Bootstrap correlation coefficient.
    
    Args:
        x: Array of x values
        y: Array of y values
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed
        
    Returns:
        Dictionary with correlation, CI, and bootstrap distribution
    """
    np.random.seed(random_state)
    n = len(x)
    bootstrap_corrs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        if np.std(x_boot) > 0 and np.std(y_boot) > 0:
            r, _ = pearsonr(x_boot, y_boot)
            if not np.isnan(r):
                bootstrap_corrs.append(r)
    
    bootstrap_corrs = np.array(bootstrap_corrs)
    
    return {
        'correlation': np.mean(bootstrap_corrs),
        'ci_lower': np.percentile(bootstrap_corrs, 2.5),
        'ci_upper': np.percentile(bootstrap_corrs, 97.5),
        'std': np.std(bootstrap_corrs),
        'bootstrap_distribution': bootstrap_corrs.tolist()
    }

def threshold_sensitivity_analysis(df, fdr_thresholds=[0.05, 0.10, 0.15, 0.20],
                                   fc_threshold=0.5):
    """
    Analyze correlation sensitivity across FDR thresholds.

    Uses ALL genes passing the pre-specified criteria at each threshold
    (no top-N selection). The |log2FC| > fc_threshold filter is applied
    alongside the FDR filter.

    Args:
        df: DataFrame with expression stats (must include fdr_pig, fdr_human)
        fdr_thresholds: List of FDR thresholds to test
        fc_threshold: Minimum absolute log2FC in both species

    Returns:
        DataFrame with sensitivity results
    """
    results = []

    # Filter to genes with both pig and human data
    df_clean = df[
        df['log2fc_pig'].notna() &
        df['log2fc_human'].notna() &
        df['gene_symbol'].notna() &
        (df['gene_symbol'] != '')
    ].copy()

    for fdr_thresh in fdr_thresholds:
        # Filter by FDR and fold-change thresholds
        df_filtered = df_clean[
            (df_clean['fdr_pig'] < fdr_thresh) &
            (df_clean['fdr_human'] < fdr_thresh) &
            (df_clean['log2fc_pig'].abs() > fc_threshold) &
            (df_clean['log2fc_human'].abs() > fc_threshold)
        ].copy()

        n = len(df_filtered)
        if n < 5:
            continue

        x = df_filtered['log2fc_pig'].values
        y = df_filtered['log2fc_human'].values

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        r, p_val = pearsonr(x, y)

        if not np.isnan(r):
            results.append({
                'fdr_threshold': fdr_thresh,
                'n_genes': n,
                'correlation': r,
                'p_value': p_val,
                'n_available': n
            })

    return pd.DataFrame(results)

def find_optimal_parameters(df):
    """
    Apply pre-specified criteria to select genes for cross-species analysis.

    Criteria: FDR < 0.10 (BH-corrected) AND |log2FC| > 0.5 in both species.
    Uses ALL passing genes (no top-N selection).
    """
    FDR_THRESHOLD = 0.10
    FC_THRESHOLD = 0.5

    df_clean = df[
        df['log2fc_pig'].notna() &
        df['log2fc_human'].notna() &
        df['gene_symbol'].notna() &
        (df['gene_symbol'] != '') &
        (df['fdr_pig'] < FDR_THRESHOLD) &
        (df['fdr_human'] < FDR_THRESHOLD) &
        (df['log2fc_pig'].abs() > FC_THRESHOLD) &
        (df['log2fc_human'].abs() > FC_THRESHOLD)
    ].copy()

    n_genes = len(df_clean)

    if n_genes >= 5:
        x = df_clean['log2fc_pig'].values
        y = df_clean['log2fc_human'].values
        r, p_val = pearsonr(x, y)
    else:
        r, p_val = 0.0, 1.0

    return {
        'fdr_threshold': FDR_THRESHOLD,
        'fc_threshold': FC_THRESHOLD,
        'n_genes': n_genes,
        'correlation': r,
        'p_value': p_val
    }

def main():
    """Run sensitivity analysis and save results."""
    print("=" * 70)
    print("Cross-Species Sensitivity Analysis")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading expression statistics...")
    df = load_expression_stats()
    print(f"   Loaded {len(df)} genes")
    
    # Find optimal parameters
    print("\n2. Finding optimal parameters...")
    optimal = find_optimal_parameters(df)
    print(f"   Optimal: FDR < {optimal['fdr_threshold']}, n = {optimal['n_genes']}, R = {optimal['correlation']:.3f}")

    # Filter to genes passing pre-specified criteria
    df_optimal = df[
        df['log2fc_pig'].notna() &
        df['log2fc_human'].notna() &
        df['gene_symbol'].notna() &
        (df['gene_symbol'] != '') &
        (df['fdr_pig'] < optimal['fdr_threshold']) &
        (df['fdr_human'] < optimal['fdr_threshold']) &
        (df['log2fc_pig'].abs() > optimal['fc_threshold']) &
        (df['log2fc_human'].abs() > optimal['fc_threshold'])
    ]
    
    print(f"   Selected {len(df_optimal)} genes for bootstrap analysis")
    
    # Bootstrap correlation
    print("\n3. Running bootstrap analysis (1000 iterations)...")
    x = df_optimal['log2fc_pig'].values
    y = df_optimal['log2fc_human'].values
    bootstrap_results = bootstrap_correlation(x, y, n_bootstrap=1000)
    
    print(f"   Bootstrap correlation: {bootstrap_results['correlation']:.3f}")
    print(f"   95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]")
    
    # Threshold sensitivity
    print("\n4. Running threshold sensitivity analysis...")
    sensitivity = threshold_sensitivity_analysis(df)
    print(f"   Tested {len(sensitivity)} parameter combinations")
    
    # Save results
    output_dir = PROJECT_ROOT / "paper/figures/output/stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'optimal_parameters': optimal,
        'bootstrap_results': bootstrap_results,
        'sensitivity_analysis': sensitivity.to_dict('records') if len(sensitivity) > 0 else []
    }
    
    output_file = output_dir / "cross_species_sensitivity.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save sensitivity as CSV for R
    if len(sensitivity) > 0:
        sensitivity_file = output_dir / "cross_species_sensitivity.csv"
        sensitivity.to_csv(sensitivity_file, index=False)
        print(f"   Saved sensitivity data to {sensitivity_file}")
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
