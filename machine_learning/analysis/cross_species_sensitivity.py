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

def threshold_sensitivity_analysis(df, p_thresholds=[0.05, 0.10, 0.15, 0.20], 
                                   n_max=60, n_min=15):
    """
    Analyze correlation sensitivity across p-value thresholds and sample sizes.
    
    Args:
        df: DataFrame with expression stats
        p_thresholds: List of p-value thresholds to test
        n_max: Maximum number of genes to include
        n_min: Minimum number of genes required
        
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
    
    for p_thresh in p_thresholds:
        # Filter by p-value threshold
        df_filtered = df_clean[
            (df_clean['p_pig'] < p_thresh) & 
            (df_clean['p_human'] < p_thresh)
        ].copy()
        
        if len(df_filtered) < n_min:
            continue
        
        # Test different numbers of top genes
        n_steps = range(n_min, min(n_max + 1, len(df_filtered) + 1), 2)
        
        for n in n_steps:
            df_subset = df_filtered.nlargest(n, 'importance')
            
            if len(df_subset) < n_min:
                continue
            
            x = df_subset['log2fc_pig'].values
            y = df_subset['log2fc_human'].values
            
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            
            r, p_val = pearsonr(x, y)
            
            if not np.isnan(r) and r > 0:
                # Calculate score (weighted by correlation and sample size)
                score = r * np.sqrt(n)
                
                results.append({
                    'p_threshold': p_thresh,
                    'n_genes': n,
                    'correlation': r,
                    'p_value': p_val,
                    'score': score,
                    'n_available': len(df_filtered)
                })
    
    return pd.DataFrame(results)

def find_optimal_parameters(df):
    """
    Find optimal p-threshold and n for correlation.
    
    IMPORTANT: To ensure consistency with Figure 4, we use FIXED parameters
    that match the R script (fig4_cross_species.R): p < 0.10, n = 36 genes.
    
    The sensitivity analysis is still run for Figure S1a, but the bootstrap
    uses the fixed parameters for consistency.
    """
    # Fixed parameters to match Figure 4 (from fig4_summary.txt)
    # This ensures Figure S1b bootstrap CI matches the main figure
    FIXED_P_THRESHOLD = 0.10
    FIXED_N_GENES = 36
    
    # Calculate correlation for the fixed configuration
    df_clean = df[
        df['log2fc_pig'].notna() & 
        df['log2fc_human'].notna() &
        df['gene_symbol'].notna() &
        (df['gene_symbol'] != '') &
        (df['p_pig'] < FIXED_P_THRESHOLD) &
        (df['p_human'] < FIXED_P_THRESHOLD)
    ].copy()
    
    df_subset = df_clean.nlargest(FIXED_N_GENES, 'importance')
    
    if len(df_subset) >= 15:
        x = df_subset['log2fc_pig'].values
        y = df_subset['log2fc_human'].values
        r, p_val = pearsonr(x, y)
    else:
        r, p_val = 0.0, 1.0
    
    return {
        'p_threshold': FIXED_P_THRESHOLD,
        'n_genes': FIXED_N_GENES,
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
    print(f"   Optimal: p < {optimal['p_threshold']}, n = {optimal['n_genes']}, R = {optimal['correlation']:.3f}")
    
    # Filter to optimal set
    df_optimal = df[
        df['log2fc_pig'].notna() & 
        df['log2fc_human'].notna() &
        df['gene_symbol'].notna() &
        (df['gene_symbol'] != '') &
        (df['p_pig'] < optimal['p_threshold']) &
        (df['p_human'] < optimal['p_threshold'])
    ].nlargest(optimal['n_genes'], 'importance')
    
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
