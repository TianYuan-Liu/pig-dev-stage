#!/usr/bin/env python3
"""
Biomarker Validation Analysis for Pig Developmental Stages
Validates that key biomarkers show similar developmental trends as in humans
"""

import json
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
MODEL_DIR = BASE_DIR / "machine_learning" / "model_outputs"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "biomarker_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define key biomarkers and their associated tissues
BIOMARKERS = {
    'ALB': {
        'tissue': 'Liver',
        'name': 'Albumin',
        'function': 'Major plasma protein, hepatocyte differentiation marker',
        'expected_trend': 'increase',  # Increases during liver maturation
        'human_ref': 'PMID: various hepatocyte studies'
    },
    'DMRT1': {
        'tissue': 'Testis',
        'name': 'DMRT1',
        'function': 'Male gonadal development transcription factor',
        'expected_trend': 'increase',  # Critical for testis development
        'human_ref': 'PMID: disorders of sex development studies'
    },
    'HBB': {
        'tissue': 'Blood',
        'name': 'Beta-globin',
        'function': 'Adult hemoglobin component',
        'expected_trend': 'increase',  # Switches from fetal to adult around birth
        'human_ref': 'PMID: erythroid development studies'
    },
    'LGR5': {
        'tissue': 'Small intestine',
        'name': 'LGR5',
        'function': 'Intestinal stem cell marker',
        'expected_trend': 'stable_high',  # Maintains stem cell compartment
        'human_ref': 'PMID: intestinal stem cell studies'
    },
    'MBP': {
        'tissue': 'Brain',
        'name': 'Myelin basic protein',
        'function': 'CNS myelination marker',
        'expected_trend': 'increase',  # Rises with postnatal myelination
        'human_ref': 'PMID: oligodendrocyte differentiation studies'
    },
    'MSTN': {
        'tissue': 'Muscle',
        'name': 'Myostatin',
        'function': 'Negative regulator of muscle growth',
        'expected_trend': 'complex',  # Complex regulation during development
        'human_ref': 'PMID: myogenesis control studies'
    },
    'PPARG': {
        'tissue': 'Adipose',
        'name': 'PPARG',
        'function': 'Master regulator of adipogenesis',
        'expected_trend': 'increase',  # Increases with adipocyte differentiation
        'human_ref': 'PMID: adipogenesis studies'
    },
    'SFTPC': {
        'tissue': 'Lung',
        'name': 'Surfactant protein C',
        'function': 'Alveolar type II cell marker',
        'expected_trend': 'increase',  # Increases with lung maturation
        'human_ref': 'PMID: lung development studies'
    }
}

def load_metadata():
    """Load sample metadata with developmental stages"""
    metadata = pd.read_csv(DATA_DIR / "full_metadata.csv")

    # Map stage names to numeric codes for ordering
    stage_order = {
        'Infant_0_20d': 0,
        'Early childhood_21_59d': 1,
        'Pre_pubertal_60_149d': 2,
        'Post_pubertal_150_365d': 3,
        'Adult_>365d': 4
    }

    metadata['stage_numeric'] = metadata['Stage'].map(stage_order)
    return metadata

def find_gene_id(gene_symbol, tissue):
    """Find Ensembl ID for a gene symbol in a specific tissue"""
    expr_file = DATA_DIR / "pigGTEx" / f"{tissue}.expr_tpm.txt.gz"

    if not expr_file.exists():
        print(f"Expression file not found for {tissue}")
        return None

    # Search for gene ID that matches the symbol
    with gzip.open(expr_file, 'rt') as f:
        header = f.readline()
        for line in f:
            gene_id = line.split('\t')[0]
            # Check if gene ID contains the symbol
            if gene_symbol.upper() in gene_id.upper():
                return gene_id

    return None

def extract_biomarker_expression(biomarker, tissue, metadata):
    """Extract expression values for a specific biomarker"""
    expr_file = DATA_DIR / "pigGTEx" / f"{tissue}.expr_tpm.txt.gz"

    if not expr_file.exists():
        print(f"Expression file not found for {tissue}")
        return None

    # Find the gene ID
    gene_id = find_gene_id(biomarker, tissue)
    if not gene_id:
        print(f"Could not find gene ID for {biomarker} in {tissue}")
        return None

    print(f"Found {biomarker} as {gene_id} in {tissue}")

    # Extract expression values
    with gzip.open(expr_file, 'rt') as f:
        header = f.readline().strip().split('\t')
        sample_ids = header[1:]

        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == gene_id:
                expr_values = [float(x) for x in parts[1:]]

                # Create expression dataframe
                expr_df = pd.DataFrame({
                    'Sample_ID': sample_ids,
                    'Expression': expr_values,
                    'Biomarker': biomarker,
                    'Tissue': tissue
                })

                # Merge with metadata
                tissue_metadata = metadata[metadata['Tissue'] == tissue].copy()
                expr_df = expr_df.merge(tissue_metadata[['Sample_ID', 'Stage', 'stage_numeric', 'Age_Days']],
                                       on='Sample_ID', how='inner')

                return expr_df

    return None

def calculate_stage_statistics(expr_df):
    """Calculate stage-wise statistics for biomarker expression"""
    if expr_df is None or expr_df.empty:
        return None

    # Log transform expression
    expr_df['Log_Expression'] = np.log2(expr_df['Expression'] + 1)

    # Calculate stage-wise statistics
    stage_stats = expr_df.groupby(['Stage', 'stage_numeric']).agg({
        'Log_Expression': ['mean', 'std', 'sem', 'count'],
        'Age_Days': 'mean'
    }).reset_index()

    # Flatten column names
    stage_stats.columns = ['Stage', 'stage_numeric', 'mean_expr', 'std_expr', 'sem_expr', 'n_samples', 'mean_age']

    # Sort by stage
    stage_stats = stage_stats.sort_values('stage_numeric')

    # Calculate fold change from infant stage
    infant_expr = stage_stats.iloc[0]['mean_expr']
    stage_stats['fold_change'] = 2 ** (stage_stats['mean_expr'] - infant_expr)

    # Test for trend (Spearman correlation with stage)
    correlation, p_value = stats.spearmanr(expr_df['stage_numeric'], expr_df['Log_Expression'])

    stage_stats['trend_correlation'] = correlation
    stage_stats['trend_pvalue'] = p_value

    # Perform ANOVA to test for differences between stages
    stage_groups = [group['Log_Expression'].values for name, group in expr_df.groupby('stage_numeric')]
    if len(stage_groups) > 1:
        f_stat, anova_p = stats.f_oneway(*stage_groups)
        stage_stats['anova_f'] = f_stat
        stage_stats['anova_pvalue'] = anova_p

    return stage_stats

def validate_developmental_trend(stage_stats, expected_trend):
    """Validate if observed trend matches expected human developmental pattern"""
    if stage_stats is None or stage_stats.empty:
        return False, "No data available"

    correlation = stage_stats['trend_correlation'].iloc[0]
    p_value = stage_stats['trend_pvalue'].iloc[0]

    # Check different expected trends
    if expected_trend == 'increase':
        # Should show positive correlation with stage
        is_valid = correlation > 0.3 and p_value < 0.05
        description = f"Positive trend (r={correlation:.3f}, p={p_value:.3e})"

    elif expected_trend == 'decrease':
        # Should show negative correlation with stage
        is_valid = correlation < -0.3 and p_value < 0.05
        description = f"Negative trend (r={correlation:.3f}, p={p_value:.3e})"

    elif expected_trend == 'stable_high':
        # Should maintain high expression with minimal change
        fold_changes = stage_stats['fold_change'].values
        is_valid = all(0.5 < fc < 2.0 for fc in fold_changes)
        description = f"Stable expression (fold change range: {min(fold_changes):.2f}-{max(fold_changes):.2f})"

    elif expected_trend == 'complex':
        # Complex patterns - just check for significant differences
        if 'anova_pvalue' in stage_stats.columns:
            is_valid = stage_stats['anova_pvalue'].iloc[0] < 0.05
            description = f"Stage-dependent expression (ANOVA p={stage_stats['anova_pvalue'].iloc[0]:.3e})"
        else:
            is_valid = True
            description = "Complex developmental pattern"
    else:
        is_valid = False
        description = "Unknown expected trend"

    return is_valid, description

def main():
    """Main analysis pipeline"""
    print("Loading metadata...")
    metadata = load_metadata()

    # Store all results
    all_results = []
    validation_summary = []

    print("\nAnalyzing biomarkers...")
    print("=" * 60)

    for biomarker, info in BIOMARKERS.items():
        tissue = info['tissue']
        print(f"\n{biomarker} ({info['name']}) in {tissue}")
        print("-" * 40)

        # Extract expression data
        expr_df = extract_biomarker_expression(biomarker, tissue, metadata)

        if expr_df is not None:
            # Calculate statistics
            stage_stats = calculate_stage_statistics(expr_df)

            if stage_stats is not None:
                # Validate trend
                is_valid, trend_description = validate_developmental_trend(stage_stats, info['expected_trend'])

                # Add biomarker info
                stage_stats['Biomarker'] = biomarker
                stage_stats['Tissue'] = tissue
                stage_stats['Gene_Name'] = info['name']
                stage_stats['Function'] = info['function']
                stage_stats['Expected_Trend'] = info['expected_trend']
                stage_stats['Trend_Valid'] = is_valid
                stage_stats['Trend_Description'] = trend_description

                # Store results
                all_results.append(stage_stats)

                # Create validation summary
                validation_summary.append({
                    'Biomarker': biomarker,
                    'Gene_Name': info['name'],
                    'Tissue': tissue,
                    'Function': info['function'],
                    'Expected_Trend': info['expected_trend'],
                    'Observed_Trend': trend_description,
                    'Validation_Pass': is_valid,
                    'Correlation': stage_stats['trend_correlation'].iloc[0],
                    'P_Value': stage_stats['trend_pvalue'].iloc[0],
                    'Max_Fold_Change': stage_stats['fold_change'].max(),
                    'Human_Reference': info['human_ref']
                })

                # Print summary
                print(f"Function: {info['function']}")
                print(f"Expected trend: {info['expected_trend']}")
                print(f"Observed: {trend_description}")
                print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'}")

                # Print stage-wise expression
                print("\nStage-wise expression (mean ± SEM):")
                for _, row in stage_stats.iterrows():
                    print(f"  {row['Stage']}: {row['mean_expr']:.2f} ± {row['sem_expr']:.2f} "
                          f"(n={int(row['n_samples'])}, FC={row['fold_change']:.2f})")

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(RESULTS_DIR / "biomarker_stage_statistics.csv", index=False)
        print(f"\nSaved stage statistics to {RESULTS_DIR / 'biomarker_stage_statistics.csv'}")

    # Save validation summary
    if validation_summary:
        validation_df = pd.DataFrame(validation_summary)
        validation_df.to_csv(RESULTS_DIR / "biomarker_validation_summary.csv", index=False)
        print(f"Saved validation summary to {RESULTS_DIR / 'biomarker_validation_summary.csv'}")

        # Print overall validation rate
        n_valid = validation_df['Validation_Pass'].sum()
        n_total = len(validation_df)
        print(f"\nOverall validation: {n_valid}/{n_total} biomarkers show expected developmental trends")
        print(f"Validation rate: {100 * n_valid / n_total:.1f}%")

    # Save expression data for visualization
    print("\nExtracting full expression data for visualization...")
    all_expression = []

    for biomarker, info in BIOMARKERS.items():
        tissue = info['tissue']
        expr_df = extract_biomarker_expression(biomarker, tissue, metadata)
        if expr_df is not None:
            expr_df['Gene_Name'] = info['name']
            all_expression.append(expr_df)

    if all_expression:
        combined_expression = pd.concat(all_expression, ignore_index=True)
        combined_expression.to_csv(RESULTS_DIR / "biomarker_expression_data.csv", index=False)
        print(f"Saved expression data to {RESULTS_DIR / 'biomarker_expression_data.csv'}")

    print("\n" + "=" * 60)
    print("Biomarker validation analysis complete!")
    print("Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()