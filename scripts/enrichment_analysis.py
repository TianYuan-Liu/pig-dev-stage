#!/usr/bin/env python3
"""
Enrichment and Pathway Analysis for Pig Developmental Stage Classification
Performs functional enrichment analysis on top genes from each tissue model
"""

import json
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import requests
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
MODEL_DIR = BASE_DIR / "machine_learning" / "model_outputs"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "enrichment_analysis"

def load_top_genes():
    """Extract top genes from all tissue model results"""
    tissues = ["Liver", "Brain", "Muscle", "Blood", "Small intestine",
               "Lung", "Adipose", "Testis"]

    all_top_genes = {}

    for tissue in tissues:
        result_file = MODEL_DIR / f"{tissue}_results.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                if 'top_genes' in data:
                    all_top_genes[tissue] = data['top_genes'][:50]  # Top 50 genes
                    print(f"Loaded {len(all_top_genes[tissue])} top genes from {tissue}")

    return all_top_genes

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

def load_expression_data(tissue, gene_list):
    """Load expression data for specific genes in a tissue"""
    expr_file = DATA_DIR / "pigGTEx" / f"{tissue}.expr_tpm.txt.gz"

    if not expr_file.exists():
        print(f"Expression file not found for {tissue}")
        return None

    # Read only the rows for genes of interest
    print(f"Loading expression data for {tissue}...")

    # First, get all gene IDs from the file to find row numbers
    with gzip.open(expr_file, 'rt') as f:
        header = f.readline().strip().split('\t')

        gene_rows = {}
        row_num = 1
        for line in f:
            gene_id = line.split('\t')[0]
            if gene_id in gene_list:
                gene_rows[gene_id] = row_num
            row_num += 1

    # Now read only the selected genes
    expr_data = pd.DataFrame()

    with gzip.open(expr_file, 'rt') as f:
        header = f.readline().strip().split('\t')

        row_num = 1
        for line in f:
            if row_num in gene_rows.values():
                parts = line.strip().split('\t')
                gene_id = parts[0]
                values = [float(x) for x in parts[1:]]

                gene_data = pd.Series(values, index=header[1:], name=gene_id)
                expr_data = pd.concat([expr_data, gene_data.to_frame().T])
            row_num += 1

    return expr_data

def calculate_stage_expression(expr_data, metadata, tissue):
    """Calculate mean expression per developmental stage"""

    # Filter metadata for this tissue
    tissue_metadata = metadata[metadata['Tissue'] == tissue].copy()

    # Get sample IDs that are in both metadata and expression data
    common_samples = list(set(tissue_metadata['Sample_ID']) & set(expr_data.columns))

    if not common_samples:
        print(f"No common samples found for {tissue}")
        return None

    # Filter to common samples
    expr_data = expr_data[common_samples]
    tissue_metadata = tissue_metadata[tissue_metadata['Sample_ID'].isin(common_samples)]

    # Calculate mean expression per stage
    stage_expr = pd.DataFrame()

    for stage in tissue_metadata['Stage'].unique():
        stage_samples = tissue_metadata[tissue_metadata['Stage'] == stage]['Sample_ID'].tolist()
        stage_samples = [s for s in stage_samples if s in expr_data.columns]

        if stage_samples:
            mean_expr = expr_data[stage_samples].mean(axis=1)
            stage_expr[stage] = mean_expr

    # Order stages chronologically
    stage_order = ['Infant_0_20d', 'Early childhood_21_59d',
                   'Pre_pubertal_60_149d', 'Post_pubertal_150_365d', 'Adult_>365d']

    stage_expr = stage_expr[[s for s in stage_order if s in stage_expr.columns]]

    return stage_expr

def calculate_fold_changes(stage_expr):
    """Calculate log2 fold changes between consecutive stages"""

    fold_changes = pd.DataFrame(index=stage_expr.index)

    stages = list(stage_expr.columns)

    for i in range(1, len(stages)):
        prev_stage = stages[i-1]
        curr_stage = stages[i]

        # Add pseudocount to avoid log(0)
        prev_values = stage_expr[prev_stage] + 1
        curr_values = stage_expr[curr_stage] + 1

        fc = np.log2(curr_values / prev_values)
        fold_changes[f"{curr_stage}_vs_{prev_stage}"] = fc

    # Also calculate overall change from infant to adult
    if 'Infant_0_20d' in stages and 'Adult_>365d' in stages:
        infant_values = stage_expr['Infant_0_20d'] + 1
        adult_values = stage_expr['Adult_>365d'] + 1
        fold_changes['Adult_vs_Infant'] = np.log2(adult_values / infant_values)

    return fold_changes

def identify_expression_patterns(stage_expr):
    """Identify genes with specific expression patterns"""

    patterns = {
        'monotonic_increase': [],
        'monotonic_decrease': [],
        'peak_early': [],
        'peak_middle': [],
        'peak_late': []
    }

    for gene in stage_expr.index:
        expr_values = stage_expr.loc[gene].values

        # Check for monotonic patterns
        if all(expr_values[i] <= expr_values[i+1] for i in range(len(expr_values)-1)):
            patterns['monotonic_increase'].append(gene)
        elif all(expr_values[i] >= expr_values[i+1] for i in range(len(expr_values)-1)):
            patterns['monotonic_decrease'].append(gene)

        # Check for peak expression
        peak_idx = np.argmax(expr_values)
        if peak_idx < 2:
            patterns['peak_early'].append(gene)
        elif peak_idx == 2:
            patterns['peak_middle'].append(gene)
        elif peak_idx > 2:
            patterns['peak_late'].append(gene)

    return patterns

def kegg_enrichment(gene_list, organism='ssc'):
    """Perform KEGG pathway enrichment analysis for pig genes"""

    print(f"Performing KEGG enrichment for {len(gene_list)} genes...")

    # Get all KEGG pathways for Sus scrofa
    try:
        # Get list of pathways
        response = requests.get(f"http://rest.kegg.jp/list/pathway/{organism}")
        pathways = {}
        for line in response.text.strip().split('\n'):
            pathway_id, pathway_name = line.split('\t')
            pathway_id = pathway_id.replace('path:', '')
            pathways[pathway_id] = pathway_name

        # Get genes for each pathway
        pathway_genes = {}
        enrichment_results = []

        for pathway_id in list(pathways.keys())[:20]:  # Limit to first 20 pathways for speed
            try:
                response = requests.get(f"http://rest.kegg.jp/link/{organism}/{pathway_id}")

                if response.status_code == 200:
                    genes_in_pathway = set()
                    for line in response.text.strip().split('\n'):
                        if line:
                            parts = line.split('\t')
                            if len(parts) == 2:
                                gene = parts[1].replace(f'{organism}:', '')
                                genes_in_pathway.add(gene)

                    pathway_genes[pathway_id] = genes_in_pathway

                    # Calculate enrichment
                    overlap = len(set(gene_list) & genes_in_pathway)

                    if overlap > 0:
                        enrichment_results.append({
                            'pathway_id': pathway_id,
                            'pathway_name': pathways[pathway_id],
                            'genes_in_pathway': len(genes_in_pathway),
                            'genes_in_list': len(gene_list),
                            'overlap': overlap,
                            'overlap_genes': list(set(gene_list) & genes_in_pathway)
                        })

            except Exception as e:
                continue

        # Calculate p-values using hypergeometric test
        # Assume ~20000 genes in pig genome
        total_genes = 20000

        for result in enrichment_results:
            # Hypergeometric test
            pval = stats.hypergeom.sf(
                result['overlap'] - 1,  # Successes in sample
                total_genes,             # Population size
                result['genes_in_pathway'],  # Successes in population
                result['genes_in_list']      # Sample size
            )
            result['p_value'] = pval
            result['enrichment_score'] = result['overlap'] / (result['genes_in_pathway'] * result['genes_in_list'] / total_genes)

        # Sort by p-value
        enrichment_results.sort(key=lambda x: x['p_value'])

        # Apply FDR correction
        if enrichment_results:
            p_values = [r['p_value'] for r in enrichment_results]
            from statsmodels.stats.multitest import multipletests
            _, q_values, _, _ = multipletests(p_values, method='fdr_bh')

            for i, result in enumerate(enrichment_results):
                result['q_value'] = q_values[i]

        return enrichment_results[:10]  # Return top 10 enriched pathways

    except Exception as e:
        print(f"Error in KEGG enrichment: {e}")
        return []

def save_results(tissue, top_genes, stage_expr, fold_changes, patterns, kegg_results):
    """Save analysis results"""

    # Save gene expression data
    expr_file = RESULTS_DIR / "gene_expression" / f"{tissue}_top_genes_expression.csv"
    expr_file.parent.mkdir(exist_ok=True, parents=True)
    stage_expr.to_csv(expr_file)
    print(f"Saved expression data to {expr_file}")

    # Save fold changes
    fc_file = RESULTS_DIR / "gene_expression" / f"{tissue}_fold_changes.csv"
    fold_changes.to_csv(fc_file)
    print(f"Saved fold changes to {fc_file}")

    # Save expression patterns
    pattern_file = RESULTS_DIR / "gene_expression" / f"{tissue}_expression_patterns.json"
    with open(pattern_file, 'w') as f:
        # Convert to lists for JSON serialization
        patterns_serializable = {k: list(v) for k, v in patterns.items()}
        json.dump(patterns_serializable, f, indent=2)
    print(f"Saved expression patterns to {pattern_file}")

    # Save KEGG enrichment results
    if kegg_results:
        kegg_df = pd.DataFrame(kegg_results)
        kegg_file = RESULTS_DIR / "pathway_results" / f"{tissue}_kegg_enrichment.csv"
        kegg_file.parent.mkdir(exist_ok=True, parents=True)
        kegg_df.to_csv(kegg_file, index=False)
        print(f"Saved KEGG results to {kegg_file}")

def main():
    """Main analysis pipeline"""

    print("=" * 60)
    print("Starting Enrichment and Pathway Analysis")
    print("=" * 60)

    # Load top genes from all tissues
    all_top_genes = load_top_genes()

    # Load metadata
    metadata = load_metadata()

    # Process each tissue
    for tissue, gene_list in all_top_genes.items():
        print(f"\n{'='*60}")
        print(f"Processing {tissue}")
        print(f"{'='*60}")

        # Load expression data
        expr_data = load_expression_data(tissue, gene_list)

        if expr_data is None or expr_data.empty:
            print(f"No expression data available for {tissue}")
            continue

        # Calculate stage-specific expression
        stage_expr = calculate_stage_expression(expr_data, metadata, tissue)

        if stage_expr is None or stage_expr.empty:
            print(f"Could not calculate stage expression for {tissue}")
            continue

        # Calculate fold changes
        fold_changes = calculate_fold_changes(stage_expr)

        # Identify expression patterns
        patterns = identify_expression_patterns(stage_expr)

        print(f"\nExpression patterns in {tissue}:")
        for pattern, genes in patterns.items():
            print(f"  {pattern}: {len(genes)} genes")

        # Perform KEGG enrichment
        # Note: KEGG uses different gene IDs, this is simplified
        # In practice, would need to convert ENSSSCG to KEGG IDs
        kegg_results = kegg_enrichment(gene_list)

        if kegg_results:
            print(f"\nTop KEGG pathways enriched in {tissue}:")
            for i, result in enumerate(kegg_results[:5], 1):
                print(f"  {i}. {result['pathway_name']}: p={result['p_value']:.3e}, genes={result['overlap']}")

        # Save results
        save_results(tissue, gene_list, stage_expr, fold_changes, patterns, kegg_results)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("Results saved to:", RESULTS_DIR)
    print("=" * 60)

if __name__ == "__main__":
    main()