#!/usr/bin/env python3
"""
Create publication-quality figures for enrichment analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 300
})

# Paths
BASE_DIR = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
RESULTS_DIR = BASE_DIR / "results" / "enrichment_analysis"
FIGURES_DIR = RESULTS_DIR / "figures"

def create_expression_heatmap(tissue):
    """Create heatmap of gene expression across developmental stages"""

    # Load stage expression data
    expr_file = RESULTS_DIR / "gene_expression" / f"{tissue}_stage_expression.csv"

    if not expr_file.exists():
        print(f"Expression file not found for {tissue}")
        return

    expr_data = pd.read_csv(expr_file, index_col=0)

    # Normalize by row (z-score)
    expr_norm = expr_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 10))

    # Create heatmap
    sns.heatmap(expr_norm, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Z-score'},
                xticklabels=True, yticklabels=False,
                vmin=-2, vmax=2, ax=ax)

    # Format x-axis labels
    stage_labels = ['Infant\n(0-20d)', 'Early Child\n(21-59d)',
                    'Pre-pubertal\n(60-149d)', 'Post-pubertal\n(150-365d)',
                    'Adult\n(>365d)']
    ax.set_xticklabels(stage_labels, rotation=0, ha='center')

    ax.set_xlabel('Developmental Stage', fontweight='bold')
    ax.set_ylabel('Top 50 Genes', fontweight='bold')
    ax.set_title(f'{tissue} - Gene Expression Dynamics', fontweight='bold', pad=20)

    # Save figure
    output_file = FIGURES_DIR / "expression_heatmaps" / f"{tissue}_expression_heatmap.png"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Created expression heatmap: {output_file.name}")

def create_kegg_bubble_plot(tissue):
    """Create bubble plot for KEGG pathway enrichment"""

    # Load KEGG results
    kegg_file = RESULTS_DIR / "pathway_results" / f"{tissue}_kegg_enrichment.csv"

    if not kegg_file.exists():
        print(f"KEGG results not found for {tissue}")
        return

    kegg_data = pd.read_csv(kegg_file)

    # Select top 10 pathways with significant enrichment
    kegg_top = kegg_data[kegg_data['Q_value'] < 0.05].head(10)

    if kegg_top.empty:
        print(f"  No significant KEGG pathways for {tissue}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data for bubble plot
    x = np.arange(len(kegg_top))
    y = -np.log10(kegg_top['Q_value'].values)
    sizes = kegg_top['Overlap'].values * 30  # Scale for visibility
    colors = kegg_top['Fold_Enrichment'].values

    # Create bubble plot
    scatter = ax.scatter(x, y, s=sizes, c=colors,
                        cmap='YlOrRd', alpha=0.7,
                        edgecolors='black', linewidth=0.5)

    # Add pathway names
    ax.set_xticks(x)
    pathway_names = kegg_top['Pathway_Name'].str[:30].tolist()  # Truncate long names
    ax.set_xticklabels(pathway_names, rotation=45, ha='right')

    ax.set_ylabel('-log10(FDR)', fontweight='bold')
    ax.set_title(f'{tissue} - KEGG Pathway Enrichment', fontweight='bold', pad=20)

    # Add significance threshold line
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--',
              alpha=0.5, label='FDR = 0.05')

    # Add colorbar for fold enrichment
    cbar = plt.colorbar(scatter, ax=ax, label='Fold Enrichment')

    # Add legend for bubble size
    legend_sizes = [5, 10, 15]
    legend_bubbles = []
    for size in legend_sizes:
        legend_bubbles.append(plt.scatter([], [], s=size*30, c='gray',
                                         alpha=0.6, edgecolors='black'))

    ax.legend(legend_bubbles, legend_sizes,
             title='Gene Count', loc='upper left',
             frameon=False, fontsize=8)

    # Save figure
    output_file = FIGURES_DIR / "enrichment_plots" / f"{tissue}_kegg_bubble.png"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Created KEGG bubble plot: {output_file.name}")

def create_go_bubble_plot(tissue):
    """Create bubble plot for GO enrichment"""

    # Load GO results
    go_file = RESULTS_DIR / "pathway_results" / f"{tissue}_go_enrichment.csv"

    if not go_file.exists():
        print(f"GO results not found for {tissue}")
        return

    go_data = pd.read_csv(go_file)

    # Select top 10 terms with significant enrichment
    go_top = go_data[go_data['Q_value'] < 0.05].head(10)

    if go_top.empty:
        print(f"  No significant GO terms for {tissue}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data for bubble plot
    x = np.arange(len(go_top))
    y = -np.log10(go_top['Q_value'].values)
    sizes = go_top['Overlap'].values * 30  # Scale for visibility
    colors = go_top['Fold_Enrichment'].values

    # Create bubble plot
    scatter = ax.scatter(x, y, s=sizes, c=colors,
                        cmap='RdPu', alpha=0.7,
                        edgecolors='black', linewidth=0.5)

    # Add GO terms
    ax.set_xticks(x)
    go_terms = go_top['GO_Term'].str[:25].tolist()  # Truncate long names
    ax.set_xticklabels(go_terms, rotation=45, ha='right')

    ax.set_ylabel('-log10(FDR)', fontweight='bold')
    ax.set_title(f'{tissue} - GO Biological Process Enrichment', fontweight='bold', pad=20)

    # Add significance threshold line
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--',
              alpha=0.5, label='FDR = 0.05')

    # Add colorbar for fold enrichment
    cbar = plt.colorbar(scatter, ax=ax, label='Fold Enrichment')

    # Add legend for bubble size
    legend_sizes = [5, 10, 15]
    legend_bubbles = []
    for size in legend_sizes:
        legend_bubbles.append(plt.scatter([], [], s=size*30, c='gray',
                                         alpha=0.6, edgecolors='black'))

    ax.legend(legend_bubbles, legend_sizes,
             title='Gene Count', loc='upper left',
             frameon=False, fontsize=8)

    # Save figure
    output_file = FIGURES_DIR / "enrichment_plots" / f"{tissue}_go_bubble.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Created GO bubble plot: {output_file.name}")

def create_comparative_enrichment_plot():
    """Create comparative plot showing top pathways across all tissues"""

    tissues = ["Liver", "Brain", "Muscle", "Blood", "Lung", "Adipose", "Testis", "Small_intestine"]

    # Collect top pathways from each tissue
    all_pathways = {}

    for tissue in tissues:
        kegg_file = RESULTS_DIR / "pathway_results" / f"{tissue}_kegg_enrichment.csv"

        if kegg_file.exists():
            kegg_data = pd.read_csv(kegg_file)
            # Get top 3 significant pathways
            top_pathways = kegg_data[kegg_data['Q_value'] < 0.05].head(3)

            for _, row in top_pathways.iterrows():
                pathway_name = row['Pathway_Name']
                if pathway_name not in all_pathways:
                    all_pathways[pathway_name] = {}
                all_pathways[pathway_name][tissue] = -np.log10(row['Q_value'])

    # Create matrix for heatmap
    pathway_matrix = pd.DataFrame(all_pathways).T
    pathway_matrix = pathway_matrix.fillna(0)  # Fill missing values with 0

    # Filter to pathways appearing in at least 2 tissues
    pathway_matrix = pathway_matrix[pathway_matrix.astype(bool).sum(axis=1) >= 2]

    if pathway_matrix.empty:
        print("Not enough shared pathways for comparative plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(pathway_matrix, cmap='YlOrRd', cbar_kws={'label': '-log10(FDR)'},
                xticklabels=True, yticklabels=True,
                linewidths=0.5, linecolor='gray',
                vmin=0, ax=ax)

    ax.set_xlabel('Tissue', fontweight='bold')
    ax.set_ylabel('KEGG Pathway', fontweight='bold')
    ax.set_title('Comparative Pathway Enrichment Across Tissues', fontweight='bold', pad=20)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save figure
    output_file = FIGURES_DIR / "comparative_pathway_enrichment.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Created comparative enrichment plot: {output_file.name}")

def create_fold_change_barplot(tissue):
    """Create barplot showing fold changes between stages"""

    # Load fold change data
    fc_file = RESULTS_DIR / "gene_expression" / f"{tissue}_fold_changes.csv"

    if not fc_file.exists():
        print(f"Fold change file not found for {tissue}")
        return

    fc_data = pd.read_csv(fc_file, index_col=0)

    # Get overall adult vs infant changes
    if 'Adult_vs_Infant' in fc_data.columns:
        adult_vs_infant = fc_data['Adult_vs_Infant']

        # Sort by absolute fold change
        top_genes = adult_vs_infant.abs().nlargest(20)
        top_values = adult_vs_infant[top_genes.index]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create barplot
        colors = ['red' if x < 0 else 'blue' for x in top_values]
        bars = ax.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)

        # Add gene names (truncated)
        gene_labels = [g[:15] + '...' if len(g) > 15 else g for g in top_genes.index]
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(gene_labels, fontsize=8)

        ax.set_xlabel('Log2 Fold Change (Adult vs Infant)', fontweight='bold')
        ax.set_title(f'{tissue} - Top Differentially Expressed Genes', fontweight='bold', pad=20)

        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Add legend
        ax.legend(['Downregulated', 'Upregulated'], loc='lower right')

        # Save figure
        output_file = FIGURES_DIR / "expression_heatmaps" / f"{tissue}_fold_change_barplot.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Created fold change barplot: {output_file.name}")

def main():
    """Generate all figures for enrichment analysis"""

    print("="*60)
    print("CREATING ENRICHMENT ANALYSIS FIGURES")
    print("="*60)

    tissues = ["Liver", "Brain", "Muscle", "Blood", "Lung",
               "Adipose", "Testis", "Small_intestine"]

    for tissue in tissues:
        print(f"\nProcessing {tissue}...")

        # Create expression heatmap
        create_expression_heatmap(tissue)

        # Create KEGG bubble plot
        create_kegg_bubble_plot(tissue)

        # Create GO bubble plot
        create_go_bubble_plot(tissue)

        # Create fold change barplot
        create_fold_change_barplot(tissue)

    # Create comparative plot
    print("\nCreating comparative plots...")
    create_comparative_enrichment_plot()

    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()