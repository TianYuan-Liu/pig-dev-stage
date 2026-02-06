#!/usr/bin/env python3
"""
Cross-species stage alignment analysis.

Compares each pig developmental stage with human infant/adult to identify
optimal alignment windows for cross-species experiments.

Outputs:
    - fig4_stage_alignment.csv: Stage alignment statistics
    - figS_stage_alignment.pdf: Supplementary figure showing alignment
"""

import gzip
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures" / "output"
STATS_DIR = OUTPUT_DIR / "stats"
PDF_DIR = OUTPUT_DIR / "pdf"

STATS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Key genes for alignment analysis
KEY_GENES = {
    "ENSSSCG00000037539": "SORCS2",
    "ENSSSCG00000036512": "FGFRL1",
    "ENSSSCG00000030303": "ACHE",
    "ENSSSCG00000009972": "KREMEN1",
    "ENSSSCG00000035805": "DLK1",
    "ENSSSCG00000039557": "TRIM54",
    "ENSSSCG00000004454": "ME1"
}

STAGE_ORDER = [
    "Infant_0_20d",
    "Early childhood_21_59d",
    "Pre_pubertal_60_149d",
    "Post_pubertal_150_365d",
    "Adult_>365d"
]

STAGE_LABELS = ["Infant\n(0-20d)", "Early\n(21-59d)", "Pre-pub\n(60-149d)",
                "Post-pub\n(150-365d)", "Adult\n(>365d)"]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_pig_expression():
    """Load pig muscle expression data."""
    print("Loading pig expression data...")

    meta = pd.read_csv(DATA_DIR / "full_metadata.csv")
    muscle_meta = meta[meta["Tissue"] == "Muscle"]

    expr_path = DATA_DIR / "pigGTEx" / "Muscle.expr_tpm.txt.gz"
    with gzip.open(expr_path, 'rt') as f:
        expr = pd.read_csv(f, sep='\t', index_col=0)

    return expr, muscle_meta


def load_human_expression():
    """Load human muscle expression data."""
    print("Loading human expression data...")

    path = DATA_DIR / "human_muscle" / "GSE257558_read_counts_healty.csv"
    expr = pd.read_csv(path, index_col=0)

    # Simple CPM normalization
    cpm = (expr / expr.sum()) * 1e6

    infant_cols = [c for c in expr.columns if c.startswith("I")]
    adult_cols = [c for c in expr.columns if c.startswith("A")]

    return cpm, infant_cols, adult_cols


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_stage_alignment(pig_expr, human_expr, metadata, key_genes,
                           h_infant_cols, h_adult_cols):
    """
    For each pig developmental stage, calculate:
    1. Mean expression similarity to human infant vs adult
    2. Correlation with human infant/adult profiles
    3. Recommended alignment (which human stage it best matches)
    """
    # Get gene symbols that exist in both species
    pig_gene_ids = list(key_genes.keys())
    human_symbols = [key_genes[g].upper() for g in pig_gene_ids]

    # Find common genes
    common_genes = []
    pig_ids_common = []
    human_symbols_common = []

    for pig_id, symbol in key_genes.items():
        if pig_id in pig_expr.index:
            for s in [symbol, symbol.upper()]:
                if s in human_expr.index:
                    common_genes.append(symbol)
                    pig_ids_common.append(pig_id)
                    human_symbols_common.append(s)
                    break

    print(f"  Found {len(common_genes)} genes in common: {common_genes}")

    if len(common_genes) < 3:
        print("  Warning: Too few common genes for robust analysis")
        return None

    # Calculate human infant and adult mean profiles
    human_infant_mean = human_expr.loc[human_symbols_common, h_infant_cols].mean(axis=1)
    human_adult_mean = human_expr.loc[human_symbols_common, h_adult_cols].mean(axis=1)

    results = []

    for i, stage in enumerate(STAGE_ORDER):
        stage_samples = metadata[metadata["Stage"] == stage]["Sample_ID"]
        valid = [s for s in stage_samples if s in pig_expr.columns]

        if len(valid) < 3:
            print(f"  Skipping {stage}: only {len(valid)} samples")
            continue

        # Get pig stage mean for common genes
        pig_stage_mean = pig_expr.loc[pig_ids_common, valid].mean(axis=1)

        # Reset indices to align for correlation
        pig_values = pig_stage_mean.values
        human_infant_values = human_infant_mean.values
        human_adult_values = human_adult_mean.values

        # Calculate correlations
        r_infant, p_infant = stats.pearsonr(pig_values, human_infant_values)
        r_adult, p_adult = stats.pearsonr(pig_values, human_adult_values)

        # Calculate normalized Euclidean distance
        # Normalize by z-scoring
        pig_z = (pig_values - pig_values.mean()) / (pig_values.std() + 1e-10)
        infant_z = (human_infant_values - human_infant_values.mean()) / (human_infant_values.std() + 1e-10)
        adult_z = (human_adult_values - human_adult_values.mean()) / (human_adult_values.std() + 1e-10)

        dist_infant = np.sqrt(((pig_z - infant_z)**2).sum())
        dist_adult = np.sqrt(((pig_z - adult_z)**2).sum())

        results.append({
            "pig_stage": stage,
            "pig_stage_label": STAGE_LABELS[i],
            "n_samples": len(valid),
            "r_human_infant": r_infant,
            "p_human_infant": p_infant,
            "r_human_adult": r_adult,
            "p_human_adult": p_adult,
            "dist_human_infant": dist_infant,
            "dist_human_adult": dist_adult,
            "best_match": "infant" if r_infant > r_adult else "adult",
            "match_confidence": abs(r_infant - r_adult)
        })

    return pd.DataFrame(results)


def create_alignment_figure(alignment_df, output_path):
    """
    Create supplementary figure showing pig-human stage alignment.

    Panel A: Heatmap of correlations (pig stages vs human infant/adult)
    Panel B: Bar chart showing distance to human infant vs adult
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Use short labels for display
    short_labels = ["Infant", "Early", "Pre-pub", "Post-pub", "Adult"]
    display_labels = alignment_df["pig_stage"].apply(
        lambda x: short_labels[STAGE_ORDER.index(x)] if x in STAGE_ORDER else x
    ).tolist()

    # Panel A: Correlation heatmap
    corr_data = alignment_df[["r_human_infant", "r_human_adult"]].copy()
    corr_data.index = display_labels
    corr_data.columns = ["Human Infant", "Human Adult"]

    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[0],
                cbar_kws={'label': 'Pearson r'})
    axes[0].set_title("A. Correlation with Human Stages", fontweight='bold', fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Pig Developmental Stage")

    # Panel B: Distance comparison bar chart
    x = np.arange(len(display_labels))
    width = 0.35

    bars1 = axes[1].bar(x - width/2, alignment_df["dist_human_infant"], width,
                        label='Human Infant', color='#3498db', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, alignment_df["dist_human_adult"], width,
                        label='Human Adult', color='#e74c3c', alpha=0.8)

    axes[1].set_title("B. Expression Distance to Human Stages", fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Pig Developmental Stage")
    axes[1].set_ylabel("Normalized Euclidean Distance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[1].legend(loc='upper right')

    # Add best match indicators
    for i, row in alignment_df.iterrows():
        best = row["best_match"]
        if best == "infant":
            axes[1].annotate('*', (i - width/2, row["dist_human_infant"] + 0.1),
                           ha='center', fontsize=14, fontweight='bold')
        else:
            axes[1].annotate('*', (i + width/2, row["dist_human_adult"] + 0.1),
                           ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("Cross-Species Stage Alignment Analysis")
    print("=" * 70)

    # Load data
    pig_expr, muscle_meta = load_pig_expression()
    human_expr, h_infant, h_adult = load_human_expression()

    print(f"\nPig samples by stage:")
    for stage in STAGE_ORDER:
        n = len(muscle_meta[muscle_meta["Stage"] == stage])
        print(f"  {stage}: {n}")

    print(f"\nHuman samples: {len(h_infant)} infant, {len(h_adult)} adult")

    # Run alignment analysis
    print("\nAnalyzing stage alignment...")
    alignment_df = analyze_stage_alignment(
        pig_expr, human_expr, muscle_meta, KEY_GENES,
        h_infant, h_adult
    )

    if alignment_df is not None and len(alignment_df) > 0:
        # Save results
        alignment_df.to_csv(STATS_DIR / "fig4_stage_alignment.csv", index=False)
        print(f"\nSaved: {STATS_DIR / 'fig4_stage_alignment.csv'}")

        # Print summary
        print("\nStage Alignment Summary:")
        print("-" * 60)
        for _, row in alignment_df.iterrows():
            print(f"{row['pig_stage']}:")
            print(f"  Correlation with human infant: r={row['r_human_infant']:.3f}")
            print(f"  Correlation with human adult:  r={row['r_human_adult']:.3f}")
            print(f"  Best match: {row['best_match']}")

        # Create figure
        print("\nGenerating alignment figure...")
        create_alignment_figure(alignment_df, PDF_DIR / "figS_stage_alignment.pdf")
    else:
        print("Warning: Could not complete alignment analysis")

    print("\n" + "=" * 70)
    print("Stage alignment analysis completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
