#!/usr/bin/env python3
"""
Within-Project Fold-Change Validation
======================================
Tests whether developmental signals are real biology or batch artifacts by
computing fold-changes within single-project cohorts (same lab/protocol/sequencer)
and correlating them with full-dataset estimates.

Two large BioProjects each span 4 developmental stages within uniform protocols:
  - PRJNA488311 (n=81): Infant -> Early childhood -> Pre-pubertal -> Post-pubertal
  - PRJNA486202 (n=75): Infant -> Early childhood -> Pre-pubertal -> Post-pubertal

Since these projects reach Post-pubertal but not Adult, comparisons use Infant
as baseline against each later stage (Early childhood, Pre-pubertal, Post-pubertal).
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "machine_learning" / "analysis" / "results" / "within_project"
EXPRESSION_STATS = PROJECT_ROOT / "paper" / "figures" / "output" / "stats" / "fig4_expression_stats.csv"

VALIDATION_PROJECTS = ["PRJNA488311", "PRJNA486202"]

# All stage comparisons use Infant as baseline
STAGE_COMPARISONS = [
    ("Infant", "Early childhood"),
    ("Infant", "Pre-pubertal"),
    ("Infant", "Post-pubertal"),
]


def load_muscle_data():
    """Load muscle expression matrix and metadata with stage/BioProject info."""
    from machine_learning.data_processing.data_loader import DataLoader

    loader = DataLoader(DATA_DIR / "pigGTEx", DATA_DIR / "PigGTEx_v0.MetaTable.csv")
    loader.load_metadata()
    meta = loader.metadata

    muscle_meta = meta[
        (meta["Tissue"] == "Muscle")
        & (meta["Tissue_Main"] == "Muscle")
        & (meta["Sub_categories"] == "Muscle")
    ].copy()

    expr_path = DATA_DIR / "pigGTEx" / "Muscle.expr_tpm.txt.gz"
    with gzip.open(expr_path, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)

    common = [s for s in expr.columns if s in muscle_meta.index]
    expr = expr[common]
    muscle_meta = muscle_meta.loc[common]

    print(f"Loaded {len(common)} muscle samples, {expr.shape[0]} genes")
    return expr, muscle_meta


def compute_fold_changes(expr, sample_ids_young, sample_ids_old):
    """Compute log2FC and Mann-Whitney p-value for each gene (Old / Young).

    Returns DataFrame with columns: log2fc, pvalue
    """
    y = [s for s in sample_ids_young if s in expr.columns]
    o = [s for s in sample_ids_old if s in expr.columns]

    if len(y) < 2 or len(o) < 2:
        return pd.DataFrame(columns=["log2fc", "pvalue"])

    results = []
    for gene in expr.index:
        v_young = expr.loc[gene, y].astype(float)
        v_old = expr.loc[gene, o].astype(float)
        fc = np.log2((v_old.mean() + 0.01) / (v_young.mean() + 0.01))
        try:
            p = stats.mannwhitneyu(v_young, v_old)[1]
        except ValueError:
            p = 1.0
        results.append({"gene": gene, "log2fc": fc, "pvalue": p})

    df = pd.DataFrame(results).set_index("gene")
    _, fdr, _, _ = multipletests(df["pvalue"], method="fdr_bh")
    df["fdr"] = fdr
    return df


def load_cross_species_genes():
    """Load the 66 cross-species validated genes from fig4 expression stats."""
    df = pd.read_csv(EXPRESSION_STATS)
    sig = df[
        (df["fdr_pig"] < 0.10)
        & (df["fdr_human"] < 0.10)
        & (df["log2fc_pig"].abs() > 0.5)
        & (df["log2fc_human"].abs() > 0.5)
    ]
    return sig[["pig_gene_id", "gene_symbol"]].copy()


def run_within_project_validation():
    """Main analysis: within-project FC vs full-dataset FC correlation across stage comparisons."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expr, meta = load_muscle_data()

    # Cross-species gene list
    cs_genes = load_cross_species_genes()
    cs_ids = cs_genes["pig_gene_id"].tolist()
    print(f"Cross-species genes: {len(cs_genes)}")

    results = {"comparisons": {}, "full_dataset_stages": {}}

    # Record per-stage sample counts for full dataset
    for stage in ["Infant", "Early childhood", "Pre-pubertal", "Post-pubertal"]:
        n = len(meta[meta["Stage"] == stage])
        results["full_dataset_stages"][stage] = n
        print(f"Full dataset — {stage}: {n}")

    for stage_young, stage_old in STAGE_COMPARISONS:
        comparison_key = f"{stage_young} vs {stage_old}"
        print(f"\n{'='*60}")
        print(f"Comparison: {comparison_key}")
        print(f"{'='*60}")

        # Full-dataset FC for this comparison
        young_all = meta[meta["Stage"] == stage_young].index.tolist()
        old_all = meta[meta["Stage"] == stage_old].index.tolist()
        print(f"Full dataset: {len(young_all)} {stage_young}, {len(old_all)} {stage_old}")

        fc_full = compute_fold_changes(expr, young_all, old_all)
        fc_full = fc_full.rename(columns={"log2fc": "fc_full", "pvalue": "p_full", "fdr": "fdr_full"})

        project_fcs = {}
        project_results = {}

        for proj in VALIDATION_PROJECTS:
            proj_mask = meta["BioProject"] == proj
            proj_meta = meta[proj_mask]

            young_proj = proj_meta[proj_meta["Stage"] == stage_young].index.tolist()
            old_proj = proj_meta[proj_meta["Stage"] == stage_old].index.tolist()

            print(f"\n  {proj}: {len(young_proj)} {stage_young}, {len(old_proj)} {stage_old}")

            fc_proj = compute_fold_changes(expr, young_proj, old_proj)
            fc_proj = fc_proj.rename(
                columns={"log2fc": f"fc_{proj}", "pvalue": f"p_{proj}", "fdr": f"fdr_{proj}"}
            )
            project_fcs[proj] = fc_proj

            # Correlate with full dataset
            merged = fc_full[["fc_full"]].join(fc_proj[[f"fc_{proj}"]], how="inner").dropna()
            r_all, p_all = stats.pearsonr(merged["fc_full"], merged[f"fc_{proj}"])
            n_all = len(merged)

            # Cross-species gene subset
            merged_cs = merged.loc[merged.index.isin(cs_ids)]
            if len(merged_cs) >= 3:
                r_cs, p_cs = stats.pearsonr(merged_cs["fc_full"], merged_cs[f"fc_{proj}"])
                n_cs = len(merged_cs)
            else:
                r_cs, p_cs, n_cs = np.nan, np.nan, len(merged_cs)

            project_results[proj] = {
                "n_young": len(young_proj),
                "n_old": len(old_proj),
                "n_total": len(proj_meta),
                "all_genes": {"r": round(r_all, 4), "p": float(p_all), "n": n_all},
                "cross_species_genes": {
                    "r": round(r_cs, 4) if not np.isnan(r_cs) else None,
                    "p": float(p_cs) if not np.isnan(p_cs) else None,
                    "n": n_cs,
                },
            }
            print(f"    All genes:           r = {r_all:.4f} (n = {n_all})")
            if not np.isnan(r_cs):
                print(f"    Cross-species genes: r = {r_cs:.4f} (n = {n_cs})")

        # Pooled analysis: average within-project FC across both projects
        fc_combined = fc_full[["fc_full"]].copy()
        for proj in VALIDATION_PROJECTS:
            fc_combined = fc_combined.join(project_fcs[proj][[f"fc_{proj}"]], how="inner")

        fc_combined = fc_combined.dropna()
        fc_combined["fc_within_mean"] = fc_combined[
            [f"fc_{proj}" for proj in VALIDATION_PROJECTS]
        ].mean(axis=1)

        r_pooled, p_pooled = stats.pearsonr(fc_combined["fc_full"], fc_combined["fc_within_mean"])
        n_pooled = len(fc_combined)

        # Bootstrap CI for pooled correlation
        boot_rs = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            idx = rng.choice(n_pooled, size=n_pooled, replace=True)
            br, _ = stats.pearsonr(
                fc_combined["fc_full"].values[idx], fc_combined["fc_within_mean"].values[idx]
            )
            boot_rs.append(br)
        ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])

        # Cross-species subset for pooled
        cs_pooled = fc_combined[fc_combined.index.isin(cs_ids)]
        if len(cs_pooled) >= 3:
            r_cs_pooled, p_cs_pooled = stats.pearsonr(
                cs_pooled["fc_full"], cs_pooled["fc_within_mean"]
            )
        else:
            r_cs_pooled, p_cs_pooled = np.nan, np.nan

        print(f"\n  Pooled within-project vs full-dataset:")
        print(f"    All genes:           r = {r_pooled:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}] (n = {n_pooled})")
        if not np.isnan(r_cs_pooled):
            print(f"    Cross-species genes: r = {r_cs_pooled:.4f} (n = {len(cs_pooled)})")

        results["comparisons"][comparison_key] = {
            "stage_young": stage_young,
            "stage_old": stage_old,
            "full_dataset": {
                "n_young": len(young_all),
                "n_old": len(old_all),
            },
            "projects": project_results,
            "pooled": {
                "r": round(r_pooled, 4),
                "p": float(p_pooled),
                "n": n_pooled,
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "cross_species_genes": {
                    "r": round(r_cs_pooled, 4) if not np.isnan(r_cs_pooled) else None,
                    "p": float(p_cs_pooled) if not np.isnan(p_cs_pooled) else None,
                    "n": len(cs_pooled),
                },
            },
        }

        # Save per-comparison CSV
        csv_out = fc_combined.copy()
        safe_key = comparison_key.replace(" ", "_")
        csv_out.to_csv(OUTPUT_DIR / f"within_project_fold_changes_{safe_key}.csv")

    # Save results
    results_file = OUTPUT_DIR / "within_project_validation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    run_within_project_validation()
