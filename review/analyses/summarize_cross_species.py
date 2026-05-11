#!/usr/bin/env python3
"""Print a concise summary of cross-species results + muscle 66-gene overlap.

Reads:
  - review/analyses/results/cross_species_all_tissues.json
  - paper/figures/output/stats/fig4_expression_stats.csv (muscle 66-gene set)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PATH = ROOT / "review" / "analyses" / "results" / "cross_species_all_tissues.json"
GENES_CSV = ROOT / "review" / "analyses" / "results" / "cross_species_per_tissue_genes.csv"
MUSCLE_CSV = ROOT / "paper" / "figures" / "output" / "stats" / "fig4_expression_stats.csv"

# Muscle 66-gene set
mdf = pd.read_csv(MUSCLE_CSV)
mask = ((mdf["fdr_pig"] < 0.10) & (mdf["fdr_human"] < 0.10) &
        (mdf["log2fc_pig"].abs() > 0.5) & (mdf["log2fc_human"].abs() > 0.5))
muscle_66 = set(mdf[mask]["gene_symbol"].dropna().str.upper())
print(f"Muscle 66-gene reference set: {len(muscle_66)} symbols\n")

# Load main JSON
data = json.loads(JSON_PATH.read_text())
print(f"Ortholog one-to-one pairs: {data['n_one_to_one_orthologs']}")
print(f"Muscle reference (from paper): r = {data['muscle_reference']['pearson_r']}, "
      f"CI = [{data['muscle_reference']['ci_low']}, {data['muscle_reference']['ci_high']}], "
      f"n = {data['muscle_reference']['n_genes']}")
print()

# Per-tissue table
print(f"{'Tissue':<10} {'n_tested':>10} {'n_strict':>10} {'r_strict':>10} {'CI_low':>8} {'CI_hi':>8} {'concord%':>10} {'n_pa':>6} {'r_pa':>8}")
print('-' * 100)
genes_df = pd.read_csv(GENES_CSV)
overlap_summary = {}
for tissue, info in data["per_tissue"].items():
    if "skipped" in info:
        print(f"{tissue:<10} SKIPPED: {info['skipped']}")
        continue
    s = info["strict"]
    pa = info["pig_anchored"]
    print(f"{tissue:<10} "
          f"{info['n_orthologs_tested']:>10} "
          f"{s['n_genes']:>10} "
          f"{(s['pearson_r'] if s['pearson_r'] is not None else 'NA'):>10} "
          f"{(s['ci_low'] if s['ci_low'] is not None else 'NA'):>8} "
          f"{(s['ci_high'] if s['ci_high'] is not None else 'NA'):>8} "
          f"{(s['directional_concordance'] if s['directional_concordance'] is not None else 'NA'):>10} "
          f"{pa['n_genes']:>6} "
          f"{(pa['pearson_r'] if pa['pearson_r'] is not None else 'NA'):>8}")

    # Overlap with muscle 66 set
    sub = genes_df[(genes_df["tissue"] == tissue) & genes_df["passes_strict"]]
    if len(sub) == 0:
        sub = genes_df[(genes_df["tissue"] == tissue) & genes_df["passes_pig_anchored"]]
    tissue_genes = set(sub["gene_symbol"].dropna().str.upper())
    overlap = tissue_genes & muscle_66
    overlap_summary[tissue] = sorted(overlap)
    print(f"           Genes overlapping with muscle 66-set: {len(overlap)} -> {sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
    print()

# Save overlap to a file
overlap_path = ROOT / "review" / "analyses" / "results" / "muscle66_overlap_by_tissue.json"
overlap_path.write_text(json.dumps(overlap_summary, indent=2))
print(f"\nMuscle-66 overlap by tissue: {overlap_path}")
