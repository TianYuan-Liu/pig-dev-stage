#!/usr/bin/env python3
"""Comprehensive gene-selection sweep — answers what knobs actually matter for cross-species r.

Tests:
  - 2 ortholog sets:    Compara 1:1 strict (7,485) vs symbol-based relaxed (15,200)
  - 4 gene filters:     pig_anchored | pig_fdr_only | fc_only | no_filter
  - 2 statistics:       Pearson | Spearman
  - 5 tissues:          Muscle, Lung, Liver, Testis, Adipose, Spleen, Heart (skip Small Intestine)

Each combination is one row. Loads data once per tissue, then iterates.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs
from autoresearch.run_uniform_cross_species import apply_uniform_drop, compute_pig_fc
from autoresearch.run_weighted_fc import median_fc
from autoresearch.load_dgtex import load_dgtex_tissue
from autoresearch.stage_mapping_v2 import (
    PIG_YOUNG_STAGES, PIG_OLD_STAGES, INCLUDED_TISSUES,
)

xs.PIG_YOUNG = set(PIG_YOUNG_STAGES)
xs.PIG_OLD = set(PIG_OLD_STAGES)

# Load ortholog tables
ORTHO_STRICT  = pd.read_csv(ROOT / "review/analyses/results/pig_human_one_to_one_orthologs.csv")
ORTHO_RELAXED = pd.read_csv(ROOT / "review/analyses/results/pig_human_orthologs_symbol_based.csv")

FDR_T = 0.10
FC_T = 0.5


def build_per_gene_table(pig_tissue: str, dgtex_smts: str, tissue_file: str,
                         ortho: pd.DataFrame) -> pd.DataFrame | None:
    """Compute pig + dGTEx log2FC per ortholog, return merged table."""
    try:
        p_expr, p_y, p_o = xs.load_pig_tissue(pig_tissue)
    except Exception as e:
        print(f"  pig load failed: {e}")
        return None
    if len(p_y) < 3 or len(p_o) < 3:
        return None
    p_expr = p_expr.loc[p_expr.median(axis=1) >= 1.0]
    p_y, p_o, _qc = apply_uniform_drop(p_expr, p_y, p_o, pig_tissue, 30.0)
    if len(p_y) < 3 or len(p_o) < 3:
        return None
    p_fc = compute_pig_fc(p_expr, p_y, p_o, "weighted", pig_tissue)

    h_expr, h_y, h_o = load_dgtex_tissue(dgtex_smts, tissue_file)
    if len(h_y) < 3 or len(h_o) < 3:
        return None
    h_expr = h_expr.loc[h_expr.median(axis=1) >= 10.0]
    h_fc = median_fc(h_expr, h_y, h_o)

    # Merge on ortholog table provided
    o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index) &
               ortho["human_gene_id"].isin(h_fc.index)]
    rows = []
    for _, r in o_.iterrows():
        rows.append({
            "pig_gene_id":   r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig":    p_fc.loc[r["pig_gene_id"], "log2fc"],
            "log2fc_human":  h_fc.loc[r["human_gene_id"], "log2fc"],
            "p_pig":         p_fc.loc[r["pig_gene_id"], "pvalue"],
            "p_human":       h_fc.loc[r["human_gene_id"], "pvalue"],
        })
    merged = pd.DataFrame(rows)
    if len(merged) < 50: return None
    merged = xs.add_fdr(merged, "p_pig",   "fdr_pig")
    merged = xs.add_fdr(merged, "p_human", "fdr_human")
    return merged


def apply_filter(merged: pd.DataFrame, filt: str) -> pd.DataFrame:
    if filt == "pig_anchored":
        return merged[(merged["fdr_pig"] < FDR_T) &
                      (merged["log2fc_pig"].abs() > FC_T) &
                      (merged["log2fc_human"].abs() > FC_T)]
    if filt == "pig_fdr_only":
        return merged[(merged["fdr_pig"] < FDR_T) &
                      (merged["log2fc_pig"].abs() > FC_T)]
    if filt == "fc_only":
        return merged[(merged["log2fc_pig"].abs() > FC_T) &
                      (merged["log2fc_human"].abs() > FC_T)]
    if filt == "no_filter":
        return merged
    if filt == "strict_bidirectional":
        return merged[(merged["fdr_pig"] < FDR_T) &
                      (merged["fdr_human"] < FDR_T) &
                      (merged["log2fc_pig"].abs() > FC_T) &
                      (merged["log2fc_human"].abs() > FC_T)]
    raise ValueError(filt)


def stats_of(sub: pd.DataFrame) -> dict:
    if len(sub) < 5:
        return {"n": len(sub), "pearson_r": None, "spearman_r": None, "dc": None}
    x = sub["log2fc_pig"].values
    y = sub["log2fc_human"].values
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    dc = 100.0 * (np.sign(x) == np.sign(y)).sum() / len(sub)
    return {
        "n": len(sub),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "dc": float(dc),
    }


def main():
    TISSUES = [
        ("Muscle",          "Muscle",          "muscle"),
        ("Lung",            "Lung",            "lung"),
        ("Liver",           "Liver",           "liver"),
        ("Testis",          "Testis",          "testis"),
        ("Adipose Tissue",  "Adipose",         "adipose_tissue"),
        ("Spleen",          "Spleen",          "spleen"),
        ("Heart",           "Heart",           "heart"),
    ]
    FILTERS = ["strict_bidirectional", "pig_anchored", "pig_fdr_only", "fc_only", "no_filter"]
    ORTHO_SETS = [("strict_1to1",   ORTHO_STRICT),
                  ("relaxed_symbol", ORTHO_RELAXED)]

    results = []
    # Cache per-(tissue, ortholog) merged tables
    for dgtex_smts, pig_tissue, tissue_file in TISSUES:
        for oname, ortho in ORTHO_SETS:
            print(f"\n=== {dgtex_smts} | ortholog={oname} ===")
            merged = build_per_gene_table(pig_tissue, dgtex_smts, tissue_file, ortho)
            if merged is None:
                print(f"  SKIPPED (insufficient data)")
                continue
            print(f"  merged orthologs: {len(merged)}")
            for filt in FILTERS:
                sub = apply_filter(merged, filt)
                s = stats_of(sub)
                row = {
                    "tissue": dgtex_smts,
                    "ortholog": oname,
                    "filter": filt,
                    **s,
                }
                results.append(row)
                pr = s.get("pearson_r")
                sr = s.get("spearman_r")
                dc = s.get("dc")
                print(f"    {filt:<22} n={s['n']:>5}  "
                      f"Pearson r={pr if pr is None else f'{pr:+.3f}'}  "
                      f"Spearman r={sr if sr is None else f'{sr:+.3f}'}  "
                      f"dc={dc if dc is None else f'{dc:.0f}%'}")

    df = pd.DataFrame(results)
    out = ROOT / "review/analyses/results/gene_selection_sweep_results.csv"
    df.to_csv(out, index=False)
    print(f"\n\nWrote {out}")
    return df


if __name__ == "__main__":
    main()
