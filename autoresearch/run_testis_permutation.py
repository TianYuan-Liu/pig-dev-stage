#!/usr/bin/env python3
"""Permutation control for testis median-FC cross-species signal.

Shuffle pig young/old labels (preserving group sizes), recompute median log2FC,
and correlate against the unchanged human log2FC. If the observed r=0.664
sits well above the permutation distribution, the signal is real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402
from autoresearch.run_weighted_fc import median_fc  # noqa: E402


def main() -> int:
    ortho = xs.build_one_to_one_orthologs()
    h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, "Testis")
    p_expr, p_young, p_old = xs.load_pig_tissue("Testis")
    print(f"Pig testis: young n={len(p_young)}, old n={len(p_old)}")
    print(f"Human testis: young n={len(h_young)}, old n={len(h_old)}")
    h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    # Observed median FC and r
    p_fc_obs = median_fc(p_expr, p_young, p_old)
    o_ = ortho[ortho["pig_gene_id"].isin(p_fc_obs.index) &
               ortho["human_gene_id"].isin(h_fc.index)].copy()
    # Apply strict filter
    pig_fdr = xs.add_fdr(p_fc_obs.copy().rename(columns={"pvalue": "pvalue"}),
                         "pvalue", "fdr")
    pig_strict_genes = pig_fdr[(pig_fdr["fdr"] < xs.FDR_THRESHOLD)
                               & (p_fc_obs["log2fc"].abs() > xs.FC_THRESHOLD)].index
    h_strict = (h_fc["pvalue"]
                .pipe(lambda s: pd.Series(np.where(s.isna(), 1.0, s.values), index=s.index)))
    h_fdr_full = h_fc.copy()
    h_fdr_full["fdr"] = pd.Series(index=h_fdr_full.index, dtype=float)
    from statsmodels.stats.multitest import multipletests
    msk = h_fdr_full["pvalue"].notna()
    if msk.sum():
        _, fdrs, _, _ = multipletests(h_fdr_full.loc[msk, "pvalue"], method="fdr_bh")
        h_fdr_full.loc[msk, "fdr"] = fdrs

    def merge_and_corr(pig_fc_df: pd.DataFrame) -> tuple[float, int]:
        rows = []
        for _, r in o_.iterrows():
            rows.append({
                "log2fc_pig": pig_fc_df.loc[r["pig_gene_id"], "log2fc"],
                "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
                "p_pig": pig_fc_df.loc[r["pig_gene_id"], "pvalue"],
                "p_human": h_fc.loc[r["human_gene_id"], "pvalue"],
            })
        merged = pd.DataFrame(rows)
        merged = xs.add_fdr(merged, "p_pig", "fdr_pig")
        merged = xs.add_fdr(merged, "p_human", "fdr_human")
        strict = merged[
            (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
            (merged["fdr_human"] < xs.FDR_THRESHOLD) &
            (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
            (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
        ]
        if len(strict) < 5:
            return (np.nan, len(strict))
        r, _ = stats.pearsonr(strict["log2fc_pig"], strict["log2fc_human"])
        return (float(r), int(len(strict)))

    r_obs, n_obs = merge_and_corr(p_fc_obs)
    print(f"\nObserved: median FC strict r={r_obs:.4f}, n={n_obs}")

    # Permutation: shuffle young/old labels (preserve group sizes)
    rng = np.random.default_rng(42)
    all_pig = list(p_young) + list(p_old)
    n_young, n_old = len(p_young), len(p_old)
    n_perm = 50
    perm_r = []
    perm_n = []
    for i in range(n_perm):
        rng.shuffle(all_pig)
        perm_young = all_pig[:n_young]
        perm_old = all_pig[n_young:]
        try:
            pfc = median_fc(p_expr, perm_young, perm_old)
            r, n = merge_and_corr(pfc)
        except Exception:
            r, n = np.nan, 0
        perm_r.append(r)
        perm_n.append(n)
        if i < 5 or (i + 1) % 10 == 0:
            print(f"  perm {i+1:3d}: r={r:.4f} n={n}")
    perm_r_clean = [v for v in perm_r if not np.isnan(v)]
    perm_r_arr = np.array(perm_r_clean)
    print(f"\nPermutation null (n={len(perm_r_clean)}/{n_perm}):")
    print(f"  mean = {perm_r_arr.mean():+.4f}")
    print(f"  std  = {perm_r_arr.std():.4f}")
    print(f"  min  = {perm_r_arr.min():+.4f}")
    print(f"  max  = {perm_r_arr.max():+.4f}")
    print(f"  observed r = {r_obs:.4f}")
    n_above = sum(1 for v in perm_r_clean if v >= r_obs)
    print(f"  observed >= {n_above}/{len(perm_r_clean)} permutations (empirical p~{n_above/len(perm_r_clean):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
