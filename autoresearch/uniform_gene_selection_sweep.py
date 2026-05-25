#!/usr/bin/env python3
"""Loop A — uniform gene-selection autoresearch.

Picks the SINGLE uniform (filter × ortholog × statistic) config that maximizes
mean cross-species r across the 5-tissue dGTEx panel, with min_r as tie-breaker.
No hidden hard thresholds — just one explainable objective.

Grid: 5 filters × 2 orthologs × 2 statistics = 20 cells (× 5 tissues each)

Per-tissue evaluation is the same uniform protocol from Phase 1:
  pig side  — Infant+Early-childhood vs Post-pubertal+Adult, drop% 30, weighted FC
  human side — dGTEx cohort 1 vs cohort 4, median FC
  merge on orthologs, apply filter, compute statistic.

The per-tissue MERGED FC tables are built ONCE per (tissue, ortholog) and cached.
Filters and statistics are applied as table operations (no recomputation).

Outputs:
  review/analyses/results/v2_uniform_sweep_full_grid.csv  — every cell evaluated
  review/analyses/results/v2_uniform_winner.json          — chosen config + per-tissue stats
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402

from autoresearch.run_uniform_cross_species import (  # noqa: E402
    apply_uniform_drop, compute_pig_fc,
)
from autoresearch.run_weighted_fc import median_fc  # noqa: E402
from autoresearch.load_dgtex import load_dgtex_tissue  # noqa: E402
from autoresearch.stage_mapping_v2 import (  # noqa: E402
    PIG_OLD_STAGES, PIG_YOUNG_STAGES,
)

xs.PIG_YOUNG = set(PIG_YOUNG_STAGES)
xs.PIG_OLD = set(PIG_OLD_STAGES)

# ─────────────────────────────────────────────────────────────────────────────
# Grid + tissue panel
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_PANEL = [
    # (dGTEx SMTS,        PigGTEx file name, dGTEx file slug)
    ("Muscle",            "Muscle",          "muscle"),
    ("Lung",              "Lung",            "lung"),
    ("Testis",            "Testis",          "testis"),
    ("Spleen",            "Spleen",          "spleen"),
    ("Adipose Tissue",    "Adipose",         "adipose_tissue"),
]

ORTHOLOG_SETS = {
    "strict_1to1":    ROOT / "review/analyses/results/pig_human_one_to_one_orthologs.csv",
    "relaxed_symbol": ROOT / "review/analyses/results/pig_human_orthologs_symbol_based.csv",
}
FILTERS = ["strict_bidirectional", "pig_anchored", "pig_fdr_only", "fc_only", "no_filter"]
STATISTICS = ["pearson", "spearman"]

FDR_T = 0.10
FC_T = 0.5
PROTOCOL = dict(
    pig_method="weighted",
    pig_min_tpm=1.0,
    pig_drop_pct=30.0,
    human_min_tpm=10.0,
    human_method="median",
    human_young_cohort=[1],
    human_old_cohort=[4],
)


def build_merged_fc(pig_tissue: str, dgtex_smts: str, tissue_file: str,
                    ortho: pd.DataFrame) -> pd.DataFrame | None:
    """Build merged pig×dGTEx FC table for one (tissue, ortholog) combo. Returns None on failure."""
    p_expr, p_y, p_o = xs.load_pig_tissue(pig_tissue)
    if len(p_y) < 3 or len(p_o) < 3:
        print(f"    pig {pig_tissue}: low initial n y={len(p_y)} o={len(p_o)}")
        return None
    p_expr = p_expr.loc[p_expr.median(axis=1) >= PROTOCOL["pig_min_tpm"]]
    p_y, p_o, _ = apply_uniform_drop(p_expr, p_y, p_o, pig_tissue, PROTOCOL["pig_drop_pct"])
    if len(p_y) < 3 or len(p_o) < 3:
        return None
    p_fc = compute_pig_fc(p_expr, p_y, p_o, PROTOCOL["pig_method"], pig_tissue)

    h_expr, h_y, h_o = load_dgtex_tissue(dgtex_smts, tissue_file)
    if len(h_y) < 3 or len(h_o) < 3:
        return None
    h_expr = h_expr.loc[h_expr.median(axis=1) >= PROTOCOL["human_min_tpm"]]
    h_fc = median_fc(h_expr, h_y, h_o)

    o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index)
               & ortho["human_gene_id"].isin(h_fc.index)]
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
    merged = xs.add_fdr(merged, "p_pig", "fdr_pig")
    merged = xs.add_fdr(merged, "p_human", "fdr_human")
    return merged


def apply_filter(m: pd.DataFrame, filt: str) -> pd.DataFrame:
    if filt == "strict_bidirectional":
        return m[(m["fdr_pig"] < FDR_T) & (m["fdr_human"] < FDR_T)
                 & (m["log2fc_pig"].abs() > FC_T) & (m["log2fc_human"].abs() > FC_T)]
    if filt == "pig_anchored":
        return m[(m["fdr_pig"] < FDR_T)
                 & (m["log2fc_pig"].abs() > FC_T) & (m["log2fc_human"].abs() > FC_T)]
    if filt == "pig_fdr_only":
        return m[(m["fdr_pig"] < FDR_T) & (m["log2fc_pig"].abs() > FC_T)]
    if filt == "fc_only":
        return m[(m["log2fc_pig"].abs() > FC_T) & (m["log2fc_human"].abs() > FC_T)]
    if filt == "no_filter":
        return m
    raise ValueError(filt)


def compute_stat(sub: pd.DataFrame, statistic: str) -> dict:
    """Compute correlation + 95% bootstrap CI + directional concordance."""
    n = len(sub)
    if n < 3:
        return {"n": n, "r": None, "p": None, "dc": None, "ci_low": None, "ci_high": None}
    x = sub["log2fc_pig"].values
    y = sub["log2fc_human"].values
    if statistic == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)
    dc = 100.0 * (np.sign(x) == np.sign(y)).sum() / n
    if n >= 5:
        _, ci_low, ci_high = xs.bootstrap_pearson_ci(x, y)
        ci_low = float(ci_low) if not np.isnan(ci_low) else None
        ci_high = float(ci_high) if not np.isnan(ci_high) else None
    else:
        ci_low = ci_high = None
    return {"n": int(n), "r": float(r), "p": float(p),
            "dc": float(dc), "ci_low": ci_low, "ci_high": ci_high}


def main():
    # ---- Phase 1: build per-(tissue, ortholog) merged FC tables, cache them ----
    print("=" * 80)
    print("Loop A — Uniform gene-selection autoresearch")
    print("=" * 80)
    print(f"Panel: {len(TISSUE_PANEL)} tissues")
    print(f"Grid: {len(FILTERS)} filters × {len(ORTHOLOG_SETS)} orthologs × "
          f"{len(STATISTICS)} statistics = {len(FILTERS) * len(ORTHOLOG_SETS) * len(STATISTICS)} cells")
    print()

    ortho_tables = {name: pd.read_csv(path) for name, path in ORTHOLOG_SETS.items()}
    print("Building merged FC tables (one per tissue × ortholog):")
    merged_cache = {}
    for dgtex_smts, pig_tissue, file_slug in TISSUE_PANEL:
        for oname, ortho in ortho_tables.items():
            print(f"  {dgtex_smts:<16} | ortho={oname}", end=" ... ", flush=True)
            m = build_merged_fc(pig_tissue, dgtex_smts, file_slug, ortho)
            if m is None:
                print("SKIPPED")
            else:
                merged_cache[(dgtex_smts, oname)] = m
                print(f"merged n={len(m)}")
    print()

    # ---- Phase 2: evaluate every (filter, ortholog, statistic) config across all tissues ----
    print("Evaluating grid:")
    grid_rows = []
    for ortho_name, filt, stat in product(ORTHOLOG_SETS, FILTERS, STATISTICS):
        per_tissue = {}
        for dgtex_smts, _, _ in TISSUE_PANEL:
            merged = merged_cache.get((dgtex_smts, ortho_name))
            if merged is None:
                per_tissue[dgtex_smts] = {"n": 0, "r": None, "p": None, "dc": None,
                                          "ci_low": None, "ci_high": None}
                continue
            sub = apply_filter(merged, filt)
            per_tissue[dgtex_smts] = compute_stat(sub, stat)
        # Aggregate
        rs = [per_tissue[t]["r"] for t, _, _ in TISSUE_PANEL if per_tissue[t]["r"] is not None]
        ns = [per_tissue[t]["n"] for t, _, _ in TISSUE_PANEL if per_tissue[t]["r"] is not None]
        ns_all = [per_tissue[t]["n"] for t, _, _ in TISSUE_PANEL]
        row = {
            "config":   f"{ortho_name} | {filt} | {stat}",
            "ortholog": ortho_name,
            "filter":   filt,
            "statistic": stat,
            "n_tissues_with_r":  len(rs),
            "mean_r": float(np.mean(rs)) if rs else None,
            "min_r":  float(min(rs))   if rs else None,
            "max_r":  float(max(rs))   if rs else None,
            "min_n":  int(min(ns))   if ns else 0,
            "max_n":  int(max(ns_all)),
        }
        for t, _, _ in TISSUE_PANEL:
            row[f"{t}_n"]   = per_tissue[t]["n"]
            row[f"{t}_r"]   = per_tissue[t]["r"]
            row[f"{t}_p"]   = per_tissue[t]["p"]
            row[f"{t}_dc"]  = per_tissue[t]["dc"]
            row[f"{t}_cilo"] = per_tissue[t]["ci_low"]
            row[f"{t}_cihi"] = per_tissue[t]["ci_high"]
        grid_rows.append(row)
        print(f"  {row['config']:<55}  mean_r={row['mean_r']:+.3f}  min_r={row['min_r']:+.3f}  "
              f"(n_range {row['min_n']}-{row['max_n']})")
    print()

    grid_df = pd.DataFrame(grid_rows)
    grid_out = ROOT / "review/analyses/results/v2_uniform_sweep_full_grid.csv"
    grid_df.to_csv(grid_out, index=False)
    print(f"Wrote: {grid_out}")

    # ---- Phase 3: pick winner ----
    valid = grid_df[grid_df["n_tissues_with_r"] == len(TISSUE_PANEL)].copy()
    if valid.empty:
        print("\nERROR: no config gives results for all 5 tissues")
        return 1
    # primary: max mean_r ; tie-break: max min_r (within 0.005 of max mean_r)
    max_mean = valid["mean_r"].max()
    near_top = valid[valid["mean_r"] >= max_mean - 0.005]
    winner_row = near_top.loc[near_top["min_r"].idxmax()]
    print(f"\nWINNER: {winner_row['config']}")
    print(f"  mean_r = {winner_row['mean_r']:+.4f}")
    print(f"  min_r  = {winner_row['min_r']:+.4f}")
    print(f"  Per tissue:")
    for t, _, _ in TISSUE_PANEL:
        n = winner_row[f"{t}_n"]
        r = winner_row[f"{t}_r"]
        p = winner_row[f"{t}_p"]
        dc = winner_row[f"{t}_dc"]
        cilo = winner_row[f"{t}_cilo"]
        cihi = winner_row[f"{t}_cihi"]
        ci_str = f"[{cilo:+.3f}, {cihi:+.3f}]" if pd.notna(cilo) else "[—, —]"
        print(f"    {t:<16} n={int(n):>4}  r={r:+.3f}  CI={ci_str}  p={p:.2e}  dc={dc:.0f}%")

    winner_out = ROOT / "review/analyses/results/v2_uniform_winner.json"
    winner_dict = {
        "config": {
            "ortholog":  winner_row["ortholog"],
            "filter":    winner_row["filter"],
            "statistic": winner_row["statistic"],
        },
        "protocol": PROTOCOL,
        "objective": "max mean_r; tie-break max min_r within 0.005 of max",
        "aggregate": {
            "mean_r": float(winner_row["mean_r"]),
            "min_r":  float(winner_row["min_r"]),
            "max_r":  float(winner_row["max_r"]),
            "n_tissues": int(winner_row["n_tissues_with_r"]),
        },
        "per_tissue": {
            t: {
                "n_genes": int(winner_row[f"{t}_n"]),
                "r":       float(winner_row[f"{t}_r"]),
                "p":       float(winner_row[f"{t}_p"]),
                "dc":      float(winner_row[f"{t}_dc"]),
                "ci_low":  float(winner_row[f"{t}_cilo"]) if pd.notna(winner_row[f"{t}_cilo"]) else None,
                "ci_high": float(winner_row[f"{t}_cihi"]) if pd.notna(winner_row[f"{t}_cihi"]) else None,
            }
            for t, _, _ in TISSUE_PANEL
        },
    }
    winner_out.write_text(json.dumps(winner_dict, indent=2))
    print(f"\nWrote: {winner_out}")

    # Tier 1 / Tier 2 reporting summary
    tier1 = [t for t, _, _ in TISSUE_PANEL if winner_row[f"{t}_r"] >= 0.40]
    print(f"\nTIER 1 (strong, r ≥ 0.40): {tier1}")
    if tier1:
        t1_rs = [winner_row[f"{t}_r"] for t in tier1]
        print(f"  mean Tier-1 r = {np.mean(t1_rs):+.4f}")
    print(f"TIER 2 (extended panel, all 5): mean r = {winner_row['mean_r']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
