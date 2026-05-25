#!/usr/bin/env python3
"""Uniform cross-species protocol across all 5 tissues.

This orchestrator applies ONE protocol identically to Muscle, Lung, Testis,
Brain, Liver. It takes uniform flags and dispatches to the appropriate human
loader per tissue (Schaiter CSV for muscle, LungMAP for lung, Cardoso-Moreira
for brain/liver/testis). The pig side is fully uniform.

Usage:
  python autoresearch/run_uniform_cross_species.py \\
    --method weighted --bins developmental \\
    --min-pig-tpm 5 --min-human-norm 50 --drop-pct 0 \\
    --out review/analyses/results/uniform_R1.json

Writes:
  - <out>.json    — per-tissue stats block
  - <out>_summary.tsv  — one row per tissue (tissue, n, r, ci_low, ci_high, sp, dc%)
"""

from __future__ import annotations

import argparse
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
from autoresearch.run_weighted_fc import (  # noqa: E402
    median_fc, weighted_fc, limma_fc,
)
from autoresearch.run_purity_generic import (  # noqa: E402
    TISSUE_MARKERS, load_pig_symbol_map, compute_marker_score,
    select_keep_stratified,
)
from autoresearch.run_lung_cross_species import (  # noqa: E402
    load_human_lungmap, remap_human_expr_to_ensg,
)

# ---------------------------------------------------------------------------
# Uniform protocol definitions
# ---------------------------------------------------------------------------

TISSUES = ["Muscle", "Lung", "Testis", "Brain", "Liver"]

# Cardoso-Moreira (brain/liver/testis) human stage bins
HUMAN_BINS_CM = {
    "developmental":         ({"newborn", "infant", "toddler"},
                              {"youngAdult", "youngMidAge"}),
    "developmental_strict":  ({"newborn", "infant", "toddler"},
                              {"youngAdult"}),
    "extended_old":          ({"newborn", "infant", "toddler"},
                              {"teenager", "oldTeenager", "youngAdult",
                               "youngMidAge", "olderMidAge", "senior",
                               "Senior"}),
}

# Pig stage bins (all tissues)
PIG_BINS = {
    "developmental":         ({"Infant", "Early childhood"},
                              {"Post-pubertal", "Adult"}),
    "developmental_strict":  ({"Infant"}, {"Adult"}),
    "extended_old":          ({"Infant", "Early childhood"},
                              {"Pre-pubertal", "Post-pubertal", "Adult"}),
}

MUSCLE_CSV = ROOT / "paper" / "figures" / "output" / "stats" / "fig4_expression_stats.csv"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_purity_z(p_expr: pd.DataFrame, p_young: list[str],
                      p_old: list[str], tissue: str) -> pd.Series:
    """Per-sample purity-z from the canonical tissue marker panel.
    z is computed within each YOUNG/OLD group so it doesn't leak stage."""
    sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel = TISSUE_MARKERS.get(tissue, [])
    ids = [sym2id[s.upper()] for s in panel if s.upper() in sym2id]
    if not ids:
        return pd.Series(0.0, index=p_expr.columns)
    candidate = sorted(set(p_young) | set(p_old))
    score = compute_marker_score(p_expr[candidate], ids)
    purity_z = pd.Series(0.0, index=p_expr.columns)
    for group in (p_young, p_old):
        if len(group) < 2:
            continue
        g = score.reindex(group)
        mu, sd = g.mean(), max(g.std(ddof=0), 1e-6)
        purity_z.loc[group] = ((g - mu) / sd).values
    return purity_z


def apply_uniform_drop(p_expr: pd.DataFrame, p_young: list[str],
                        p_old: list[str], tissue: str,
                        drop_pct: float) -> tuple:
    """Stratified-purity drop% per stage, applied uniformly using the
    tissue's canonical marker panel."""
    if drop_pct <= 0:
        return p_young, p_old, {"drop_pct": 0.0, "n_dropped_young": 0,
                                 "n_dropped_old": 0}
    sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel = TISSUE_MARKERS.get(tissue, [])
    ids = [sym2id[s.upper()] for s in panel if s.upper() in sym2id]
    if not ids:
        return p_young, p_old, {"drop_pct": drop_pct, "skipped": "no_markers"}
    candidate = sorted(set(p_young) | set(p_old))
    score = compute_marker_score(p_expr[candidate], ids)
    kept, _ = select_keep_stratified(
        score, {"young": p_young, "old": p_old}, drop_pct, drop_low=True)
    new_young = [s for s in p_young if s in kept]
    new_old = [s for s in p_old if s in kept]
    return new_young, new_old, {
        "drop_pct": drop_pct,
        "panel_present": len(ids),
        "n_dropped_young": len(p_young) - len(new_young),
        "n_dropped_old": len(p_old) - len(new_old),
    }


def compute_pig_fc(p_expr: pd.DataFrame, p_young: list[str], p_old: list[str],
                    method: str, tissue: str) -> pd.DataFrame:
    """Apply the uniform pig FC method."""
    if method == "median":
        return median_fc(p_expr, p_young, p_old)
    if method == "weighted":
        purity_z = compute_purity_z(p_expr, p_young, p_old, tissue)
        weights = 1.0 / (1.0 + np.exp(-purity_z))
        return weighted_fc(p_expr, p_young, p_old, weights)
    if method == "limma":
        purity_z = compute_purity_z(p_expr, p_young, p_old, tissue)
        return limma_fc(p_expr, p_young, p_old, purity_z)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Per-tissue analysis
# ---------------------------------------------------------------------------

def analyse_tissue(tissue: str, method: str, bins_key: str,
                    min_pig_tpm: float, min_human_norm: float,
                    drop_pct: float, ortho: pd.DataFrame) -> dict:
    """Apply the uniform protocol to one tissue. Returns the per-tissue
    stats dict (mirrors the schema produced by the other autoresearch scripts)."""
    # ---- 1. Pig side: bins, expression, filter, drop --------------------
    pb = PIG_BINS[bins_key]
    xs.PIG_YOUNG, xs.PIG_OLD = pb[0], pb[1]
    p_expr, p_young, p_old = xs.load_pig_tissue(tissue)
    if len(p_young) < 3 or len(p_old) < 3:
        return {"tissue": tissue, "skipped": "pig_low_n_initial",
                "n_pig_young": len(p_young), "n_pig_old": len(p_old)}
    if min_pig_tpm > 0:
        keep = p_expr.median(axis=1) >= min_pig_tpm
        p_expr = p_expr.loc[keep]
    p_young, p_old, qc = apply_uniform_drop(
        p_expr, p_young, p_old, tissue, drop_pct)
    if len(p_young) < 3 or len(p_old) < 3:
        return {"tissue": tissue, "skipped": "pig_low_n_after_drop", "qc": qc}
    p_fc = compute_pig_fc(p_expr, p_young, p_old, method, tissue)

    # ---- 2. Human side: 3-way dispatch ----------------------------------
    if tissue == "Muscle":
        muscle = pd.read_csv(MUSCLE_CSV)
        # Re-derive pig FC from our (potentially different) method, but keep
        # the precomputed Schaiter human FC.
        ortho_m = ortho[ortho["pig_gene_id"].isin(muscle["pig_gene_id"])]
        # Build per-row table
        muscle_dict = muscle.set_index("pig_gene_id")[
            ["log2fc_human", "p_human", "fdr_human"]
        ].to_dict("index")
        rows = []
        for _, r in ortho_m.iterrows():
            pid = r["pig_gene_id"]
            if pid not in p_fc.index or pid not in muscle_dict:
                continue
            mh = muscle_dict[pid]
            rows.append({
                "gene_symbol": r["symbol"],
                "pig_gene_id": pid,
                "human_gene_id": r["human_gene_id"],
                "log2fc_pig":   p_fc.loc[pid, "log2fc"],
                "log2fc_human": mh["log2fc_human"],
                "p_pig":        p_fc.loc[pid, "pvalue"],
                "p_human":      mh["p_human"],
            })
        merged = pd.DataFrame(rows)
        merged = merged.dropna(
            subset=["log2fc_pig", "log2fc_human", "p_pig", "p_human"])
        n_y_h = "Schaiter-fixed"
        n_o_h = "Schaiter-fixed"
        merged = xs.add_fdr(merged, "p_pig",   "fdr_pig")
        merged = xs.add_fdr(merged, "p_human", "fdr_human")
    elif tissue == "Lung":
        h_expr_sym, h_young, h_old, _ = load_human_lungmap("BPS", drop_adult=True)
        h_expr = remap_human_expr_to_ensg(h_expr_sym, ortho)
        if min_human_norm > 0:
            h_expr = h_expr.loc[h_expr.median(axis=1) >= min_human_norm]
        h_fc = median_fc(h_expr, h_young, h_old)
        n_y_h, n_o_h = len(h_young), len(h_old)
        o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index)
                   & ortho["human_gene_id"].isin(h_fc.index)]
        rows = []
        for _, r in o_.iterrows():
            rows.append({
                "gene_symbol": r["symbol"],
                "pig_gene_id": r["pig_gene_id"],
                "human_gene_id": r["human_gene_id"],
                "log2fc_pig":   p_fc.loc[r["pig_gene_id"], "log2fc"],
                "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
                "p_pig":        p_fc.loc[r["pig_gene_id"], "pvalue"],
                "p_human":      h_fc.loc[r["human_gene_id"], "pvalue"],
            })
        merged = pd.DataFrame(rows)
        merged = xs.add_fdr(merged, "p_pig",   "fdr_pig")
        merged = xs.add_fdr(merged, "p_human", "fdr_human")
    else:  # Brain, Liver, Testis
        hb = HUMAN_BINS_CM[bins_key]
        xs.HUMAN_YOUNG, xs.HUMAN_OLD = hb[0], hb[1]
        h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, tissue)
        if min_human_norm > 0:
            h_expr = h_expr.loc[h_expr.median(axis=1) >= min_human_norm]
        h_fc = median_fc(h_expr, h_young, h_old)
        n_y_h, n_o_h = len(h_young), len(h_old)
        o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index)
                   & ortho["human_gene_id"].isin(h_fc.index)]
        rows = []
        for _, r in o_.iterrows():
            rows.append({
                "gene_symbol": r["symbol"],
                "pig_gene_id": r["pig_gene_id"],
                "human_gene_id": r["human_gene_id"],
                "log2fc_pig":   p_fc.loc[r["pig_gene_id"], "log2fc"],
                "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
                "p_pig":        p_fc.loc[r["pig_gene_id"], "pvalue"],
                "p_human":      h_fc.loc[r["human_gene_id"], "pvalue"],
            })
        merged = pd.DataFrame(rows)
        merged = xs.add_fdr(merged, "p_pig",   "fdr_pig")
        merged = xs.add_fdr(merged, "p_human", "fdr_human")

    # ---- 3. Uniform pig-anchored filter ---------------------------------
    pa = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    def _stats(df_):
        if len(df_) < 5:
            return {"n_genes": int(len(df_)), "pearson_r": None,
                    "pearson_p": None, "spearman_r": None,
                    "ci_low": None, "ci_high": None,
                    "directional_concordance": None}
        x = df_["log2fc_pig"].values
        y = df_["log2fc_human"].values
        r, p = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        _, lo, hi = xs.bootstrap_pearson_ci(x, y)
        return {
            "n_genes": int(len(df_)),
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "ci_low":  float(lo) if not np.isnan(lo) else None,
            "ci_high": float(hi) if not np.isnan(hi) else None,
            "directional_concordance": 100.0 * nc / len(df_),
        }

    pa_stats = _stats(pa)

    return {
        "tissue": tissue,
        "method": method, "bins": bins_key,
        "min_pig_tpm": min_pig_tpm, "min_human_norm": min_human_norm,
        "drop_pct": drop_pct,
        "qc": qc,
        "n_orthologs_tested": int(len(merged)),
        "sample_sizes": {
            "pig_young": len(p_young), "pig_old": len(p_old),
            "human_young": n_y_h, "human_old": n_o_h,
        },
        "pig_anchored": pa_stats,
        "_gene_table": pa[
            ["gene_symbol", "pig_gene_id", "human_gene_id",
             "log2fc_pig", "log2fc_human", "p_pig", "p_human",
             "fdr_pig", "fdr_human"]
        ].to_dict("records") if len(pa) < 5000 else None,
    }


# ---------------------------------------------------------------------------
# Driver: run all 5 tissues uniformly
# ---------------------------------------------------------------------------

def run_uniform(method: str, bins_key: str, min_pig_tpm: float,
                 min_human_norm: float, drop_pct: float,
                 out_path: Path) -> dict:
    print(f"=== Uniform protocol: method={method}, bins={bins_key}, "
          f"min_pig_tpm={min_pig_tpm}, min_human_norm={min_human_norm}, "
          f"drop_pct={drop_pct} ===")
    ortho = xs.build_one_to_one_orthologs()
    per_tissue = {}
    summary_rows = []
    for tissue in TISSUES:
        print(f"\n--- {tissue} ---")
        res = analyse_tissue(tissue, method, bins_key, min_pig_tpm,
                              min_human_norm, drop_pct, ortho)
        per_tissue[tissue] = res
        pa = res.get("pig_anchored", {}) or {}
        print(f"  pig-anchored: n={pa.get('n_genes')} r={pa.get('pearson_r')} "
              f"sp={pa.get('spearman_r')} CI=[{pa.get('ci_low')}, "
              f"{pa.get('ci_high')}] dc={pa.get('directional_concordance')}")
        ss = res.get("sample_sizes", {})
        summary_rows.append({
            "tissue":       tissue,
            "method":       method,
            "bins":         bins_key,
            "min_pig_tpm":  min_pig_tpm,
            "min_human_norm": min_human_norm,
            "drop_pct":     drop_pct,
            "n_genes":      pa.get("n_genes"),
            "pearson_r":    pa.get("pearson_r"),
            "pearson_p":    pa.get("pearson_p"),
            "spearman_r":   pa.get("spearman_r"),
            "ci_low":       pa.get("ci_low"),
            "ci_high":      pa.get("ci_high"),
            "dc_pct":       pa.get("directional_concordance"),
            "n_pig_young":  ss.get("pig_young"),
            "n_pig_old":    ss.get("pig_old"),
            "n_human_young": ss.get("human_young"),
            "n_human_old":   ss.get("human_old"),
            "n_dropped_young": (res.get("qc") or {}).get("n_dropped_young"),
            "n_dropped_old":   (res.get("qc") or {}).get("n_dropped_old"),
        })
    # Ensemble metrics
    rs = [s["pearson_r"] for s in summary_rows if s["pearson_r"] is not None]
    summary = {
        "config": {
            "method": method, "bins": bins_key,
            "min_pig_tpm": min_pig_tpm, "min_human_norm": min_human_norm,
            "drop_pct": drop_pct,
        },
        "ensemble": {
            "mean_r":   float(np.mean(rs))   if rs else None,
            "median_r": float(np.median(rs)) if rs else None,
            "min_r":    float(min(rs))       if rs else None,
            "max_r":    float(max(rs))       if rs else None,
            "sum_r":    float(np.sum(rs))    if rs else None,
            "n_tissues_with_r": len(rs),
        },
        "per_tissue": per_tissue,
    }
    print(f"\n=== Ensemble: mean r = {summary['ensemble']['mean_r']:.3f}, "
          f"min r = {summary['ensemble']['min_r']:.3f}, "
          f"max r = {summary['ensemble']['max_r']:.3f} ===")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    tsv_path = out_path.with_suffix(".summary.tsv")
    pd.DataFrame(summary_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    print(f"Wrote {tsv_path}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method",
                    choices=["median", "weighted", "limma"], default="median")
    ap.add_argument("--bins",
                    choices=list(PIG_BINS.keys()), default="developmental")
    ap.add_argument("--min-pig-tpm",   type=float, default=0.0)
    ap.add_argument("--min-human-norm", type=float, default=0.0)
    ap.add_argument("--drop-pct",      type=float, default=0.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    run_uniform(args.method, args.bins, args.min_pig_tpm,
                 args.min_human_norm, args.drop_pct, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
