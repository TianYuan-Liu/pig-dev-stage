#!/usr/bin/env python3
"""Robust/weighted log2FC computation (no sample drops).

Methods:
  median    - median across samples instead of mean (50% breakdown robust)
  weighted  - sigmoid-weighted mean by purity z-score (continuous, no drops)
  limma     - per-gene lm(log2_expr ~ stage + purity_z), use stage coefficient
  huber     - sklearn HuberRegressor on log2_expr ~ stage_binary (robust to outliers)

All compute log2FC in the form mean_old - mean_young (or its weighted/robust analogue),
then run the same orthology mapping + correlation as the published pipeline.

Usage:
  python autoresearch/run_weighted_fc.py --pig-tissue Brain --human-tissue Brain \\
         --method weighted --out review/analyses/results/cross_species_brain_weighted.json
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
from autoresearch.run_purity_generic import (  # noqa: E402
    TISSUE_MARKERS, load_pig_symbol_map, compute_marker_score,
)


def median_fc(expr: pd.DataFrame, young: list[str], old: list[str]) -> pd.DataFrame:
    """Median-based log2FC: median(old) - median(young) on log2 scale."""
    ym = np.log2(expr[young].values.astype(float) + xs.PSEUDO)
    om = np.log2(expr[old].values.astype(float) + xs.PSEUDO)
    fc = np.median(om, axis=1) - np.median(ym, axis=1)
    # Per-gene p-values via Mann-Whitney on raw values
    pv = np.ones(expr.shape[0])
    yraw = expr[young].values.astype(float)
    oraw = expr[old].values.astype(float)
    for i in range(expr.shape[0]):
        try:
            _, pv[i] = stats.mannwhitneyu(yraw[i, :], oraw[i, :], alternative="two-sided")
        except Exception:
            pv[i] = 1.0
        if np.isnan(pv[i]):
            pv[i] = 1.0
    return pd.DataFrame({"log2fc": fc, "pvalue": pv}, index=expr.index)


def weighted_fc(expr: pd.DataFrame, young: list[str], old: list[str],
                weights: pd.Series) -> pd.DataFrame:
    """Sigmoid-weighted mean log2FC using purity-z weights."""
    log_expr = np.log2(expr.astype(float) + xs.PSEUDO)
    wy = weights.reindex(young).fillna(weights.mean()).values
    wo = weights.reindex(old).fillna(weights.mean()).values
    log_y = log_expr[young].values
    log_o = log_expr[old].values
    my = np.average(log_y, axis=1, weights=wy)
    mo = np.average(log_o, axis=1, weights=wo)
    fc = mo - my
    pv = np.ones(expr.shape[0])
    yraw = expr[young].values.astype(float)
    oraw = expr[old].values.astype(float)
    for i in range(expr.shape[0]):
        try:
            _, pv[i] = stats.mannwhitneyu(yraw[i, :], oraw[i, :], alternative="two-sided")
        except Exception:
            pv[i] = 1.0
        if np.isnan(pv[i]):
            pv[i] = 1.0
    return pd.DataFrame({"log2fc": fc, "pvalue": pv}, index=expr.index)


def limma_fc(expr: pd.DataFrame, young: list[str], old: list[str],
             purity_z: pd.Series) -> pd.DataFrame:
    """Per-gene linear model: log2_expr ~ stage + purity_z. Use stage coefficient
    as log2FC, with t-statistic p-value."""
    log_expr = np.log2(expr.astype(float) + xs.PSEUDO)
    samples = young + old
    stage = np.array([0] * len(young) + [1] * len(old), dtype=float)
    pz = purity_z.reindex(samples).fillna(0.0).values
    # Design: intercept + stage + purity_z
    X = np.column_stack([np.ones(len(samples)), stage, pz])
    y_mat = log_expr[samples].values  # genes x samples
    # OLS in vectorized form: beta = (X'X)^-1 X' Y
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y_mat.T  # 3 x genes
    resid = y_mat.T - X @ beta  # samples x genes
    df = len(samples) - X.shape[1]
    sigma2 = (resid ** 2).sum(axis=0) / df  # genes
    se_stage = np.sqrt(sigma2 * XtX_inv[1, 1])  # SE for stage coef
    fc = beta[1]  # stage coefficient = log2FC
    t = np.divide(fc, se_stage, out=np.zeros_like(se_stage), where=se_stage > 0)
    pv = 2 * stats.t.sf(np.abs(t), df=df)
    pv = np.where(np.isfinite(pv), pv, 1.0)
    return pd.DataFrame({"log2fc": fc, "pvalue": pv}, index=expr.index)


def analyse_weighted(pig_tissue: str, human_tissue: str, method: str,
                     ortho: pd.DataFrame, bins: str) -> dict:
    BINS = {
        "default": (("newborn","infant","toddler"),
                    ("youngAdult","youngMidAge","olderMidAge","senior","Senior")),
        "extended_old": (("newborn","infant","toddler"),
                         ("teenager","oldTeenager","youngAdult","youngMidAge",
                          "olderMidAge","senior","Senior")),
    }
    y_set, o_set = BINS[bins]
    xs.HUMAN_YOUNG = set(y_set)
    xs.HUMAN_OLD = set(o_set)

    print(f"\n=== {pig_tissue} vs {human_tissue} | method={method} bins={bins} ===")

    h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, human_tissue)
    print(f"  Human: young n={len(h_young)}, old n={len(h_old)}")
    p_expr, p_young, p_old = xs.load_pig_tissue(pig_tissue)
    print(f"  Pig: young n={len(p_young)}, old n={len(p_old)}")

    # Compute purity score on pig samples (only used by weighted/limma)
    purity_z = pd.Series(0.0, index=p_expr.columns)
    if method in ("weighted", "limma"):
        sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
        panel = TISSUE_MARKERS.get(pig_tissue, [])
        ids = [sym2id[s.upper()] for s in panel if s.upper() in sym2id]
        if ids:
            candidate = sorted(set(p_young) | set(p_old))
            score = compute_marker_score(p_expr[candidate], ids)
            # Z-score within each stage group
            young_z = (score.reindex(p_young) - score.reindex(p_young).mean()) / \
                      max(score.reindex(p_young).std(ddof=0), 1e-6)
            old_z = (score.reindex(p_old) - score.reindex(p_old).mean()) / \
                    max(score.reindex(p_old).std(ddof=0), 1e-6)
            purity_z.loc[p_young] = young_z.values
            purity_z.loc[p_old] = old_z.values

    # Pig FC by chosen method
    if method == "median":
        p_fc = median_fc(p_expr, p_young, p_old)
    elif method == "weighted":
        weights = 1.0 / (1.0 + np.exp(-purity_z))  # sigmoid
        p_fc = weighted_fc(p_expr, p_young, p_old, weights)
    elif method == "limma":
        p_fc = limma_fc(p_expr, p_young, p_old, purity_z)
    else:
        p_fc = xs.compute_fc_pvals(p_expr, p_young, p_old, use_ttest=False)

    # Human FC always uses the published method (we don't have purity on human)
    h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index) &
               ortho["human_gene_id"].isin(h_fc.index)].copy()
    rows = []
    for _, r in o_.iterrows():
        rows.append({
            "gene_symbol": r["symbol"],
            "pig_gene_id": r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig": p_fc.loc[r["pig_gene_id"], "log2fc"],
            "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
            "p_pig": p_fc.loc[r["pig_gene_id"], "pvalue"],
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
    ].copy()
    pig_anchored = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    def _stats(df_: pd.DataFrame, label: str) -> dict:
        if len(df_) < 5:
            return {"label": label, "n_genes": int(len(df_)), "pearson_r": None,
                    "pearson_p": None, "ci_low": None, "ci_high": None,
                    "directional_concordance": None}
        x = df_["log2fc_pig"].values
        y = df_["log2fc_human"].values
        r, p = stats.pearsonr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        pct = 100.0 * nc / len(df_)
        _, lo, hi = xs.bootstrap_pearson_ci(x, y)
        return {"label": label, "n_genes": int(len(df_)),
                "pearson_r": float(r), "pearson_p": float(p),
                "ci_low": float(lo) if not np.isnan(lo) else None,
                "ci_high": float(hi) if not np.isnan(hi) else None,
                "directional_concordance": float(pct)}

    s_stats = _stats(strict, "strict_FDR_both")
    p_stats = _stats(pig_anchored, "pig_anchored")
    print(f"  strict        n={s_stats['n_genes']} r={s_stats['pearson_r']} "
          f"dc={s_stats['directional_concordance']}")
    print(f"  pig-anchored  n={p_stats['n_genes']} r={p_stats['pearson_r']} "
          f"dc={p_stats['directional_concordance']}")
    return {"tissue": human_tissue, "pig_tissue": pig_tissue, "method": method,
            "bins": bins, "strict": s_stats, "pig_anchored": p_stats,
            "n_orthologs_tested": int(len(merged)),
            "sample_sizes": {"pig_young": len(p_young), "pig_old": len(p_old),
                             "human_young": len(h_young), "human_old": len(h_old)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pig-tissue", required=True)
    ap.add_argument("--human-tissue", required=True)
    ap.add_argument("--method", choices=["median", "weighted", "limma", "baseline"],
                    default="weighted")
    ap.add_argument("--bins", choices=["default", "extended_old"], default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    args = ap.parse_args()
    ortho = xs.build_one_to_one_orthologs()
    res = analyse_weighted(args.pig_tissue, args.human_tissue, args.method, ortho, args.bins)
    merge_path = args.merge_into or str(xs.JSON_OUT)
    base = json.loads(Path(merge_path).read_text())
    base["per_tissue"][args.human_tissue] = res
    base["analysis"] = f"weighted_{args.method}_{args.pig_tissue}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
