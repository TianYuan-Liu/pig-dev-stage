#!/usr/bin/env python3
"""Generic purity-based cross-species analysis: parameterized by tissue and
canonical marker panel. Replaces the brain-specific QC script with a unified
implementation.

Markers per tissue (canonical identity markers; samples low in these are
removed as low-purity). The choice of markers is BLIND to the cross-species
outcome and based on standard cell-type identity literature.

Usage:
  python autoresearch/run_purity_generic.py --pig-tissue Liver --human-tissue Liver \\
         --method stratified_purity --drop-pct 30 \\
         --out review/analyses/results/cross_species_liver_purity30.json
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

# Canonical tissue-identity marker panels (samples with LOW expression are
# considered low-purity).
TISSUE_MARKERS = {
    "Brain": ["NEFL", "NEFM", "NEFH", "MAP2", "SYN1", "SYP", "SNAP25", "STMN2",
              "GFAP", "OLIG2", "MBP", "PLP1", "VGAT", "VGLUT1", "GAD1",
              "SLC17A7", "RBFOX3", "TUBB3", "DLG4", "GRIN1"],
    "Liver": ["ALB", "TTR", "AFP", "HNF4A", "HNF1A", "HNF1B", "CYP3A4",
              "CYP2E1", "CYP7A1", "F2", "F7", "F9", "APOA1", "APOA2", "APOB",
              "ASS1", "OTC", "CPS1", "ARG1", "G6PC", "PCK1", "SERPINA1"],
    "Heart": ["MYH6", "MYH7", "TNNT2", "TNNI3", "TNNC1", "ACTC1", "MYL2",
              "MYL7", "NPPA", "NPPB", "NKX2-5", "TBX5", "GATA4", "MEF2C",
              "DES", "CSRP3"],
    "Testis": ["PRM1", "PRM2", "TNP1", "TNP2", "SYCP3", "SYCP1", "DDX4",
               "DAZL", "BOLL", "STRA8", "PIWIL1", "SOHLH1", "SOHLH2"],
    "Kidney": ["NPHS1", "NPHS2", "PODXL", "WT1", "PAX2", "PAX8", "AQP1",
               "AQP2", "SLC22A6", "SLC22A8", "UMOD", "CDH16", "GATA3"],
    "Muscle": ["MYH1", "MYH2", "MYH7", "ACTN3", "MB", "DES", "MYOG", "MYOD1",
               "CKM", "MYL1", "TNNT3", "TNNI2", "MYBPC1", "TNNC2"],
    "Lung": ["SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "SFTPD", "FOXA2", "NKX2-1",
             "SCGB1A1", "SCGB3A2", "MUC5B", "AGER", "AQP5", "FOXJ1"],
}


def load_pig_symbol_map(cache: Path) -> dict[str, str]:
    d = json.loads(cache.read_text())
    sym2id: dict[str, str] = {}
    for pid, sym in d.items():
        if not sym:
            continue
        sym2id.setdefault(sym.upper(), pid)
    return sym2id


def compute_marker_score(expr: pd.DataFrame, marker_pig_ids: list[str]) -> pd.Series:
    present = [g for g in marker_pig_ids if g in expr.index]
    if not present:
        raise SystemExit("ERROR: no marker IDs found in expression matrix")
    sub = np.log2(expr.loc[present].astype(float).values + 1.0)
    return pd.Series(sub.mean(axis=0), index=expr.columns, name="marker_score")


def select_keep_stratified(score: pd.Series, per_group: dict[str, list[str]],
                            drop_pct: float, drop_low: bool = True
                            ) -> tuple[list[str], list[str]]:
    """Within each group, drop drop_pct of samples. If drop_low=True drop the
    LOWEST-purity samples (i.e. keep the highest); if False drop the highest.
    """
    kept_all, dropped_all = [], []
    for samples in per_group.values():
        sg = score.reindex(samples).dropna()
        if sg.empty:
            kept_all.extend(samples)
            continue
        thresh = float(np.percentile(sg.values, drop_pct)) if drop_low \
                 else float(np.percentile(sg.values, 100.0 - drop_pct))
        if drop_low:
            kept_all.extend(sg.index[sg.values >= thresh].tolist())
            dropped_all.extend(sg.index[sg.values < thresh].tolist())
        else:
            kept_all.extend(sg.index[sg.values <= thresh].tolist())
            dropped_all.extend(sg.index[sg.values > thresh].tolist())
    return kept_all, dropped_all


def analyse_with_purity(pig_tissue: str, human_tissue: str, ortho: pd.DataFrame,
                        method: str, drop_pct: float, bin_set: str
                        ) -> dict:
    # Set bins via patching the module
    BINS = {
        "default": (("newborn","infant","toddler"),
                    ("youngAdult","youngMidAge","olderMidAge","senior","Senior")),
        "extended_old": (("newborn","infant","toddler"),
                         ("teenager","oldTeenager","youngAdult","youngMidAge",
                          "olderMidAge","senior","Senior")),
    }
    y, o = BINS[bin_set]
    xs.HUMAN_YOUNG = set(y)
    xs.HUMAN_OLD = set(o)

    print(f"\n=== {pig_tissue} (pig) vs {human_tissue} (human) | "
          f"method={method} drop={drop_pct}% bins={bin_set} ===")

    # ---- HUMAN ----
    h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, human_tissue)
    print(f"  Human: young n={len(h_young)}, old n={len(h_old)}")
    if len(h_young) < 3 or len(h_old) < 3:
        return {"tissue": human_tissue, "skipped": "insufficient_human"}
    h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    # ---- PIG ----
    p_expr, p_young, p_old = xs.load_pig_tissue(pig_tissue)
    print(f"  Pig: young n={len(p_young)}, old n={len(p_old)}")
    candidate = sorted(set(p_young) | set(p_old))

    # ---- Purity filter ----
    qc_info: dict = {}
    if method == "none":
        p_young_f, p_old_f, dropped = p_young, p_old, []
    elif method == "stratified_purity":
        sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
        panel = TISSUE_MARKERS.get(pig_tissue)
        if panel is None:
            raise SystemExit(f"No marker panel defined for {pig_tissue}")
        ids = [sym2id[s.upper()] for s in panel if s.upper() in sym2id]
        if not ids:
            raise SystemExit(f"No marker IDs found for {pig_tissue}")
        score = compute_marker_score(p_expr[candidate], ids)
        kept, dropped = select_keep_stratified(
            score, {"young": p_young, "old": p_old}, drop_pct, drop_low=True)
        p_young_f = [s for s in p_young if s in kept]
        p_old_f = [s for s in p_old if s in kept]
        qc_info = {"panel_size": len(panel), "panel_present": len(ids),
                   "n_dropped_young": len([s for s in dropped if s in p_young]),
                   "n_dropped_old": len([s for s in dropped if s in p_old])}
    else:
        raise SystemExit(f"unknown method: {method}")
    print(f"  After QC: young n={len(p_young_f)}, old n={len(p_old_f)} "
          f"(dropped {len(dropped)})")

    if len(p_young_f) < 3 or len(p_old_f) < 3:
        return {"tissue": human_tissue, "skipped": "insufficient_after_qc",
                "qc": qc_info}

    p_fc = xs.compute_fc_pvals(p_expr, p_young_f, p_old_f, use_ttest=False)

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
                    "directional_concordance": None, "n_concordant":
                    int((np.sign(df_["log2fc_pig"]) == np.sign(df_["log2fc_human"])).sum())
                    if len(df_) > 0 else 0}
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
                "directional_concordance": float(pct), "n_concordant": nc}

    s_stats = _stats(strict, "strict_FDR_both")
    p_stats = _stats(pig_anchored, "pig_anchored")
    print(f"  strict       n={s_stats['n_genes']} r={s_stats['pearson_r']} dc={s_stats['directional_concordance']}")
    print(f"  pig-anchored n={p_stats['n_genes']} r={p_stats['pearson_r']} dc={p_stats['directional_concordance']}")
    return {
        "tissue": human_tissue, "pig_tissue": pig_tissue,
        "qc_method": method, "drop_pct": drop_pct, "bins": bin_set,
        "qc": qc_info,
        "n_orthologs_tested": int(len(merged)),
        "sample_sizes": {"pig_young": len(p_young_f), "pig_old": len(p_old_f),
                         "human_young": len(h_young), "human_old": len(h_old)},
        "strict": s_stats, "pig_anchored": p_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pig-tissue", required=True)
    ap.add_argument("--human-tissue", required=True)
    ap.add_argument("--method", choices=["none", "stratified_purity"], default="stratified_purity")
    ap.add_argument("--drop-pct", type=float, default=30.0)
    ap.add_argument("--bins", choices=["default", "extended_old"], default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None,
                    help="Existing alltis JSON to merge tissue result into (default: cross_species_all_tissues.json)")
    args = ap.parse_args()
    ortho = xs.build_one_to_one_orthologs()
    res = analyse_with_purity(args.pig_tissue, args.human_tissue, ortho,
                              args.method, args.drop_pct, args.bins)
    merge_path = args.merge_into or str(xs.JSON_OUT)
    base = json.loads(Path(merge_path).read_text())
    base["per_tissue"][args.human_tissue] = res
    base["analysis"] = f"purity_generic_{args.pig_tissue}_{args.method}_drop{int(args.drop_pct)}_bins{args.bins}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
