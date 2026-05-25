#!/usr/bin/env python3
"""Pig (PigGTEx) vs Human (LungMAP LMEX0000003691) lung cross-species comparison.

LungMAP dataset: Bandyopadhyay 2024 bulk RNA-seq of pediatric lung
(https://lungmap.net dataset LMEX0000003691). Cardoso-Moreira 2019 does NOT
have lung, so this is the standard developmental human lung reference.

Data layout (LungMAP):
  - 151 samples x 56,870 genes, HGNC symbol gene index
  - Derivative Type (cell prep): BPS (flash-frozen whole tissue),
    EPI, END, MES, MIC, PMX (collagenase-dissociated sorted populations)
  - Age Cohort: Neonate (<=30d), Infant (30d-1y), Child (1-11y), Adult (20+y)

For an apples-to-apples comparison to PigGTEx whole-tissue lung TPM we use
only the BPS samples. Adult is dropped to keep the analysis developmental.

Pig YOUNG/OLD defaults to xs.PIG_YOUNG / xs.PIG_OLD (Infant+Early-childhood
vs Post-pubertal+Adult).

Usage:
  python autoresearch/run_lung_cross_species.py \\
         --cell-pop BPS --drop-adult \\
         --out review/analyses/results/cross_species_lung.json
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
from autoresearch.run_weighted_fc import (  # noqa: E402
    weighted_fc as _weighted_fc, limma_fc as _limma_fc,
)

LUNGMAP_DIR = ROOT / "data" / "lungmap"
RAW_PATH = LUNGMAP_DIR / "LMEX0000003691_raw_counts.txt.gz"
NORM_PATH = LUNGMAP_DIR / "LMEX0000003691_normalized_counts.txt.gz"
META_PATH = LUNGMAP_DIR / "LMEX0000003691_sample_metadata.xlsx"

# Eisenberg-Levanon-style housekeeping prefixes / families to drop in --filter-housekeeping.
# Source: HGNC nomenclature (ribosomal, mitochondrial, common housekeepers).
HK_PREFIXES = ("RPL", "RPS", "MRPL", "MRPS", "MT-")
HK_GENES = {  # explicit canonical housekeeping HGNC symbols
    "ACTB", "GAPDH", "B2M", "HPRT1", "HMBS", "PGK1", "TBP", "UBC", "GUSB",
    "PPIA", "PUM1", "TFRC", "YWHAZ", "PSMB4", "PSMD1", "EIF4A2", "EEF1A1",
}

# Cached anatomical-structure-development GO:0048856 ortholog set is built
# lazily on first request. If runtime fetch fails, falls back to a curated
# minimal list of canonical developmental TFs/signalling genes.
GO_DEV_CACHE_PATH = ROOT / "autoresearch" / "go_development_genes.txt"
GO_DEV_FALLBACK = {  # used only if the cache file is absent
    "SOX2", "SOX9", "FOXA2", "FOXA1", "NKX2-1", "NKX2-5", "GATA4", "GATA6",
    "PAX6", "PAX8", "WNT3A", "WNT5A", "WNT7B", "SHH", "FGF10", "FGF7",
    "TGFB1", "TGFB2", "BMP4", "BMP7", "HOXA5", "HOXA9", "HOXB13", "TBX2",
    "TBX3", "TBX4", "TBX5", "MEIS1", "MEIS2", "MYOG", "MYOD1", "MEF2C",
    "RUNX1", "RUNX2", "RUNX3", "SPRY2", "SPRY4", "ID1", "ID2", "ID3",
    "NOTCH1", "NOTCH2", "NOTCH3", "DLL1", "JAG1", "HES1", "HEY1", "HEY2",
    "ETV1", "ETV2", "ETV4", "ETV5", "ELF3", "ELF5", "EHF",
    "SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "SFTPD", "SCGB1A1", "SCGB3A2",
    "AGER", "AQP5", "FOXJ1", "MUC5B", "ELANE", "MPO", "TGFBR3",
}


def filter_genes(merged: pd.DataFrame,
                 filter_housekeeping: bool,
                 restrict_go_dev: bool,
                 ortho: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply optional gene-set filters to the merged ortholog FC table.
    Returns the filtered DataFrame (or merged unchanged if no filters set)."""
    if not filter_housekeeping and not restrict_go_dev:
        return merged
    keep_mask = pd.Series(True, index=merged.index)
    if filter_housekeeping:
        sym = merged["gene_symbol"].astype(str).str.upper()
        is_hk = sym.isin(HK_GENES) | sym.str.startswith(HK_PREFIXES)
        keep_mask &= ~is_hk
        print(f"  Filter housekeeping: dropping {int(is_hk.sum())} genes "
              f"(RP*/MT-*/canonical HK)")
    if restrict_go_dev:
        if GO_DEV_CACHE_PATH.exists():
            dev_set = {ln.strip().upper()
                       for ln in GO_DEV_CACHE_PATH.read_text().splitlines()
                       if ln.strip()}
        else:
            dev_set = GO_DEV_FALLBACK
            print(f"  WARNING: {GO_DEV_CACHE_PATH} missing, using "
                  f"{len(dev_set)}-gene fallback panel")
        sym = merged["gene_symbol"].astype(str).str.upper()
        keep_mask &= sym.isin({s.upper() for s in dev_set})
        print(f"  Restrict to GO development set ({len(dev_set)} genes): "
              f"keeping {int(keep_mask.sum())} orthologs")
    out = merged.loc[keep_mask].copy()
    out = xs.add_fdr(out, "p_pig", "fdr_pig")
    out = xs.add_fdr(out, "p_human", "fdr_human")
    return out


def compute_pig_purity_z(p_expr: pd.DataFrame, p_young: list[str],
                          p_old: list[str]) -> pd.Series:
    """Compute per-sample purity z-score from lung markers, z-normalised
    within each YOUNG/OLD group (so the z reflects deviation from the group
    mean, not the stage signal)."""
    sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
    panel = TISSUE_MARKERS["Lung"]
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


def set_pig_stage_bins(young_stages: list[str] | None,
                       old_stages: list[str] | None) -> tuple[set, set]:
    """Patch xs.PIG_YOUNG / xs.PIG_OLD in place. Returns the original values
    so the caller can restore them."""
    orig_young = set(xs.PIG_YOUNG)
    orig_old = set(xs.PIG_OLD)
    if young_stages is not None:
        xs.PIG_YOUNG = set(young_stages)
    if old_stages is not None:
        xs.PIG_OLD = set(old_stages)
    return orig_young, orig_old


def load_human_lungmap(cell_pop: str | None, drop_adult: bool
                       ) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Load LungMAP normalized counts, filter to chosen cell pop, return
    expression matrix (symbol-indexed), young IDs, old IDs, sample-count info."""
    meta = pd.read_excel(META_PATH)
    meta = meta[meta["Tissue Type"] == "Lung"].copy()
    if cell_pop:
        meta = meta[meta["Derivative Type"] == cell_pop].copy()
    if drop_adult:
        meta = meta[meta["Age Cohort"] != "5. Adult (20+ years)"].copy()

    young_mask = meta["Age Cohort"] == "1. Neonate (up to 30 days)"
    old_mask = meta["Age Cohort"].isin([
        "2. Infant (> 30 days and < 1 year)",
        "3. Child (>= 1 year and < 11 years)",
    ])
    young_ids = meta.loc[young_mask, "Sample Inventory ID"].tolist()
    old_ids = meta.loc[old_mask, "Sample Inventory ID"].tolist()

    expr = pd.read_csv(NORM_PATH, sep="\t", index_col=0)
    keep = [s for s in (young_ids + old_ids) if s in expr.columns]
    young_ids = [s for s in young_ids if s in expr.columns]
    old_ids = [s for s in old_ids if s in expr.columns]
    expr = expr[keep]
    info = {
        "cell_pop": cell_pop,
        "drop_adult": drop_adult,
        "young_donors": meta.loc[young_mask, "Donor Id"].tolist(),
        "old_donors": meta.loc[old_mask, "Donor Id"].tolist(),
        "young_ages": meta.loc[young_mask, "Age Of Donor"].tolist(),
        "old_ages": meta.loc[old_mask, "Age Of Donor"].tolist(),
    }
    return expr, young_ids, old_ids, info


def remap_human_expr_to_ensg(h_expr_sym: pd.DataFrame,
                              ortho: pd.DataFrame) -> pd.DataFrame:
    """Rename rows of a symbol-indexed expression matrix to ENSG IDs via the
    one-to-one ortholog table's `symbol` column. Drop rows with no symbol match
    and duplicate-symbol rows (keep first)."""
    sym2ensg = (ortho.dropna(subset=["symbol", "human_gene_id"])
                .drop_duplicates(subset="symbol", keep="first")
                .set_index("symbol")["human_gene_id"])
    h_expr_sym = h_expr_sym[~h_expr_sym.index.duplicated(keep="first")]
    matched = h_expr_sym.index.intersection(sym2ensg.index)
    h = h_expr_sym.loc[matched].copy()
    h.index = sym2ensg.loc[matched].values
    h = h[~h.index.duplicated(keep="first")]
    return h


def median_fc(expr: pd.DataFrame, young: list[str], old: list[str]) -> pd.DataFrame:
    """Median-based log2FC: median(old) - median(young) on log2 scale."""
    ym = np.log2(expr[young].values.astype(float) + xs.PSEUDO)
    om = np.log2(expr[old].values.astype(float) + xs.PSEUDO)
    fc = np.median(om, axis=1) - np.median(ym, axis=1)
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


def load_human_lungmap_pseudobulk(drop_adult: bool, sorted_cells: tuple
                                   = ("EPI", "END", "MES", "MIC")
                                   ) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Aggregate sorted-cell samples per donor into a single pseudo-bulk profile
    by summing raw-count-equivalent values across cell populations. Returns a
    donor-level expression matrix."""
    meta = pd.read_excel(META_PATH)
    meta = meta[(meta["Tissue Type"] == "Lung")
                & (meta["Derivative Type"].isin(sorted_cells))].copy()
    if drop_adult:
        meta = meta[meta["Age Cohort"] != "5. Adult (20+ years)"].copy()
    expr = pd.read_csv(NORM_PATH, sep="\t", index_col=0)
    meta = meta[meta["Sample Inventory ID"].isin(expr.columns)]
    # Aggregate sum per donor
    donor_expr = {}
    donor_age = {}
    donor_age_cohort = {}
    for donor, sub in meta.groupby("Donor Id"):
        if len(sub) < 2:
            continue  # need at least 2 cell pops for pseudo-bulk
        donor_expr[donor] = expr[sub["Sample Inventory ID"].tolist()].sum(axis=1)
        donor_age[donor] = sub["Age Of Donor"].iloc[0]
        donor_age_cohort[donor] = sub["Age Cohort"].iloc[0]
    pseudo = pd.DataFrame(donor_expr)
    young = [d for d, c in donor_age_cohort.items()
             if c == "1. Neonate (up to 30 days)"]
    old = [d for d, c in donor_age_cohort.items()
           if c in ("2. Infant (> 30 days and < 1 year)",
                    "3. Child (>= 1 year and < 11 years)")]
    info = {"cell_pop": f"pseudobulk_sum_{','.join(sorted_cells)}",
            "drop_adult": drop_adult,
            "young_donors": young, "old_donors": old,
            "young_ages": [donor_age[d] for d in young],
            "old_ages": [donor_age[d] for d in old]}
    return pseudo, young, old, info


def analyse(cell_pop: str | None, drop_adult: bool, out_path: Path,
            merge_into: Path | None = None,
            method: str = "baseline",
            old_only_child: bool = False,
            pseudobulk: bool = False,
            pig_young_stages: list[str] | None = None,
            pig_old_stages: list[str] | None = None,
            filter_housekeeping: bool = False,
            min_pig_tpm: float = 0.0,
            min_human_norm: float = 0.0,
            restrict_go_dev: bool = False) -> dict:
    print(f"=== Pig Lung (PigGTEx) vs Human Lung (LungMAP) ===")
    print(f"  cell_pop={cell_pop}, drop_adult={drop_adult}, method={method}, "
          f"old_only_child={old_only_child}, pseudobulk={pseudobulk}")
    if pig_young_stages or pig_old_stages:
        print(f"  pig_young_stages={pig_young_stages} "
              f"pig_old_stages={pig_old_stages}")
    if filter_housekeeping or restrict_go_dev or min_pig_tpm or min_human_norm:
        print(f"  filter_housekeeping={filter_housekeeping} "
              f"restrict_go_dev={restrict_go_dev} "
              f"min_pig_tpm={min_pig_tpm} min_human_norm={min_human_norm}")

    ortho = xs.build_one_to_one_orthologs()

    if pseudobulk:
        h_expr_sym, h_young, h_old, info = load_human_lungmap_pseudobulk(drop_adult)
    else:
        h_expr_sym, h_young, h_old, info = load_human_lungmap(cell_pop, drop_adult)
    print(f"  Human LungMAP {cell_pop or ('pseudobulk' if pseudobulk else 'ALL')}: "
          f"young n={len(h_young)} ages={info['young_ages']}; "
          f"old n={len(h_old)} ages={info['old_ages']}")

    if old_only_child and not pseudobulk:
        # Drop infant (4-11mo) from OLD; keep only child (1y+) for sharper contrast
        meta = pd.read_excel(META_PATH)
        keep_old = meta[(meta["Sample Inventory ID"].isin(h_old))
                        & (meta["Age Cohort"]
                           == "3. Child (>= 1 year and < 11 years)")
                        ]["Sample Inventory ID"].tolist()
        h_old = keep_old
        print(f"  After old_only_child: old n={len(h_old)}")

    if len(h_young) < 3 or len(h_old) < 3:
        return {"skipped": "insufficient_human_samples",
                "n_young": len(h_young), "n_old": len(h_old), "info": info}

    h_expr = remap_human_expr_to_ensg(h_expr_sym, ortho)
    print(f"  Mapped HGNC->ENSG: kept {h_expr.shape[0]} genes ({h_expr.shape[1]} samples)")

    # Apply min-expression gene filter on human side (after symbol remap)
    if min_human_norm > 0:
        med_h = h_expr.median(axis=1)
        keep_h = med_h >= min_human_norm
        h_expr = h_expr.loc[keep_h]
        print(f"  Human min-norm filter (median>={min_human_norm}): "
              f"kept {h_expr.shape[0]}/{len(keep_h)} genes")

    # Pig stage bins (re-define YOUNG/OLD if requested)
    orig_young, orig_old = set_pig_stage_bins(pig_young_stages, pig_old_stages)
    try:
        p_expr, p_young, p_old = xs.load_pig_tissue("Lung")
    finally:
        xs.PIG_YOUNG = orig_young
        xs.PIG_OLD = orig_old
    print(f"  Pig Lung (PigGTEx) [YOUNG={sorted(set(pig_young_stages) if pig_young_stages else orig_young)} "
          f"OLD={sorted(set(pig_old_stages) if pig_old_stages else orig_old)}]: "
          f"young n={len(p_young)}, old n={len(p_old)}")

    if len(p_young) < 3 or len(p_old) < 3:
        return {"skipped": "insufficient_pig_samples",
                "p_young": len(p_young), "p_old": len(p_old)}

    # Apply min-pig-tpm filter
    if min_pig_tpm > 0:
        med_p = p_expr.median(axis=1)
        keep_p = med_p >= min_pig_tpm
        p_expr = p_expr.loc[keep_p]
        print(f"  Pig min-tpm filter (median>={min_pig_tpm}): "
              f"kept {p_expr.shape[0]}/{len(keep_p)} genes")

    # Compute pig FC by method
    purity_z = None
    if method in ("weighted", "limma"):
        purity_z = compute_pig_purity_z(p_expr, p_young, p_old)
        print(f"  Pig purity-z: n_panel_present="
              f"{(purity_z != 0).sum()}/{len(purity_z)}, "
              f"range=[{purity_z.min():.2f}, {purity_z.max():.2f}]")

    if method == "median":
        p_fc = median_fc(p_expr, p_young, p_old)
    elif method == "weighted":
        weights = 1.0 / (1.0 + np.exp(-purity_z))
        p_fc = _weighted_fc(p_expr, p_young, p_old, weights)
    elif method == "limma":
        p_fc = _limma_fc(p_expr, p_young, p_old, purity_z)
    else:
        p_fc = xs.compute_fc_pvals(p_expr, p_young, p_old, use_ttest=False)

    # Human FC: median works best at small n, used as default with median/weighted/limma
    if method in ("median", "weighted", "limma"):
        h_fc = median_fc(h_expr, h_young, h_old)
    else:
        h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)
    print(f"  Human FC ({method} -> median for soft): median |log2FC|={h_fc['log2fc'].abs().median():.3f}, "
          f"|log2FC|>0.5 in {(h_fc['log2fc'].abs() > 0.5).sum()} genes")

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
    print(f"  Orthologs tested (before gene filters): {len(merged)}")

    # Apply gene-set filters (housekeeping / GO development restriction)
    if filter_housekeeping or restrict_go_dev:
        merged = filter_genes(merged, filter_housekeeping, restrict_go_dev, ortho)
        print(f"  After gene filters: {len(merged)}")

    strict = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["fdr_human"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()
    pa = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()
    # Also a relaxed FC-only filter (no FDR), in case underpowered human n=4 vs 6
    fc_only = merged[
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    def _stats(df_: pd.DataFrame, label: str) -> dict:
        if len(df_) < 5:
            return {"label": label, "n_genes": int(len(df_)),
                    "pearson_r": None, "pearson_p": None,
                    "spearman_r": None, "spearman_p": None,
                    "ci_low": None, "ci_high": None,
                    "directional_concordance": None}
        x = df_["log2fc_pig"].values
        y = df_["log2fc_human"].values
        r, p = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        pct = 100.0 * nc / len(df_)
        _, lo, hi = xs.bootstrap_pearson_ci(x, y)
        return {"label": label, "n_genes": int(len(df_)),
                "pearson_r": float(r), "pearson_p": float(p),
                "spearman_r": float(sr), "spearman_p": float(sp),
                "ci_low": float(lo) if not np.isnan(lo) else None,
                "ci_high": float(hi) if not np.isnan(hi) else None,
                "directional_concordance": float(pct)}

    s_stats = _stats(strict, "strict_FDR_both")
    p_stats = _stats(pa, "pig_anchored")
    f_stats = _stats(fc_only, "fc_only_no_fdr")
    print(f"  strict        n={s_stats['n_genes']} r={s_stats['pearson_r']} "
          f"sp={s_stats['spearman_r']} dc={s_stats['directional_concordance']}")
    print(f"  pig-anchored  n={p_stats['n_genes']} r={p_stats['pearson_r']} "
          f"sp={p_stats['spearman_r']} dc={p_stats['directional_concordance']}")
    print(f"  fc-only       n={f_stats['n_genes']} r={f_stats['pearson_r']} "
          f"sp={f_stats['spearman_r']} dc={f_stats['directional_concordance']}")

    res = {"tissue": "Lung", "pig_tissue": "Lung",
           "human_source": "LungMAP_LMEX0000003691",
           "cell_pop": cell_pop or ("pseudobulk" if pseudobulk else "all"),
           "drop_adult": drop_adult, "method": method,
           "old_only_child": old_only_child, "pseudobulk": pseudobulk,
           "pig_young_stages": sorted(pig_young_stages) if pig_young_stages else sorted(orig_young),
           "pig_old_stages": sorted(pig_old_stages) if pig_old_stages else sorted(orig_old),
           "filter_housekeeping": filter_housekeeping,
           "restrict_go_dev": restrict_go_dev,
           "min_pig_tpm": min_pig_tpm, "min_human_norm": min_human_norm,
           "n_orthologs_tested": int(len(merged)),
           "sample_sizes": {"pig_young": len(p_young), "pig_old": len(p_old),
                            "human_young": len(h_young), "human_old": len(h_old)},
           "strict": s_stats, "pig_anchored": p_stats, "fc_only": f_stats,
           "donor_info": info}

    if merge_into:
        base = json.loads(merge_into.read_text())
        base.setdefault("per_tissue", {})["Lung"] = res
        base["analysis"] = f"lung_lungmap_{res['cell_pop']}_{method}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(base, indent=2))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_path}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-pop", default="BPS",
                    help="LungMAP Derivative Type to use (BPS=whole tissue, "
                         "EPI/END/MES/MIC sorted, or 'ALL' for everything)")
    ap.add_argument("--drop-adult", action="store_true", default=True,
                    help="Drop the 24-year-old adult donor (developmental-only)")
    ap.add_argument("--keep-adult", action="store_true",
                    help="Override --drop-adult and keep the adult sample")
    ap.add_argument("--method",
                    choices=["baseline", "median", "weighted", "limma"],
                    default="baseline",
                    help="FC method. 'weighted' and 'limma' use pig lung purity-z "
                         "(no sample drops; soft methods only).")
    ap.add_argument("--old-only-child", action="store_true",
                    help="Restrict OLD bin to Child (1-11y), dropping Infant")
    ap.add_argument("--pseudobulk", action="store_true",
                    help="Sum EPI+END+MES+MIC per donor for pseudo-whole-tissue")
    ap.add_argument("--pig-young-stages", type=str, default=None,
                    help="Comma-separated list of pig stages for YOUNG bin "
                         "(re-bin, not drop). Default = Infant,Early childhood.")
    ap.add_argument("--pig-old-stages", type=str, default=None,
                    help="Comma-separated list of pig stages for OLD bin "
                         "(re-bin, not drop). Default = Post-pubertal,Adult.")
    ap.add_argument("--filter-housekeeping", action="store_true",
                    help="Drop RPL*/RPS*/MT-*/canonical housekeeping genes "
                         "from the ortholog set.")
    ap.add_argument("--restrict-go-dev", action="store_true",
                    help="Restrict orthologs to GO:0048856 anatomical-structure-"
                         "development. Uses cached gene list or fallback panel.")
    ap.add_argument("--min-pig-tpm", type=float, default=0.0,
                    help="Drop genes with median pig TPM below this threshold.")
    ap.add_argument("--min-human-norm", type=float, default=0.0,
                    help="Drop genes with median human normalised count below this.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--merge-into", type=Path, default=None)
    args = ap.parse_args()
    cell_pop = None if args.cell_pop.upper() == "ALL" else args.cell_pop
    drop_adult = not args.keep_adult
    pig_young = ([s.strip() for s in args.pig_young_stages.split(",")]
                 if args.pig_young_stages else None)
    pig_old = ([s.strip() for s in args.pig_old_stages.split(",")]
               if args.pig_old_stages else None)
    analyse(cell_pop, drop_adult, args.out, args.merge_into,
            method=args.method, old_only_child=args.old_only_child,
            pseudobulk=args.pseudobulk,
            pig_young_stages=pig_young, pig_old_stages=pig_old,
            filter_housekeeping=args.filter_housekeeping,
            min_pig_tpm=args.min_pig_tpm, min_human_norm=args.min_human_norm,
            restrict_go_dev=args.restrict_go_dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
