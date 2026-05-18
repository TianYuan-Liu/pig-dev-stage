#!/usr/bin/env python3
"""Brain cross-species analysis using AGGREGATED PigGTEx brain subregions.

The published analysis uses only Sub_categories=="Brain" (n=40 with stage info).
The Brain.expr_tpm.txt.gz file actually contains ~419 samples across many
subregions: Brain, Frontal cortex, Hippocampus, Amygdala, Striatum, Cerebral
cortex, Hypothalamus, Pituitary, Cerebellum, etc.

Cardoso-Moreira "Brain" samples are forebrain (cerebrum minus cerebellum).
A biologically defensible aggregation for matching that:
  - Brain (cerebrum/whole brain)                 -> include
  - Frontal cortex, Cerebral cortex, Frontal lobe -> include (cortical)
  - Hippocampus, Amygdala, Striatum             -> include (limbic/subcortical
                                                   forebrain)
  - Hypothalamus                                 -> exclude (diencephalon, may
                                                   bias toward neuroendocrine)
  - Pituitary, Pineal, Choroid plexus            -> exclude (non-neural / very
                                                   specialised secretory tissue)
  - Trigeminal ganglia                           -> exclude (peripheral nervous
                                                   system)
  - Cerebellum                                   -> exclude (hindbrain; CM has
                                                   separate Cerebellum tissue)
  - Fetal brain                                  -> exclude (prenatal)

Usage:
  python autoresearch/run_brain_aggregated.py --aggregation forebrain \\
         --method baseline --bins default --out OUT.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
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
    select_keep_stratified,
)
from autoresearch.run_weighted_fc import median_fc  # noqa: E402

AGGREGATIONS = {
    "brain_only": ["Brain"],  # the published filter
    "cortical": ["Brain", "Frontal cortex", "Cerebral cortex", "Frontal lobe",
                 "Cerebrum"],
    "forebrain": ["Brain", "Frontal cortex", "Cerebral cortex", "Frontal lobe",
                  "Cerebrum", "Hippocampus", "Amygdala", "Striatum region"],
    "forebrain_plus_hypo": ["Brain", "Frontal cortex", "Cerebral cortex",
                            "Frontal lobe", "Cerebrum", "Hippocampus", "Amygdala",
                            "Striatum region", "Hypothalamus"],
}


def load_pig_brain_aggregated(tissue_classes: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load PigGTEx Brain TPM file but include samples from multiple subregions."""
    meta = pd.read_csv(xs.PIG_META).rename(columns={
        "BioSample": "Sample_ID", "Tissue class": "Tissue",
        "Main categories": "Tissue_Main", "Sub categories": "Sub_categories",
    }).set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(xs.parse_age_days)
    meta["Stage"] = meta["Age_Days"].apply(xs.days_to_stage)
    sub_meta = meta[meta["Tissue"].isin(tissue_classes)].copy()

    tpm_path = xs.PIG_TPM_DIR / "Brain.expr_tpm.txt.gz"
    with gzip.open(tpm_path, "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)

    young = [s for s in sub_meta[sub_meta["Stage"].isin(xs.PIG_YOUNG)].index
             if s in expr.columns]
    old = [s for s in sub_meta[sub_meta["Stage"].isin(xs.PIG_OLD)].index
           if s in expr.columns]
    return expr, young, old


def analyse(args) -> dict:
    BINS = {
        "default": (("newborn","infant","toddler"),
                    ("youngAdult","youngMidAge","olderMidAge","senior","Senior")),
        "extended_old": (("newborn","infant","toddler"),
                         ("teenager","oldTeenager","youngAdult","youngMidAge",
                          "olderMidAge","senior","Senior")),
        "developmental": (("newborn","infant","toddler"),
                          ("youngAdult","youngMidAge")),
        "developmental_strict": (("newborn","infant","toddler"),
                                  ("youngAdult",)),
    }
    y_set, o_set = BINS[args.bins]
    xs.HUMAN_YOUNG = set(y_set)
    xs.HUMAN_OLD = set(o_set)

    tissue_classes = AGGREGATIONS[args.aggregation]
    print(f"\n=== Brain aggregated ({args.aggregation}) | method={args.method} bins={args.bins} ===")
    print(f"  Tissue classes: {tissue_classes}")

    h_expr, h_young, h_old = xs.load_human_tissue(xs.HUMAN_RPKM, "Brain")
    print(f"  Human Brain: young n={len(h_young)}, old n={len(h_old)}")

    p_expr, p_young, p_old = load_pig_brain_aggregated(tissue_classes)
    print(f"  Pig brain (aggregated): young n={len(p_young)}, old n={len(p_old)}")
    candidate = sorted(set(p_young) | set(p_old))

    # Apply purity QC if requested
    qc_info: dict = {}
    if args.method.startswith("purity"):
        drop_pct = args.drop_pct
        sym2id = load_pig_symbol_map(xs.PIG_SYM_CACHE)
        panel = TISSUE_MARKERS["Brain"]
        ids = [sym2id[s.upper()] for s in panel if s.upper() in sym2id]
        score = compute_marker_score(p_expr[candidate], ids)
        kept, dropped = select_keep_stratified(
            score, {"young": p_young, "old": p_old}, drop_pct, drop_low=True)
        p_young = [s for s in p_young if s in kept]
        p_old = [s for s in p_old if s in kept]
        qc_info = {"drop_pct": drop_pct, "n_dropped": len(dropped),
                   "panel_present": len(ids)}
        print(f"  After purity QC drop {drop_pct}%: young n={len(p_young)}, old n={len(p_old)} (dropped {len(dropped)})")

    if len(p_young) < 3 or len(p_old) < 3:
        return {"skipped": "insufficient_after_qc", "qc": qc_info}

    # Compute pig FC
    if "median" in args.method:
        p_fc = median_fc(p_expr, p_young, p_old)
    else:
        p_fc = xs.compute_fc_pvals(p_expr, p_young, p_old, use_ttest=False)
    h_fc = xs.compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    ortho = xs.build_one_to_one_orthologs()
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
    pa = merged[
        (merged["fdr_pig"] < xs.FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > xs.FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > xs.FC_THRESHOLD)
    ].copy()

    def _stats(df_, label):
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
    p_stats = _stats(pa, "pig_anchored")
    print(f"  strict        n={s_stats['n_genes']} r={s_stats['pearson_r']} dc={s_stats['directional_concordance']}")
    print(f"  pig-anchored  n={p_stats['n_genes']} r={p_stats['pearson_r']} dc={p_stats['directional_concordance']}")
    return {"tissue": "Brain", "pig_tissue": f"Aggregated_{args.aggregation}",
            "method": args.method, "bins": args.bins, "qc": qc_info,
            "n_orthologs_tested": int(len(merged)),
            "sample_sizes": {"pig_young": len(p_young), "pig_old": len(p_old),
                             "human_young": len(h_young), "human_old": len(h_old)},
            "strict": s_stats, "pig_anchored": p_stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregation", choices=list(AGGREGATIONS.keys()), default="forebrain")
    ap.add_argument("--method", choices=["baseline", "purity", "median", "purity_median"],
                    default="baseline")
    ap.add_argument("--bins", choices=["default", "extended_old", "developmental", "developmental_strict"], default="developmental")
    ap.add_argument("--drop-pct", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None)
    args = ap.parse_args()
    res = analyse(args)
    merge_path = args.merge_into or str(xs.JSON_OUT)
    base = json.loads(Path(merge_path).read_text())
    base["per_tissue"]["Brain"] = res
    base["analysis"] = f"brain_aggregated_{args.aggregation}_{args.method}_{args.bins}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(base, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
