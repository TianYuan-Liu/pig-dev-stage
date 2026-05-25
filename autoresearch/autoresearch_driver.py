#!/usr/bin/env python3
"""Parameterized autoresearch driver — evaluates ANY cross-species config over
the FROZEN evaluator WITHOUT modifying it.

This is a thin orchestration layer over the immutable ground-truth functions.
It NEVER edits the science; it only varies their *inputs*:

  FROZEN (imported unchanged, never edited here):
    xs.load_pig_tissue        pig sample binning (reads xs.PIG_YOUNG/PIG_OLD)
    apply_uniform_drop        purity-stratified sample drop
    compute_pig_fc            pig log2FC (weighted / median / limma)
    load_dgtex_tissue         human binning (reads stage_mapping_v2 cohorts)
    median_fc                 human log2FC
    xs.add_fdr                BH FDR
    apply_filter              gene filter (the 5 baseline filters)
    compute_stat              r + bootstrap CI + directional concordance

  PARAMETERIZED here (search-space knobs, set before calling the frozen code):
    pig/human stage bins, FC method, drop%, min-TPM filters, ortholog set,
    filter, statistic, tissue panel, per-tissue bin overrides, gene-restriction
    list (e.g. GO:0032502), pig sub-region pooling.

  NEW filters (operate on the merged FC table only — allowed by the search space):
    top_n_pigfc        top-N genes by |log2fc_pig| (no FDR threshold)
    bootstrap_stable   pig_anchored genes that survive in >=80% of pig bootstraps

Output: a single JSON with per-tissue stats and aggregate mean_r/min_r/max_r,
identical to 6 d.p. for the same config (seed=42 throughout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "review" / "analyses"))

import cross_species_all_tissues as xs  # noqa: E402
import autoresearch.stage_mapping_v2 as sm  # noqa: E402
from autoresearch.run_uniform_cross_species import (  # noqa: E402  (FROZEN)
    apply_uniform_drop, compute_pig_fc,
)
from autoresearch.run_weighted_fc import median_fc  # noqa: E402  (FROZEN)
from autoresearch.load_dgtex import (  # noqa: E402
    load_dgtex_tissue, load_tissue_expression, load_sample_metadata,
)
# The baseline filter + statistic are imported UNCHANGED from Loop A.
from autoresearch.uniform_gene_selection_sweep import (  # noqa: E402  (FROZEN)
    apply_filter as _apply_filter_frozen, compute_stat, FDR_T, FC_T,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tissue registry: name -> (dGTEx SMTS label, PigGTEx file stem, dGTEx slug)
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_REGISTRY = {
    "Muscle":          ("Muscle",          "Muscle",          "muscle"),
    "Lung":            ("Lung",            "Lung",            "lung"),
    "Testis":          ("Testis",          "Testis",          "testis"),
    "Spleen":          ("Spleen",          "Spleen",          "spleen"),
    "Adipose Tissue":  ("Adipose Tissue",  "Adipose",         "adipose_tissue"),
    # Loop 0 rescue tissues (dGTEx expression file must be present locally)
    "Heart":           ("Heart",           "Heart",           "heart"),
    "Liver":           ("Liver",           "Liver",           "liver"),
    "Small Intestine": ("Small Intestine", "Small_intestine", "small_intestine"),
}

BASELINE_PANEL = ["Muscle", "Lung", "Testis", "Spleen", "Adipose Tissue"]


# ─────────────────────────────────────────────────────────────────────────────
# Pig sub-region pooling (for the Small-Intestine rescue)
# ─────────────────────────────────────────────────────────────────────────────
def load_pig_pooled(pool_files: list[str], pig_young: set, pig_old: set):
    """Pool several PigGTEx sub-region files into one (young, old) tissue.

    Mirrors xs.load_pig_tissue but unions samples across `pool_files`, matching
    each file's samples in the metadata by Tissue/Sub_categories == file stem
    (with '_' <-> ' ' tolerance). Returns (expr_df, young_ids, old_ids).
    """
    import gzip
    meta = pd.read_csv(xs.PIG_META).rename(columns={
        "BioSample": "Sample_ID", "Tissue class": "Tissue",
        "Main categories": "Tissue_Main", "Sub categories": "Sub_categories"})
    meta = meta.set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(xs.parse_age_days)
    meta["Stage"] = meta["Age_Days"].apply(xs.days_to_stage)

    exprs, youngs, olds = [], [], []
    for stem in pool_files:
        variants = {stem, stem.replace("_", " "), stem.replace(" ", "_")}
        sub = meta[meta["Tissue"].isin(variants) | meta["Sub_categories"].isin(variants)]
        path = xs.PIG_TPM_DIR / f"{stem}.expr_tpm.txt.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt") as fh:
            e = pd.read_csv(fh, sep="\t", index_col=0)
        exprs.append(e)
        youngs += [s for s in sub[sub["Stage"].isin(pig_young)].index if s in e.columns]
        olds += [s for s in sub[sub["Stage"].isin(pig_old)].index if s in e.columns]
    if not exprs:
        return pd.DataFrame(), [], []
    # Outer-join on gene index (common genes across PigGTEx files are identical)
    expr = pd.concat(exprs, axis=1)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    young = sorted(set(youngs))
    old = sorted(set(olds))
    return expr, young, old


# ─────────────────────────────────────────────────────────────────────────────
# Merged FC table (mirrors uniform_gene_selection_sweep.build_merged_fc EXACTLY,
# but parameterized). Calls the FROZEN science functions unchanged.
# ─────────────────────────────────────────────────────────────────────────────
def build_merged(name: str, cfg: dict) -> tuple[pd.DataFrame | None, dict]:
    smts, pig_tissue, slug = TISSUE_REGISTRY[name]
    info = {"pig_young": 0, "pig_old": 0, "human_young": 0, "human_old": 0,
            "merged_n": 0, "skipped": None}

    # ---- bins for THIS tissue (per-tissue override or global) ----
    pby = set(cfg["per_tissue"].get(name, {}).get("pig_young", cfg["pig_young"]))
    pbo = set(cfg["per_tissue"].get(name, {}).get("pig_old", cfg["pig_old"]))
    hyc = tuple(cfg["per_tissue"].get(name, {}).get("human_young_cohorts", cfg["human_young_cohorts"]))
    hoc = tuple(cfg["per_tissue"].get(name, {}).get("human_old_cohorts", cfg["human_old_cohorts"]))
    xs.PIG_YOUNG, xs.PIG_OLD = pby, pbo
    sm.HUMAN_YOUNG_COHORTS, sm.HUMAN_OLD_COHORTS = hyc, hoc

    # ---- pig side ----
    pool = cfg["pig_pool"].get(name)
    if pool:
        p_expr, p_y, p_o = load_pig_pooled(pool, pby, pbo)
    else:
        p_expr, p_y, p_o = xs.load_pig_tissue(pig_tissue)
    info["pig_young"], info["pig_old"] = len(p_y), len(p_o)
    if len(p_y) < 3 or len(p_o) < 3:
        info["skipped"] = "pig_low_n_initial"
        return None, info
    if cfg["pig_min_tpm"] > 0:
        p_expr = p_expr.loc[p_expr.median(axis=1) >= cfg["pig_min_tpm"]]
    p_y, p_o, _ = apply_uniform_drop(p_expr, p_y, p_o, pig_tissue, cfg["drop_pct"])
    info["pig_young"], info["pig_old"] = len(p_y), len(p_o)
    if len(p_y) < 3 or len(p_o) < 3:
        info["skipped"] = "pig_low_n_after_drop"
        return None, info
    p_fc = compute_pig_fc(p_expr, p_y, p_o, cfg["pig_method"], pig_tissue)

    # ---- human side ----
    h_expr, h_y, h_o = load_dgtex_tissue(smts, slug)
    info["human_young"], info["human_old"] = len(h_y), len(h_o)
    if len(h_y) < 3 or len(h_o) < 3:
        info["skipped"] = "human_low_n"
        return None, info
    if cfg["human_min_tpm"] > 0:
        h_expr = h_expr.loc[h_expr.median(axis=1) >= cfg["human_min_tpm"]]
    h_fc = median_fc(h_expr, h_y, h_o)

    # ---- merge on orthologs ----
    ortho = cfg["ortho_table"]
    o_ = ortho[ortho["pig_gene_id"].isin(p_fc.index) & ortho["human_gene_id"].isin(h_fc.index)]
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

    # ---- optional gene restriction (e.g. GO:0032502 developmental genes) ----
    if cfg["restrict_ids"] is not None:
        col = "human_gene_id" if cfg["restrict_col"] == "human" else "pig_gene_id"
        merged = merged[merged[col].isin(cfg["restrict_ids"])]

    info["merged_n"] = len(merged)
    if len(merged) < 50:
        info["skipped"] = f"merged_n<50 ({len(merged)})"
        return None, info
    merged = xs.add_fdr(merged, "p_pig", "fdr_pig")
    merged = xs.add_fdr(merged, "p_human", "fdr_human")

    # attach pig matrices for bootstrap_stable (only needed by that filter)
    if cfg["filter"] == "bootstrap_stable":
        merged.attrs["p_expr"] = p_expr
        merged.attrs["p_y"] = p_y
        merged.attrs["p_o"] = p_o
        merged.attrs["pig_tissue"] = pig_tissue
        merged.attrs["pig_method"] = cfg["pig_method"]
    return merged, info


# ─────────────────────────────────────────────────────────────────────────────
# Filters: baseline 5 are FROZEN; NEW ones operate on the merged table
# ─────────────────────────────────────────────────────────────────────────────
def select_genes(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    filt = cfg["filter"]
    if filt in ("strict_bidirectional", "pig_anchored", "pig_fdr_only", "fc_only", "no_filter"):
        return _apply_filter_frozen(merged, filt)
    if filt == "top_n_pigfc":
        order = merged["log2fc_pig"].abs().sort_values(ascending=False, kind="mergesort").index
        return merged.loc[order[: cfg["top_n"]]]
    if filt == "bootstrap_stable":
        return _bootstrap_stable(merged, cfg)
    raise ValueError(f"unknown filter {filt}")


def _bootstrap_stable(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep pig_anchored genes whose pig-side signal survives in >= frac of
    bootstrap resamples of the pig young/old sample sets (seed=42)."""
    base = _apply_filter_frozen(merged, "pig_anchored")
    p_expr = merged.attrs.get("p_expr")
    if p_expr is None or len(base) == 0:
        return base
    p_y, p_o = merged.attrs["p_y"], merged.attrs["p_o"]
    pig_tissue, method = merged.attrs["pig_tissue"], merged.attrs["pig_method"]
    B = cfg["bootstrap_B"]
    rng = np.random.default_rng(42)
    cand = base["pig_gene_id"].tolist()
    hits = pd.Series(0, index=cand, dtype=int)
    for _ in range(B):
        by = list(rng.choice(p_y, size=len(p_y), replace=True))
        bo = list(rng.choice(p_o, size=len(p_o), replace=True))
        fc = compute_pig_fc(p_expr, by, bo, method, pig_tissue)
        fc = xs.add_fdr(fc.assign(p=fc["pvalue"]).rename(columns={"p": "pv"}), "pvalue", "fdr")
        passed = fc.index[(fc["fdr"] < FDR_T) & (fc["log2fc"].abs() > FC_T)]
        present = [g for g in cand if g in passed]
        hits.loc[present] += 1
    stable = hits[hits >= cfg["bootstrap_frac"] * B].index
    return base[base["pig_gene_id"].isin(stable)]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(cfg: dict) -> dict:
    per_tissue, infos = {}, {}
    for name in cfg["panel"]:
        merged, info = build_merged(name, cfg)
        infos[name] = info
        if merged is None:
            per_tissue[name] = {"n": 0, "r": None, "p": None, "dc": None,
                                "ci_low": None, "ci_high": None}
            continue
        sub = select_genes(merged, cfg)
        per_tissue[name] = compute_stat(sub, cfg["statistic"])

    rs = [per_tissue[t]["r"] for t in cfg["panel"] if per_tissue[t]["r"] is not None]
    valid = len(rs) == len(cfg["panel"]) and len(rs) > 0
    return {
        "label": cfg["label"],
        "config": {k: cfg[k] for k in (
            "ortholog", "filter", "statistic", "pig_young", "pig_old",
            "human_young_cohorts", "human_old_cohorts", "pig_method",
            "drop_pct", "pig_min_tpm", "human_min_tpm", "top_n")},
        "panel": cfg["panel"],
        "per_tissue": per_tissue,
        "sample_info": infos,
        "mean_r": float(np.mean(rs)) if rs else None,
        "min_r": float(min(rs)) if rs else None,
        "max_r": float(max(rs)) if rs else None,
        "n_tissues_with_r": len(rs),
        "all_tissues_valid": valid,
    }


def _parse_list(s, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()] if s else []


def build_cfg(args) -> dict:
    ortho_paths = {
        "strict_1to1": ROOT / "review/analyses/results/pig_human_one_to_one_orthologs.csv",
        "relaxed_symbol": ROOT / "review/analyses/results/pig_human_orthologs_symbol_based.csv",
    }
    ortho_table = pd.read_csv(ortho_paths[args.ortholog])

    restrict_ids = None
    if args.restrict_genes:
        ids = Path(args.restrict_genes).read_text().split()
        restrict_ids = set(x.strip() for x in ids if x.strip())

    return {
        "label": args.label,
        "panel": _parse_list(args.tissues) or list(BASELINE_PANEL),
        "ortholog": args.ortholog,
        "ortho_table": ortho_table,
        "filter": args.filter,
        "statistic": args.statistic,
        "pig_young": _parse_list(args.pig_young),
        "pig_old": _parse_list(args.pig_old),
        "human_young_cohorts": _parse_list(args.human_young_cohorts, int),
        "human_old_cohorts": _parse_list(args.human_old_cohorts, int),
        "pig_method": args.pig_method,
        "drop_pct": args.drop_pct,
        "pig_min_tpm": args.pig_min_tpm,
        "human_min_tpm": args.human_min_tpm,
        "top_n": args.top_n,
        "bootstrap_B": args.bootstrap_B,
        "bootstrap_frac": args.bootstrap_frac,
        "restrict_ids": restrict_ids,
        "restrict_col": args.restrict_col,
        "per_tissue": json.loads(args.per_tissue_bins) if args.per_tissue_bins else {},
        "pig_pool": json.loads(args.pig_pool) if args.pig_pool else {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parameterized autoresearch evaluator")
    ap.add_argument("--label", default="adhoc")
    ap.add_argument("--tissues", default="", help="comma list; default = 5-tissue baseline panel")
    ap.add_argument("--ortholog", choices=["strict_1to1", "relaxed_symbol"], default="strict_1to1")
    ap.add_argument("--filter", default="pig_anchored",
                    help="strict_bidirectional|pig_anchored|pig_fdr_only|fc_only|no_filter|top_n_pigfc|bootstrap_stable")
    ap.add_argument("--statistic", choices=["pearson", "spearman"], default="spearman")
    ap.add_argument("--pig-young", dest="pig_young", default="Infant,Early childhood")
    ap.add_argument("--pig-old", dest="pig_old", default="Post-pubertal,Adult")
    ap.add_argument("--human-young-cohorts", dest="human_young_cohorts", default="1")
    ap.add_argument("--human-old-cohorts", dest="human_old_cohorts", default="4")
    ap.add_argument("--pig-method", dest="pig_method", choices=["weighted", "median", "limma"], default="weighted")
    ap.add_argument("--drop-pct", dest="drop_pct", type=float, default=30.0)
    ap.add_argument("--pig-min-tpm", dest="pig_min_tpm", type=float, default=1.0)
    ap.add_argument("--human-min-tpm", dest="human_min_tpm", type=float, default=10.0)
    ap.add_argument("--top-n", dest="top_n", type=int, default=300)
    ap.add_argument("--bootstrap-B", dest="bootstrap_B", type=int, default=100)
    ap.add_argument("--bootstrap-frac", dest="bootstrap_frac", type=float, default=0.80)
    ap.add_argument("--restrict-genes", dest="restrict_genes", default="")
    ap.add_argument("--restrict-col", dest="restrict_col", choices=["pig", "human"], default="human")
    ap.add_argument("--per-tissue-bins", dest="per_tissue_bins", default="",
                    help='JSON: {"Heart":{"human_young_cohorts":[1,2],"human_old_cohorts":[3,4]}}')
    ap.add_argument("--pig-pool", dest="pig_pool", default="",
                    help='JSON: {"Small Intestine":["Small_intestine","Ileum","Jejunum","Duodenum"]}')
    ap.add_argument("--out", default="", help="optional path to write result JSON")
    args = ap.parse_args()

    cfg = build_cfg(args)
    result = evaluate(cfg)

    # human-readable summary to stderr; machine JSON to stdout / --out
    print(f"LABEL: {result['label']}", file=sys.stderr)
    print(f"CONFIG: {cfg['ortholog']} | {cfg['filter']} | {cfg['statistic']}  "
          f"panel={cfg['panel']}", file=sys.stderr)
    print(f"mean_r={result['mean_r']}  min_r={result['min_r']}  max_r={result['max_r']}  "
          f"all_valid={result['all_tissues_valid']}", file=sys.stderr)
    for t in cfg["panel"]:
        pt, si = result["per_tissue"][t], result["sample_info"][t]
        print(f"  {t:<16} n={pt['n']:>5}  r={pt['r']}  "
              f"(pig {si['pig_young']}/{si['pig_old']}, human {si['human_young']}/{si['human_old']}"
              f"{', SKIP=' + si['skipped'] if si['skipped'] else ''})", file=sys.stderr)

    out_json = json.dumps(result, indent=2, default=lambda o: None)
    print(out_json)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
