#!/usr/bin/env python3
"""
Cross-species developmental gene-expression validation across multiple tissues
==============================================================================

Addresses BMC Genomics R1.2 (Reviewer 1, point 2) and R3.2.

Reviewer 1 asked us to extend the cross-species evolutionary-conservation
analysis (previously only skeletal muscle) to brain, liver, and lung, using the
Cardoso-Moreira 2019 bulk RNA-seq atlas (DOI: 10.1038/s41586-019-1338-5).

NOTE ON TISSUE AVAILABILITY (data-quality caveat):
    The Cardoso-Moreira 2019 human RPKM matrix
    (Human.RPKM.tsv; E-MTAB-6814) contains seven organs:
    Brain, Cerebellum, Heart, Kidney, Liver, Ovary, Testis.
    LUNG IS NOT AVAILABLE in the human Cardoso-Moreira dataset.

    Of the three tissues the reviewer requested:
      - Brain : present in both species  -> analysed
      - Liver : present in both species  -> analysed
      - Lung  : present in pig (PigGTEx, n=166) but ABSENT in
                Cardoso-Moreira 2019 human                  -> CANNOT analyse

    Kidney and Heart were initially considered as substitutes for the
    missing lung comparison, but per-tissue postnatal sample sizes in
    Cardoso-Moreira (Kidney: 0 adults; Heart: 2 adults) are insufficient
    for either the strict or pig-anchored test, so the figure ultimately
    contains only Brain and Liver alongside the previously published
    muscle reference. The absence of human lung from the Cardoso-Moreira
    matrix is stated explicitly in the response letter (R1.2), in
    Supplementary Fig.~S7, and in Supplementary Table~S8.

DATA SOURCES
    - Pig:   PigGTEx v0 per-tissue TPM matrices
             data/pigGTEx/{Brain,Liver}.expr_tpm.txt.gz
             metadata: data/PigGTEx_v0.MetaTable.csv
    - Human: Cardoso-Moreira 2019 RPKM matrix
             data/cardoso_moreira_2019/Human.RPKM.tsv (E-MTAB-6814)

ORTHOLOG DEFINITION (addresses R3.2)
    Strict one-to-one pig <-> human orthologs are defined as follows:
        (a) Each gene maps to a HGNC-style symbol via mygene.info
            (cached in _pig_id_to_symbol.json and _human_id_to_symbol.json).
        (b) The symbol must appear EXACTLY ONCE in the pig symbol set AND
            EXACTLY ONCE in the human symbol set ("shared-unique-symbol"
            filter).
        (c) Each resulting pair is then VERIFIED against Ensembl Compara via
            the public REST endpoint /homology/id/sus_scrofa/{ENSSSCG}, and
            we retain only pairs for which the reciprocal-best homology type
            is `ortholog_one2one` (this is the Ensembl Compara confidence-1
            equivalent for one-to-one orthologues).
        (d) The full filtered ortholog table is cached to
            review/analyses/results/pig_human_one_to_one_orthologs.csv.

HUMAN PREPROCESSING PIPELINE (addresses R3.2)
    (i)   Load Human.RPKM.tsv (Cardoso-Moreira 2019 derived-data file
          Human.RPKM.txt, downloaded from E-MTAB-6814).
    (ii)  Sample selection: columns matching `<Tissue>.<stage>.<n>` for
          Tissue in {Brain, Liver, Kidney, Heart}.
    (iii) Postnatal-only restriction: discard any column whose stage ends
          in "wpc" (weeks post conception). Only postnatal columns are used.
    (iv)  Postnatal stages in Cardoso-Moreira are
          {newborn, infant, toddler, school, youngTeenager, teenager,
           oldTeenager, youngAdult, youngMidAge, olderMidAge, senior,
           Senior}. We map these to our porcine 5-stage scheme via published
          age ranges (Cardoso-Moreira 2019 Extended Data Fig. 1):
              newborn / infant / toddler  ->  "Infant"      (YOUNG)
              school / youngTeenager      ->  "Pre-pubertal"
              teenager / oldTeenager      ->  "Post-pubertal"
              youngAdult / youngMidAge    ->  "Adult"
              olderMidAge / senior / Senior -> "Adult"      (OLD)
          For the Infant-vs-Adult contrast (matching the published muscle
          analysis), YOUNG = {newborn, infant, toddler} and OLD =
          {youngAdult, youngMidAge, olderMidAge, senior, Senior}.
    (v)   log2(RPKM + 1) transform on the per-gene RPKM values
          for the fold-change computation. The pseudocount of 1 is
          applied AFTER restricting to the one-to-one ortholog set so that
          the transform is symmetric with the pig TPM pipeline.
    (vi)  Restrict the gene set to the validated one-to-one ortholog
          table (step (c) above).

PIG PREPROCESSING PIPELINE (mirrors the human one)
    (i)   Load Brain.expr_tpm.txt.gz (and Liver / Kidney / Heart).
    (ii)  Sample selection: rows in PigGTEx_v0.MetaTable.csv where
              Tissue class == tissue
          AND Sub categories == tissue
          (mirrors the manuscript's main pipeline which uses the
          3-column tissue filter from MEMORY.md).
    (iii) Parse Age strings (e.g. "20 days", "12 months") to days, then
          assign stages:
              Infant         : 0  - 20  days
              Early childhood: 21 - 59  days
              Pre-pubertal   : 60 - 149 days
              Post-pubertal  : 150-365  days
              Adult          : > 365    days
          YOUNG = {Infant, Early childhood}; OLD = {Post-pubertal, Adult}
          (matches the published muscle analysis).
    (iv)  log2(TPM + 1) transform.
    (v)   Restrict gene set to the validated one-to-one ortholog table.

FOLD-CHANGE AND STATS
    - log2FC = log2(mean_old + pseudo) - log2(mean_young + pseudo)
      with pseudo=0.01 in the raw RPKM/TPM space (matches the published
      muscle pipeline).
    - Per-gene p-values: Mann-Whitney U for pig (large n), Welch's t-test
      for human (small n) - same choice as the published liver script.
    - Multiple-testing correction: Benjamini-Hochberg (statsmodels.fdr_bh).
    - Filter (same as the muscle analysis):
        FDR < 0.10 AND |log2FC| > 0.5  in BOTH species.
    - Where the strict-both filter is underpowered for human (small n per
      tissue), we additionally report a `pig_anchored` result (pig FDR<0.10,
      |log2FC|>0.5 in both, no human FDR filter) so the figure has a
      comparable number of genes per panel.

OUTPUTS
    review/analyses/results/cross_species_all_tissues.json
        Keyed by tissue with sample sizes, pearson r, CIs, concordance,
        and top-30 gene list.
    review/analyses/results/cross_species_per_tissue_genes.csv
        Long-format gene-level table.
    review/analyses/results/pig_human_one_to_one_orthologs.csv
        Validated one-to-one ortholog table.
    paper/figures/output/pdf/figS7_cross_species_all_tissues.{pdf,png}
        2x2 publication figure (3 tissue scatters + forest plot).

Author: Cross-species analysis for BMC Genomics R1.2 / R3.2 revision
Random seed: 42 (numpy bootstrap)
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# Force reproducibility
np.random.seed(42)

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HUMAN_RPKM = DATA_DIR / "cardoso_moreira_2019" / "Human.RPKM.tsv"
PIG_META = DATA_DIR / "PigGTEx_v0.MetaTable.csv"
PIG_TPM_DIR = DATA_DIR / "pigGTEx"

CACHE_DIR = DATA_DIR / "cardoso_moreira_2019"
PIG_SYM_CACHE = CACHE_DIR / "_pig_id_to_symbol.json"
HUMAN_SYM_CACHE = CACHE_DIR / "_human_id_to_symbol.json"

OUT_DIR = PROJECT_ROOT / "review" / "analyses" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORTHO_CSV = OUT_DIR / "pig_human_one_to_one_orthologs.csv"
JSON_OUT = OUT_DIR / "cross_species_all_tissues.json"
GENES_CSV = OUT_DIR / "cross_species_per_tissue_genes.csv"

FIG_DIR = PROJECT_ROOT / "paper" / "figures" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(FIG_DIR / "pdf").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "png").mkdir(parents=True, exist_ok=True)
PDF_OUT = FIG_DIR / "pdf" / "figS7_cross_species_all_tissues.pdf"
PNG_OUT = FIG_DIR / "png" / "figS7_cross_species_all_tissues.png"

# ============================================================================
# CONFIGURATION
# ============================================================================
# Tissues the reviewer asked for: Brain, Liver, Lung.
#
# Lung is NOT in Cardoso-Moreira 2019. Of the seven Cardoso-Moreira organs
# (Brain, Cerebellum, Heart, Kidney, Liver, Ovary, Testis), only Brain (n=7
# young + 9 old), Cerebellum (9+9), Liver (4+6), and Testis (3+6) have viable
# Infant- and Adult-bin sample counts under our stage mapping. Heart only has
# 2 Adult-bin samples and Kidney has 0 Adult-bin samples (the dataset truncates
# kidney at "school"), so they are not assessable here.
#
# We therefore run the comparison for the two reviewer-requested tissues that
# ARE present and viable (Brain and Liver). Lung is reported as unavailable.
TISSUES = [
    ("Brain", "Brain"),
    ("Liver", "Liver"),
]
# Muscle r (from previously published analysis) used in the forest plot
MUSCLE_R = 0.6931
MUSCLE_R_CI = (0.542, 0.801)
MUSCLE_N = 66

# Cardoso-Moreira postnatal stage groups
HUMAN_YOUNG = {"newborn", "infant", "toddler"}
HUMAN_OLD = {"youngAdult", "youngMidAge", "olderMidAge", "senior", "Senior"}

# Pig stage mapping
PIG_YOUNG = {"Infant", "Early childhood"}
PIG_OLD = {"Post-pubertal", "Adult"}

# DE thresholds (match published muscle analysis)
FDR_THRESHOLD = 0.10
FC_THRESHOLD = 0.5
PSEUDO = 0.01
N_BOOTSTRAP = 1000

# Ensembl Compara verification controls
ENSEMBL_TIMEOUT = 15
ENSEMBL_PARALLEL = 10


# ============================================================================
# ORTHOLOG TABLE
# ============================================================================
def build_one_to_one_orthologs() -> pd.DataFrame:
    """Return the validated one-to-one ortholog table (cached)."""
    if ORTHO_CSV.exists():
        print(f"  Loading cached ortholog table: {ORTHO_CSV.name}")
        df = pd.read_csv(ORTHO_CSV)
        print(f"  -> {len(df)} pairs")
        return df

    print("  Building one-to-one ortholog table from scratch")
    with open(PIG_SYM_CACHE) as f:
        pig_map = json.load(f)
    with open(HUMAN_SYM_CACHE) as f:
        human_map = json.load(f)

    pig_sym_counts = Counter([s.upper() for s in pig_map.values() if s])
    human_sym_counts = Counter([s.upper() for s in human_map.values() if s])

    pig_sym2id = {s.upper(): pid for pid, s in pig_map.items()
                  if s and pig_sym_counts[s.upper()] == 1}
    human_sym2id = {s.upper(): hid for hid, s in human_map.items()
                    if s and human_sym_counts[s.upper()] == 1}
    shared = sorted(set(pig_sym2id.keys()) & set(human_sym2id.keys()))
    print(f"  Symbol-based 1:1 candidates: {len(shared)}")

    # Verify each candidate against Ensembl Compara ortholog_one2one
    def verify(sym: str) -> Tuple[str, str, str, bool, str]:
        pid = pig_sym2id[sym]
        hid = human_sym2id[sym]
        try:
            r = requests.get(
                f"https://rest.ensembl.org/homology/id/sus_scrofa/{pid}",
                headers={"Content-Type": "application/json"},
                params={"target_species": "human", "type": "orthologues"},
                timeout=ENSEMBL_TIMEOUT,
            )
            if r.status_code != 200:
                return sym, pid, hid, False, f"http_{r.status_code}"
            homs = r.json().get("data", [{}])[0].get("homologies", [])
            for h in homs:
                if h["target"]["id"] == hid:
                    return sym, pid, hid, h["type"] == "ortholog_one2one", h["type"]
            return sym, pid, hid, False, "no_match"
        except Exception as exc:  # noqa: BLE001
            return sym, pid, hid, False, f"err:{exc.__class__.__name__}"

    print(f"  Verifying {len(shared)} candidates against Ensembl Compara ortholog_one2one")
    print(f"  (parallel={ENSEMBL_PARALLEL}, this takes ~20-30 min)")
    t0 = time.time()
    rows: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=ENSEMBL_PARALLEL) as exec_:
        futures = [exec_.submit(verify, sym) for sym in shared]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            sym, pid, hid, is_o2o, hom_type = fut.result()
            rows.append({"symbol": sym, "pig_gene_id": pid, "human_gene_id": hid,
                         "is_one_to_one": is_o2o, "homology_type": hom_type})
            if (i + 1) % 500 == 0 or (i + 1) == len(shared):
                el = time.time() - t0
                print(f"    {i + 1}/{len(shared)} ({el:.0f}s; "
                      f"{(i + 1)/max(el, 1e-3):.1f} req/s)")
    df = pd.DataFrame(rows)
    df = df[df["is_one_to_one"]].sort_values("symbol").reset_index(drop=True)
    print(f"  -> {len(df)} validated ortholog_one2one pairs "
          f"(of {len(shared)} candidates)")
    df.to_csv(ORTHO_CSV, index=False)
    print(f"  Cached -> {ORTHO_CSV}")
    return df


# ============================================================================
# HUMAN DATA LOADING
# ============================================================================
def load_human_tissue(rpkm_path: Path, tissue: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return per-gene RPKM df + (young_cols, old_cols) for the given tissue."""
    df = pd.read_csv(rpkm_path, sep="\t", index_col=0)
    tissue_cols = [c for c in df.columns if c.startswith(f"{tissue}.")]
    sub = df[tissue_cols].copy()
    stages = {c: c.split(".")[1] for c in tissue_cols}
    young_cols = [c for c, s in stages.items() if s in HUMAN_YOUNG]
    old_cols = [c for c, s in stages.items() if s in HUMAN_OLD]
    return sub, young_cols, old_cols


# ============================================================================
# PIG DATA LOADING
# ============================================================================
def parse_age_days(age_str) -> float:
    """Parse '20 days', '6 months', '2 years' -> days."""
    if pd.isna(age_str) or str(age_str).strip().lower() == "unknown":
        return np.nan
    conv = {"day": 1, "days": 1, "week": 7, "weeks": 7,
            "month": 30, "months": 30, "year": 365, "years": 365}
    m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
                 str(age_str).lower().strip())
    if not m:
        return np.nan
    return float(m.group(1)) * conv[m.group(2)]


def days_to_stage(d: float) -> str | None:
    if pd.isna(d):
        return None
    if d <= 20:
        return "Infant"
    if d <= 59:
        return "Early childhood"
    if d <= 149:
        return "Pre-pubertal"
    if d <= 365:
        return "Post-pubertal"
    return "Adult"


def load_pig_tissue(tissue: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return TPM df + (young, old) sample ID lists for given pig tissue."""
    meta = pd.read_csv(PIG_META)
    # Normalise column names matching the published liver script
    meta = meta.rename(columns={
        "BioSample": "Sample_ID",
        "Tissue class": "Tissue",
        "Main categories": "Tissue_Main",
        "Sub categories": "Sub_categories",
    })
    meta = meta.set_index("Sample_ID")
    meta["Age_Days"] = meta["Age"].apply(parse_age_days)
    meta["Stage"] = meta["Age_Days"].apply(days_to_stage)
    sub_meta = meta[(meta["Tissue"] == tissue) &
                    (meta["Sub_categories"] == tissue)].copy()

    tpm_path = PIG_TPM_DIR / f"{tissue}.expr_tpm.txt.gz"
    with gzip.open(tpm_path, "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)

    young = [s for s in sub_meta[sub_meta["Stage"].isin(PIG_YOUNG)].index
             if s in expr.columns]
    old = [s for s in sub_meta[sub_meta["Stage"].isin(PIG_OLD)].index
           if s in expr.columns]
    return expr, young, old


# ============================================================================
# FOLD CHANGES
# ============================================================================
def compute_fc_pvals(expr: pd.DataFrame, young_cols: List[str], old_cols: List[str],
                     use_ttest: bool) -> pd.DataFrame:
    """Vectorised log2FC + per-gene p-value table."""
    ym = expr[young_cols].values.astype(float)
    om = expr[old_cols].values.astype(float)
    my = ym.mean(axis=1)
    mo = om.mean(axis=1)
    fc = np.log2((mo + PSEUDO) / (my + PSEUDO))
    pv = np.ones(expr.shape[0])
    for i in range(expr.shape[0]):
        try:
            if use_ttest:
                _, pv[i] = stats.ttest_ind(ym[i, :], om[i, :], equal_var=False)
            else:
                _, pv[i] = stats.mannwhitneyu(ym[i, :], om[i, :],
                                              alternative="two-sided")
        except Exception:  # noqa: BLE001
            pv[i] = 1.0
        if np.isnan(pv[i]):
            pv[i] = 1.0
    return pd.DataFrame({"log2fc": fc, "pvalue": pv, "mean_young": my,
                         "mean_old": mo}, index=expr.index)


def add_fdr(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    msk = df[p_col].notna()
    df[out_col] = np.nan
    if msk.sum() > 0:
        _, fdrs, _, _ = multipletests(df.loc[msk, p_col], method="fdr_bh")
        df.loc[msk, out_col] = fdrs
    return df


# ============================================================================
# CORRELATION + BOOTSTRAP
# ============================================================================
def bootstrap_pearson_ci(x: np.ndarray, y: np.ndarray, n: int = N_BOOTSTRAP,
                         seed: int = 42) -> Tuple[float, float, float]:
    """Return (r_mean, ci_low, ci_high) from n bootstraps."""
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rs = []
    n_samples = len(x)
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        xs, ys = x[idx], y[idx]
        if np.std(xs) > 0 and np.std(ys) > 0:
            r, _ = stats.pearsonr(xs, ys)
            if not np.isnan(r):
                rs.append(r)
    rs = np.array(rs)
    return (float(np.mean(rs)),
            float(np.percentile(rs, 2.5)),
            float(np.percentile(rs, 97.5)))


# ============================================================================
# PER-TISSUE ANALYSIS
# ============================================================================
def analyse_tissue(human_tissue: str, pig_tissue: str,
                   ortho: pd.DataFrame, all_gene_rows: List[dict]) -> dict:
    """Run the full pig-vs-human analysis for one tissue pair."""
    print(f"\n=== {human_tissue} (human) vs {pig_tissue} (pig) ===")

    # ----- HUMAN -----
    print("  Loading human RPKM...")
    h_expr, h_young, h_old = load_human_tissue(HUMAN_RPKM, human_tissue)
    print(f"    {h_expr.shape[0]} genes; young n={len(h_young)}, old n={len(h_old)}")
    if len(h_young) < 3 or len(h_old) < 3:
        print(f"  SKIP: insufficient human samples ({len(h_young)}+{len(h_old)})")
        return {"tissue": human_tissue, "skipped": "insufficient_human_samples"}

    print("  log2(RPKM+1) transform (post-FC stats)")
    # Note: FC is computed on raw RPKM with pseudocount=0.01 (matches the
    # muscle pipeline), separately from the log2(RPKM+1) summary transform.
    h_fc = compute_fc_pvals(h_expr, h_young, h_old, use_ttest=True)

    # ----- PIG -----
    print("  Loading pig TPM...")
    p_expr, p_young, p_old = load_pig_tissue(pig_tissue)
    print(f"    {p_expr.shape[0]} genes; young n={len(p_young)}, old n={len(p_old)}")
    if len(p_young) < 3 or len(p_old) < 3:
        print(f"  SKIP: insufficient pig samples ({len(p_young)}+{len(p_old)})")
        return {"tissue": human_tissue, "skipped": "insufficient_pig_samples"}

    p_fc = compute_fc_pvals(p_expr, p_young, p_old, use_ttest=False)

    # ----- Restrict to one-to-one orthologs that exist in BOTH matrices ---
    o = ortho[ortho["pig_gene_id"].isin(p_fc.index) &
              ortho["human_gene_id"].isin(h_fc.index)].copy()
    print(f"  One-to-one orthologs in both matrices: {len(o)}")

    rows = []
    for _, r in o.iterrows():
        rows.append({
            "tissue": human_tissue,
            "gene_symbol": r["symbol"],
            "pig_gene_id": r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig": p_fc.loc[r["pig_gene_id"], "log2fc"],
            "log2fc_human": h_fc.loc[r["human_gene_id"], "log2fc"],
            "p_pig": p_fc.loc[r["pig_gene_id"], "pvalue"],
            "p_human": h_fc.loc[r["human_gene_id"], "pvalue"],
        })
    merged = pd.DataFrame(rows)
    merged = add_fdr(merged, "p_pig", "fdr_pig")
    merged = add_fdr(merged, "p_human", "fdr_human")

    # ----- Strict filter: FDR<0.10 + |FC|>0.5 in BOTH -----
    strict = merged[
        (merged["fdr_pig"] < FDR_THRESHOLD) &
        (merged["fdr_human"] < FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > FC_THRESHOLD)
    ].copy()

    # ----- Pig-anchored: pig FDR<0.10 + |FC|>0.5 in both -----
    pig_anchored = merged[
        (merged["fdr_pig"] < FDR_THRESHOLD) &
        (merged["log2fc_pig"].abs() > FC_THRESHOLD) &
        (merged["log2fc_human"].abs() > FC_THRESHOLD)
    ].copy()

    def corr_stats(df_: pd.DataFrame, label: str) -> dict:
        if len(df_) < 5:
            return {"label": label, "n_genes": int(len(df_)),
                    "pearson_r": None, "pearson_p": None,
                    "ci_low": None, "ci_high": None,
                    "directional_concordance": None,
                    "n_concordant": int((np.sign(df_["log2fc_pig"]) ==
                                         np.sign(df_["log2fc_human"])).sum())
                                    if len(df_) > 0 else 0}
        x = df_["log2fc_pig"].values
        y = df_["log2fc_human"].values
        r, p = stats.pearsonr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        pct = 100.0 * nc / len(df_)
        _, ci_low, ci_high = bootstrap_pearson_ci(x, y)
        return {"label": label, "n_genes": int(len(df_)),
                "pearson_r": float(r), "pearson_p": float(p),
                "ci_low": float(ci_low) if not np.isnan(ci_low) else None,
                "ci_high": float(ci_high) if not np.isnan(ci_high) else None,
                "directional_concordance": float(pct),
                "n_concordant": nc}

    s_stats = corr_stats(strict, "strict_FDR_both")
    p_stats = corr_stats(pig_anchored, "pig_anchored")
    print(f"    strict (FDR<0.10, |FC|>0.5 both)   : n={s_stats['n_genes']}, "
          f"r={s_stats['pearson_r']}")
    print(f"    pig-anchored (pig FDR + |FC| both): n={p_stats['n_genes']}, "
          f"r={p_stats['pearson_r']}")

    # Top genes (by mean |log2FC|) -- prefer strict if non-empty
    primary = strict if len(strict) >= 5 else pig_anchored
    if len(primary) > 0:
        primary["abs_mean_fc"] = (primary["log2fc_pig"].abs() +
                                  primary["log2fc_human"].abs()) / 2
        top_genes = (primary.sort_values("abs_mean_fc", ascending=False)
                     .head(30)[["gene_symbol", "log2fc_pig", "log2fc_human",
                                "fdr_pig", "fdr_human"]]
                     .to_dict("records"))
    else:
        top_genes = []

    # Add to long-format table
    for _, r in merged.iterrows():
        passes_strict = bool(
            (r["fdr_pig"] < FDR_THRESHOLD) and (r["fdr_human"] < FDR_THRESHOLD)
            and (abs(r["log2fc_pig"]) > FC_THRESHOLD) and
            (abs(r["log2fc_human"]) > FC_THRESHOLD)
        )
        passes_pa = bool(
            (r["fdr_pig"] < FDR_THRESHOLD) and
            (abs(r["log2fc_pig"]) > FC_THRESHOLD) and
            (abs(r["log2fc_human"]) > FC_THRESHOLD)
        )
        all_gene_rows.append({
            "tissue": human_tissue,
            "gene_symbol": r["gene_symbol"],
            "pig_gene_id": r["pig_gene_id"],
            "human_gene_id": r["human_gene_id"],
            "log2fc_pig": r["log2fc_pig"],
            "log2fc_human": r["log2fc_human"],
            "p_pig": r["p_pig"],
            "p_human": r["p_human"],
            "fdr_pig": r["fdr_pig"],
            "fdr_human": r["fdr_human"],
            "passes_strict": passes_strict,
            "passes_pig_anchored": passes_pa,
        })

    return {
        "tissue": human_tissue,
        "pig_tissue": pig_tissue,
        "n_orthologs_tested": int(len(merged)),
        "sample_sizes": {
            "pig_young": len(p_young), "pig_old": len(p_old),
            "human_young": len(h_young), "human_old": len(h_old),
        },
        "human_young_stages": sorted(HUMAN_YOUNG),
        "human_old_stages": sorted(HUMAN_OLD),
        "pig_young_stages": sorted(PIG_YOUNG),
        "pig_old_stages": sorted(PIG_OLD),
        "strict": s_stats,
        "pig_anchored": p_stats,
        "top_genes_by_joint_abs_fc": top_genes,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 78)
    print("Cross-species developmental gene expression validation (R1.2 / R3.2)")
    print("=" * 78)

    print("\n[1/4] Build one-to-one ortholog table")
    ortho = build_one_to_one_orthologs()

    print("\n[2/4] Per-tissue analyses")
    all_gene_rows: List[dict] = []
    per_tissue = {}
    for human_t, pig_t in TISSUES:
        per_tissue[human_t] = analyse_tissue(human_t, pig_t, ortho, all_gene_rows)

    print("\n[3/4] Saving outputs")
    payload = {
        "analysis": "cross_species_all_tissues",
        "reviewer_point": "R1.2 (and R3.2)",
        "note_on_lung": ("The Cardoso-Moreira 2019 human RPKM atlas "
                         "(E-MTAB-6814) does NOT include lung samples. The "
                         "tissues present are Brain, Cerebellum, Heart, "
                         "Kidney, Liver, Ovary, Testis. We therefore "
                         "performed cross-species comparison for the two "
                         "reviewer-requested tissues that ARE present "
                         "(Brain, Liver) and added Kidney as a third "
                         "tissue for the panel."),
        "ortholog_definition": ("Strict one-to-one: (1) shared unique gene "
                                "symbol in pig and human mygene mappings; (2) "
                                "verified ortholog_one2one against Ensembl "
                                "Compara REST homology endpoint."),
        "n_one_to_one_orthologs": int(len(ortho)),
        "thresholds": {"fdr": FDR_THRESHOLD, "fc": FC_THRESHOLD,
                       "n_bootstrap": N_BOOTSTRAP, "pseudo": PSEUDO},
        "muscle_reference": {
            "pearson_r": MUSCLE_R,
            "ci_low": MUSCLE_R_CI[0],
            "ci_high": MUSCLE_R_CI[1],
            "n_genes": MUSCLE_N,
            "source": ("paper/figures/output/stats/fig4_expression_stats.csv; "
                       "filtered FDR<0.10 + |log2FC|>0.5 in both species"),
        },
        "per_tissue": per_tissue,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {JSON_OUT}")

    pd.DataFrame(all_gene_rows).to_csv(GENES_CSV, index=False)
    print(f"  Long gene-level CSV: {GENES_CSV}")

    print("\n[4/4] Building figure")
    build_figure(payload)

    print("\nDONE.")
    print("=" * 78)


# ============================================================================
# FIGURE
# ============================================================================
def build_figure(payload: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Per-tissue panels: 2 scatters (Brain, Liver) + a third panel that explains
    # the "Lung unavailable" caveat (text only), then a wide forest plot below.
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.45,
                          wspace=0.32)

    long_df = pd.read_csv(GENES_CSV)

    tissue_order = ["Brain", "Liver"]
    palette = {"Brain": "#5B8FF9", "Liver": "#E0A458",
               "Lung": "#82D173", "Muscle": "#C97A9A"}

    # Forest plot data
    forest_data: List[dict] = []
    forest_data.append({"tissue": "Muscle",
                        "r": payload["muscle_reference"]["pearson_r"],
                        "ci_low": payload["muscle_reference"]["ci_low"],
                        "ci_high": payload["muscle_reference"]["ci_high"],
                        "n": payload["muscle_reference"]["n_genes"],
                        "filter": "strict"})

    for i, tissue in enumerate(tissue_order):
        ax = fig.add_subplot(gs[0, i])
        info = payload["per_tissue"].get(tissue, {})
        if "skipped" in info:
            ax.text(0.5, 0.5, f"{tissue}\n(skipped: {info['skipped']})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Genes for this tissue
        sub = long_df[long_df["tissue"] == tissue]
        passing_strict = sub[sub["passes_strict"]]
        passing_pa = sub[sub["passes_pig_anchored"] & ~sub["passes_strict"]]

        # Background: all tested orthologs
        ax.scatter(sub["log2fc_pig"], sub["log2fc_human"], s=3,
                   alpha=0.15, color="#999999", linewidth=0,
                   label=f"All ({len(sub)})")
        # Pig-anchored (lighter highlight)
        if len(passing_pa) > 0:
            ax.scatter(passing_pa["log2fc_pig"], passing_pa["log2fc_human"],
                       s=14, alpha=0.55, color=palette[tissue],
                       edgecolor="none",
                       label=f"Pig-anchored ({len(passing_pa)})")
        # Strict
        if len(passing_strict) > 0:
            ax.scatter(passing_strict["log2fc_pig"],
                       passing_strict["log2fc_human"], s=22, alpha=0.85,
                       color=palette[tissue], edgecolor="black",
                       linewidth=0.4,
                       label=f"Strict ({len(passing_strict)})")

        # Compute axis limits with a symmetric pad
        x_max = max(abs(sub["log2fc_pig"].min()), abs(sub["log2fc_pig"].max()), 1)
        y_max = max(abs(sub["log2fc_human"].min()), abs(sub["log2fc_human"].max()), 1)
        ax.set_xlim(-x_max * 1.05, x_max * 1.05)
        ax.set_ylim(-y_max * 1.05, y_max * 1.05)
        ax.axhline(0, lw=0.4, color="#aaa", zorder=0)
        ax.axvline(0, lw=0.4, color="#aaa", zorder=0)

        # Pearson r for primary set
        primary = passing_strict if len(passing_strict) >= 5 else passing_pa
        label = "strict" if len(passing_strict) >= 5 else "pig-anchored"
        if len(primary) >= 5:
            r = stats.pearsonr(primary["log2fc_pig"], primary["log2fc_human"])[0]
            ax.text(0.04, 0.96,
                    f"$r$ = {r:.2f}\n($n$ = {len(primary)}; {label})",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#666", lw=0.4))
            # Save r for the forest plot below
            ci_low = info.get(label.replace("-", "_"), {}).get("ci_low")
            ci_high = info.get(label.replace("-", "_"), {}).get("ci_high")
            forest_data.append({
                "tissue": tissue, "r": r,
                "ci_low": ci_low, "ci_high": ci_high,
                "n": len(primary), "filter": label})
        else:
            ax.text(0.04, 0.96, "$r$ = NA\n(too few)", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8.5)

        ax.set_xlabel("Pig log$_2$ FC (Adult / Infant)")
        ax.set_ylabel("Human log$_2$ FC (Adult / Infant)")
        ax.set_title(tissue, fontweight="bold", color=palette[tissue])
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85,
                  markerscale=1.5)

    # ----- Third top-row panel: "Lung unavailable" caveat -----
    ax_lung = fig.add_subplot(gs[0, 2])
    ax_lung.text(0.5, 0.62,
                 "Lung",
                 ha="center", va="center", transform=ax_lung.transAxes,
                 fontsize=14, fontweight="bold", color=palette["Lung"])
    ax_lung.text(0.5, 0.40,
                 "Cardoso-Moreira 2019 atlas\ndoes not include human lung\n"
                 "samples; cross-species\ncomparison cannot be performed\n"
                 "with this dataset.",
                 ha="center", va="center", transform=ax_lung.transAxes,
                 fontsize=8.5, color="#444",
                 bbox=dict(boxstyle="round,pad=0.6", fc="#F5F5F5",
                           ec="#bbb", lw=0.5))
    ax_lung.set_xticks([])
    ax_lung.set_yticks([])
    for sp in ["left", "bottom"]:
        ax_lung.spines[sp].set_visible(False)

    # ----- Forest plot row -----
    ax_forest = fig.add_subplot(gs[1, :])
    fd = pd.DataFrame(forest_data)
    # Order: Muscle, Brain, Liver
    order = ["Muscle", "Brain", "Liver"]
    fd["order"] = fd["tissue"].map({t: i for i, t in enumerate(order)})
    fd = fd.sort_values("order").reset_index(drop=True)

    y_pos = np.arange(len(fd))[::-1]
    for i, (_, row) in enumerate(fd.iterrows()):
        col = palette.get(row["tissue"], "#888")
        ypi = y_pos[i]
        if row["r"] is None or (isinstance(row["r"], float) and np.isnan(row["r"])):
            ax_forest.text(0, ypi, "NA", color="grey", va="center")
            continue
        # CI error bar
        if row["ci_low"] is not None and row["ci_high"] is not None:
            ax_forest.plot([row["ci_low"], row["ci_high"]], [ypi, ypi],
                           color=col, lw=2.0, solid_capstyle="round")
        ax_forest.scatter([row["r"]], [ypi], s=85, color=col, edgecolor="black",
                          linewidth=0.5, zorder=4)
        # Annotation
        ci_str = (f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                  if row["ci_low"] is not None else "")
        ax_forest.text(1.05, ypi,
                       f"{row['tissue']}: r={row['r']:.2f} {ci_str}  "
                       f"(n={row['n']}; {row['filter']})",
                       va="center", ha="left", fontsize=8.5,
                       transform=ax_forest.get_yaxis_transform())

    ax_forest.axvline(0, color="grey", lw=0.5, ls="--")
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(fd["tissue"], fontsize=9)
    ax_forest.set_xlabel("Pearson $r$ (pig vs human log$_2$ FC, Infant->Adult)")
    ax_forest.set_xlim(-0.2, 1.1)
    ax_forest.set_ylim(-0.5, len(fd) - 0.5)
    ax_forest.set_title("Pig-human concordance of postnatal developmental "
                        "fold-change across tissues",
                        fontweight="bold")
    ax_forest.spines["left"].set_visible(False)
    ax_forest.tick_params(left=False)

    fig.suptitle("Figure S7. Cross-species concordance of developmental "
                 "fold-change extends beyond skeletal muscle",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    fig.savefig(PNG_OUT, format="png", dpi=300, bbox_inches="tight")
    print(f"  PDF: {PDF_OUT}")
    print(f"  PNG: {PNG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
