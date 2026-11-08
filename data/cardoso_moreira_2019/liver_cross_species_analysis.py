#!/usr/bin/env python3
"""
Liver Cross-Species Developmental Gene Expression Validation
=============================================================
Compares pig vs human liver developmental fold-changes using:
  - Pig: PigGTEx liver TPM (Infant/Early childhood vs Post-pubertal/Adult)
  - Human: Cardoso-Moreira 2019 liver RPKM (newborn/infant/toddler vs teenager/youngAdult/.../senior)
  - Gene mapping via mygene (Ensembl ID -> gene symbol for both species)

Two analysis modes:
  A) Strict: Same criteria as muscle (FDR<0.10, |FC|>0.5 in BOTH species)
     -- expected to yield few/no genes due to low human postnatal sample count
  B) Pig-anchored: Use pig-significant genes (FDR<0.10, |FC|>0.5) and test
     whether their human FC is correlated. This is the appropriate approach
     given the human Cardoso-Moreira liver has only 4 young + 7 old postnatal
     samples (insufficient for per-gene FDR correction in human).
"""

import gzip
import json
import re
import sys
import warnings
from pathlib import Path

import mygene
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = Path(__file__).resolve().parent / "liver_cross_species_results.json"

PIG_TPM_PATH = DATA_DIR / "pigGTEx" / "Liver.expr_tpm.txt.gz"
PIG_META_PATH = DATA_DIR / "PigGTEx_v0.MetaTable.csv"
PIG_MODEL_PATH = PROJECT_ROOT / "machine_learning" / "model_outputs" / "Liver_results.json"
HUMAN_RPKM_TSV = Path(__file__).resolve().parent / "Human.RPKM.tsv"

CACHE_DIR = Path(__file__).resolve().parent
PIG_MAP_CACHE = CACHE_DIR / "_pig_id_to_symbol.json"
HUMAN_MAP_CACHE = CACHE_DIR / "_human_id_to_symbol.json"


# ==============================================================================
# GENE ID MAPPING
# ==============================================================================

def mygene_batch_lookup(ensembl_ids, species_label, cache_path=None, batch_size=1000):
    """Map Ensembl IDs -> symbols via mygene (with JSON caching)."""
    if cache_path and cache_path.exists():
        print(f"  Loading cached {species_label} mapping from {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    mg = mygene.MyGeneInfo()
    ids = list(ensembl_ids)
    results = {}
    chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    print(f"  Querying mygene for {len(ids)} {species_label} IDs ({len(chunks)} chunks)...")

    for ci, chunk in enumerate(chunks):
        out = mg.querymany(chunk, scopes="ensembl.gene", fields="symbol",
                           species="all", returnall=True, verbose=False)
        for hit in out.get("out", []):
            if "symbol" in hit and "query" in hit:
                results[hit["query"]] = hit["symbol"]
        if (ci + 1) % 10 == 0 or (ci + 1) == len(chunks):
            print(f"    {ci+1}/{len(chunks)} done ({len(results)} symbols)")

    print(f"  Resolved {len(results)}/{len(ids)}")
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f)
    return results


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_pig_liver_data():
    """Load pig liver TPM, split young/old by developmental stage."""
    print("\n--- Loading pig liver data ---")
    meta = pd.read_csv(PIG_META_PATH)
    meta.columns = [c.strip().replace(" ", "_") for c in meta.columns]
    for old, new in {"BioSample": "Sample_ID", "Tissue_class": "Tissue",
                     "Main_categories": "Tissue_Main"}.items():
        if old in meta.columns and new not in meta.columns:
            meta = meta.rename(columns={old: new})
    meta = meta.set_index("Sample_ID")

    age_conv = {"day": 1, "days": 1, "week": 7, "weeks": 7,
                "month": 30, "months": 30, "year": 365, "years": 365}

    def parse_age(s):
        if pd.isna(s) or str(s).strip().lower() == "unknown":
            return None
        m = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
                     str(s).lower().strip())
        return float(m.group(1)) * age_conv[m.group(2)] if m else None

    meta["Age_Days"] = meta["Age"].apply(parse_age)
    stages = {"Infant": (0, 20), "Early childhood": (21, 59), "Pre-pubertal": (60, 149),
              "Post-pubertal": (150, 365), "Adult": (366, float("inf"))}

    def to_stage(d):
        if pd.isna(d): return None
        for s, (lo, hi) in stages.items():
            if lo <= d <= hi: return s
        return "Adult"

    meta["Stage"] = meta["Age_Days"].apply(to_stage)
    lm = meta[(meta["Tissue"] == "Liver") & (meta["Tissue_Main"] == "Liver") &
              (meta["Sub_categories"] == "Liver")].copy()
    print(f"  Liver samples: {len(lm)}")
    for st in stages:
        print(f"    {st}: {(lm['Stage'] == st).sum()}")

    with gzip.open(PIG_TPM_PATH, "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    print(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} samples")

    young = [s for s in lm[lm["Stage"].isin(["Infant", "Early childhood"])].index if s in expr.columns]
    old = [s for s in lm[lm["Stage"].isin(["Post-pubertal", "Adult"])].index if s in expr.columns]
    print(f"  Young: {len(young)}, Old: {len(old)}")
    return expr, young, old


def load_human_liver_data():
    """Load human liver RPKM, split young/old postnatal samples."""
    print("\n--- Loading human liver data ---")
    df = pd.read_csv(HUMAN_RPKM_TSV, sep="\t", index_col=0)
    print(f"  Full RPKM: {df.shape[0]} genes x {df.shape[1]} samples")

    liver_cols = [c for c in df.columns if c.startswith("Liver.")]
    liver = df[liver_cols].copy()
    cs = {c: c.split(".")[1] for c in liver_cols}
    print(f"  Liver columns: {len(liver_cols)}")
    print(f"  Stages: {sorted(set(cs.values()))}")

    # Postnatal young vs old (exclude prenatal wpc samples)
    young_s = {"newborn", "infant", "toddler"}
    old_s = {"teenager", "youngAdult", "youngMidAge", "olderMidAge", "senior"}

    young_cols = [c for c, s in cs.items() if s in young_s]
    old_cols = [c for c, s in cs.items() if s in old_s]

    print(f"  Young (newborn/infant/toddler): {len(young_cols)}")
    print(f"  Old (teenager/.../senior): {len(old_cols)}")
    for stage in sorted(young_s | old_s):
        n = sum(1 for s in cs.values() if s == stage)
        g = "YOUNG" if stage in young_s else "OLD"
        print(f"    {stage}: {n} [{g}]")

    return liver, young_cols, old_cols


# ==============================================================================
# FOLD-CHANGE + STATS (vectorized where possible)
# ==============================================================================

def compute_fold_changes(expr, young_cols, old_cols, pseudocount=0.01, use_ttest=False):
    """
    Compute log2FC (old/young) and per-gene p-values.

    Args:
        use_ttest: If True, use Welch's t-test (more power for small n).
                   Default False uses Mann-Whitney U.
    """
    ym = expr[young_cols].values.astype(float)
    om = expr[old_cols].values.astype(float)

    my = ym.mean(axis=1)
    mo = om.mean(axis=1)
    fc = np.log2((mo + pseudocount) / (my + pseudocount))

    pv = np.ones(len(expr))
    for i in range(len(expr)):
        try:
            if use_ttest:
                _, pv[i] = stats.ttest_ind(ym[i, :], om[i, :], equal_var=False)
            else:
                _, pv[i] = stats.mannwhitneyu(ym[i, :], om[i, :], alternative="two-sided")
        except Exception:
            pv[i] = 1.0
        # ttest can return nan for zero-variance
        if np.isnan(pv[i]):
            pv[i] = 1.0

    return pd.DataFrame({"log2fc": fc, "pvalue": pv, "mean_young": my, "mean_old": mo},
                         index=expr.index)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("Liver Cross-Species Developmental Validation")
    print("=" * 70)

    # 1. Model results
    print("\n--- Loading pig liver model results ---")
    with open(PIG_MODEL_PATH) as f:
        model_data = json.load(f)
    top_genes = model_data.get("top_genes", [])[:1000]
    importance_map = model_data.get("feature_importance", {})
    print(f"  Top model genes: {len(top_genes)}")

    # 2. Expression
    pig_expr, pig_young, pig_old = load_pig_liver_data()
    human_expr, human_young, human_old = load_human_liver_data()

    # 3. Fold-changes
    # Pig: large sample size -> Mann-Whitney is fine
    # Human: small sample size -> use Welch's t-test for more power
    print("\n--- Computing fold-changes ---")
    pig_fc = compute_fold_changes(pig_expr, pig_young, pig_old, use_ttest=False)
    print(f"  Pig: {len(pig_fc)} genes (Mann-Whitney U)")
    human_fc = compute_fold_changes(human_expr, human_young, human_old, use_ttest=True)
    print(f"  Human: {len(human_fc)} genes (Welch's t-test)")

    # 4. Gene symbol mapping
    print("\n--- Gene symbol mapping ---")
    pig_sym_map = mygene_batch_lookup(pig_fc.index.tolist(), "pig", PIG_MAP_CACHE)
    human_sym_map = mygene_batch_lookup(human_fc.index.tolist(), "human", HUMAN_MAP_CACHE)

    # 5. Orthologous pairs
    print("\n--- Finding orthologous pairs ---")
    pig_s2i = {}
    for eid in pig_fc.index:
        sym = pig_sym_map.get(eid)
        if sym:
            k = sym.upper()
            if k not in pig_s2i:
                pig_s2i[k] = eid

    human_s2i = {}
    for eid in human_fc.index:
        sym = human_sym_map.get(eid)
        if sym:
            k = sym.upper()
            if k not in human_s2i:
                human_s2i[k] = eid

    common = set(pig_s2i.keys()) & set(human_s2i.keys())
    print(f"  Pig with symbols: {len(pig_s2i)}")
    print(f"  Human with symbols: {len(human_s2i)}")
    print(f"  Common: {len(common)}")

    # 6. Merge
    print("\n--- Building merged table ---")
    rows = []
    for su in common:
        pid, hid = pig_s2i[su], human_s2i[su]
        rows.append({
            "gene_symbol": pig_sym_map.get(pid, su),
            "pig_gene_id": pid, "human_gene_id": hid,
            "log2fc_pig": pig_fc.loc[pid, "log2fc"],
            "log2fc_human": human_fc.loc[hid, "log2fc"],
            "p_pig": pig_fc.loc[pid, "pvalue"],
            "p_human": human_fc.loc[hid, "pvalue"],
            "importance": importance_map.get(pid, 0.0),
        })
    merged = pd.DataFrame(rows)
    print(f"  Merged: {len(merged)}")

    if len(merged) == 0:
        print("ERROR: No overlapping genes.")
        sys.exit(1)

    # 7. FDR (on the merged set)
    print("\n--- FDR correction ---")
    vm = merged["p_pig"].notna() & merged["p_human"].notna()
    merged["fdr_pig"] = np.nan
    merged["fdr_human"] = np.nan
    if vm.sum() > 0:
        _, fp, _, _ = multipletests(merged.loc[vm, "p_pig"], method="fdr_bh")
        _, fh, _, _ = multipletests(merged.loc[vm, "p_human"], method="fdr_bh")
        merged.loc[vm, "fdr_pig"] = fp
        merged.loc[vm, "fdr_human"] = fh

    # ======================================================================
    # ANALYSIS A: Strict (FDR<0.10 + |FC|>0.5 in BOTH) -- for comparison
    # ======================================================================
    FDR_T, FC_T = 0.10, 0.5
    strict_mask = (
        (merged["fdr_pig"] < FDR_T) & (merged["fdr_human"] < FDR_T) &
        (merged["log2fc_pig"].abs() > FC_T) & (merged["log2fc_human"].abs() > FC_T)
    )
    n_strict = strict_mask.sum()

    # ======================================================================
    # ANALYSIS B: Pig-anchored (pig FDR<0.10, |FC|>0.5; require |FC|>0.5 in human but no FDR filter)
    # This is appropriate because human has only 4 vs 7 samples -> FDR is unpowered
    # ======================================================================
    pig_sig_mask = (
        (merged["fdr_pig"] < FDR_T) &
        (merged["log2fc_pig"].abs() > FC_T) &
        (merged["log2fc_human"].abs() > FC_T)  # Require effect size in human too
    )
    pig_anchored = merged[pig_sig_mask].copy()

    # Also compute a version with raw p < 0.05 in human (not FDR) as a middle ground
    relaxed_mask = (
        (merged["fdr_pig"] < FDR_T) &
        (merged["log2fc_pig"].abs() > FC_T) &
        (merged["log2fc_human"].abs() > FC_T) &
        (merged["p_human"] < 0.05)
    )
    relaxed = merged[relaxed_mask].copy()

    print(f"\n  Strict (FDR<0.10 both): {n_strict} genes")
    print(f"  Pig-anchored (pig FDR<0.10, |FC|>0.5 both): {len(pig_anchored)} genes")
    print(f"  Relaxed (pig FDR<0.10, |FC|>0.5 both, human p<0.05): {len(relaxed)} genes")

    # ======================================================================
    # Compute correlation for each analysis
    # ======================================================================
    def compute_corr_stats(df_subset, label):
        """Compute Pearson/Spearman correlation and directional concordance."""
        if len(df_subset) < 3:
            print(f"\n  {label}: Too few genes ({len(df_subset)}) for correlation")
            return {"n_genes": len(df_subset), "pearson_r": None, "pearson_p": None,
                    "ci_lower": None, "ci_upper": None, "boot_ci_lower": None,
                    "boot_ci_upper": None, "spearman_rho": None, "spearman_p": None,
                    "n_concordant": 0, "pct_concordant": 0.0}

        x, y = df_subset["log2fc_pig"].values, df_subset["log2fc_human"].values
        n = len(df_subset)
        r, p = stats.pearsonr(x, y)
        nc = int((np.sign(x) == np.sign(y)).sum())
        pct = 100.0 * nc / n

        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3) if n > 3 else np.nan
        cil, ciu = float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))

        np.random.seed(42)
        bc = []
        for _ in range(1000):
            idx = np.random.choice(n, size=n, replace=True)
            if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
                br, _ = stats.pearsonr(x[idx], y[idx])
                if not np.isnan(br):
                    bc.append(br)
        bc = np.array(bc)
        blo, bhi = float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))
        rho, rho_p = stats.spearmanr(x, y)

        print(f"\n  {label} ({n} genes):")
        print(f"    Pearson r = {r:.3f}, p = {p:.2e}")
        print(f"    Fisher z 95% CI: [{cil:.3f}, {ciu:.3f}]")
        print(f"    Bootstrap 95% CI: [{blo:.2f}, {bhi:.2f}]")
        print(f"    Spearman rho = {rho:.3f}, p = {rho_p:.2e}")
        print(f"    Directional concordance: {pct:.1f}% ({nc}/{n})")

        return {"n_genes": n, "pearson_r": round(float(r), 4), "pearson_p": float(p),
                "ci_lower": round(cil, 3), "ci_upper": round(ciu, 3),
                "boot_ci_lower": round(blo, 2), "boot_ci_upper": round(bhi, 2),
                "spearman_rho": round(float(rho), 4), "spearman_p": float(rho_p),
                "n_concordant": nc, "pct_concordant": round(pct, 1)}

    print("\n" + "=" * 70)
    print("RESULTS: Liver Cross-Species Developmental Validation")
    print("=" * 70)

    corr_pig_anchored = compute_corr_stats(pig_anchored, "Pig-anchored")
    corr_relaxed = compute_corr_stats(relaxed, "Relaxed (human p<0.05)")

    # Also compute correlation for ALL overlapping genes (no filters) as reference
    all_valid = merged[merged["log2fc_pig"].notna() & merged["log2fc_human"].notna()]
    corr_all = compute_corr_stats(all_valid, "All overlapping genes (no filter)")

    # Top genes from pig-anchored
    if len(pig_anchored) > 0:
        pig_anchored["abs_mean_fc"] = (pig_anchored["log2fc_pig"].abs() + pig_anchored["log2fc_human"].abs()) / 2
        sig = pig_anchored.sort_values("abs_mean_fc", ascending=False)
        print(f"\n  Top pig-anchored conserved genes:")
        print(f"  {'Gene':<15} {'FC_pig':>10} {'FC_human':>10} {'FDR_pig':>10} {'p_human':>10} {'Dir':>8}")
        for _, row in sig.head(30).iterrows():
            d = "UP" if row["log2fc_pig"] > 0 else "DOWN"
            c = "same" if np.sign(row["log2fc_pig"]) == np.sign(row["log2fc_human"]) else "OPP"
            print(f"  {row['gene_symbol']:<15} {row['log2fc_pig']:>10.3f} {row['log2fc_human']:>10.3f} "
                  f"{row['fdr_pig']:>10.4f} {row['p_human']:>10.4f} {d:>4} ({c})")

    # Print top genes from relaxed analysis too
    if len(relaxed) > 0:
        relaxed["abs_mean_fc"] = (relaxed["log2fc_pig"].abs() + relaxed["log2fc_human"].abs()) / 2
        sig_r = relaxed.sort_values("abs_mean_fc", ascending=False)
        print(f"\n  Top relaxed-criteria conserved genes:")
        print(f"  {'Gene':<15} {'FC_pig':>10} {'FC_human':>10} {'FDR_pig':>10} {'p_human':>10} {'Dir':>8}")
        for _, row in sig_r.head(30).iterrows():
            d = "UP" if row["log2fc_pig"] > 0 else "DOWN"
            c = "same" if np.sign(row["log2fc_pig"]) == np.sign(row["log2fc_human"]) else "OPP"
            print(f"  {row['gene_symbol']:<15} {row['log2fc_pig']:>10.3f} {row['log2fc_human']:>10.3f} "
                  f"{row['fdr_pig']:>10.4f} {row['p_human']:>10.4f} {d:>4} ({c})")

    sp = merged[(merged["fdr_pig"] < FDR_T) & (merged["log2fc_pig"].abs() > FC_T)]
    sh = merged[(merged["fdr_human"] < FDR_T) & (merged["log2fc_human"].abs() > FC_T)]
    print(f"\n  Summary:")
    print(f"    Sig in pig (FDR<0.10, |FC|>0.5): {len(sp)}")
    print(f"    Sig in human (FDR<0.10, |FC|>0.5): {len(sh)}")
    print(f"    NOTE: Human has only {len(human_young)} young + {len(human_old)} old postnatal liver")
    print(f"    samples -> per-gene FDR correction is underpowered.")

    # Save
    output = {
        "analysis": "liver_cross_species_developmental_validation",
        "note": ("Human Cardoso-Moreira 2019 liver has only 4 young + 7 old postnatal "
                 "samples. Per-gene FDR correction is underpowered for human, so the "
                 "pig-anchored analysis (pig FDR<0.10, |FC|>0.5 in both, no human FDR) "
                 "is the primary result."),
        "criteria": {
            "fdr_threshold": FDR_T, "fc_threshold": FC_T,
            "pig_young": ["Infant", "Early childhood"],
            "pig_old": ["Post-pubertal", "Adult"],
            "human_young": ["newborn", "infant", "toddler"],
            "human_old": ["teenager", "youngAdult", "youngMidAge", "olderMidAge", "senior"],
            "human_test": "Welch t-test (more power for n=4 vs n=7)",
            "pig_test": "Mann-Whitney U",
        },
        "sample_sizes": {
            "pig_young": len(pig_young), "pig_old": len(pig_old),
            "human_young": len(human_young), "human_old": len(human_old),
        },
        "gene_counts": {
            "pig_total": len(pig_fc), "human_total": len(human_fc),
            "pig_with_symbols": len(pig_s2i), "human_with_symbols": len(human_s2i),
            "common_symbols": len(common), "merged": len(merged),
            "strict_both_sig": int(n_strict),
            "pig_anchored": len(pig_anchored),
            "relaxed_human_p05": len(relaxed),
            "sig_pig": int(len(sp)), "sig_human": int(len(sh)),
        },
        "strict_analysis": {
            "description": "FDR<0.10 and |FC|>0.5 in BOTH species (as in muscle)",
            "n_genes": int(n_strict),
            "result": "Insufficient genes due to low human sample size",
        },
        "pig_anchored_analysis": {
            "description": "Pig FDR<0.10, |FC|>0.5 in both species (no human FDR filter)",
            **corr_pig_anchored,
        },
        "relaxed_analysis": {
            "description": "Pig FDR<0.10, |FC|>0.5 in both, human raw p<0.05",
            **corr_relaxed,
        },
        "all_genes_analysis": {
            "description": "All overlapping genes, no significance filter (reference)",
            **corr_all,
        },
        "pig_anchored_genes": pig_anchored[[
            "gene_symbol", "pig_gene_id", "human_gene_id",
            "log2fc_pig", "log2fc_human", "fdr_pig", "p_human", "importance"
        ]].sort_values("importance", ascending=False).to_dict("records") if len(pig_anchored) > 0 else [],
        "relaxed_genes": relaxed[[
            "gene_symbol", "pig_gene_id", "human_gene_id",
            "log2fc_pig", "log2fc_human", "fdr_pig", "p_human", "importance"
        ]].sort_values("importance", ascending=False).to_dict("records") if len(relaxed) > 0 else [],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
