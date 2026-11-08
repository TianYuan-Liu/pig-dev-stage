#!/usr/bin/env python3
"""Regenerate the within-stage subgroup-heterogeneity CSVs at 6-stage.

Writes (drop-in replacements for figS8_heterogeneity.R):
  review/analyses/results/heterogeneity_stratification.csv
      tissue, stage, dimension, level, n_samples       (composition per stage)
  review/analyses/results/heterogeneity_subgroup_performance.csv
      tissue, dimension, level, n_samples, n_stages_represented, balanced_accuracy
  review/analyses/results/heterogeneity_permutation_pvalues.csv
      tissue, dimension, n_levels_used, usable_levels, observed_std_ba,
      perm_null_mean_std, perm_null_q95_std, pvalue, n_permutations

Model: OrdinalPLS (default 10 comp), 5-fold StratifiedKFold OOF predictions
(seed 42) -- the same in-distribution stratified protocol the heterogeneity
analysis always used. Subgroup BA = BA of the OOF predictions within each
Breed/Sex/BioProject level (n>=10, >=2 true stages). Permutation null = labels
shuffled WITHIN stage (1000x) so it preserves the stage composition of each
subgroup. 6-stage drop-small (<10). Real data only.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
sys.path.insert(0, str(ROOT))
from machine_learning.data_processing.data_loader import DataLoader
from machine_learning.model_training.models import OrdinalPLS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

DATA_DIR = ROOT / "data" / "pigGTEx"
META = ROOT / "data" / "PigGTEx_v0.MetaTable.csv"
RESULTS = ROOT / "review" / "analyses" / "results"
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
DIMS = ["Breed", "Sex", "BioProject"]
MIN_PER_STAGE = 10
N_COMPONENTS = 10
SEED = 42
N_FOLDS = 5
MIN_GROUP_N = 10
N_PERM = 1000
ORDER6 = ["Neonatal", "Suckling", "Juvenile", "Pre-pubertal", "Post-pubertal", "Adult"]


def stage6(d):
    if pd.isna(d): return None
    if d <= 3:   return "Neonatal"
    if d <= 28:  return "Suckling"
    if d <= 90:  return "Juvenile"
    if d <= 165: return "Pre-pubertal"
    if d <= 540: return "Post-pubertal"
    return "Adult"


def drop_small(labels, order):
    counts = pd.Series(labels).value_counts()
    return [s for s in order if counts.get(s, 0) >= MIN_PER_STAGE]


def norm_sex(v):
    m = {"male": "Male", "female": "Female", "pooled": "Pooled", "neuter": "Neuter",
         "unknown": "Unknown", "nan": "Unknown", "": "Unknown",
         "hermaphrodite": "Hermaphrodite"}
    return m.get(str(v).strip().lower(), "Unknown")


def main():
    raw_idx = pd.read_csv(META).set_index("BioSample")
    dl = DataLoader(data_dir=DATA_DIR, metadata_path=META)
    dl.load_metadata()

    strat_rows, perf_rows, perm_rows = [], [], []

    for tissue in TISSUES:
        expr, meta = dl.load_expression(tissue)
        samples = list(expr.columns)
        X = np.log2(expr.T.values + 1.0)
        age = pd.to_numeric(meta["Age_Days"] if "Age_Days" in meta else meta["Age"],
                            errors="coerce").values
        bp = meta["BioProject"].astype(str).values
        breed = raw_idx.reindex(samples)["Breed"].astype(str).values
        sex = np.array([norm_sex(v) for v in raw_idx.reindex(samples)["Sex"].values])

        y6 = pd.Series([stage6(a) for a in age])
        kept = drop_small(y6, ORDER6)
        mask = y6.isin(kept).values & ~pd.isna(age)
        Xk = X[mask]
        yk = y6[mask].values
        bpk, breedk, sexk = bp[mask], breed[mask], sex[mask]
        code = {s: i for i, s in enumerate(kept)}
        yi = np.array([code[v] for v in yk])
        dimvals = {"Breed": breedk, "Sex": sexk, "BioProject": bpk}

        # ---- stratification composition (per stage) ----
        for dim in DIMS:
            dv = dimvals[dim]
            for st in kept:
                sm = (yk == st)
                for lvl, cnt in pd.Series(dv[sm]).value_counts().items():
                    strat_rows.append({"tissue": tissue, "stage": st, "dimension": dim,
                                       "level": lvl, "n_samples": int(cnt)})

        # ---- 5-fold stratified OOF predictions ----
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        oof = np.full(len(yi), -1, int)
        for tr, te in skf.split(Xk, yi):
            if len(np.unique(yi[tr])) < 2:
                continue
            m = OrdinalPLS(n_components=N_COMPONENTS, seed=SEED)
            m.fit(pd.DataFrame(Xk[tr]), yi[tr])
            oof[te] = np.asarray(m.predict(pd.DataFrame(Xk[te])), int)
        ok = oof >= 0

        # overall marginal BA
        perf_rows.append({"tissue": tissue, "dimension": "Overall", "level": "All",
                          "n_samples": int(ok.sum()),
                          "n_stages_represented": int(np.unique(yi[ok]).size),
                          "balanced_accuracy": balanced_accuracy_score(yi[ok], oof[ok])})

        # ---- subgroup BA + permutation null per dimension ----
        for dim in DIMS:
            dv = dimvals[dim]
            usable = []
            for lvl in pd.unique(dv):
                sm = (dv == lvl) & ok
                if sm.sum() < MIN_GROUP_N or np.unique(yi[sm]).size < 2:
                    continue
                usable.append(lvl)
                perf_rows.append({"tissue": tissue, "dimension": dim, "level": str(lvl),
                                  "n_samples": int(sm.sum()),
                                  "n_stages_represented": int(np.unique(yi[sm]).size),
                                  "balanced_accuracy": balanced_accuracy_score(yi[sm], oof[sm])})
            if len(usable) < 2:
                continue

            def spread(labels):
                bas = []
                for lvl in usable:
                    sm = (labels == lvl) & ok
                    if sm.sum() < MIN_GROUP_N or np.unique(yi[sm]).size < 2:
                        continue
                    bas.append(balanced_accuracy_score(yi[sm], oof[sm]))
                return float(np.std(bas)) if len(bas) >= 2 else np.nan

            obs = spread(dv)
            rng = np.random.default_rng(SEED)
            stage_idx = {s: np.where(yk == s)[0] for s in np.unique(yk)}  # within-stage groups
            null = []
            for _ in range(N_PERM):
                perm = dv.copy()
                for idx in stage_idx.values():
                    sub = perm[idx].copy()
                    rng.shuffle(sub)
                    perm[idx] = sub
                s = spread(perm)
                if not np.isnan(s):
                    null.append(s)
            null = np.asarray(null)
            pval = float((null >= obs).sum() / len(null)) if len(null) else np.nan
            perm_rows.append({"tissue": tissue, "dimension": dim,
                              "n_levels_used": len(usable),
                              "usable_levels": ";".join(map(str, usable)),
                              "observed_std_ba": obs,
                              "perm_null_mean_std": float(np.mean(null)) if len(null) else np.nan,
                              "perm_null_q95_std": float(np.quantile(null, 0.95)) if len(null) else np.nan,
                              "pvalue": pval, "n_permutations": int(len(null))})

        print(f"  {tissue}: kept {kept} | overall BA "
              f"{perf_rows[-1]['balanced_accuracy'] if perf_rows[-1]['dimension']=='Overall' else '?'}")
        sys.stdout.flush()

    pd.DataFrame(strat_rows).to_csv(RESULTS / "heterogeneity_stratification.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(RESULTS / "heterogeneity_subgroup_performance.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(RESULTS / "heterogeneity_permutation_pvalues.csv", index=False)
    print("\nWrote 3 heterogeneity CSVs (6-stage) to", RESULTS)
    print("perm p-values:")
    print(pd.DataFrame(perm_rows)[["tissue", "dimension", "observed_std_ba", "pvalue"]].to_string(index=False))


if __name__ == "__main__":
    main()
