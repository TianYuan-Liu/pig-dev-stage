#!/usr/bin/env python3
"""Recompute the within-project fold-change validation (Supp Table S3) under the
6-stage scheme. Mirrors machine_learning/analysis within-project methodology:
two largest single-project muscle cohorts; per-gene log2(TPM+1) stage means;
log2FC = mean(old) - mean(young); Pearson r of project-FC vs full-dataset-FC over
all genes; pooled (mean of the two project FCs) with 1000x bootstrap 95% CI.
6-stage baseline = the earliest stage with >=5 samples in BOTH cohorts.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # provenance dir: sibling modules
import transfer_6stage as t6

COHORTS = ["PRJNA488311", "PRJNA486202"]
ORDER = t6.ORDER6


def stage_means(Xg, samples):
    return np.log2(Xg.loc[samples].values + 0.0 + 1.0) if False else Xg.loc[samples].mean(axis=0).values


def main():
    X = t6.load_pig_tissue("Muscle")          # log2(TPM+1) samples x genes (+Age_Days,BioProject)
    stage = X["Age_Days"].apply(t6.stage6)
    bp = X["BioProject"].astype(str)
    genes = X.drop(columns=["Age_Days", "BioProject"])

    # 6-stage composition of the two cohorts
    comp = {c: stage[bp == c].value_counts().reindex(ORDER).fillna(0).astype(int).to_dict() for c in COHORTS}
    print("Cohort 6-stage composition:")
    for c in COHORTS:
        print(f"  {c}: " + ", ".join(f"{s}={comp[c][s]}" for s in ORDER if comp[c][s] > 0))

    # baseline = earliest stage with >=5 in both cohorts
    base = next((s for s in ORDER if all(comp[c][s] >= 5 for c in COHORTS)), None)
    later = [s for s in ORDER if ORDER.index(s) > ORDER.index(base)
             and all(comp[c][s] >= 4 for c in COHORTS)]
    print(f"\nbaseline={base}; contrasts vs {later}")

    def fc(mask_young, mask_old):
        return genes.loc[mask_old].mean(axis=0).values - genes.loc[mask_young].mean(axis=0).values

    out = {"baseline": base, "scheme": "6stage", "cohorts": COHORTS,
           "composition": comp, "comparisons": {}}
    rng = np.random.default_rng(42)
    for old in later:
        full_y = (stage == base).values
        full_o = (stage == old).values
        fc_full = fc(full_y, full_o)
        proj_fcs = []
        per_proj = {}
        for c in COHORTS:
            y = ((stage == base) & (bp == c)).values
            o = ((stage == old) & (bp == c)).values
            f = fc(y, o)
            proj_fcs.append(f)
            r = stats.pearsonr(f, fc_full)[0]
            per_proj[c] = {"n_young": int(y.sum()), "n_old": int(o.sum()), "r": round(float(r), 4)}
        pooled = np.mean(proj_fcs, axis=0)
        r_pool = stats.pearsonr(pooled, fc_full)[0]
        # bootstrap CI over genes
        n = len(pooled); boots = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            boots.append(stats.pearsonr(pooled[idx], fc_full[idx])[0])
        ci = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
        out["comparisons"][f"{base} vs {old}"] = {
            "n_full_young": int(full_y.sum()), "n_full_old": int(full_o.sum()),
            "per_project": per_proj, "pooled_r": round(float(r_pool), 4), "pooled_ci95": ci,
            "n_genes": int(n)}
        print(f"  {base} vs {old}: pooled r={r_pool:.4f} CI{ci} "
              f"(n_full {int(full_y.sum())}/{int(full_o.sum())})")

    p = ROOT / "paper" / "figures" / "data" / "within_project_6stage.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
