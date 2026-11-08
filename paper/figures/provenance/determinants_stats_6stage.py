#!/usr/bin/env python3
"""Reproduce the prose-only determinant statistics quoted in the manuscript from
the committed per-cell transfer data (paper/figures/data/determinants_6stage.json),
closing the provenance gap for the crossed mixed-model variance components and the
random-six-level-factor permutation null. Real data only; deterministic seed.

Reproduces, from the 40 Cardoso pig-to-target cells:
  - OLS variance partition: organ-only R^2, Type-II ANOVA F/p (organ, divergence),
    df-adjusted unique R^2 and raw unique R^2.
  - Random six-level-factor permutation null (mean R^2, 95th pct, p) vs observed
    one-way organ R^2.
  - Crossed mixed model rho ~ divergence + (1|organ) + (1|species): variance
    components for organ and species.
  - Species-level and cell-level Spearman(rho, divergence).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
SRC = ROOT / "paper" / "figures" / "data" / "determinants_6stage.json"
SEED = 42
N_PERM = 5000


def main() -> None:
    pts = json.loads(SRC.read_text())["points"]
    df = pd.DataFrame(pts)[["organ", "species", "rho", "divergence_mya"]]
    n = len(df)
    print(f"Loaded {n} Cardoso cells from {SRC.name}\n")

    # --- OLS variance partition -------------------------------------------------
    full = smf.ols("rho ~ C(organ) + divergence_mya", data=df).fit()
    organ_only = smf.ols("rho ~ C(organ)", data=df).fit()
    div_only = smf.ols("rho ~ divergence_mya", data=df).fit()
    aov = sm.stats.anova_lm(full, typ=2)
    r2_organ_only = organ_only.rsquared
    # df-adjusted unique R^2 = increment in adjusted R^2 from adding the term
    uniq_organ_adj = full.rsquared_adj - div_only.rsquared_adj
    uniq_div_adj = full.rsquared_adj - organ_only.rsquared_adj
    uniq_organ_raw = full.rsquared - div_only.rsquared
    uniq_div_raw = full.rsquared - organ_only.rsquared
    print("[OLS variance partition]")
    print(f"  organ-only R^2            = {r2_organ_only:.4f}  (paper 0.56)")
    print(f"  organ  Type-II F, p       = {aov.loc['C(organ)','F']:.3f}, "
          f"{aov.loc['C(organ)','PR(>F)']:.2e}  (paper 9.0, 2e-5)")
    print(f"  divergence Type-II F, p   = {aov.loc['divergence_mya','F']:.3f}, "
          f"{aov.loc['divergence_mya','PR(>F)']:.3f}  (paper 3.9, 0.056)")
    print(f"  df-adj unique R^2 organ   = {uniq_organ_adj:.4f}  (paper 0.49)")
    print(f"  df-adj unique R^2 diverg. = {uniq_div_adj:.4f}  (paper 0.04)")
    print(f"  raw unique R^2 organ/div  = {uniq_organ_raw:.4f} / {uniq_div_raw:.4f}  (paper 0.54 / 0.05)")

    # --- random six-level-factor permutation null ------------------------------
    rng = np.random.default_rng(SEED)
    y = df["rho"].values
    r2s = []
    for _ in range(N_PERM):
        lab = rng.integers(0, 6, n)
        d2 = pd.DataFrame({"rho": y, "f": lab})
        r2s.append(smf.ols("rho ~ C(f)", data=d2).fit().rsquared)
    r2s = np.array(r2s)
    p_perm = (np.sum(r2s >= r2_organ_only) + 1) / (N_PERM + 1)
    print("\n[random six-level-factor null]")
    print(f"  mean R^2 = {r2s.mean():.4f}  (paper 0.13);  95th pct = {np.percentile(r2s,95):.4f}  (thesis 0.26)")
    print(f"  observed organ R^2 = {r2_organ_only:.4f};  permutation p = {p_perm:.4f}  (paper 0.0002)")

    # --- crossed mixed model ----------------------------------------------------
    # variance components for organ and species (divergence as fixed effect)
    df_vc = df.copy()
    df_vc["grp"] = 1
    md = smf.mixedlm("rho ~ divergence_mya", df_vc, groups="grp",
                     vc_formula={"organ": "0 + C(organ)", "species": "0 + C(species)"})
    mf = md.fit(reml=True)
    vc = mf.cov_re  # variance components
    organ_v = float(mf.vcomp[0]); species_v = float(mf.vcomp[1])
    print("\n[crossed mixed model rho ~ divergence + (1|organ) + (1|species)]")
    print(f"  organ variance component   = {organ_v:.4f}  (paper 0.039)")
    print(f"  species variance component = {species_v:.4f}  (paper 0.002)")
    print(f"  divergence beta = {mf.fe_params['divergence_mya']:.5f}/Myr  (paper -0.0007)")

    # --- Spearman tests ---------------------------------------------------------
    sp_cell = spearmanr(df["divergence_mya"], df["rho"])
    sm_mean = df.groupby("species")["rho"].mean()
    div_by_sp = df.groupby("species")["divergence_mya"].first()
    sp_species = spearmanr(div_by_sp.values, sm_mean.values)
    print("\n[phylogeny]")
    print(f"  cell-level Spearman   = {sp_cell.statistic:.3f}, p={sp_cell.pvalue:.3f}, n={n}  (paper -0.25, 0.11)")
    print(f"  species-level Spearman= {sp_species.statistic:.3f}, p={sp_species.pvalue:.3f}, n={len(sm_mean)}  (paper -0.76, 0.049)")


if __name__ == "__main__":
    main()
