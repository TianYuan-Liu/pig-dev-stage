#!/usr/bin/env python3
"""Feature-selection stability of the 6-stage PLS-VIP biomarkers (Fig S1b/c).

Reproducible, self-contained 6-stage regeneration of the feature-stability
panels. The previous feature_stability_results.json / expression_correlation_
summary.csv on disk were a pre-migration (5-stage, Feb-6) run whose generator was
never committed; this script replaces them with the canonical 6-stage analysis,
using the SAME OrdinalPLS / VIP definition as the biomarker table
(review/analyses/pls_vip_features_6stage.py).

PROTOCOL (per tissue: Muscle, Brain, Liver, Blood, Lung)
  - Load pig expression, assign the six ordinal stages (boundaries 3/28/90/165/540,
    drop-small <10), expression = log2(TPM+1) (identical to the biomarker pipeline).
  - Full-data fit: OrdinalPLS(n_components=10) on all retained samples; rank genes
    by VIP. The top-50 of this fit define panel-c's feature-stage correlation
    (mean |Spearman(gene, ordinal stage)|) and the cross-tissue overlap.
  - Stability (panel b): for each of 10 seeds, draw a bootstrap resample of the
    samples (n with replacement), refit OrdinalPLS, take the top-50 VIP genes.
    Report the mean +/- s.d. of the C(10,2)=45 pairwise Jaccard similarities of
    these top-50 sets, plus the fraction of top-50 genes recurring in >=80% (>=8)
    of the bootstrap fits (the "stable" fraction quoted in the main text).
  - Cross-tissue overlap: mean pairwise Jaccard of the five full-fit top-50 sets.

OUTPUTS
  machine_learning/analysis/results/feature_stability_results.json
  machine_learning/analysis/results/expression_correlation_summary.csv
  machine_learning/analysis/results/feature_stability_cross_tissue.json

Deterministic (fixed seed list). Real data only; no resampling of gene labels.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/Users/tianyuan/Desktop/github_dev/pig-dev-stage")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper" / "figures" / "provenance"))

from transfer_6stage import (  # noqa: E402
    assign_ordinal_stages, load_pig_tissue, N_COMPONENTS,
)
from machine_learning.model_training.models import OrdinalPLS  # noqa: E402

RESULTS_DIR = ROOT / "machine_learning" / "analysis" / "results"
TISSUES = ["Muscle", "Brain", "Liver", "Blood", "Lung"]
TOP_K = 50
SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]  # 10 seeds
SCHEME = "6stage"


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def prep_tissue(tissue: str):
    pig_X = load_pig_tissue(tissue)
    stage, kept = assign_ordinal_stages(pig_X["Age_Days"], SCHEME)
    code = {s: i for i, s in enumerate(kept)}
    mask = stage.notna().values
    feat = pig_X.drop(columns=["Age_Days", "BioProject"]).loc[mask]  # log2(TPM+1)
    y = np.array([code[s] for s in stage[mask].values])
    return feat.reset_index(drop=True), y, kept


def top_vip(feat: pd.DataFrame, y: np.ndarray, k: int = TOP_K):
    pls = OrdinalPLS(n_components=N_COMPONENTS, seed=42)
    pls.fit(feat, y)
    return pls.get_feature_importance().index[:k].tolist()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Feature-selection stability at 6-stage (Fig S1b/c)")
    print("=" * 78)

    results: dict = {}
    corr_rows: list = []
    full_top: dict = {}

    for tissue in TISSUES:
        feat, y, kept = prep_tissue(tissue)
        n = len(y)

        # --- full-data fit: defines panel-c correlation + cross-tissue overlap
        full = top_vip(feat, y)
        full_top[tissue] = set(full)
        abs_rhos = []
        for g in full:
            r = spearmanr(feat[g].values, y).statistic
            if r == r:  # not NaN
                abs_rhos.append(abs(float(r)))
        mean_abs_r = float(np.mean(abs_rhos)) if abs_rhos else float("nan")
        corr_rows.append({"Tissue": tissue, "Mean |r|": round(mean_abs_r, 4)})

        # --- bootstrap stability across 10 seeds
        per_seed: dict = {}
        for s in SEEDS:
            rng = np.random.default_rng(s)
            idx = rng.integers(0, n, n)  # bootstrap resample (with replacement)
            sub = feat.iloc[idx].reset_index(drop=True)
            ysub = y[idx]
            per_seed[str(s)] = top_vip(sub, ysub)

        sets = {s: set(g) for s, g in per_seed.items()}
        pairwise = [jaccard(sets[a], sets[b]) for a, b in combinations(sets, 2)]
        # gene recurrence across the 10 bootstrap fits
        freq: dict = {}
        for genes in per_seed.values():
            for g in genes:
                freq[g] = freq.get(g, 0) + 1
        # "stable" = recurring in >=80% of the full-fit top-50 genes' fits
        n_stable = sum(1 for g in full if freq.get(g, 0) >= 0.8 * len(SEEDS))
        stable_fraction = n_stable / TOP_K

        results[tissue] = {
            "scheme": SCHEME,
            "n_seeds": len(SEEDS),
            "n_samples": int(n),
            "kept_stages": kept,
            "top_k": TOP_K,
            "jaccard_similarity": {
                "mean": round(float(np.mean(pairwise)), 4),
                "std": round(float(np.std(pairwise)), 4),
                "all_values": [round(float(v), 4) for v in pairwise],
            },
            "n_stable_80pct": int(n_stable),
            "stable_fraction": round(stable_fraction, 4),
            "expression_correlation": {"mean_abs_spearman": round(mean_abs_r, 4)},
            "selected_genes_per_seed": per_seed,
            "gene_frequency": freq,
        }
        print(f"  {tissue:7s} n={n:4d}  Jaccard={results[tissue]['jaccard_similarity']['mean']:.3f}"
              f"  stable={n_stable}/{TOP_K} ({stable_fraction*100:.0f}%)"
              f"  |rho|={mean_abs_r:.3f}", flush=True)

    # --- cross-tissue overlap (mean pairwise Jaccard of full-fit top-50 sets)
    ct = [jaccard(full_top[a], full_top[b]) for a, b in combinations(full_top, 2)]
    cross = {
        "top_k": TOP_K,
        "pairwise_jaccard": {f"{a}|{b}": round(jaccard(full_top[a], full_top[b]), 5)
                             for a, b in combinations(full_top, 2)},
        "mean_jaccard": round(float(np.mean(ct)), 5),
        "mean_jaccard_pct": round(float(np.mean(ct)) * 100, 3),
    }

    (RESULTS_DIR / "feature_stability_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(corr_rows).to_csv(RESULTS_DIR / "expression_correlation_summary.csv", index=False)
    (RESULTS_DIR / "feature_stability_cross_tissue.json").write_text(json.dumps(cross, indent=2))

    jvals = [results[t]["jaccard_similarity"]["mean"] for t in TISSUES]
    svals = [results[t]["stable_fraction"] for t in TISSUES]
    rvals = [r["Mean |r|"] for r in corr_rows]
    print("\nSUMMARY (for reconciling the manuscript):")
    print(f"  Within-tissue Jaccard range : {min(jvals):.2f} - {max(jvals):.2f}")
    print(f"  Stable fraction range       : {min(svals)*100:.0f}% - {max(svals)*100:.0f}%")
    print(f"  Mean |Spearman| range       : {min(rvals):.2f} - {max(rvals):.2f}")
    print(f"  Cross-tissue Jaccard        : {cross['mean_jaccard_pct']:.2f}%")


if __name__ == "__main__":
    main()
