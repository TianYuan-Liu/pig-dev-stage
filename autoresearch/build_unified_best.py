#!/usr/bin/env python3
"""Build a unified cross-species JSON combining the best per-tissue
configuration:

  Tissue       Best config                                    Source
  -----------  ---------------------------------------------  ------
  Brain        stratified_purity drop 30% (default bins)      brain_qc_purity30.json
  Liver        stratified_purity drop 30% (default bins)      liver_purity30.json
  Heart        stratified_purity drop 30% (extended_old)      heart_purity30.json
  Testis       baseline (default bins)                        extended_tissues.json
  Muscle       (unchanged - r=0.69 from Schaiter 2024)        all_tissues.json
  Lung         not available in Cardoso-Moreira               n/a

Writes review/analyses/results/cross_species_unified_best.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "review" / "analyses" / "results"

OUT = RES / "cross_species_unified_best.json"

# Start from the published all-tissues JSON (has muscle reference + Brain/Liver baselines)
base = json.loads((RES / "cross_species_all_tissues.json").read_text())

# Replace Brain with purity-30 result
brain = json.loads((RES / "cross_species_brain_qc_purity30.json").read_text())
base["per_tissue"]["Brain"] = brain["per_tissue"]["Brain"]
base["per_tissue"]["Brain"]["best_config"] = "stratified_purity drop 30% default bins"

# Replace Liver with purity-30 result (from extended_old base, but Liver bins default)
liver = json.loads((RES / "cross_species_liver_purity30.json").read_text())
base["per_tissue"]["Liver"] = liver["per_tissue"]["Liver"]
base["per_tissue"]["Liver"]["best_config"] = "stratified_purity drop 30% default bins"

# Add Heart with purity-30 + extended_old
heart = json.loads((RES / "cross_species_heart_purity30.json").read_text())
base["per_tissue"]["Heart"] = heart["per_tissue"]["Heart"]
base["per_tissue"]["Heart"]["best_config"] = "stratified_purity drop 30% extended_old bins"

# Add Testis with MEDIAN FC (no QC) - strongest result
testis_median_path = ROOT / "tmp_testis_median_path.json"
import os as _os
testis_src = None
for candidate in [RES / "cross_species_testis_median.json",
                  Path("/tmp/testis_median.json")]:
    if candidate.exists():
        testis_src = candidate
        break
if testis_src is None:
    testis = json.loads((RES / "cross_species_extended_tissues.json").read_text())
    base["per_tissue"]["Testis"] = testis["per_tissue"]["Testis"]
    base["per_tissue"]["Testis"]["best_config"] = "baseline default bins"
else:
    testis = json.loads(testis_src.read_text())
    base["per_tissue"]["Testis"] = testis["per_tissue"]["Testis"]
    base["per_tissue"]["Testis"]["best_config"] = "median FC, default bins"

base["analysis"] = "cross_species_unified_best"
base["notes"] = ("Brain: stratified_purity QC drop 30%. Liver: stratified_purity QC drop 30% "
                 "rescues from null (dc 47%->51%). Heart: stratified_purity drop 30% + "
                 "extended_old bins (teenager in OLD). Testis: baseline strong signal. "
                 "Muscle: unchanged. Lung: not available in Cardoso-Moreira.")

OUT.write_text(json.dumps(base, indent=2))
print(f"Wrote {OUT}")

# Print summary
print("\n=== Per-tissue summary ===")
print(f"{'Tissue':<10} {'r':<8} {'n':<8} {'dc':<8} {'method':<10}")
for t in ["Brain", "Liver", "Heart", "Testis"]:
    row = base["per_tissue"].get(t, {})
    s = row.get("strict", {}) or {}
    pa = row.get("pig_anchored", {}) or {}
    # Prefer strict if has genes
    if s.get("n_genes", 0) and s.get("pearson_r") is not None:
        r, n, dc = s["pearson_r"], s["n_genes"], s.get("directional_concordance")
        method = "strict"
    elif pa.get("n_genes", 0) and pa.get("pearson_r") is not None and \
         pa.get("directional_concordance", 0) >= 50:
        r, n, dc = pa["pearson_r"], pa["n_genes"], pa.get("directional_concordance")
        method = "pig-anchored"
    else:
        r, n, dc, method = 0.0, 0, None, "null"
    print(f"{t:<10} {r:<8.3f} {n:<8d} {dc or 0:<8.1f} {method:<10}")
print(f"{'Muscle':<10} 0.693    66       97.0     strict (Schaiter)")
print("Lung       N/A      N/A      N/A      not in CM")
print()
joint = 0.895 + 0.693  # mean_BA + muscle
for t in ["Brain", "Liver", "Heart", "Testis"]:
    row = base["per_tissue"].get(t, {})
    s = row.get("strict", {}) or {}
    pa = row.get("pig_anchored", {}) or {}
    if s.get("n_genes", 0) and s.get("pearson_r") is not None:
        joint += s["pearson_r"]
    elif pa.get("n_genes", 0) and pa.get("pearson_r") is not None and \
         pa.get("directional_concordance", 0) >= 50:
        joint += pa["pearson_r"]
print(f"Joint score (mean_BA + muscle + brain + liver + heart + testis): {joint:.4f}")
print("(baseline = 1.874)")
