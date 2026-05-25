# Verification Re-Analysis — dGTEx Cross-Species Autoresearch

14-agent independent verification (workflow `verify_reanalysis.mjs`): 6 step audits +
4 experiments + 3 adversarial reviewers + synthesis. Every step independently
re-derived from raw data (not by re-running repo code).

## Verdict on "ML-feature r < DE-gene r": EXPECTED, not a bug

The cross-species metric IS `Spearman(log2fc_pig, log2fc_human)`. DE selection ranks
genes on `|log2fc_pig|`/pig-FDR — i.e. it directly selects the large, reliable pig
fold-changes that are exactly the quantity being correlated. ML importance optimises a
different objective (developmental-stage discrimination). Proof (Muscle, matched n=117):

| Gene-selection criterion | cross-species Spearman r |
|---|---|
| `|log2fc_pig|` descending (DE magnitude) | 0.521 |
| pig FDR ascending (DE significance) | 0.488 |
| ML ridge `|coef|*std` importance | 0.307 |
| random orthologs (50-draw mean) | 0.214 |

So DE > ML > random is mechanistic, not an artifact. The ML-feature signal is still
**genuine** (ortholog-scramble permutation: Muscle r=0.293 p=0.003; Lung r=0.246 p=0.015).

### Fair-comparison nuance (partially vindicates the original concern)
The DE `pig_anchored` filter conditions on `|log2fc_human|>0.5`, which partly conditions
on the outcome and inflates DE r by ~0.14. Under a **pig-only** filter (no human-FC gate):

| Tissue | DE (pig-only) r | ML ridge_std r |
|---|---|---|
| Muscle | 0.391 | 0.307 (DE wins) |
| Lung | 0.227 | **0.263 (ML edges ahead)** |

⟹ "DE > ML" is robust for Muscle / on the conditioned comparison, but narrows sharply
(and reverses for Lung) on a strictly apples-to-apples footing. State as a caveat, not a law.

## Pipeline audit: 4/6 steps bug-free, 2 minor bugs

- **gene-ID matching — CLEAN.** No format bug. Only ~23% of pig genes have 1:1 human
  orthologs expressed in both species; ridge top-1000 → ~30% in the testable universe
  (above the 23% random rate = mild enrichment). The small overlap is biology, not a bug.
- **pig FC — CORRECT** (verified vs ACTC1↓, NEXN↑, MYH3↓ ground truth).
- **human FC — CORRECT** (max delta 0.000000 vs independent recompute; young/old not swapped).
- **pig_anchored filter — CORRECT** (100% gene-set match; FDR on pig p-values only).

### Bug 1 (minor): Spearman winner reported with a Pearson bootstrap CI
`compute_stat` (uniform_gene_selection_sweep.py:152) calls `bootstrap_pearson_ci`
unconditionally. Point estimates/p/perm-p/LOO unchanged; only the CI was mislabelled.
**Fixed** via `fix_spearman_ci.py` (correct Spearman percentile bootstrap, n=1000, seed=42);
`v2_uniform_winner.json` + `fig4_v2_dgtex_summary.tsv` CIs regenerated. The frozen evaluator
was NOT modified. Corrected CIs (e.g. Adipose now [0.044, 0.281], excludes 0 — consistent
with perm p=0.009).

### Bug 2 (minor, hygiene): `ridge_continuous` top_genes ranked by scale-confounded `|coef|`
`OrdinalRidge.get_feature_importance` returns raw `|coef|` on unstandardised log2(TPM+1)
(models.py:553); `ridge_std` returns `|coef|*std` (correct; models.py:570). Same model,
identical BA. All ML-feature cross-species analysis uses `ridge_std`. No analytic output
changes; use `ridge_std` for any biological gene ranking.

## What changed in committed results
- Reported Spearman CIs corrected (Bug 1). No headline number changes (Muscle 0.535,
  Lung 0.362, panel mean 0.356 stand; ML-feature ridge_std Muscle 0.307 / Lung 0.263).
- No data regeneration or re-runs required beyond the CI recompute.

Confidence: high (Bug 1/2 confirmed at source; DE>ML mechanism empirically demonstrated;
3/3 reviewers reproduced all numbers; the Lung pig-only reversal is the sole caveat).
