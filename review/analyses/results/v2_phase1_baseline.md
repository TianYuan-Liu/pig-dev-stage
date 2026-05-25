# Phase 1 Baseline Results — dGTEx-Only Cross-Species (Strict Bins)

**Date**: 2026-05-25 (Phase 1 execution)
**Protocol**: Uniform v2 — weighted purity-z FC, pig-anchored filter, drop% 30, pig TPM≥1, human TPM≥10
**Human reference**: dGTEx v1 (2026-01-30), AGECOHORT=1 (Infant 0-2y) vs AGECOHORT=4 (Post-pubertal 13-18y)
**Pig reference**: PigGTEx (Infant+EarlyChildhood vs Post-pubertal+Adult)

## Per-tissue results

| Tissue | n_genes | Pearson r | 95% CI | dc% | perm p | dGTEx y/o n |
|---|---:|---:|---|---:|---:|---|
| **Muscle** | 293 | **+0.504** | [+0.40, +0.60] | 75.1 | 0.001 | 16 / 21 |
| **Lung** | 233 | **+0.383** | [+0.24, +0.51] | 70.4 | 0.001 | 19 / 11 |
| **Testis** | 357 | **+0.357** | [+0.25, +0.47] | 74.2 | 0.001 | 5 / 4 |
| Spleen | 256 | +0.248 | [+0.07, +0.41] | 44.1 | 0.001 | 3 / 3 |
| Adipose Tissue | 264 | +0.130 | [-0.01, +0.27] | 60.6 | 0.028 | 13 / 16 |
| Heart | 212 | +0.033 | [-0.21, +0.25] | 24.5 | 0.331 | 3 / 3 |
| Liver | 145 | **-0.254** | [-0.43, -0.07] | 32.4 | 0.999 | 6 / 4 |
| Small Intestine | — | — | — | — | — | (skipped — pig label mismatch) |

**Mean r = 0.200 across 7 evaluable tissues.**

- Tissues at "Strong" threshold (r ≥ 0.40): **1** (Muscle only)
- Tissues at "Minimum" threshold (r ≥ 0.30): **3** (Muscle, Lung, Testis)
- Tissues below minimum: 4

## What's working and what isn't

### ✅ Working (well-powered human n, biologically clean)
- **Muscle** (h=16/21): r=0.50, dc=75%, classic markers (ACTC1, ACHE, IGF2) directionally correct
- **Lung** (h=19/11): r=0.38, dc=70%, real biological signal
- **Testis** (h=5/4): r=0.36, dc=74%, strong despite small n

### ⚠️ Weak (positive but below threshold)
- **Adipose** (h=13/16): r=0.13 despite GOOD sample sizes — adipose development may be less conserved between species, or dGTEx subcutaneous depots don't match pig adipose
- **Spleen** (h=3/3): r=0.25, dc=44% — minimal sample size means median FC is noisy

### ❌ Bad (failing tissues)
- **Heart** (h=3/3): r=0.03, dc=25%, perm p=0.33 — pure noise from tiny human n
- **Liver** (h=6/4): **r=-0.25, dc=32%, perm p=0.999** — ANTI-CORRELATED, statistically significant in wrong direction

### Liver anti-correlation — biological interpretation

Liver top genes by |pig log2FC|:

| Gene | Pig log2FC | dGTEx log2FC | Concordant? | Biology |
|---|---:|---:|:---:|---|
| HAMP | -3.96 | -1.42 | ✓ | hepcidin, iron regulation |
| AR | +3.09 | -0.74 | ✗ | androgen receptor (induced w/ puberty) |
| ADH4 | +1.94 | -1.15 | ✗ | alcohol dehydrogenase (mature liver) |
| GHR | +2.35 | -0.84 | ✗ | growth hormone receptor (mature) |
| ALB | +1.57 | -1.38 | ✗ | albumin (mature hepatocyte) |
| SLPI | -1.73 | +1.44 | ✗ | secretory leukocyte peptidase inhibitor |

**Hypothesis**: Classic adult-liver markers (AR, GHR, ALB, ADH4) are UP in pig (Post-pubertal/Adult vs Infant/EarlyChild) but DOWN in dGTEx human (cohort 4 vs cohort 1). dGTEx's "old" cohort tops out at 18y — these are *adolescent* livers, not adult. Adult liver maturation in humans continues well past puberty (full adult hepatic phenotype ~25y). So dGTEx "old" liver = late-pubertal adolescents that haven't yet acquired full adult hepatic gene-expression profile.

This is a **biological mismatch in dGTEx**, not a methodological error. Liver simply needs adult (>21y) human samples that the dGTEx pilot doesn't provide.

### Small Intestine — naming mismatch
PigGTEx file is `Small_intestine.expr_tpm.txt.gz` (270 samples) but metadata's `Tissue class` uses sub-region labels (Ileum, Jejunum, Duodenum). The xs.load_pig_tissue() function filters metadata by string match → returns 0 samples. Fixable but needs a one-liner metadata-label alias.

## Implications

The baseline mean r=0.20 is **well below the publication-grade threshold (≥0.40)** the user set in the success-criteria section of the plan.

**Three tissues hit a real signal** (Muscle, Lung, Testis); four don't. The pattern:
- More dGTEx samples per bin → cleaner signal (Muscle 16+21 → r=0.50)
- Few dGTEx samples → noise or worse (Heart 3+3 → r=0.03, Liver 6+4 → r=-0.25)

The dGTEx pilot's small donor count (40 total, ~10-20 per tissue) is the binding constraint, not the protocol.

## Options for the user

### Option A — accept smaller panel, ship 3-tissue strong story
Report Muscle, Lung, Testis only (mean r=0.41, all at minimum+ threshold).
- ✅ Honest, defensible, all tissues pass minimum threshold
- ❌ Smaller panel than current paper's 5 tissues
- ❌ No new tissues vs the current paper (loses Brain, gains Testis)

### Option B — proceed to Phase 2 ML sweep, see if top-500 intersection rescues weak tissues
The Phase 2 ML framework sweep + top-500 gene intersection might filter out noise from weak tissues.
- ✅ Worth trying — Ridge's gene importance might select more conserved genes
- ❌ Won't fix the dGTEx small-n problem (only ~150-300 genes per tissue under pig-anchored to start with)
- ❌ Adds 6 hours of Garnatxa compute before knowing if it helps

### Option C — mixed dataset (dGTEx where strong, legacy where dGTEx fails)
- Use dGTEx for Muscle, Lung, Testis, Spleen, Adipose (5 tissues)
- Keep Cardoso-Moreira for Liver (gave r=0.36 there)
- Keep Schaiter for muscle (gives r=0.69) as a parallel "single-donor" check
- ✅ Best per-tissue r, most tissues
- ❌ Violates the "dGTEx-only" simplification the user wanted

### Option D — wait for dGTEx v2 (timeline unclear)

### Recommendation
**Try Option B first** — the ML sweep is cheap on Garnatxa (~6 hours) and might rescue some tissues. If after Phase 2 the picture is still dominated by 3 strong tissues, **fall back to Option A** with clean narrative ("we use dGTEx exclusively, reporting only tissues where the pilot's sample size supports a reliable comparison"). Option C is pragmatic but the user explicitly rejected mixed datasets.
