# dGTEx Discovery Report (Stage A)

**Date**: 2026-05-25
**Source**: dGTEx Analysis 2026-01-30 v1 (Developmental GTEx, NIH Common Fund pilot)
**Purpose**: Verify dGTEx data structure, define stage mapping, build tissue inclusion list — before any Stage B compute.

---

## 1. What was downloaded

4 metadata XLSX files from `https://storage.googleapis.com/developmental-gtex/annotations/v1/metadata-files/`:

| File | Size | Rows × Cols | Purpose |
|---|---|---|---|
| `Sample_Attributes_Public_DD.xlsx` | 123 KB | data dictionary | Column definitions for sample-attributes file |
| `Subject_Phenotypes_Public_DD.xlsx` | 43 KB | 195 × 50 | Column definitions for subject-phenotypes file |
| `Sample_Attributes_Dataset_DS_open.xlsx` | 320 KB | 379 × 106 | Per-sample data (tissue, donor, batch, QC) |
| `Subject_Phenotypes_Dataset_DS_open.xlsx` | 11 KB | **40 × 6** | Per-donor phenotype data (AGECOHORT, SEX, BMI, …) |

**Expression data NOT yet downloaded** — see Section 6 (Open issues).

---

## 2. The dGTEx pilot is small but ideally structured for our purpose

### Donor cohort (40 total donors)

dGTEx encodes developmental age as `AGECOHORT` ∈ {1, 2, 3, 4} with the following **dictionary-confirmed mapping** (verbatim from `Subject_Phenotypes_Public_DD.xlsx`):

| AGECOHORT | Label | Human age range | dGTEx donor count |
|---|---|---|---:|
| 1 | **Infant** | 0 to 2 years | 14 |
| 2 | **Early Childhood** | 2 to 8 years | 3 |
| 3 | **Pre-pubertal** | 8 to ~13 years | 4 |
| 4 | **Post-pubertal** | ~13 to 18 years | 19 |

**This is a HUGE win**: dGTEx uses the **exact same 4-stage developmental scheme** as our pig staging (Infant, Early childhood, Pre-pubertal, Post-pubertal). Pig stages 1–4 map directly to dGTEx AGECOHORT 1–4. No translation layer needed.

dGTEx also has continuous-age columns (`AGEYEARS`, `AGEMONTHS`, `AGEDAYS`) but these are **not in the open-access release** — only AGECOHORT is publicly available. This is fine for our binned comparison.

Additional donor-level fields available: `SEX` (32 M / 8 F — heavily male-biased), `BMI`, `DTHHRDY` (Hardy death-classification scale), `PUBSTAGE` (Tanner stage 1–3).

### Sample table (379 rows)

The 379 rows in `Sample_Attributes_Dataset_DS_open.xlsx` include:
- 332 RNA-seq samples (`SMAFRZE == "RNASEQ"`)
- 37 WGS DNA samples (genotyping; ignore)
- 10 samples flagged EXCLUDE (quality fail; ignore)

Key columns identified:
- `SAMPLE_ID` — unique sample identifier (format: `DGTEX###-Aliquot-NN-SM-Lane`)
- `SUBJECT_ID` — links to donor table (DGTEX001, DGTEX002, …)
- `SMTS` — tissue (broad, 24 unique values) → use this for tissue matching
- `SMTSD` — tissue (detailed sub-region, 52 unique values) → for sub-region awareness
- `SMTSUBRTRM`, `SMTSDUBRTRM` — UBERON terms (useful for ontology mapping)
- `SMNABTCH`, `SMGEBTCH` — batch IDs (for batch-effect QC)
- 95 other QC columns (mapping rate, RIN, intron/intergenic ratios, etc.) for downstream sample-QC filtering

---

## 3. Stage mapping (uniform pig ↔ human)

### Pig (UNCHANGED from current paper)
- **Young** = Infant + Early childhood (PigGTEx Age 0–59 days)
- **Old**   = Post-pubertal + Adult (PigGTEx Age ≥150 days)

### Human (dGTEx)
- **Young (strict)** = AGECOHORT 1 only (Infant, 0–2 years)
- **Old (strict)**   = AGECOHORT 4 only (Post-pubertal, 13–18 years)

Permissive alternative (if power needed):
- Young (permissive) = AGECOHORT 1 + 2 (Infant + Early Childhood)
- Old (permissive) = AGECOHORT 3 + 4 (Pre-pubertal + Post-pubertal)

### Biological rationale for the strict scheme

Both species' "young" anchor lands at the youngest developmental cohort available; both species' "old" anchor lands at the most mature pre-adult cohort. Critically:
- dGTEx pilot does NOT include adult (>18 years) samples, so dGTEx "old" = pubertal adolescents
- Pig "old" extends to truly adult animals (≥365 days)
- This is an asymmetry to flag: dGTEx's young-vs-old comparison spans 0→16 years; pig's spans 0→365+ days. Both cover the full developmental trajectory available in each dataset, but neither is "to senescence". This should be stated in Methods.

### Permissive vs strict — recommendation

Use **strict bins** (cohort 1 only / cohort 4 only) as the primary protocol. Reason: the maximally separated developmental contrast gives the cleanest log2FC signal. Permissive bins are a robustness check in Supp.

---

## 4. Tissue feasibility matrix (pig × dGTEx)

**8 tissues are strict-feasible** (n≥3 in both bins on both species), and the same 8 are permissive-feasible (permissive doesn't add new tissues — the missing pieces are on the pig side, not human):

### ✅ Strict-feasible panel (8 tissues — proposed Stage B headline)

| Tissue | Pig young | Pig old | dGTEx c1 | dGTEx c4 | Status |
|---|---:|---:|---:|---:|---|
| Muscle | 303 | 401 | 16 | 21 | very strong both sides |
| Lung | 56 | 27 | 19 | 11 | strong |
| Small Intestine | 107 | 28 | 9 | 11 | strong |
| Liver | 145 | 81 | 6 | 4 | strong pig, marginal human |
| Adipose Tissue | 30 | 76 | 13 | 16 | strong both sides |
| Testis | 23 | 25 | 5 | 4 | adequate |
| Spleen | 44 | 12 | 3 | 3 | adequate (human at minimum) |
| Heart | 16 | 25 | 3 | 3 | adequate (both at minimum) |

### ❌ Excluded — why each fails

| Tissue | Failure mode | Note |
|---|---|---|
| **Brain** | dGTEx only has **4 RNA samples total** (1 + 1 + 0 + 2) | **Major loss** — current paper has Brain as one of 5 tissues |
| **Kidney** | Pig Infant TPM n=2 (the day-old samples lack TPM) AND dGTEx kidney 2+0+0+1 | Double failure — same conclusion as previous analyses |
| **Colon** | Pig Colon TPM has **zero** Post-pubertal+Adult samples | dGTEx colon has 25 samples but pig dataset truncates |
| **Blood Vessel** | Pig "Artery" TPM has **zero** Infant+Early samples | Pig dataset truncates the other direction |
| **Ovary** | dGTEx ovary 0+0+1+3 (no Infant) | All dGTEx ovary samples are pubertal+ |
| **Lymph Node** | Pig Post-pubertal+Adult = 1 | Pig lymph_node truncates |
| **Uterus** | Pig Infant+Early = 0 AND dGTEx 0+0+0+3 | Both sides too sparse on young |
| **Blood** | dGTEx whole-blood RNA n=0 (only DNA samples) | dGTEx pilot didn't include blood RNA |
| Pituitary | Pig PostPub+Adult = 1; dGTEx c1 = 0 | Insufficient on both sides |
| Skin, Esophagus, Thyroid, Stomach, Adrenal Gland, Pancreas, Nerve | No pig TPM file | PigGTEx doesn't have these tissues |

### Net change from current paper

| | Current paper (5 tissues) | Proposed dGTEx panel (8 tissues) |
|---|---|---|
| Common | Muscle, Lung, Liver | ✓ all kept |
| Dropped from current | **Brain** (no dGTEx samples), Blood (no dGTEx RNA) | — |
| New in proposed | — | Heart, Adipose, Spleen, Small Intestine, Testis |
| Net | 5 → **8 tissues**, but losing Brain is significant |

### Brain alternative — open question

Should we keep brain in the panel using the current Cardoso-Moreira reference as an *exception*? This would violate the "dGTEx-only" rule the user set but preserve brain coverage. Three options:

1. **Strict dGTEx-only**: drop brain. 8-tissue panel. Cleanest story but loses brain.
2. **dGTEx + 1 exception**: include brain via Cardoso-Moreira; note as "the only non-dGTEx tissue, because dGTEx pilot has only 4 brain RNA samples". 9-tissue panel.
3. **Wait for dGTEx v2**: dGTEx is actively expanding; brain coverage may improve. Risky for the current revision.

**Recommendation**: Option 1 (strict dGTEx-only) for the headline. Discuss brain explicitly in Limitations: "dGTEx pilot v1 had insufficient brain RNA samples (n=4) to support cross-species inference; brain conservation can be re-evaluated when dGTEx v2 releases."

---

## 5. Sample size considerations for Stage B

dGTEx is a **pilot** dataset with only 40 donors. Per-tissue × per-cohort cell sizes are mostly 3–20. This is sufficient for **group-level log2FC** between bins (our cross-species comparison), but is **NOT sufficient for**:
- Per-gene FDR in the human side (no power at n=3–5)
- Within-stage subgroup analyses (breed, sex)
- Per-individual outlier detection

This means our existing **pig-anchored protocol** (pig FDR<0.10 + |log2FC|>0.5 both sides, no human FDR) is the right choice. Strict bidirectional filter is infeasible here.

For per-tissue Pearson r between pig and dGTEx log2FC, the gene-level n (after pig-anchored selection) typically lands at 100–500 genes per tissue, which is plenty for stable Pearson correlation estimation.

---

## 6. Open issues — requires user input

### 6a. Expression data URLs

**Problem**: I tried 14 candidate URL patterns; all returned HTTP 404. Examples:
- `https://storage.googleapis.com/developmental-gtex/data/v1/gene_tpm_dgtex_v1_brain.gct.gz` → 404
- `https://storage.googleapis.com/developmental-gtex/expression/v1/...` → 404
- `https://storage.googleapis.com/developmental-gtex/rna-seq/v1/...` → 404

The metadata files live under `/annotations/v1/metadata-files/` but expression data is at an unknown subpath within the same bucket.

**Action needed from user**: On the dGTEx download page, right-click "Copy link address" on ONE of the per-tissue gct.gz files (e.g., `gene_tpm_dgtex_v1_brain.gct.gz`) and paste the URL here. Once I have one, I can derive the pattern for all 24 tissues.

### 6b. Confirmation of proposed approach

The 8-tissue strict-feasible panel above replaces the current 5-tissue panel (Muscle, Brain, Liver, Lung, Blood) with **{Muscle, Lung, Liver, Small Intestine, Adipose, Testis, Spleen, Heart}** — a net gain of 3 tissues but losing Brain and Blood. User should confirm whether:

- Option 1 (strict dGTEx-only, 8 tissues, no Brain): cleanest story, recommended
- Option 2 (dGTEx + 1 Cardoso-Moreira exception for Brain, 9 tissues): violates "only dGTEx" but keeps brain
- Option 3 (something else): user proposes

### 6c. Permissive vs strict bin choice

Both strict (c1/c4 only) and permissive (c1+c2/c3+c4) yield the same 8-tissue panel — no tissues gained by going permissive. Decision: **use strict for headline** (cleaner contrast), report permissive in Supp as robustness.

---

## 7. Files produced by Stage A

- `data/dgtex/metadata/dGTEx_Analysis_2026-01-30_v1_*.xlsx` (4 files, committed to git for reproducibility)
- `review/analyses/results/dgtex_human_inventory.csv` (dGTEx tissue × cohort matrix)
- `review/analyses/results/dgtex_pig_overlap_matrix.csv` (pig × dGTEx feasibility — the critical file)
- `review/analyses/results/dgtex_discovery_report.md` (this document)

---

## 8. Stage B (experimental plan) — recommended scope based on Stage A

Now that the actual data has been inspected, the Stage B plan can be made concrete. Pending user confirmation of the open issues above, the Stage B plan should be:

1. **Tissue panel: 8 tissues** (the strict-feasible list)
2. **Stage mapping: pig Inf+Early vs PostPub+Adult; dGTEx cohort 1 vs cohort 4**
3. **Gene-selection protocol: pig-anchored uniform** (FDR_pig<0.10, |log2FC|>0.5 both, weighted purity-z, drop% 30, expression filter)
4. **ML framework sweep: 19 frameworks × 8 tissues = 152 cells** on Garnatxa SLURM array (instead of 190 cells planned in plan file)
5. **Validation: top-3 per tissue → ensemble + isotonic calibration → permutation + bootstrap re-validation**
6. **Paper outputs: updated R1.2 response emphasizing the dGTEx single-canonical-reference approach**

---

## STOP — awaiting user review

Before any Stage B compute is consumed, please confirm:
1. Expression URL pattern (Section 6a)
2. Brain inclusion decision (Section 6b)
3. Strict vs permissive bins (Section 6c — recommendation: strict)
