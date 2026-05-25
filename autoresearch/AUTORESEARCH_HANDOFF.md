# Autoresearch Handoff — dGTEx-Only Cross-Species Validation

This is an experiment to have the LLM do its own research, autonomously, on Garnatxa HPC. The mission is to maximise the cross-species developmental conservation signal between pig (PigGTEx) and human (dGTEx pilot v1) using a SINGLE uniform protocol applied across a maximally-sized tissue panel.

---

## Setup

Before starting any experimentation, work through this setup:

### 1. Confirm git state
- **Branch**: `autoresearch/dgtex-v2` (already exists in this worktree at `/Users/tianyuan/Desktop/github_dev/.claude-worktree-dgtex-v2/`)
- **Remote**: `git@github.com:TianYuan-Liu/pig-dev-stage.git`
- Push the branch to GitHub if not already: `git push -u origin autoresearch/dgtex-v2`

### 2. Read the in-scope files (don't skip — the loops depend on them)

| File | What it is |
|---|---|
| `CLAUDE.md` | Project rules — NEVER MODIFY the bib file; never simulate data; sync paper+thesis |
| `autoresearch/stage_mapping_v2.py` | Pig + human stage bins, 5-tissue panel definition. **MODIFIABLE** — this is one of the search dimensions. |
| `autoresearch/load_dgtex.py` | dGTEx loader (parses metadata XLSX, cross-joins, pools sub-regions). **DO NOT MODIFY** — fixed data interface. |
| `autoresearch/uniform_gene_selection_sweep.py` | Loop A — the autoresearch grid search. **DO NOT MODIFY** the objective function (max mean_r, tie-break max min_r). You CAN extend the grid. |
| `autoresearch/run_v2_dgtex.py` | Per-tissue cross-species evaluator. **DO NOT MODIFY** the merge/filter/stat logic — this IS the evaluator. |
| `autoresearch/run_uniform_cross_species.py:apply_uniform_drop, compute_pig_fc` | Pig-side helpers. **DO NOT MODIFY**. |
| `autoresearch/run_purity_generic.py` | TISSUE_MARKERS panel + purity-z. **DO NOT MODIFY** — uses existing curated panels. |
| `review/analyses/results/v2_uniform_winner.json` | Loop A current best. **DO NOT DELETE** — it's the baseline to beat. |
| `review/analyses/results/v2_phase1_baseline.md` | Why the panel is what it is. Read fully before considering tissue changes. |

### 3. Verify data exists on Garnatxa

```bash
ssh garnatxa 'ls -lh ~/pig-dev-stage/data/pigGTEx/Muscle.expr_tpm.txt.gz ~/pig-dev-stage/data/dgtex/expression/gene_tpm_dgtex_v1_muscle.gct.gz ~/pig-dev-stage/data/dgtex/metadata/*.xlsx 2>&1 | head'
```

If any file missing → rsync from local:
```bash
rsync -avz /Users/tianyuan/Desktop/github_dev/.claude-worktree-dgtex-v2/data/ garnatxa:~/pig-dev-stage/data/
```

### 4. Initialize results.tsv

Create `autoresearch/autoresearch_results.tsv` with header:

```
commit	loop	cell_id	mean_r	min_r	max_r	panel_size	status	description
```

The Loop A baseline will be recorded after the first sanity-check run.

### 5. Confirm setup looks good

Run the sanity check:
```bash
ssh garnatxa 'bash -lc "cd ~/pig-dev-stage && module load anaconda/anaconda3_2026 && mamba activate pig-dev-stage && python3 -m autoresearch.uniform_gene_selection_sweep"'
```

Expected output (verify these numbers within 0.005):
- `WINNER: strict_1to1 | pig_anchored | spearman`
- mean_r = 0.3559, min_r = 0.1619
- Muscle r=0.535, Testis r=0.420, Lung r=0.362, Spleen r=0.300, Adipose r=0.162

If sanity check passes, log first row in results.tsv with status=`keep` and description=`baseline Loop A winner`. Then proceed to experimentation.

---

## The target (one number to maximise)

**Primary metric**: `mean_r` = mean Spearman correlation across the included tissue panel.

**Tie-breaker**: `min_r` = minimum per-tissue Spearman r (protects against configs that wreck one tissue).

**Selection rule** (lexicographic):
```
1. Among all configs that produce valid r for all panel tissues,
   pick the one with highest mean_r.
2. Among configs within 0.005 of the top mean_r,
   pick the one with highest min_r.
3. Among ties on (mean_r, min_r), prefer the simpler config
   (see "Simplicity criterion" below).
```

**No hard thresholds.** Every tissue's number is reported regardless of n, p, or dc — the panel can be re-sized via the inclusion loop, but per-tissue stats are never censored.

Current best to beat (Loop A baseline):
```
mean_r = 0.3559   min_r = 0.1619   panel_size = 5
config = strict_1to1 | pig_anchored | spearman
```

---

## The evaluator (ground truth — DO NOT MODIFY)

The evaluator is a single deterministic function defined by these three pieces:

1. **Pig-side log2FC** — `autoresearch/run_uniform_cross_species.py:compute_pig_fc`
   - Inputs: pig expression matrix, young samples, old samples, method ∈ {weighted, median, limma}, tissue
   - Output: per-gene `log2fc` + `pvalue`
   - Defaults: method=`weighted` (purity-z weighted), drop%=30 (stratified by marker purity), pig median TPM ≥ 1

2. **Human-side log2FC** — `autoresearch/run_weighted_fc.py:median_fc`
   - Inputs: dGTEx expression matrix, young samples, old samples
   - Output: per-gene `log2fc` + `pvalue`
   - Defaults: human median TPM ≥ 10

3. **Cross-species correlation** — `autoresearch/uniform_gene_selection_sweep.py:apply_filter + compute_stat`
   - Inputs: merged pig×dGTEx FC table, filter ∈ {strict_bidirectional, pig_anchored, pig_fdr_only, fc_only, no_filter}, statistic ∈ {pearson, spearman}
   - Output: r, p, n_genes, dc, bootstrap CI

Seed = 42 throughout. Two invocations of this evaluator with the same config + data give identical results to 6 decimal places.

You **CANNOT modify** any of these three functions — they are the ground truth metric, just like the `evaluate_bpb` function in the user's template. You optimise OVER them, not WITHIN them.

---

## What you CAN modify (the search space)

You may freely modify these — each is a degree of freedom of the autoresearch:

### Gene-selection knobs (Loop A search space)
| Knob | Current | Other values to try |
|---|---|---|
| `filter` | pig_anchored | strict_bidirectional, pig_fdr_only, fc_only, no_filter, plus any NEW filter you can implement on the merged FC table (e.g., quantile-based, rank-based, top-N-by-pig-FC) |
| `ortholog` | strict_1to1 (7,485) | relaxed_symbol (15,200), plus any NEW ortholog set you can build (e.g., relaxed_with_paralogs, OrthoFinder result) |
| `statistic` | spearman | pearson, weighted Pearson (weights ∝ combined \|FC\|), Fisher-Z bootstrap |
| `pig_young_stages` | Infant + Early-childhood | Infant only, Infant + Early-childhood + Pre-pubertal |
| `pig_old_stages` | Post-pubertal + Adult | Adult only, Pre-pubertal + Post-pubertal + Adult |
| `human_young_cohorts` | [1] (Infant only) | [1, 2] (Infant + Early Childhood) |
| `human_old_cohorts` | [4] (Post-pubertal only) | [3, 4] (Pre-pubertal + Post-pubertal) |
| `pig_FC_method` | weighted | median, limma |
| `drop_pct` | 30 | 0, 10, 20, 40 |
| `pig_min_tpm` | 1 | 0, 0.5, 3, 5 |
| `human_min_tpm` | 10 | 0, 5, 30, 50 |

### Tissue panel (Loop 0 — tissue expansion attempts)

The current 5-tissue panel is not sacred. Below is the COMPLETE inventory of every tissue ever considered (24 dGTEx tissues + 1 pig-only Blood), with its exact failure mode and whether a rescue is allowed.

#### Complete tissue inventory — all 25 candidate tissues

| # | dGTEx tissue (SMTS) | dGTEx RNA n (c1/c2/c3/c4) | Pig TPM n (young/old) | Status | Reason |
|---|---|---|---|---|---|
| 1 | Muscle | 48 (16/7/4/21) | 1321 (303/401) | **IN PANEL** ✓ | Strong cross-species signal (r=0.535) |
| 2 | Lung | 36 (19/5/0/12) | 149 (56/27) | **IN PANEL** ✓ | Strong signal (r=0.362) |
| 3 | Testis | 10 (5/0/1/4) | 184 (23/25) | **IN PANEL** ✓ | Strong signal (r=0.420) |
| 4 | Spleen | 8 (3/0/2/3) | 91 (44/12) | **IN PANEL** ✓ | Moderate signal (r=0.300) |
| 5 | Adipose Tissue | 40 (13/5/4/17) | 285 (30/76) | **IN PANEL** ✓ | Weak but valid signal (r=0.162) |
| 6 | Small Intestine | 24 (9/3/1/11) | 270 (n/a — label mismatch) | **LOOP 0 RESCUE A** | Pig metadata uses sub-region labels (Ileum/Jejunum/Duodenum) — loader returns n=0 |
| 7 | Heart | 7 (3/0/1/3) | 164 (16/25) | **LOOP 0 RESCUE B** | dGTEx n=3+3 strict bins; try permissive bins (3+4) |
| 8 | Colon | 25 (7/1/3/14) | 67 (27/0) | **LOOP 0 RESCUE C** | Pig has 0 PostPub+Adult samples; try broader pig old anchor |
| 9 | Lymph Node | 18 (6/2/1/9) | 50 (16/1) | **LOOP 0 RESCUE D** | Pig PostPub+Adult=1; try broader pig old anchor |
| 10 | Liver | 11 (6/1/0/4) | 501 (145/81) | **LOOP 0 RESCUE E** | Adult-hepatic genes flip direction in dGTEx adolescents; try GO:developmental restriction |
| 11 | Blood Vessel | 7 (3/0/0/4) | 59 (0/30) — `Artery` file | **LOOP 0 RESCUE F** | Pig Artery has 0 Infant+Early samples; try broader pig young anchor |
| 12 | Pituitary | 2 (0/0/0/2) | 53 (3/1) | **LOOP 0 RESCUE G** | Small n both sides; try permissive bins to see if r>0 with valid CI |
| 13 | Brain | 4 (1/1/0/2) | 419 (160/22) | **UNRESCUABLE** ✗ | dGTEx n=4 total. Even permissive bins give 2+2. No algorithmic fix manufactures power. Awaits dGTEx v2. |
| 14 | Kidney | 3 (2/0/0/1) | 44 (2/22) | **UNRESCUABLE** ✗ | BOTH sides fail: pig has only n=2 Infant in TPM (the 0-day samples lack TPM data); dGTEx kidney has n=3 total |
| 15 | Ovary | 4 (0/0/1/3) | 204 (20/58) | **UNRESCUABLE** ✗ | dGTEx ovary has 0 Infant samples — even permissive bins can't produce a young anchor. Pig side fine but no human partner |
| 16 | Uterus | 4 (0/0/1/3) | 213 (0/15) | **UNRESCUABLE** ✗ | BOTH sides fail on young anchor: dGTEx 0 Infants, pig 0 Infant+Early uterus samples |
| 17 | Thymus | 1 (1/0/0/0) | 48 (`Fetal_thymus` — fetal only, irrelevant) | **UNRESCUABLE** ✗ | dGTEx n=1 (single Infant sample); pig has only fetal data |
| 18 | Skin | 22 (6/3/2/11) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no skin RNA-seq matrix. Out of scope without new pig data. |
| 19 | Esophagus | 18 (3/2/2/11) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no esophagus TPM |
| 20 | Thyroid | 15 (2/1/0/12) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no thyroid TPM (n=1 sample in metadata, no file) |
| 21 | Stomach | 14 (2/1/3/8) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no stomach TPM |
| 22 | Nerve | 10 (5/2/0/2) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no nerve TPM |
| 23 | Adrenal Gland | 6 (3/1/0/2) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no adrenal TPM |
| 24 | Pancreas | 5 (3/0/1/1) | — (no pig TPM file) | **UNRESCUABLE** ✗ | PigGTEx has no pancreas TPM |
| 25 | Blood (pig-only) | 0 RNA (DNA only) | 386 (97/39) | **UNRESCUABLE** ✗ | dGTEx pilot has 0 whole-blood RNA samples (only WGS DNA). Cross-species impossible. Pig Blood is published as BA-only. |

Summary:
- **5 IN PANEL** (Muscle, Lung, Testis, Spleen, Adipose Tissue)
- **7 LOOP 0 RESCUE ATTEMPTS** (Small Intestine, Heart, Colon, Lymph Node, Liver, Blood Vessel, Pituitary) — each gets ONE shot, possibly grows panel to 6–12
- **13 UNRESCUABLE** with a documented reason. Of these:
  - 7 lack pig TPM files (Skin, Esophagus, Thyroid, Stomach, Nerve, Adrenal Gland, Pancreas) — would need new pig data
  - 6 fail on the dGTEx side or both sides (Brain, Kidney, Ovary, Uterus, Thymus, Blood) — would need dGTEx v2 expansion

#### Loop 0 rescue protocol (each tissue tried EXACTLY ONCE)

| Code | Tissue | Concrete intervention | Acceptance criterion |
|---|---|---|---|
| A | Small Intestine | Add metadata alias in `cross_species_all_tissues.py:load_pig_tissue` so `Small_intestine` matches PigGTEx `Tissue class ∈ {Small intestine, Ileum, Jejunum, Duodenum}` | n_pig_young ≥ 3 AND n_pig_old ≥ 3 AND r > 0 with valid p |
| B | Heart | In `stage_mapping_v2.py`, set `HUMAN_YOUNG_COHORTS=(1,2)` and `HUMAN_OLD_COHORTS=(3,4)` ONLY for Heart cell | r > 0.10 with bootstrap CI excluding 0 |
| C | Colon | In `stage_mapping_v2.py`, set `PIG_OLD_STAGES=("Pre-pubertal","Post-pubertal","Adult")` ONLY for Colon cell | n_pig_old ≥ 3 AND r > 0.10 |
| D | Lymph Node | Same as C (broader pig old anchor) | n_pig_old ≥ 3 AND r > 0.10 |
| E | Liver | Restrict gene set to GO:0032502 (developmental process) descendants. Implementation: query `mygene.info` `go.MF + go.BP` field for `GO:0032502`, intersect with pig-anchored set, recompute r. | r > 0 AND CI excludes 0 (must reverse the −0.25 baseline) |
| F | Blood Vessel | Set `PIG_YOUNG_STAGES=("Infant","Early childhood","Pre-pubertal")` ONLY for Blood Vessel cell | n_pig_young ≥ 3 AND r > 0.10 |
| G | Pituitary | Apply permissive bins (cohort 1+2 vs 3+4 on dGTEx; Infant+EC+PrePub vs PostPub+Adult on pig) | n ≥ 30 genes after pig-anchored filter AND r > 0.10 |

**Important constraint on Loop 0 rescues**: the per-tissue interventions B–G use tissue-specific stage bins. **This breaks the "uniform across all tissues" constraint of Loop A.** If a rescue is kept, the paper must report this as a Per-Tissue Bin Customization in Supplementary, with the explicit justification that dGTEx pilot's small per-tissue sample distribution forces stage-bin choices to be tissue-aware. The cross-species r itself is still computed by the SAME uniform protocol (pig_anchored filter, strict 1:1 orthologs, Spearman).

If a rescued tissue fails its acceptance criterion → revert to the documented exclusion. Log to results.tsv with status=`discard`.

### ML framework (Loop B search space)
| Knob | Current frameworks |
|---|---|
| `model_type` | ordinal_lgb, ridge_continuous, random_forest, elastic_net_lr, multiclass_lgb |
| `n_cv_folds` | 5 (outer) |
| `n_trials` | 5 (Optuna) — increase to 25 for the winner only |

---

## What you CANNOT modify (immutable inputs)

These are off-limits. Modifying any of them invalidates the autoresearch:

- **Raw data**: `data/pigGTEx/*.expr_tpm.txt.gz`, `data/dgtex/expression/*.gct.gz`, `data/dgtex/metadata/*.xlsx`, `data/PigGTEx_v0.MetaTable.csv`
- **The evaluator functions** listed above (`compute_pig_fc`, `median_fc`, `apply_filter`, `compute_stat`)
- **The Pearson/Spearman implementations** (`scipy.stats.pearsonr`, `scipy.stats.spearmanr`)
- **The dGTEx AGECOHORT definitions** (1=Infant 0-2y, 2=Early Childhood 2-8y, 3=Pre-pubertal 8-13y, 4=Post-pubertal 13-18y — these are dictionary-defined)
- **The PigGTEx 5-stage scheme** (Infant 0-20d, Early childhood 21-59d, Pre-pubertal 60-149d, Post-pubertal 150-364d, Adult ≥365d)
- **The 1:1 ortholog source** (`pig_human_one_to_one_orthologs.csv` — Ensembl Compara verified; immutable)
- **The bibliography** (`paper/pig-age-human/pig-age-human.bib` — CLAUDE.md rule; new citations go in `review/new_bib_entries.bib`)
- **The Garnatxa cluster configuration** (modules, SLURM partitions, paths)
- **The fact that you cannot install new packages** — only what's in `environment_garnatxa.yml` is available

---

## Simplicity criterion

All else being equal, simpler is better:

- Fewer knobs touched > more knobs touched
- Strict 1:1 orthologs > relaxed (more defensible)
- Default stage bins (Infant+EC vs PostPub+Adult) > custom bins
- Pearson > Spearman (more standard)
- pig_anchored > strict_bidirectional (no selection-on-both-sides bias)

When evaluating whether to keep a change, weigh complexity vs improvement:
- +0.005 mean_r from one knob flip: KEEP if simpler config, otherwise DISCARD
- +0.02 mean_r from one knob flip: KEEP regardless of simplicity
- +0.005 mean_r from adding 50 lines of hacky code: DISCARD
- 0 mean_r change but removed knob (simpler protocol): KEEP — that's a simplification win

---

## Output format (what each script prints)

### Loop A — `uniform_gene_selection_sweep.py`

```
Loop A — Uniform gene-selection autoresearch
Grid: 5 filters × 2 orthologs × 2 statistics = 20 cells

WINNER: <ortholog> | <filter> | <statistic>
  mean_r = <number>
  min_r  = <number>
  Per tissue:
    Muscle           n=<int>  r=<+number>  CI=[<lo>, <hi>]  p=<sci>  dc=<int>%
    Lung             ...
    Testis           ...
    Spleen           ...
    Adipose Tissue   ...

TIER 1 (strong, r ≥ 0.40): [list]
TIER 2 (extended panel, all 5): mean r = <number>
```

Extract key metrics with:
```bash
grep "^WINNER:\|^  mean_r\|^  min_r" run.log
```

### Loop B — `aggregate_ml_simple.py`

```
framework      mean    min     max
ridge_cont     0.92    0.88    0.96
ordinal_lgb    0.89    0.81    0.96
...

WINNER: ridge_continuous
```

### Loop C — `validate_v2_winner.py` per tissue

```
TISSUE: Muscle
  r = 0.535
  bootstrap 95% CI = [0.40, 0.60]
  permutation empirical p = 0.001
  LOO range = [0.49, 0.58]
  marker hits = 3/14 (2 concordant)
```

---

## Logging results

Write to `autoresearch/autoresearch_results.tsv` (tab-separated). DO NOT commit this file — leave it untracked.

Schema:

| Column | Type | Notes |
|---|---|---|
| commit | str | 7-char git short SHA |
| loop | str | A, A_ext, 0, B, C |
| cell_id | str | filter\|ortholog\|stat OR framework\|tissue OR rescue\|tissue |
| mean_r | float | mean of per-tissue r across panel (for Loop A); per-tissue BA (for Loop B); per-tissue r (for Loop C) |
| min_r | float | min of per-tissue r (Loop A only; blank otherwise) |
| max_r | float | max of per-tissue r (Loop A only; blank otherwise) |
| panel_size | int | tissues in panel for this cell (5 unless tissue expansion changed it) |
| status | str | keep, discard, crash |
| description | str | short narrative (no commas) |

Example rows:
```
a1b2c3d	A	strict_1to1|pig_anchored|spearman	0.3559	0.1619	0.5352	5	keep	baseline Loop A winner
b2c3d4e	A_ext	relaxed|pig_anchored|spearman	0.3559	0.1456	0.5206	5	discard	tie on mean_r lower min_r
c3d4e5f	0	rescue_small_intestine_metadata_alias	0.380	0.220	0.580	6	keep	added Small Intestine after metadata fix; new mean_r > baseline
d4e5f6g	0	rescue_heart_permissive_bins	0.000	0.000	0.000	5	discard	Heart still noise under permissive bins
e5f6g7h	B	ridge_continuous|muscle	0.940	0.000	0.000	1	keep	per-tissue BA cell
f6g7h8i	B	ordinal_lgb|liver	0.000	0.000	0.000	0	crash	OOM with 16G, increase to 32G
```

---

## The experiment loop

Each LOOP RUNS AUTONOMOUSLY. After setup, you do not stop until ALL loops complete or you hit an unrecoverable block.

### Loop 0 — Tissue expansion (run FIRST, before Loop A re-runs)

Iterate over the 7 rescue attempts (A–G in the tissue inventory table). Each attempt is ONE shot — if it fails its acceptance criterion, revert and move on.

```
FOR rescue in [A:SmallIntestine, B:Heart, C:Colon, D:LymphNode, E:Liver, F:BloodVessel, G:Pituitary]:
  1. git checkout autoresearch/dgtex-v2 (current best tip)
  2. Implement the rescue per the "Loop 0 rescue protocol" table:
       - A: edit cross_species_all_tissues.py:load_pig_tissue to alias Small_intestine
       - B,F,G: edit stage_mapping_v2.py to apply per-tissue bin override
       - C,D: edit stage_mapping_v2.py for broader pig old anchor on that tissue
       - E: add GO:0032502 restriction in run_v2_dgtex.py for Liver cell only
  3. Add the tissue to INCLUDED_TISSUES in stage_mapping_v2.py
  4. git commit -m "Loop 0 rescue <code>: <tissue> via <intervention>"
  5. Re-run: python3 -m autoresearch.uniform_gene_selection_sweep
  6. grep "^WINNER:\|^  mean_r\|<tissue>" run.log → check that the rescued tissue produced valid stats
  7. Apply the rescue's acceptance criterion (from rescue protocol table):
       - If PASSES: status=keep in results.tsv, advance branch (tissue stays in panel)
       - If FAILS:  status=discard in results.tsv, git reset --hard HEAD~1 (tissue stays excluded)
  8. Move on to next rescue. Do NOT re-test failed rescues with different interventions in this pass.
```

After Loop 0 completes, the panel is final (5–12 tissues). The Loop A baseline is automatically re-scored on the final panel because each Loop 0 commit re-ran the sweep.

### Loop A — Uniform gene-selection sweep (extend the grid)

```
FOR each new (filter, ortholog, statistic, bins, drop%, expression filter) variant:
  1. git checkout autoresearch/dgtex-v2 (current best branch tip)
  2. Modify autoresearch/uniform_gene_selection_sweep.py to add the new variant to the grid
  3. git commit -m "Loop A extend: <variant>"
  4. Re-run sweep: python3 -m autoresearch.uniform_gene_selection_sweep
  5. grep "^WINNER" run.log
  6. Log row to results.tsv with mean_r/min_r/max_r/panel_size of the NEW winner
  7. IF (new mean_r, new min_r) > (current best mean_r, current best min_r) lexicographically:
       status=keep, advance branch (commit becomes new baseline)
     ELSE:
       status=discard, git reset --hard <previous best>
  8. Move on to next variant
```

Variants to try in order (cheap first):
1. permissive human bins (cohort 1+2 vs 3+4) — change `HUMAN_YOUNG_COHORTS=(1,2)` and `HUMAN_OLD_COHORTS=(3,4)`
2. permissive pig bins (Infant+EC+PrePub vs PostPub+Adult) — change `PIG_YOUNG_STAGES`
3. median pig FC instead of weighted — change `UNIFORM_METHOD = "median"` in `run_v2_dgtex.py`
4. drop_pct = 0 (no marker-purity drop)
5. lower expression filters (pig_min_tpm=0.5, human_min_tpm=5)
6. higher expression filters (pig_min_tpm=3, human_min_tpm=30)
7. NEW filter variant: top-300-by-pig-FC (no FDR, just top N by |pig log2FC|)
8. NEW filter variant: bootstrap-stable pig_anchored (gene must pass in ≥80% of pig bootstraps)

If after 8 variants no improvement: Loop A is converged. Lock the winner. Move on to Loop B.

### Loop B — ML framework sweep on Garnatxa

This loop is SLURM-driven, NOT git-iterative. Submit array, wait, aggregate, pick winner.

```
1. Write autoresearch/run_ml_simple_cell.py (one cell = one framework × tissue)
2. Write autoresearch/slurm/run_ml_simple.sbatch (5 frameworks × panel_size cells)
3. git commit + push
4. ssh garnatxa "cd ~/pig-dev-stage && git pull && sbatch autoresearch/slurm/run_ml_simple.sbatch"
5. Poll squeue every 5 minutes until array done (or 60 min wall-clock)
6. Aggregate with autoresearch/aggregate_ml_simple.py → winner.json
7. Log per-cell rows to results.tsv with loop=B
8. If any cell crashed (FAILED/OOM/TIMEOUT):
     - If trivial fix (OOM → bump --mem; typo → fix syntax): resubmit failed indices ONCE
     - If fundamental: status=crash, exclude that (framework, tissue) cell
9. Winner = framework with highest mean BA AND no tissue below 0.80
```

If Loop B winner is the currently-published model (ordinal_lgb): keep published methodology, document.
If Loop B winner is different (likely Ridge): document the switch in paper Methods.

### Loop C — Validation (SLURM)

```
1. Write autoresearch/validate_v2_winner.py (Loop A winner + bootstrap + perm + LOO + marker sanity, per tissue)
2. Write autoresearch/run_ml_winner_full_cv.py (Loop B winner only, full nested CV with 25 Optuna trials)
3. Write autoresearch/slurm/run_validation.sbatch
4. Submit, wait, aggregate
5. Log to results.tsv with loop=C
```

If a tissue's Loop C bootstrap CI no longer excludes zero, demote it from Tier 1 to Tier 2 in the paper framing.

### Loop D — Paper outputs (local, NOT autoresearch — just delivery)

```
1. autoresearch/aggregate_paper_data.py → paper/figures/output/stats/fig4_v2_dgtex_summary.tsv
2. Regenerate Fig 4 forest plot (R script)
3. Update paper.tex §2.4 + Limitations
4. Update response_to_reviewers.tex R1.2
5. Append dGTEx citation to review/new_bib_entries.bib
6. Mirror to thesis/
7. Compile all PDFs
8. Final commit + push
```

---

## Crash handling

**OOM / timeout / SLURM FAILED**:
- Trivial fix (bump memory, fix typo, missing import): apply fix, resubmit failed cells ONCE, log `crash` only if it fails again.
- Fundamental (algorithm doesn't converge on this data, framework incompatible with feature count): log `crash`, exclude that cell, continue.

**Reproducibility check (Loop A re-run on Garnatxa) gives different numbers than local**:
- STOP. Compare data file sizes (`md5sum`) between local and Garnatxa. Re-rsync if mismatched.

**git conflict on push**:
- `git pull --rebase`, resolve, push again. Don't force-push without confirmation.

**Anything that touches `paper/pig-age-human/pig-age-human.bib`**:
- STOP IMMEDIATELY. Revert. CLAUDE.md prohibits any modification.

---

## NEVER STOP rule

Once the experiment loop has begun (after the setup sanity check passes), do NOT pause to ask the user if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The user may be away from the computer and expects you to continue working until you are manually stopped or all loops complete.

You are autonomous. The loop terminates when:
1. Loop 0 has tried all 7 rescue attempts (each at most once — A through G in the tissue inventory)
2. Loop A has tried all 8 variants without improvement, OR an improvement was kept and re-scored on the (possibly expanded) panel
3. Loop B has completed all (5 frameworks × final panel size) cells
4. Loop C has validated the winners
5. Loop D has produced compilable paper outputs

If you run out of ideas mid-loop: re-read the in-scope files for new angles, try combining previous near-misses, try the next variant in the queue. The natural stopping criterion is "loops are exhausted", not "I feel finished".

---

## Justification: tissue panel decisions (full accounting of all 25 candidates)

The 5-tissue minimum is NOT a design choice — it's the consequence of a tissue-by-tissue feasibility audit. The full inventory above shows every dGTEx tissue (and Blood from pig-side) with its specific fate. Here's the high-level breakdown:

- **5 IN PANEL** (Muscle, Lung, Testis, Spleen, Adipose) — all have a positive cross-species signal under the default uniform protocol, with documented per-tissue r and CI.
- **7 LOOP 0 RESCUE ATTEMPTS** (Small Intestine, Heart, Colon, Lymph Node, Liver, Blood Vessel, Pituitary) — each gets ONE concrete intervention. Successful rescues grow the panel; failures revert.
- **13 UNRESCUABLE** with concrete data-availability reasons. Split into two groups:
  - **7 lack pig TPM data**: Skin, Esophagus, Thyroid, Stomach, Nerve, Adrenal Gland, Pancreas. Would require new pig RNA-seq generation — out of scope for this paper.
  - **6 lack dGTEx data**: Brain (n=4 total), Kidney (n=3, both sides fail), Ovary (no Infant cohort), Uterus (both sides fail on young), Thymus (n=1), Blood (no whole-blood RNA — DNA only). Would require dGTEx v2 expansion.

**The panel could grow from 5 → 12 if all 7 rescue attempts succeed**, or stay at 5 if none do. The autoresearch decides; the human reviewer sees a complete audit trail in `autoresearch_results.tsv` showing exactly what was tried and why each tissue ended up where it did.

**Honest expectation**: 2–4 rescues likely succeed (Small Intestine is almost certain — the metadata fix is mechanical; Heart and Pituitary are likely to stay marginal; Liver under GO restriction is the most interesting open question because if it works it inverts the original anti-correlation finding into a positive signal restricted to truly developmental genes).

This audit is the answer to "why not more tissues" — the constraint is data availability on EITHER the pig side OR the dGTEx pilot side, not methodology.

---

## Definition of done

The handoff is complete when ALL of these are true:

- [ ] `git log autoresearch/dgtex-v2 --oneline` shows commits for Loop 0, Loop A extensions, Loop B, Loop C, Loop D
- [ ] `autoresearch/autoresearch_results.tsv` has rows for every experiment attempted (kept + discarded + crashed)
- [ ] `review/analyses/results/v2_uniform_winner.json` reflects the final Loop A winner (≥ baseline)
- [ ] `autoresearch/ml_sweep_simple/winner.json` exists with the Loop B winner framework
- [ ] `paper/figures/output/stats/fig4_v2_dgtex_summary.tsv` exists with one row per panel tissue
- [ ] `paper/paper.pdf`, `review/response_to_reviewers.pdf` compile cleanly
- [ ] `git diff paper/pig-age-human/pig-age-human.bib` returns empty (no modifications)
- [ ] `ssh garnatxa "sacct -u tyuan --starttime=<startdate> --format=jobid,state | grep -v -E 'COMPLETED|RUNNING|PENDING'"` returns empty (no failed jobs)
- [ ] Final summary report written to chat with:
  - Final per-tissue table (panel_size rows: tissue, n_genes, r, CI, perm_p, LOO range)
  - Loop B winning framework
  - Tissues added in Loop 0 (if any)
  - Loop A variants tried and outcomes
  - Git commit hashes

---

## Quick reference — current baseline numbers (the targets to verify reproducibility against)

```
Loop A baseline (locked):
  config:        strict_1to1 | pig_anchored | spearman
  mean_r:        0.3559
  min_r:         0.1619 (Adipose)
  max_r:         0.5352 (Muscle)
  panel_size:    5

Per-tissue Spearman r:
  Muscle:         0.535   CI [0.40, 0.60]   p=4e-23   dc=75%
  Testis:         0.420   CI [0.25, 0.47]   p=1e-16   dc=74%
  Lung:           0.362   CI [0.24, 0.51]   p=1e-8    dc=70%
  Spleen:         0.300   CI [0.07, 0.41]   p=1e-6    dc=44%
  Adipose:        0.162   CI [-0.01, 0.27]  p=8e-3    dc=61%

Tier 1 (r≥0.40): Muscle, Testis     mean = 0.478
Tier 2 (all 5):                      mean = 0.356
```

If after Loop 0 + Loop A extensions, `mean_r < 0.3559` for the chosen config → KEEP the baseline (`git reset --hard <baseline commit>`). The baseline is the floor.
