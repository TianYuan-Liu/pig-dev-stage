# Joint cross-species ML feature selection

Pipeline that selects developmental-program genes predictive of postnatal
developmental stage in both pig and human, by training species-specific
LightGBM models in a shared 7,485-ortholog feature space and intersecting
their seed-stable top-K importance sets.

## Replication
From the project root:
```bash
python3 review/analyses/cross_species_joint_ml/pipeline.py
python3 review/analyses/cross_species_joint_ml/recategorize.py
python3 review/analyses/cross_species_joint_ml/figures.py
python3 review/analyses/cross_species_joint_ml/generate_drafts.py
```
`pipeline.py` skips any tissue with an existing `results_{tissue}.json`
unless the file records an error, so re-runs are incremental. Pig matrices
are cached as parquet under `cache/`.

## Per-tissue summary (final run)
| Tissue | Pig n | Human n | Joint set | Pearson r |
|---|---|---|---|---|
| Muscle | 761 | 11 (4+7, GSE257558) | 2 (ACHE, ABAT; synaptic) | n<5 |
| Brain  | 38 (post-filter) | 16 (7+9, Cardoso-Moreira) | 16 (lipid/metabolic + synaptic) | 0.14 [-0.29, 0.53] |
| Liver  | 309 | 10 (4+6, Cardoso-Moreira) | 1 (PTS) | n<5 |
| Lung   | 149 | 0 (no human matrix) | n/a | n/a |
| Blood  | 76 (post-filter) | 0 (no human matrix) | n/a | n/a |

Brain pig was filtered to keep only stages with ≥5 samples per class
(Infant + Post-pubertal); Blood pig was filtered the same way (Early
childhood + Pre-pubertal + Post-pubertal).

## Files
- `pipeline.py` - main pipeline
- `figures.py` - figure generator
- `generate_drafts.py` - markdown draft generator (data-driven)
- `recategorize.py` - post-hoc gene-category refinement
- `features_pig_{tissue}.csv` - per-seed top-100 with mean rank, stability
- `features_human_{tissue}.csv` - same for human (when available)
- `joint_features_{tissue}.csv` - intersection set (pig∩human top-100 stable)
- `results_{tissue}.json` - per-tissue stability + concordance + annotation
- `fig_top_features_{tissue}.pdf` - pig top-30 | human top-30 | joint
- `fig_concordance_panel.pdf` - joint-ML Pearson r per tissue with 95% CI
- `fig_method_comparison.pdf` - baseline FDR gene-set size vs joint ML
- `log.tsv` - autoresearch attempt log
- `DRAFT_methods.md`, `DRAFT_results.md`, `DRAFT_supplementary.md` -
  data-driven draft text for paper integration

## Why no pickled models
Per-fold LightGBM ensembles for ten seeds × five folds (pig) plus ten seeds
× leave-pair-out CV (human) total several hundred fitted gradient-boosted
models per tissue, which would inflate the repository by ~1-2 GB without a
downstream use-case. The actionable model output (per-seed gain ranking,
top-100 stability counts, joint-set definition) is preserved in
`features_*.csv` and `joint_features_*.csv`. Re-fitting from
`pipeline.py` is ~20 min wall-clock on 5 tissues.

## Hyperparameter sweep
If joint @100 is empty for a tissue, the pipeline retries the human binary
model under up to three regularisation configurations (`num_leaves` ∈ {6,
8, 12}, `min_data_in_leaf` ∈ {2, 3, 5}). The configuration that produced
the reported joint set is logged as `human_sweep_idx` in `results_*.json`.
