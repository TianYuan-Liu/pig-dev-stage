# `review/analyses/` — Revision-driven analysis scripts

Each script below was written to address a specific BMC Genomics R1 reviewer
point. All scripts:

- read raw data exclusively from `data/PigGTEx_v0.MetaTable.csv`,
  `data/pigGTEx/*.expr_tpm.txt.gz`, and `data/cardoso_moreira_2019/`;
- write results to `review/analyses/results/`;
- write figures to `paper/figures/output/pdf/` and `.../png/`;
- expose a `python -m review.analyses.<name>` CLI;
- log to stdout and `review/analyses/results/<name>.log`.

| Script | Reviewer point | Outputs |
|---|---|---|
| `batch_effect_visualization.py` | R3.1 | PCA + UMAP per tissue coloured by BioProject; variance-decomposition table |
| `cross_species_all_tissues.py` | R1.2, R3.2, R3.4 | Pig↔human Pearson r per tissue (brain/liver/lung); ortholog filter counts |
| `heterogeneity_sensitivity.py` | R2.1 | Per-stage performance stratified by breed, sex, BioProject |
| `marker_gene_validation.py` | R3.3 | Stage-wise expression of curated weaning/puberty marker genes |
| `tissue_specific_features.py` | R3.4 | Top-20 features per tissue with UniProt-derived functional annotation |

Re-run order (some scripts depend on outputs of others):

```bash
cd /Users/tianyuan/Desktop/github_dev/pig-dev-stage
python review/analyses/tissue_specific_features.py
python review/analyses/batch_effect_visualization.py
python review/analyses/heterogeneity_sensitivity.py
python review/analyses/marker_gene_validation.py
python review/analyses/cross_species_all_tissues.py
```
