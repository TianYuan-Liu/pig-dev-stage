# `review/analyses/` — Revision-driven analysis scripts

> **6-stage migration index (2026-05-30).** The current 6-stage / single-PLS paper is
> built from the **canonical generators** in the table just below. The original R1
> reviewer scripts further down are kept as provenance but are **superseded** (5-stage).
> The cross-species transfer + heterogeneity generators now live in `paper/figures/provenance/` (the retired `autoresearch/` folder).
>
> **Canonical (LIVE) generators:**
>
> | Script | Produces | Consumed by |
> |--------|----------|-------------|
> | `pls_vip_features_6stage.py` | `results/pls_vip_features.csv` (top-20 PLS-VIP/tissue, 6 per-stage TPM bins, signed) | Fig 3, Fig S5, Table S5 |
> | `build_tableS5_tex.py` | `output/supplementary/tableS7_tissue_features.tex` (Table S5) | Table S5 |
> | `marker_gene_validation_v3.py` | `figS6_marker_genes_v3.{pdf,png}` — invoked by the figures Makefile | Fig S6 |
> | `batch_effect_visualization.py` | `results/batch_effect_embeddings_*.csv` | Fig S4 |
>
> Fig S8's `heterogeneity_*.csv` are produced by `paper/figures/provenance/heterogeneity_6stage.py`, not here.
>
> **Superseded (kept as provenance):** `marker_gene_validation.py` + `_v2` → `_v3`;
> `pls_vip_features.py` → `_6stage`; `heterogeneity_{sensitivity,figure}.py` →
> `paper/figures/provenance/heterogeneity_6stage.py`; `cross_species_all_tissues.py`,
> `summarize_cross_species.py`, `tissue_specific_features.py` → exploration / older
> fold-change method (do **not** reuse for the transfer matrix).
>
> `results/` (81 JSONs, ~5 MB) mixes the live CSVs above with pure-provenance exploration
> JSONs (34 `cross_species_lung_*`, 18 `cross_species_brain_qc_*`, liver/testis runs).
>
> ---
>
> ### Original R1 reviewer scripts (5-stage — superseded, kept as provenance)

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
