# PigDevStage: developmental staging of pig tissues from RNA-seq

Code and processed results for:

> Liu T, Lei R, Khan IM, Theobald P. Tissue identity is the dominant determinant of cross-species transferability of a porcine developmental programme. *BMC Genomics* 27, 919 (2026). https://doi.org/10.1186/s12864-026-13103-0

Web application: https://pigdevstage.streamlit.app

The first commit of this repository (tag `bmc-genomics-2026`) is the analysis code as used for the published paper. Later commits update the web application.

## What is here

| Folder | Contents |
|---|---|
| `machine_learning/` | Data loading, the ordinal models (`OrdinalPLS` and the benchmark estimators) and evaluation helpers |
| `paper/figures/provenance/` | Scripts that compute the values behind every figure and table: the cross-species transfer matrix, its determinants, the pig-independent baselines, within-pig evaluation, the 18-model benchmark (`hpc_benchmark/`) |
| `paper/figures/data/` | Outputs of those scripts (the numbers the figures and tables are drawn from) |
| `paper/figures/R/`, `paper/figures/python/` | Plotting scripts |
| `review/analyses/` | Supplementary analyses (batch-effect embeddings, PLS-VIP biomarkers, marker panel, heterogeneity) and their results |
| `streamlit_app/` | The web application; models are stored as plain numpy arrays (`models/*.npz`) |
| `data/` | PigGTEx sample metadata and ortholog symbol maps |

## Data to download

The expression matrices are public but not stored here:

| Data | Source | Place under |
|---|---|---|
| PigGTEx v0 gene TPM matrices (`PigGTEx_v0.Gene.TPM_matrix.tar`, 461 MB) | https://piggtex.farmgtex.org (Downloads) | `data/pigGTEx/<Tissue>.expr_tpm.txt.gz` |
| Cardoso-Moreira et al. 2019 RPKM matrices | ArrayExpress E-MTAB-6814 (human), E-MTAB-6798 (mouse), E-MTAB-6811 (rat), E-MTAB-6782 (rabbit), E-MTAB-6833 (opossum), E-MTAB-6769 (chicken), E-MTAB-6813 (rhesus) | `data/cardoso_moreira_2019/` with the file names in `TARGET_SPECIES_FILES` of `paper/figures/provenance/transfer_6stage.py` |
| Developmental GTEx v1 gene TPMs by tissue and annotations | https://gtexportal.org (bucket `developmental-gtex`: `bulk-gex/v1/rna-seq/tpms-by-tissue/`, `annotations/v1/metadata-files/`) | `data/dgtex/expression/`, `data/dgtex/metadata/` |

## Running

Python 3.12 with the packages in `requirements.txt`. Run scripts from the repository root with the root on `PYTHONPATH`, for example:

```bash
PYTHONPATH=. python paper/figures/provenance/transfer_6stage.py --exp matrix   # transfer matrix + determinants
PYTHONPATH=. python paper/figures/provenance/within_pig_6stage_figdata.py      # within-pig evaluation
```

A few scripts in the published snapshot set `ROOT` to the original author's local path; change that line to your checkout.

## Web application

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
python streamlit_app/export_pls_models.py   # rebuild the models (needs data/pigGTEx)
pytest streamlit_app/tests -q
```

## Citation

```bibtex
@article{liu2026tissue,
  title   = {Tissue identity is the dominant determinant of cross-species transferability of a porcine developmental programme},
  author  = {Liu, Tianyuan and Lei, Rujing and Khan, Ilyas M. and Theobald, Peter},
  journal = {BMC Genomics},
  volume  = {27},
  pages   = {919},
  year    = {2026},
  doi     = {10.1186/s12864-026-13103-0}
}
```

Please also cite PigGTEx: Teng J et al. A compendium of genetic regulatory effects across pig tissues. *Nat Genet* 56, 112–123 (2024).

## License

MIT.
