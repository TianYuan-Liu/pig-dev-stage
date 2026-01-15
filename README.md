# A Calibrated Transcriptomic Atlas of Porcine Development

[![DOI](https://img.shields.io/badge/Data-GSE257558-blue)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE257558)
[![PigGTEx](https://img.shields.io/badge/Resource-PigGTEx-green)](https://piggtex.farmgtex.org/)

Machine learning framework for inferring developmental stage from pig transcriptomic data, with cross-species validation against human. This repository accompanies the paper:

> **A Calibrated Transcriptomic Atlas of Porcine Development Reveals Conserved Molecular Programs with Humans**
> 
> Tianyuan Liu, Rujing Lei, Ilyas M. Khan, Peter Theobald

## Overview

Current developmental staging in pig research relies on chronological age or subjective morphological criteria, which fail to capture inter-individual variability. This project provides a **calibrated transcriptomic atlas** that infers biological developmental stage directly from gene expression profiles using machine learning.

### Key Findings

- **1,924 samples** across **5 tissues** (brain, liver, muscle, lung, blood) from [PigGTEx](https://piggtex.farmgtex.org/)
- **5 developmental stages**: Infant (0–20d), Early childhood (21–59d), Pre-pubertal (60–149d), Post-pubertal (150–365d), Adult (>365d)
- **High-precision staging**: Balanced accuracy 0.64–0.92 (mean: 0.83) using LightGBM ordinal classification
- **Cross-species conservation**: 89% directional concordance with human skeletal muscle (Pearson r = 0.62, p < 10⁻⁴)

## Project Structure

```
pig-dev-stage/
├── machine_learning/           # Python ML Pipeline
│   ├── data_processing/       # Data loading & preprocessing
│   ├── feature_engineering/   # Feature selection (variance, MI)
│   ├── model_training/        # LightGBM ordinal classification
│   ├── model_evaluation/      # Metrics & bootstrap CI
│   ├── cross_tissue_analysis/ # Cross-tissue validation
│   ├── analysis/              # Additional analyses
│   ├── config.yaml            # Pipeline configuration
│   └── run_pipeline.py        # Main entry point
│
├── paper/                      # LaTeX Manuscript
│   ├── paper.tex              # Main manuscript (Nature format)
│   ├── supplementary.tex      # Supplementary materials
│   └── figures/               # Publication figures
│
├── visualization/              # R Publication Figures
│   ├── theme_configs/         # Nature journal themes
│   └── nature_figures/        # Figure generation scripts
│
├── slides/                     # Presentation Materials
│   ├── R/                     # Slide figure generation
│   └── Figures/               # Generated panels
│
├── data/                       # Data Files
│   ├── pigGTEx/               # Expression files
│   └── human_muscle/          # Human validation data
│
├── scripts/                    # Utility Scripts
└── results/                    # Analysis Outputs
```

## Quick Start

### Run ML Pipeline

```bash
python machine_learning/run_pipeline.py
```

### Run with Custom Parameters

```bash
python machine_learning/run_pipeline.py \
    --max-features 2000 \
    --seed 42 \
    --tissues muscle liver brain blood lung
```

### Generate Publication Figures

```bash
cd paper/figures/R
Rscript fig1_study_design.R
Rscript fig2_model_performance.R
Rscript fig3_functional_enrichment.R
Rscript fig4_cross_species.R
```

## Methods

### Machine Learning Framework

We implemented a **reduction-based ordinal classification** framework using LightGBM. Given the ordinal nature of developmental stages, we decomposed the K-class problem into K-1 binary classification subtasks (Frank & Hall method).

**Key parameters:**
- Feature selection: Top 2000 genes by information gain
- Train/test split: 70%/30% stratified
- Hyperparameters: num_leaves=31, learning_rate=0.05, n_estimators=200

### Cross-Species Validation

Developmental conservation was validated by comparing pig muscle transcriptomes with human skeletal muscle data from [Schaiter et al. (2024)](https://doi.org/10.1038/s41598-024-73893-5). We identified 36 genes with conserved developmental trajectories (p < 0.10 in both species).

## Requirements

### Python

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, scikit-learn, lightgbm, scipy

### R (for visualization)

```r
install.packages(c("ggplot2", "patchwork", "jsonlite", "viridis"))
```

## Data Availability

- **PigGTEx data**: https://piggtex.farmgtex.org/
- **Raw RNA-seq**: NCBI GEO accession [GSE257558](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE257558)

## Citation

If you use this pipeline or atlas, please cite:

```bibtex
@article{liu2025pig,
  title={A Calibrated Transcriptomic Atlas of Porcine Development Reveals Conserved Molecular Programs with Humans},
  author={Liu, Tianyuan and Lei, Rujing and Khan, Ilyas M. and Theobald, Peter},
  journal={In preparation},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, please contact:
- **Tianyuan Liu** - Primary developer
- **Peter Theobald** - Corresponding author (TheobaldP@cardiff.ac.uk)
