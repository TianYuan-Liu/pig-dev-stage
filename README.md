# Pig Developmental Stage Classification Pipeline

Machine learning pipeline for classifying developmental stages across pig tissues using transcriptomic data from pigGTEx.

## Project Structure

```
pig-dev-stage/
├── machine_learning/           # Python ML Pipeline
│   ├── data_processing/       # Data loading & preprocessing
│   ├── feature_engineering/   # Feature selection (variance, MI)
│   ├── model_training/        # ML models (LogisticRegression)
│   ├── model_evaluation/      # Metrics & bootstrap CI
│   ├── cross_tissue_analysis/ # Cross-tissue validation
│   ├── model_outputs/         # Results JSON files
│   └── run_pipeline.py        # Main entry point
│
├── visualization/              # R Publication Figures
│   ├── theme_configs/         # Nature journal themes
│   ├── figure1_methodology.R
│   ├── figure2_performance.R
│   ├── figure3_molecular_signatures.R
│   └── figure4_biological_validation.R
│
├── slides/                     # Presentation Materials
│   ├── R/                     # Slide figure generation
│   │   ├── slides_theme.R     # Cardiff-styled theme
│   │   └── generate_figures.R # Panel generator
│   ├── Figures/               # Generated PNG panels
│   ├── templates/             # LaTeX templates
│   └── theme/                 # Beamer theme files
│
├── paper/                      # LaTeX Manuscript
│   ├── main_modular.tex       # Main document
│   ├── sections/              # Paper sections
│   └── supplementary/         # Supplementary materials
│
├── data/                       # Data Files
│   ├── pigGTEx/               # Expression files (34 tissues)
│   └── full_metadata.csv      # Sample metadata (n=2,467)
│
├── scripts/                    # Utility Scripts
│   ├── enrichment_analysis.py
│   ├── biomarker_validation.py
│   └── run_cross_tissue_validation.py
│
├── results/                    # Analysis Outputs
├── configs/                    # Configuration Files
├── docs/                       # Documentation
└── logs/                       # Pipeline Logs
```

## Quick Start

### Run ML Pipeline
```bash
python machine_learning/run_pipeline.py
```

### Run with Custom Parameters
```bash
python machine_learning/run_pipeline.py \
    --max-features 1000 \
    --max-iter 1500 \
    --train-ratio 0.7 \
    --seed 42
```

### Cross-tissue Validation
```bash
python scripts/run_cross_tissue_validation.py
```

### Generate Publication Figures
```bash
cd visualization
Rscript figure1_methodology.R
Rscript figure2_performance.R
```

### Generate Slide Figures
```bash
cd slides/R
Rscript generate_figures.R
```

## Key Results

- **8 tissues analyzed**: Muscle, Brain, Liver, Blood, Small intestine, Lung, Adipose, Testis
- **Classification schemes**: 4-class (Infant/Early/Pre-pub/Post-pub), 3-class, or 2-class depending on sample availability
- **Performance**: Balanced accuracy 0.70-0.95 across tissues
- **Validation**: Bootstrap confidence intervals, GO enrichment, marker gene validation

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, scikit-learn, scipy, matplotlib

For R visualization:
```r
source("visualization/install_visualization_packages.R")
```

## Citation

If you use this pipeline, please cite the associated publication (in preparation).
