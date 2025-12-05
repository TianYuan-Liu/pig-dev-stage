# Developmental Stage Model Overview

This document describes how the machine learning pipeline under `machine_learning/` predicts porcine developmental stages from RNA-seq expression matrices. The goal is to classify each sample into age-based stages (Infant → Adult) while adapting to tissue-specific sample availability and providing transparent performance reporting.

## Data Inputs and Stage Definitions
- Expression matrices come from pigGTEx (`data/pigGTEx/<tissue>.expr_tpm.txt.gz`) with genes in rows and samples in columns.
- Sample-level metadata (`data/full_metadata.csv`) supplies age, tissue labels, and optional covariates. Ages are converted into the ordered stage labels `['Infant', 'Early childhood', 'Pre-pubertal', 'Post-pubertal', 'Adult']`.
- `DataLoader` (`machine_learning/data_processing/data_loader.py`) aligns expression matrices and metadata, filters out genes with low detection rates, and adds derived categorical fields such as developmental stage.

## Pipeline Orchestration
`machine_learning/run_pipeline.py` is the entry point. For each tissue it executes the phases below, coordinated by structured logging (`utils/logging_config.py`) and a progress tracker (`utils/progress_tracker.py`). Command-line flags control the run (e.g., `--tissues`, `--max-features`, `--train-ratio`, `--seed`, `--skip-cv`).

## Step-by-Step Workflow

### 1. Data Loading and Quality Checks
- `DataLoader.load_expression` loads the tissue matrix, aligns samples to metadata, and optionally substitutes underscores for spaced tissue names.
- Quality hooks log summary statistics (`log_data_statistics`) and flag data issues such as missing values, zero-variance genes, stage metadata mismatches, or extreme sample means.

### 2. Stage Granularity Selection
- `StageGranularitySelector` (`data_processing/stage_selection.py`) inspects stage counts to decide whether the tissue supports a 4-, 3-, or 2-class problem. Adult samples can be merged into late-stage buckets when counts are sparse.
- The selected scheme supplies both the merged class labels and mapping rules so downstream targets stay ordinal.

### 3. Expression Preprocessing
- `ExpressionPreprocessor` (`data_processing/preprocessing.py`) applies log2(TPM+1) transformation, variance-based gene filtering (default 20th percentile), optional covariate regression, and per-gene z-score scaling.
- It tracks selected genes so the same transformation can be applied to hold-out or cross-tissue data.

### 4. Feature Selection
- `StableFeatureSelector` (`feature_engineering/feature_selection.py`) runs Elastic Net logistic regression with variance pre-filtering to prioritize informative genes.
- Coefficients determine feature importance; the selector limits the final feature set to `max_features` (2,000 by default) while preserving stability attributes for reporting.

### 5. Model Training
- `StageClassifier` (`model_training/models.py`) chooses an estimator based on the number of classes:
  - ≥3 classes: `OrdinalLogisticRegression`, implemented as cumulative link models with Elastic Net regularization and class weighting.
  - 2 classes: binary logistic regression (scikit-learn `LogisticRegression` with `saga` solver) using Elastic Net defaults.
- The pipeline performs a stratified train/test split when class counts allow (70/30 by default) and captures warnings or class-imbalance messages.

### 6. Evaluation and Reporting
- `MetricCalculator` (`model_evaluation/evaluation.py`) reports accuracy, balanced accuracy, macro/weighted F1, confusion matrices, and ordinal-aware metrics (MAE in stage units, quadratic-weighted Cohen’s κ, near-miss accuracy).
- Bootstrap resampling (1,000 samples by default) produces confidence intervals for key metrics.
- Phase timings, memory usage, and top-ranked genes are added to a per-tissue JSON artifact under `machine_learning/model_outputs/`.

## Cross-Tissue Generalization
When `--skip-cv` is not set, `CrossTissueValidator` (`cross_tissue_analysis/cross_tissue_validation.py`) trains models on all eligible tissues and evaluates:
- **Leave-one-tissue-out (LOTO):** hold each tissue out, train on the rest, and report balanced accuracy / macro F1.
- **Directed transfer:** optional source → target experiments with gene intersection, shared preprocessing, and cloned models.
Results are summarized into JSON and logged, highlighting transferable signatures and tissues that require additional data.

## Instrumentation and Observability
- `setup_logging` wires console, rotating file, and JSON logging with optional colorization. `PerformanceLogger` measures durations and memory peaks per phase.
- `PipelineProgressTracker` maintains phase-level ETAs, persists timing history (`machine_learning/timing_history.json`), and can render a Rich-based dashboard or fallback text progress when Rich is unavailable.

## Output Artifacts
For each processed tissue the pipeline writes `<Tissue>_results.json`, containing:
- Stage scheme, sample counts, preprocessing summaries, and the selected gene list (top 50 included in logs).
- Evaluation metrics plus bootstrap confidence intervals when enabled.
- Optional performance telemetry (phase timings, memory usage) when performance logging is active.
A combined `pipeline_summary.json` aggregates run-level information.

## Customization Tips
- **Feature budget:** adjust `--max-features` or the selector’s Elastic Net hyperparameters to trade off sparsity vs. accuracy.
- **Class balance:** override `StageClassifier` parameters (e.g., custom class weights) or supply `--train-ratio` / `--seed` to explore alternate splits.
- **Covariates:** enable regression of metadata columns by instantiating `ExpressionPreprocessor` with `remove_covariates` inside custom scripts.
- **New analyses:** extend `CrossTissueValidator` to plug in alternative dimensionality reduction (it already supports PCA/UMAP hooks) or add new transfer scenarios.

Together these components provide a reproducible, observable pipeline for developmental-stage prediction that scales across tissues while adapting its label space to the available data.
