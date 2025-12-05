# Machine Learning Module

This module contains all Python-based machine learning components for developmental stage classification.

## Structure

```
machine_learning/
├── data_processing/         # Data loading and preprocessing
│   ├── data_loader.py      # Handles expression data loading
│   ├── preprocessing.py    # Expression data preprocessing
│   └── stage_selection.py  # Stage granularity selection
├── feature_engineering/     # Feature selection and engineering
│   └── feature_selection.py # Stable feature selection
├── model_training/          # Model implementation
│   └── models.py           # Ordinal and binary classifiers
├── model_evaluation/        # Evaluation metrics
│   └── evaluation.py       # Comprehensive metric calculation
├── cross_tissue_analysis/   # Cross-tissue validation
│   └── cross_tissue_validation.py # Transfer learning validation
├── model_outputs/          # Saved models and results
└── run_pipeline.py         # Main pipeline script
```

## Quick Start

### Test the pipeline
```bash
python machine_learning/test_pipeline_simple.py
```

### Run full pipeline
```bash
python machine_learning/run_pipeline.py
```

## Components

### Data Processing
- **DataLoader**: Loads expression data and metadata from pigGTEx
- **ExpressionPreprocessor**: Log transformation, variance filtering, standardization
- **StageGranularitySelector**: Selects appropriate classification scheme

### Feature Engineering
- **StableFeatureSelector**: Elastic Net-based feature selection with stability

### Model Training
- **OrdinalLogisticRegression**: For ordered developmental stages
- **BinaryLogisticRegression**: For simplified 2-class problems
- **StageClassifier**: Unified interface for all classifiers

### Model Evaluation
- **MetricCalculator**: Comprehensive metrics with bootstrap CIs
- Includes ordinal-specific metrics (MAE, weighted kappa)

### Cross-Tissue Analysis
- **CrossTissueValidator**: Leave-one-tissue-out validation
- Tests model transferability across tissues

## Dependencies
- numpy, pandas
- scikit-learn
- scipy
- tqdm

## Output Format
Results are saved as JSON files in `model_outputs/` containing:
- Classification metrics
- Selected features
- Model parameters
- Cross-validation results