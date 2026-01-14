# Developmental Stage Classification Pipeline

Technical reference for the machine learning pipeline that predicts porcine developmental stages from RNA-seq expression data.

---

## 1. System Objective

The pipeline classifies RNA-seq samples into ordered developmental stages based on gene expression profiles. This is an **ordinal classification** problem where the target variable has a natural ordering:

```
Infant (0-20d) < Early childhood (21-59d) < Pre-pubertal (60-149d) < Post-pubertal (150-365d) < Adult (>365d)
```

**Key design goals:**
- Adapt classification granularity to tissue-specific sample availability
- Prevent data leakage between training and evaluation sets
- Provide ordinal-aware metrics that respect stage ordering
- Enable cross-tissue generalization assessment

---

## 2. Pipeline Architecture

Entry point: `machine_learning/run_pipeline.py`

The pipeline executes 7 sequential phases per tissue:

```
Data Loading → Stage Selection → Train/Test Split → Preprocessing → Feature Selection → Model Training → Evaluation
```

### Phase 1: Data Loading

**Module:** `data_processing/data_loader.py`

| Operation | Implementation |
|-----------|----------------|
| Expression loading | `DataLoader.load_expression()` reads `data/pigGTEx/{tissue}.expr_tpm.txt.gz` |
| Matrix format | Genes (rows) x Samples (columns) as TPM values |
| Detection filtering | Removes genes with TPM > 0.1 in < 10% of samples (`data_loader.py:285-290`) |
| Metadata alignment | Intersects expression columns with metadata sample IDs (`data_loader.py:297-315`) |

**Alignment validation** (`data_loader.py:307-310`):
```python
common_samples = [s for s in expr_df.columns if s in tissue_metadata.index]
if len(common_samples) == 0:
    raise ValueError("No matching sample IDs between expression and metadata")
```

**Age-to-stage conversion** (`data_loader.py:101-118`):
| Age Range | Stage |
|-----------|-------|
| 0-20 days | Infant |
| 21-59 days | Early childhood |
| 60-149 days | Pre-pubertal |
| 150-365 days | Post-pubertal |
| >365 days | Adult |

### Phase 2: Stage Granularity Selection

**Module:** `data_processing/stage_selection.py`

`StageGranularitySelector` (`stage_selection.py:25-169`) adaptively selects classification granularity based on sample counts:

| Scheme | Merging Rule | Min Samples/Class | Total Min |
|--------|--------------|-------------------|-----------|
| 5-class | None (original) | 40 | - |
| 4-class | Adult → Post-pubertal | 40 | - |
| 3-class | Infant+Early childhood → Early; Post-pubertal+Adult → Late | 30 | - |
| 2-class | <150d → Pre-pubertal; >=150d → Post-pubertal | 15 | 40 |

Selection proceeds from highest to lowest granularity until minimum sample requirements are met.

### Phase 3: Train/Test Split

**Location:** `run_pipeline.py:331-374`

| Parameter | Default | Notes |
|-----------|---------|-------|
| Train ratio | 0.7 | 70/30 split |
| Stratification | Yes | Preserves class proportions when min_samples >= 2 |
| Random seed | 42 | Reproducible splits |

**Data leakage prevention:** Split occurs BEFORE any preprocessing transformations.

```python
# Stratified split if possible (run_pipeline.py:342-349)
X_train, X_test, y_train, y_test = train_test_split(
    X.T, y, train_size=train_ratio, stratify=y, random_state=seed
)
```

**Fallback:** If any class has < 2 samples, uses non-stratified random split with warning.

### Phase 4: Preprocessing

**Module:** `data_processing/preprocessing.py`

`ExpressionPreprocessor` (`preprocessing.py:19-167`) applies transformations in sequence:

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Log transformation | log2(TPM + 1) |
| 2 | Variance filtering | Remove genes below 20th percentile variance |
| 3 | Z-score standardization | Per-gene mean=0, std=1 |

**Data leakage prevention** (`run_pipeline.py:391`):
```python
X_train_processed = preprocessor.fit_transform(X_train_raw)  # Fits on training
X_test_processed = preprocessor.transform(X_test_raw)        # Uses fitted parameters
```

Variance threshold and scaling statistics are computed exclusively from training data.

### Phase 5: Feature Selection

**Location:** `run_pipeline.py:447-474`

Feature selection uses **LightGBM feature importance** to rank and select genes:

```python
# Fit LightGBM on training data to derive feature importance
importance_model = create_lightgbm_model(seed=random_state)
importance_model.fit(X_train_full, y_train)

# Rank genes by importance and select top N
importance = importance_model.get_feature_importance()
selected_genes = importance.head(max_features).index.to_numpy()
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_features` | 2000 | Maximum genes to retain |
| Importance type | "gain" | LightGBM gain-based importance |

**How it works:**
1. Fit OrdinalLightGBM on all preprocessed training genes
2. Extract feature importance (average gain across binary thresholds)
3. Select top `max_features` genes ranked by importance
4. Apply same gene subset to test set

**Data leakage prevention:** Feature importance computed on training data only.

### Phase 6: Model Training

**Module:** `model_training/models.py`

**Algorithm:** `OrdinalLightGBM` (`models.py:20-257`)

Implements ordinal reduction via binary threshold decomposition (CORAL/CORN approach):
- For K classes, trains K-1 binary LightGBM classifiers
- Classifier k predicts P(class > k)
- Cumulative probabilities converted to class probabilities

**Default hyperparameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_leaves` | 31 | Tree complexity |
| `max_depth` | -1 | Unlimited |
| `learning_rate` | 0.1 | Boosting step size |
| `n_estimators` | 100 | Max boosting rounds |
| `min_data_in_leaf` | 20 | Leaf sample threshold |
| `feature_fraction` | 0.9 | Feature bagging |
| `bagging_fraction` | 0.9 | Data bagging |
| `lambda_l1` | 0.0 | L1 regularization |
| `lambda_l2` | 0.0 | L2 regularization |

**Class balancing** (`models.py:104-110`):
```python
if self.class_weight == 'balanced':
    n_pos = np.sum(y_binary)
    n_neg = len(y_binary) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
```

**Early stopping** (`models.py:130-148`):
- Internal 80/20 validation split per binary classifier
- Patience: 10 rounds without improvement
- Uses different random seed per threshold: `seed + k`

### Phase 7: Evaluation

**Module:** `model_evaluation/evaluation.py`

`MetricCalculator` (`evaluation.py:266-404`) computes comprehensive metrics:

**Classification metrics:**
| Metric | Function |
|--------|----------|
| Accuracy | `accuracy_score` |
| Balanced accuracy | `balanced_accuracy_score` |
| F1 (macro) | `f1_score(average='macro')` |
| F1 (weighted) | `f1_score(average='weighted')` |
| Matthews correlation | `matthews_corrcoef` |
| Confusion matrix | `confusion_matrix` |

**Ordinal-specific metrics** (`evaluation.py:24-75`):
| Metric | Purpose |
|--------|---------|
| MAE | Mean absolute error in stage units |
| Quadratic kappa | Distance-weighted agreement |
| Linear kappa | Linearly-weighted agreement |
| Spearman rho | Rank correlation |
| Kendall tau | Ordinal concordance |
| Near-miss accuracy (pm1) | Correct if within 1 stage |

**Bootstrap confidence intervals** (`evaluation.py:215-263`):
- Default: 1000 resamples, 95% CI
- Computed for balanced accuracy, F1 macro, and MAE

---

## 3. Model Details

### OrdinalLightGBM Architecture

The model treats ordinal classification as a series of cumulative binary problems:

```
Class 0 vs (1,2,3,4)  →  P(y > 0)
Class (0,1) vs (2,3,4) →  P(y > 1)
Class (0,1,2) vs (3,4) →  P(y > 2)
Class (0,1,2,3) vs 4   →  P(y > 3)
```

**Probability calculation** (`models.py:154-193`):
```python
# Cumulative probabilities from binary classifiers
cumulative_probs[:, k+1] = P(class > k)

# Convert to class probabilities
class_probs[:, 0] = 1 - cumulative_probs[:, 1]
class_probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k+1]
class_probs[:, -1] = cumulative_probs[:, -1]
```

**Monotonic post-processing** (`models.py:232-256`):
- Enforces unimodality: probabilities decrease from the mode
- Ensures ordinal consistency in predictions

### Feature Importance

Aggregated across binary classifiers (`models.py:209-230`):
```python
importance = pd.concat([model.feature_importance() for model in models_]).mean(axis=1)
```

---

## 4. Validation Strategy

### 4.1 Within-Tissue Validation

Standard stratified train/test split (70/30) applied independently per tissue.

### 4.2 Nested Cross-Validation

**Module:** `cross_tissue_analysis/cross_validation.py`

`NestedCrossValidator` (`cross_validation.py:32-177`) implements proper nested CV:

| Loop | Purpose | Default Folds |
|------|---------|---------------|
| Outer | Unbiased model evaluation | 5 |
| Inner | Hyperparameter tuning | 3 |

**Data leakage prevention** (`cross_validation.py:263-268`):
```python
# Feature selection fit ONLY on training fold
if self.feature_selector:
    selector_clone = clone(self.feature_selector)
    selector_clone.fit(X_train, y_train)
    X_train = selector_clone.transform(X_train)
    X_test = selector_clone.transform(X_test)
```

### 4.3 Leave-One-Tissue-Out (LOTO) Validation

**Module:** `cross_tissue_analysis/cross_tissue_validation.py`

`CrossTissueValidator.leave_one_tissue_out()` (`cross_tissue_validation.py:52-156`):
- Trains on all tissues except one
- Tests on held-out tissue
- Reports balanced accuracy and F1 macro

### 4.4 Cross-Tissue Transfer

`CrossTissueValidator.cross_tissue_transfer()` (`cross_tissue_validation.py:158-232`):
- Directed source → target experiments
- Uses gene intersection between tissues
- Evaluates transferability of developmental signatures

---

## 5. Critical Implementation Notes

### 5.1 Known Issues

#### HIGH SEVERITY: LOTO Common Gene Data Leakage

**Location:** `cross_tissue_validation.py:77-78`

```python
# ISSUE: Includes held-out test tissue in gene intersection
common_genes = self._find_common_genes(
    list(train_tissues.values()) + [test_data]  # ← test_data should NOT be here
)
```

**Impact:** Information from the held-out tissue leaks into training through gene selection. This causes **optimistic bias** in cross-tissue generalization estimates.

**Recommendation:** Compute common genes from training tissues only:
```python
common_genes = self._find_common_genes(list(train_tissues.values()))
```

#### MEDIUM SEVERITY: Non-Stratified Early Stopping Split

**Location:** `models.py:131-133`

```python
n_train = int(0.8 * len(y_binary))
indices = np.random.RandomState(self.seed + k).permutation(len(y_binary))
train_idx, val_idx = indices[:n_train], indices[n_train:]
```

**Impact:** For sparse classes, the 80/20 internal validation split may not preserve class proportions, potentially affecting early stopping behavior.

#### LOW SEVERITY: Stratification Failure Handling

Uses `stratify=y` in train_test_split without explicit handling when any class has < 2 samples. The pipeline falls back to non-stratified splits, but this should be explicitly documented in logs.

### 5.2 Edge Case Handling

**Missing stages in a tissue:**
- Filtered by `metadata['Stage'].notna()` check (`cross_tissue_validation.py:94-98`)
- Samples with missing stage labels are excluded from training/testing

**Extreme class imbalance:**
- `StageGranularitySelector` automatically merges sparse stages into coarser schemes
- Minimum 15 samples per class for binary; 30-40 for multi-class

**Zero-variance genes:**
- Removed during variance filtering (20th percentile threshold)
- Logged as data quality warning (`run_pipeline.py:100-106`)

### 5.3 Reproducibility

| Component | Seed Control |
|-----------|--------------|
| Train/test split | `--seed` argument (default: 42) |
| Feature selection | Same seed propagated |
| Model training | `seed + k` per binary classifier |
| Bootstrap resampling | Fixed seed (42) |

### 5.4 Memory and Performance

- Expression matrices transposed between genes x samples and samples x genes formats as needed
- Large gene sets filtered progressively (detection rate → variance → feature importance)
- Timing logged via `PerformanceLogger` and `PipelineProgressTracker`

---

## 6. Configuration Reference

Default parameters from `machine_learning/config.yaml`:

```yaml
preprocessing:
  log_transform: true
  log_base: 2
  pseudocount: 1
  variance_filter_percentile: 20
  standardize: true

feature_selection:
  max_features: 2000
  stability_threshold: 0.6

model:
  num_leaves: 31
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 200
  min_data_in_leaf: 10
  feature_fraction: 0.8
  bagging_fraction: 0.8
  lambda_l1: 0.1
  lambda_l2: 0.1
  early_stopping_patience: 10

cross_validation:
  n_folds: 5
  shuffle: true

stage_thresholds:
  min_samples_4class: 40
  min_samples_3class: 30
  min_samples_2class: 15
  min_total_2class: 40
```

---

## 7. Output Artifacts

Per-tissue results saved to `machine_learning/model_outputs/{Tissue}_results.json`:

| Field | Content |
|-------|---------|
| `stage_scheme` | Selected classification scheme (5/4/3/2-class) |
| `sample_counts` | Samples per stage after selection |
| `preprocessing_summary` | Genes retained at each filtering step |
| `selected_genes` | Top features ranked by importance |
| `metrics` | All evaluation metrics with bootstrap CIs |
| `confusion_matrix` | True vs predicted stage counts |
| `timing` | Phase-level execution times |

Aggregated summary: `machine_learning/model_outputs/pipeline_summary.json`

---

## 8. Command-Line Interface

```bash
python -m machine_learning.run_pipeline \
    --tissues Liver Brain Heart \
    --max-features 2000 \
    --train-ratio 0.7 \
    --seed 42 \
    --skip-cv \
    --log-level INFO
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tissues` | All available | Specific tissues to process |
| `--max-features` | 2000 | Maximum features to select |
| `--train-ratio` | 0.7 | Training set proportion |
| `--seed` | 42 | Random seed |
| `--skip-cv` | False | Skip cross-tissue validation |
| `--log-level` | INFO | Logging verbosity |
| `--simple-logging` | False | Disable rich formatting |
