# Fix Cross-Tissue Validation Label Space Mismatch

## Problem Summary

The cross-tissue validation in `cross_tissue_validation.py` uses raw stage labels directly without harmonizing them to a common scheme. When tissues have different stage granularities (e.g., Liver has 4-class, Brain has 2-class), the model trained on one label space cannot properly predict/evaluate on a different label space.

**Current behavior**: Raw `metadata['Stage']` values (e.g., "Infant", "Pre-pubertal", "Adult") are combined from different tissues without any mapping. Different tissues may have samples from different subsets of stages, causing label space mismatches during evaluation.

## Solution Overview

1. Determine a **common stage scheme** (lowest common denominator) across all participating tissues
2. **Map all raw stage labels** to the common scheme before training/evaluation
3. **Exclude tissues** that can't support the common scheme (with warnings)

## Implementation Steps

### Step 1: Add Common Scheme Determination to StageGranularitySelector

**File:** `machine_learning/data_processing/stage_selection.py`

Add method after `get_summary_table()` (around line 340):

```python
def determine_common_scheme(
    self,
    tissue_metadata: Dict[str, pd.DataFrame]
) -> Tuple[str, List[str], List[str]]:
    """
    Determine lowest common denominator scheme for cross-tissue validation.

    Args:
        tissue_metadata: Dict mapping tissue name -> metadata DataFrame with 'Stage' column

    Returns:
        Tuple of (scheme_name, eligible_tissues, excluded_tissues)
    """
    tissue_schemes = {}

    for tissue_name, metadata in tissue_metadata.items():
        # Count samples per stage from raw labels
        stage_counts = metadata['Stage'].value_counts()
        scheme_name, _ = self.select_scheme(stage_counts, tissue_name)
        tissue_schemes[tissue_name] = scheme_name

    # Find tissues that can't support any scheme
    excluded = [t for t, s in tissue_schemes.items() if s is None]
    eligible_schemes = {t: s for t, s in tissue_schemes.items() if s is not None}

    if not eligible_schemes:
        return None, [], list(tissue_metadata.keys())

    # Determine common scheme (lowest common denominator)
    # Priority: 2-class < 3-class < 4-class
    scheme_priority = {'2-class': 0, '3-class': 1, '4-class': 2}
    min_scheme = min(eligible_schemes.values(), key=lambda s: scheme_priority.get(s, 0))

    # Filter tissues that can support the common scheme
    eligible = [t for t, s in eligible_schemes.items()
                if scheme_priority.get(s, 0) >= scheme_priority.get(min_scheme, 0)]

    return min_scheme, eligible, excluded
```

### Step 2: Add Label Harmonization to CrossTissueValidator

**File:** `machine_learning/cross_tissue_analysis/cross_tissue_validation.py`

#### 2.1 Add import at top of file (after line 16)
```python
from machine_learning.data_processing.stage_selection import StageGranularitySelector
```

#### 2.2 Modify `__init__` (line 24-47) - add new parameters
```python
def __init__(
    self,
    model,
    preprocessor,
    min_common_genes: int = 1000,
    seed: int = 42,
    stage_selector: Optional[StageGranularitySelector] = None,
    common_scheme: str = 'auto'  # 'auto', '2-class', '3-class', '4-class'
):
    """
    Initialize cross-tissue validator.

    Args:
        model: Base model for classification
        preprocessor: Data preprocessor
        min_common_genes: Minimum common genes required
        seed: Random seed
        stage_selector: StageGranularitySelector for label harmonization
        common_scheme: Stage scheme for cross-tissue validation
                      'auto' = determine lowest common denominator
                      '2-class', '3-class', '4-class' = use specific scheme
    """
    self.model = model
    self.preprocessor = preprocessor
    self.min_common_genes = min_common_genes
    self.seed = seed
    self.stage_selector = stage_selector or StageGranularitySelector()
    self.common_scheme = common_scheme

    self.results = {}
    self.common_genes = None
    self.tissue_models = {}
```

#### 2.3 Add helper methods after `__init__`
```python
def _harmonize_labels(self, stages: pd.Series, scheme: str) -> pd.Series:
    """
    Map raw stage labels to common scheme.

    Args:
        stages: Series with raw stage labels (e.g., 'Infant', 'Pre-pubertal')
        scheme: Target scheme ('2-class', '3-class', '4-class')

    Returns:
        Series with harmonized labels
    """
    return self.stage_selector.prepare_labels(stages, scheme)

def _determine_common_scheme(
    self,
    tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
) -> Tuple[str, List[str], List[str]]:
    """
    Auto-determine common scheme from tissue data.

    Args:
        tissue_data: Dict of tissue_name -> (expression_df, metadata_df)

    Returns:
        (common_scheme, eligible_tissues, excluded_tissues)
    """
    # Extract metadata for each tissue
    tissue_metadata = {name: meta for name, (_, meta) in tissue_data.items()}
    return self.stage_selector.determine_common_scheme(tissue_metadata)
```

### Step 3: Modify `leave_one_tissue_out()` Method

**File:** `machine_learning/cross_tissue_analysis/cross_tissue_validation.py` (lines 49-153)

Replace the entire method with this updated version:

```python
def leave_one_tissue_out(
    self,
    tissue_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    target_tissue: str,
    common_scheme: Optional[str] = None
) -> Dict:
    """
    Train on all tissues except one, test on held-out tissue.

    Args:
        tissue_data: Dictionary of tissue -> (expression, metadata) tuples
        target_tissue: Tissue to hold out for testing
        common_scheme: Override common scheme (None uses self.common_scheme)

    Returns:
        Dictionary of performance metrics
    """
    logger.info(f"LOTO validation: Testing on {target_tissue}")

    # === NEW: Determine common scheme ===
    effective_scheme = common_scheme or self.common_scheme
    excluded_tissues = []

    if effective_scheme == 'auto':
        effective_scheme, eligible, excluded_tissues = self._determine_common_scheme(tissue_data)
        if effective_scheme is None:
            raise ValueError("No common scheme available for any tissues")
        logger.info(f"Auto-selected common scheme: {effective_scheme}")
        if excluded_tissues:
            logger.warning(f"Excluding {len(excluded_tissues)} tissues: {excluded_tissues}")
            # Remove excluded tissues from data
            tissue_data = {k: v for k, v in tissue_data.items() if k not in excluded_tissues}

    # Separate training and test tissues
    train_tissues = {k: v for k, v in tissue_data.items() if k != target_tissue}
    test_data = tissue_data[target_tissue]

    if not train_tissues:
        raise ValueError("No training tissues available")

    # Find common genes across all tissues (excluding test tissue - fix data leakage)
    common_genes = self._find_common_genes(list(train_tissues.values()))

    if len(common_genes) < self.min_common_genes:
        logger.warning(f"Only {len(common_genes)} common genes found")

    # Combine training data from all tissues
    X_train_combined = []
    y_train_combined = []
    tissue_labels_train = []

    for tissue_name, (expr, metadata) in train_tissues.items():
        # Filter to common genes
        expr_filtered = expr.loc[expr.index.intersection(common_genes)]

        # Remove samples with missing stages
        valid_idx = metadata['Stage'].notna()
        valid_idx_array = valid_idx.values
        expr_valid = expr_filtered.iloc[:, valid_idx_array]
        metadata_valid = metadata[valid_idx_array]

        # Transpose to samples x genes
        X_tissue = expr_valid.T

        # === NEW: Harmonize labels to common scheme ===
        y_tissue_raw = metadata_valid['Stage']
        y_tissue = self._harmonize_labels(y_tissue_raw, effective_scheme)

        X_train_combined.append(X_tissue)
        y_train_combined.extend(y_tissue.tolist())
        tissue_labels_train.extend([tissue_name] * len(y_tissue))

    # Concatenate all training data
    X_train = pd.concat(X_train_combined, axis=0)
    y_train = pd.Series(y_train_combined)

    # Prepare test data
    X_test_expr, test_metadata = test_data
    X_test_expr = X_test_expr.loc[X_test_expr.index.intersection(common_genes)]
    valid_test = test_metadata['Stage'].notna()
    valid_test_array = valid_test.values
    X_test = X_test_expr.iloc[:, valid_test_array].T

    # === NEW: Harmonize test labels to common scheme ===
    y_test_raw = test_metadata[valid_test_array]['Stage']
    y_test = self._harmonize_labels(y_test_raw, effective_scheme)

    # Align gene order
    common_genes_ordered = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_genes_ordered]
    X_test = X_test[common_genes_ordered]

    # Preprocess if needed
    if self.preprocessor:
        X_train_proc = self.preprocessor.fit_transform(X_train.T).T
        X_test_proc = self.preprocessor.transform(X_test.T).T
    else:
        X_train_proc = X_train
        X_test_proc = X_test

    # Train model
    model_clone = clone(self.model)
    model_clone.fit(X_train_proc, y_train)

    # Predict on test tissue
    y_pred = model_clone.predict(X_test_proc)

    # Calculate metrics
    metrics = {
        'target_tissue': target_tissue,
        'common_scheme': effective_scheme,  # NEW: Record scheme used
        'n_train_tissues': len(train_tissues),
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'n_common_genes': len(common_genes_ordered),
        'excluded_tissues': excluded_tissues,  # NEW: Record exclusions
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

    # Store results
    self.results[f'LOTO_{target_tissue}'] = metrics
    logger.info(f"LOTO {target_tissue}: Acc={metrics['balanced_accuracy']:.3f}")

    return metrics
```

### Step 4: Modify `cross_tissue_transfer()` Method

**File:** `machine_learning/cross_tissue_analysis/cross_tissue_validation.py` (lines 155-229)

Add harmonization to the existing method. Key changes at lines ~190 and ~197:

```python
def cross_tissue_transfer(
    self,
    source_data: Tuple[pd.DataFrame, pd.DataFrame],
    target_data: Tuple[pd.DataFrame, pd.DataFrame],
    source_name: str = "source",
    target_name: str = "target",
    common_scheme: Optional[str] = None  # NEW parameter
) -> Dict:
    """Train on one tissue, test on another with harmonized labels."""
    logger.info(f"Transfer: {source_name} -> {target_name}")

    # === NEW: Determine common scheme if auto ===
    effective_scheme = common_scheme or self.common_scheme
    if effective_scheme == 'auto':
        tissue_metadata = {
            source_name: source_data[1],
            target_name: target_data[1]
        }
        effective_scheme, eligible, excluded = self.stage_selector.determine_common_scheme(tissue_metadata)
        if effective_scheme is None:
            raise ValueError(f"No common scheme for {source_name} and {target_name}")
        logger.info(f"Using common scheme: {effective_scheme}")

    # ... (existing gene intersection code) ...

    # Prepare source data
    # ... (existing code until y_source assignment) ...
    y_source_raw = source_meta[valid_source_array]['Stage']
    y_source = self._harmonize_labels(y_source_raw, effective_scheme)  # NEW

    # Prepare target data
    # ... (existing code until y_target assignment) ...
    y_target_raw = target_meta[valid_target_array]['Stage']
    y_target = self._harmonize_labels(y_target_raw, effective_scheme)  # NEW

    # ... (rest of method unchanged) ...

    # Add scheme to metrics
    metrics['common_scheme'] = effective_scheme  # NEW
```

### Step 5: Update Pipeline Integration

**File:** `machine_learning/run_pipeline.py`

In `run_cross_tissue_validation()` function:
```python
# Initialize stage selector
stage_selector = StageGranularitySelector()

# Pass to validator
validator = CrossTissueValidator(
    model=model,
    preprocessor=preprocessor,
    stage_selector=stage_selector,
    common_scheme='auto'  # or from config
)
```

### Step 6: Add Configuration Options

**File:** `machine_learning/config.yaml`

```yaml
cross_tissue_validation:
  common_scheme: 'auto'  # 'auto', '2-class', '3-class', '4-class'
  min_tissues: 3
```

## Files to Modify

| File | Changes |
|------|---------|
| `machine_learning/data_processing/stage_selection.py` | Add `determine_common_scheme()` method |
| `machine_learning/cross_tissue_analysis/cross_tissue_validation.py` | Add harmonization, modify LOTO & transfer methods |
| `machine_learning/run_pipeline.py` | Pass stage_selector to validator |
| `machine_learning/config.yaml` | Add cross_tissue_validation section |

## Verification Plan

1. **Quick sanity check**: After implementation, add temporary logging:
   ```python
   logger.info(f"y_train unique labels: {y_train.unique()}")
   logger.info(f"y_test unique labels: {y_test.unique()}")
   ```
   Both should show identical labels from the common scheme.

2. **Run pipeline on tissues with known different schemes**:
   ```bash
   python -m machine_learning.run_pipeline --log-level DEBUG 2>&1 | grep -E "(common scheme|harmoniz|LOTO)"
   ```
   Verify logs show:
   - "Auto-selected common scheme: X-class"
   - Labels are being harmonized
   - No label mismatch errors

3. **Verify metrics are reasonable**:
   - Balanced accuracy should be > 0.5 (better than random)
   - No warnings about unknown labels during evaluation

## Bonus Fix Included

The updated `leave_one_tissue_out()` method also fixes the **data leakage issue** documented in the overview:
```python
# OLD (data leakage - test tissue included):
common_genes = self._find_common_genes(list(train_tissues.values()) + [test_data])

# NEW (fixed - only training tissues):
common_genes = self._find_common_genes(list(train_tissues.values()))
```

## Summary of Changes

| File | Method/Section | Change |
|------|----------------|--------|
| `stage_selection.py` | New `determine_common_scheme()` | Finds lowest common scheme |
| `cross_tissue_validation.py` | `__init__` | Add `stage_selector`, `common_scheme` params |
| `cross_tissue_validation.py` | New `_harmonize_labels()` | Maps raw labels to scheme |
| `cross_tissue_validation.py` | New `_determine_common_scheme()` | Wrapper for stage selector |
| `cross_tissue_validation.py` | `leave_one_tissue_out()` | Harmonize train/test labels |
| `cross_tissue_validation.py` | `cross_tissue_transfer()` | Harmonize source/target labels |
| `run_pipeline.py` | `run_cross_tissue_validation()` | Pass stage selector to validator |
| `config.yaml` | New section | `cross_tissue_validation.common_scheme` |
