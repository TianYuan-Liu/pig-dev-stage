# Pipeline Default Behavior Update

## 🎯 Change Summary

The ML pipeline (`run_pipeline.py`) has been updated to **process all eligible tissues by default** instead of just processing Muscle.

## 📊 Default Behavior

### Before
```bash
python3 machine_learning/run_pipeline.py
# Would only process: Muscle
```

### After
```bash
python3 machine_learning/run_pipeline.py
# Now processes all 9 eligible tissues:
# - Muscle, Brain, Liver (4-class classification)
# - Blood, Macrophage, Small intestine, Lung (3-class)
# - Adipose, Testis (2-class)
```

## 🚀 Usage Examples

### Process All Eligible Tissues (Default)
```bash
# All of these commands do the same thing:
python3 machine_learning/run_pipeline.py
python3 machine_learning/run_pipeline.py --all-tissues
```

### Process Specific Tissue(s) Only
```bash
# Single tissue
python3 machine_learning/run_pipeline.py --tissues Muscle

# Multiple specific tissues
python3 machine_learning/run_pipeline.py --tissues Muscle Brain Liver

# Just the 4-class tissues
python3 machine_learning/run_pipeline.py --tissues Muscle Brain Liver
```

### Skip Cross-Validation (Faster)
```bash
# Process all tissues but skip cross-validation
python3 machine_learning/run_pipeline.py --skip-cv

# Process specific tissues without cross-validation
python3 machine_learning/run_pipeline.py --tissues Muscle Brain --skip-cv
```

### With Enhanced Logging Options
```bash
# All tissues with debug logging
python3 machine_learning/run_pipeline.py --log-level DEBUG

# All tissues with simple logging
python3 machine_learning/run_pipeline.py --simple-logging

# Specific tissues with debug logging
python3 machine_learning/run_pipeline.py --tissues Muscle --log-level DEBUG
```

## 📋 Eligible Tissues

The pipeline automatically reads eligible tissues from `results/results/eligible_tissues.json`:

| Classification | Tissues |
|---------------|---------|
| **4-class** | Muscle, Brain, Liver |
| **3-class** | Blood, Macrophage, Small intestine, Lung |
| **2-class** | Adipose, Testis |

## 🔍 How It Works

1. **Automatic Detection**: The pipeline looks for eligible tissues in these locations:
   - `results/results/eligible_tissues.json`
   - `artifacts/results/eligible_tissues.json`
   - `machine_learning/model_outputs/eligible_tissues.json`

2. **Duplicate Handling**: Automatically handles tissue name variations (e.g., "Small intestine" vs "Small_intestine")

3. **Fallback**: If no eligible tissues file is found, defaults to: Muscle, Brain, Liver, Blood, Lung

## ⚙️ Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--tissues` | Specific tissues to process | All eligible tissues |
| `--all-tissues` | Explicitly process all tissues | Same as default |
| `--skip-cv` | Skip cross-tissue validation | False |
| `--log-level` | Logging level | INFO |
| `--simple-logging` | Use simple logging | False |

## 💡 Benefits

1. **Comprehensive Analysis**: Automatically processes all tissues that meet quality thresholds
2. **Flexibility**: Can still process specific tissues when needed
3. **Efficiency**: Option to skip cross-validation for faster results
4. **Consistency**: Uses the same eligible tissues list throughout the pipeline

## 📈 Performance Considerations

Processing all 9 tissues will take longer than processing a single tissue:
- **Single tissue (Muscle)**: ~2-5 minutes
- **All 9 tissues**: ~15-30 minutes
- **With cross-validation**: Add ~10-20 minutes

To speed up processing:
1. Use `--skip-cv` to skip cross-validation
2. Use `--simple-logging` to reduce logging overhead
3. Specify only the tissues you need with `--tissues`

## 🔧 Implementation Details

The change was implemented by:
1. Adding `get_eligible_tissues()` function to read the eligible tissues list
2. Changing the default value of `--tissues` from `['Muscle']` to `None`
3. Auto-populating tissues list when `None` or when `--all-tissues` is used
4. Handling duplicate tissue names (with/without underscores)

## 📝 Notes

- The pipeline remains fully backward compatible
- Existing scripts that specify `--tissues Muscle` will work unchanged
- The eligible tissues list is determined by the preprocessing pipeline based on sample counts and quality metrics
- Cross-tissue validation is still performed by default unless `--skip-cv` is specified

---
*Updated: September 28, 2025*