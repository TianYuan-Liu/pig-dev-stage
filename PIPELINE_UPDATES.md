# ML Pipeline Updates Summary

## Recent Changes (September 28, 2025)

### 1. ✅ Enhanced Logging Integration
- Merged enhanced logging from `run_pipeline_with_logging.py` into main `run_pipeline.py`
- Added performance tracking, memory monitoring, and detailed stage logging
- Preserved all recent bug fixes including `_to_serializable()` function
- Made enhanced logging optional with `--simple-logging` flag
- Removed duplicate file to avoid confusion

### 2. ✅ Default Behavior Change
- **NEW DEFAULT**: Pipeline now processes ALL eligible tissues automatically
- **BEFORE**: Only processed Muscle tissue by default
- **AFTER**: Processes all 9 eligible tissues (Muscle, Brain, Liver, Blood, Macrophage, Small intestine, Lung, Adipose, Testis)

## Quick Usage Guide

### Run with All Eligible Tissues (Default)
```bash
python3 machine_learning/run_pipeline.py
```

### Run with Specific Tissues
```bash
python3 machine_learning/run_pipeline.py --tissues Muscle
python3 machine_learning/run_pipeline.py --tissues Muscle Brain Liver
```

### Run Faster (Skip Cross-Validation)
```bash
python3 machine_learning/run_pipeline.py --skip-cv
```

### Debug Mode with Enhanced Logging
```bash
python3 machine_learning/run_pipeline.py --log-level DEBUG
```

### Simple Logging (Original Behavior)
```bash
python3 machine_learning/run_pipeline.py --simple-logging
```

## Key Features

| Feature | Description | Command Flag |
|---------|-------------|--------------|
| **All Tissues** | Process all eligible tissues | Default (no flag needed) |
| **Specific Tissues** | Process only specified tissues | `--tissues Muscle Brain` |
| **Enhanced Logging** | Detailed performance tracking | Default (use `--simple-logging` to disable) |
| **Debug Mode** | Maximum verbosity | `--log-level DEBUG` |
| **Skip Cross-Validation** | Faster processing | `--skip-cv` |

## File Structure

```
machine_learning/
├── run_pipeline.py              # Main pipeline (updated)
├── utils/
│   └── logging_config.py        # Logging utilities
└── model_outputs/               # Results directory

logs/                            # Log files directory
├── ml_pipeline_*.log           # Text logs
└── ml_pipeline_*.json          # Structured JSON logs

docs/
├── logging_guide.md            # Comprehensive logging documentation
├── logging_integration_summary.md
└── default_tissues_update.md   # This update documentation
```

## Performance Expectations

| Configuration | Approximate Time |
|---------------|-----------------|
| Single tissue (e.g., Muscle) | 2-5 minutes |
| All 9 tissues | 15-30 minutes |
| All tissues + cross-validation | 25-50 minutes |

## Eligible Tissues

Automatically determined from `results/results/eligible_tissues.json`:

- **4-class**: Muscle, Brain, Liver
- **3-class**: Blood, Macrophage, Small intestine, Lung
- **2-class**: Adipose, Testis

## Testing

Verify the installation:
```bash
# Test enhanced logging
python3 machine_learning/test_enhanced_logging.py

# Test default tissue selection
python3 machine_learning/test_default_tissues.py
```

## Backward Compatibility

✅ All existing scripts and commands remain functional
✅ Scripts specifying `--tissues Muscle` work unchanged
✅ Enhanced logging can be disabled with `--simple-logging`

---
*Last Updated: September 28, 2025*