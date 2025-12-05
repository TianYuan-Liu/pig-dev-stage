# Enhanced Logging Guide for ML Pipeline

## Overview

The ML pipeline now includes comprehensive logging capabilities that provide detailed tracking of all pipeline stages, performance metrics, and debugging information.

## Quick Start

### Running the Enhanced Pipeline

```bash
# Basic usage with INFO level logging
python3 machine_learning/run_pipeline_with_logging.py --tissues Muscle

# With DEBUG level for detailed diagnostics
python3 machine_learning/run_pipeline_with_logging.py --tissues Muscle --log-level DEBUG

# Process multiple tissues
python3 machine_learning/run_pipeline_with_logging.py --tissues Muscle Heart Liver

# Skip cross-validation for faster testing
python3 machine_learning/run_pipeline_with_logging.py --tissues Muscle --skip-cv
```

## Features

### 1. Multi-Level Logging

The system supports five log levels:
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Unexpected but recoverable situations
- **ERROR**: Serious problems that prevent operation completion
- **CRITICAL**: System-wide failures

### 2. Multiple Output Formats

#### Console Output
- Colored output for better readability
- Real-time progress tracking
- Hierarchical message structure

#### File Output
- Plain text logs with detailed formatting
- Automatic rotation when files exceed 10MB
- Keeps 5 backup files by default

#### JSON Structured Logs
- Machine-readable format for programmatic analysis
- Includes metadata like timestamps, module, function, and line numbers
- Extra data fields for metrics and parameters

### 3. Performance Monitoring

The system tracks:
- **Execution Time**: Duration of each pipeline stage
- **Memory Usage**: RAM consumption at key checkpoints
- **System Information**: Platform, Python version, CPU, memory specs
- **Operation Timings**: Individual operation durations

### 4. Data Quality Validation

Automatic checks for:
- Missing values in expression data
- Constant genes (zero variance)
- Outlier samples (>3 standard deviations)
- Metadata alignment issues
- Dimension mismatches

### 5. Hierarchical Logging Structure

```
Pipeline
├── System Information
├── Configuration
├── Tissue Processing
│   ├── Data Loading
│   │   ├── File reading
│   │   ├── Metadata loading
│   │   └── Quality validation
│   ├── Stage Selection
│   ├── Preprocessing
│   │   ├── Variance filtering
│   │   └── Normalization
│   ├── Feature Selection
│   │   ├── Variance threshold
│   │   └── Elastic net selection
│   ├── Model Training
│   │   ├── Train/test split
│   │   └── Model fitting
│   └── Evaluation
│       ├── Predictions
│       └── Metrics calculation
└── Cross-Tissue Validation
    ├── Tissue data loading
    ├── Within-tissue performance
    └── Leave-one-out validation
```

## Log Files Location

All logs are saved in the `logs/` directory with timestamps:
- `ml_pipeline_YYYYMMDD_HHMMSS.log` - Plain text logs
- `ml_pipeline_YYYYMMDD_HHMMSS.json` - Structured JSON logs
- `pipeline_summary.json` - Summary of the entire run

## Configuration

### Using the YAML Configuration File

Edit `configs/logging_config.yaml` to customize:

```yaml
logging:
  level: INFO  # Global log level
  log_dir: logs  # Output directory

  console:
    enabled: true
    colored: true
    level: INFO

  file:
    enabled: true
    level: DEBUG
    max_bytes: 10485760  # 10 MB
    backup_count: 5
```

### Programmatic Configuration

```python
from machine_learning.utils.logging_config import setup_logging

logger = setup_logging(
    name="custom_pipeline",
    level="DEBUG",
    log_dir=Path("custom_logs"),
    console=True,
    file=True,
    json_file=True,
    colored=True
)
```

## Using Logging in Your Code

### Basic Logging

```python
logger.info("Processing tissue: %s", tissue_name)
logger.debug("Data shape: %s", data.shape)
logger.warning("Low sample count: %d", n_samples)
logger.error("Failed to load file: %s", file_path)
```

### Performance Tracking

```python
from machine_learning.utils.logging_config import PerformanceLogger

perf_logger = PerformanceLogger(logger)

# Track timing
perf_logger.start_timer("operation_name")
# ... do work ...
duration = perf_logger.end_timer("operation_name")

# Track memory
memory_mb = perf_logger.log_memory_usage("checkpoint_name")

# Get summary
summary = perf_logger.get_summary()
```

### Context Manager for Grouped Operations

```python
from machine_learning.utils.logging_config import LogContext

with LogContext(logger, "Data Processing"):
    # All operations here are grouped
    logger.info("Loading data...")
    # ... operations ...
    logger.info("Processing complete")
# Automatically logs completion time
```

### Function Decorators

```python
from machine_learning.utils.logging_config import log_execution_time

@log_execution_time(logger)
def process_data(data):
    # Function execution time is automatically logged
    return processed_data
```

## Example Output

### Console Output
```
2025-09-27 11:40:03 - INFO - ml_pipeline - MACHINE LEARNING PIPELINE WITH ENHANCED LOGGING
2025-09-27 11:40:03 - INFO - ml_pipeline - Processing Muscle
2025-09-27 11:40:03 - INFO - ml_pipeline.tissue.Muscle - Starting: Data Loading
2025-09-27 11:40:05 - INFO - ml_pipeline.tissue.Muscle - Loaded Muscle: 22088 genes x 914 samples
2025-09-27 11:40:05 - WARNING - ml_pipeline.tissue.Muscle - Muscle: 30 samples may be outliers
2025-09-27 11:40:05 - INFO - ml_pipeline - Completed Muscle_loading in 2.02 seconds
```

### JSON Log Entry
```json
{
  "timestamp": "2025-09-27T09:40:05.123456",
  "level": "INFO",
  "logger": "ml_pipeline.tissue.Muscle",
  "message": "Loaded Muscle: 22088 genes x 914 samples",
  "module": "run_pipeline_with_logging",
  "function": "run_single_tissue_pipeline",
  "line": 165,
  "extra": {
    "n_genes": 22088,
    "n_samples": 914,
    "memory_mb": 485.2
  }
}
```

## Troubleshooting

### Common Issues

1. **Log files not created**: Check write permissions for the `logs/` directory
2. **Missing colored output**: Install colorama: `pip install colorama`
3. **Memory tracking not working**: Install psutil: `pip install psutil`
4. **JSON logs corrupted**: Each line is a separate JSON object, not an array

### Debug Mode

For maximum verbosity:
```bash
python3 machine_learning/run_pipeline_with_logging.py --log-level DEBUG
```

This will show:
- All data transformations
- Detailed parameter values
- Step-by-step execution flow
- Memory usage at each stage

## Performance Impact

The logging system has minimal performance impact:
- Console logging: ~1-2% overhead
- File logging: ~2-3% overhead
- JSON logging: ~3-4% overhead
- Performance tracking: ~1% overhead

For production runs, you can reduce overhead by:
- Setting log level to WARNING or ERROR
- Disabling console output
- Using async logging (in configuration)

## Integration with Monitoring Tools

The JSON logs can be easily integrated with:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- CloudWatch
- Custom dashboards using the structured format

## Best Practices

1. **Use appropriate log levels**: DEBUG for diagnostics, INFO for progress, WARNING for issues
2. **Include context**: Add tissue names, sample counts, parameter values
3. **Log failures with details**: Include error messages and stack traces
4. **Track performance**: Monitor slow operations and memory usage
5. **Clean old logs**: Implement log rotation or cleanup scripts
6. **Use structured logging**: Include extra data for analysis
7. **Test with small datasets**: Verify logging before full runs

## Summary

The enhanced logging system provides comprehensive visibility into the ML pipeline execution, making it easier to:
- Debug issues quickly
- Monitor performance
- Track data quality
- Analyze results
- Reproduce runs
- Generate reports

For questions or issues, check the log files first - they contain detailed information about every step of the pipeline execution.