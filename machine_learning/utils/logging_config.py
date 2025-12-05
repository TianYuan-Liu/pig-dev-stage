#!/usr/bin/env python3
"""
Enhanced logging configuration for the ML pipeline.
Provides detailed logging with rotation, colored output, and performance tracking.
"""

import logging
import logging.handlers
import sys
import json
import time
import traceback
import psutil
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from functools import wraps


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_data)


class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, logger: logging.Logger):
        """Initialize performance logger."""
        self.logger = logger
        self.start_times = {}
        self.metrics = {}

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")

    def end_timer(self, operation: str) -> float:
        """End timing and log duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0

        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]

        self.metrics[operation] = duration
        self.logger.info(f"Completed {operation} in {duration:.2f} seconds")
        return duration

    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        self.logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")
        return memory_mb

    def log_system_info(self):
        """Log system information."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024**3,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3
        }

        self.logger.info(f"System Information: {json.dumps(info, indent=2)}")
        return info

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'timings': self.metrics,
            'total_time': sum(self.metrics.values()),
            'slowest_operation': max(self.metrics.items(), key=lambda x: x[1]) if self.metrics else None
        }


def setup_logging(
    name: str = "ml_pipeline",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console: bool = True,
    file: bool = True,
    json_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    colored: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Enable console logging
        file: Enable file logging
        json_file: Enable JSON file logging
        max_bytes: Max size for log files before rotation
        backup_count: Number of backup files to keep
        colored: Use colored console output

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create log directory if needed
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Generate timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if colored:
            console_format = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file:
        file_path = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)

        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Log file location
        logger.info(f"Logging to file: {file_path}")

    # JSON file handler for structured logging
    if json_file:
        json_path = log_dir / f"{name}_{timestamp}.json"
        json_handler = logging.handlers.RotatingFileHandler(
            json_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JsonFormatter())
        logger.addHandler(json_handler)

        logger.info(f"JSON logging to file: {json_path}")

    return logger


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {duration:.2f} seconds: {str(e)}")
                raise

        return wrapper
    return decorator


def log_data_shape(logger: logging.Logger, data_name: str):
    """Decorator to log data shape changes."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log input shape if available
            if args and hasattr(args[0], 'shape'):
                logger.debug(f"{func.__name__} input {data_name} shape: {args[0].shape}")

            result = func(*args, **kwargs)

            # Log output shape if available
            if hasattr(result, 'shape'):
                logger.debug(f"{func.__name__} output {data_name} shape: {result.shape}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if hasattr(r, 'shape'):
                        logger.debug(f"{func.__name__} output[{i}] {data_name} shape: {r.shape}")

            return result
        return wrapper
    return decorator


class LogContext:
    """Context manager for grouped logging operations."""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """Initialize log context."""
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} ({duration:.2f}s) - {exc_val}")

        return False  # Don't suppress exceptions


def create_module_logger(module_name: str, parent_logger: Optional[logging.Logger] = None) -> logging.Logger:
    """
    Create a child logger for a specific module.

    Args:
        module_name: Name of the module
        parent_logger: Parent logger to inherit from

    Returns:
        Module-specific logger
    """
    if parent_logger:
        return parent_logger.getChild(module_name)
    else:
        return logging.getLogger(module_name)