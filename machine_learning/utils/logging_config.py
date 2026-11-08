"""Logging configuration for the ML pipeline."""

import logging
import time
from typing import Optional
from functools import wraps


def setup_logging(name: str = "ml_pipeline", level: str = "INFO") -> logging.Logger:
    """Set up logging with a simple console handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


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
                logger.error(f"Failed {func.__name__} after {duration:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator


class LogContext:
    """Context manager for grouped logging operations."""

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.operation} ({duration:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} ({duration:.2f}s) - {exc_val}")
        return False


def create_module_logger(module_name: str, parent_logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Create a child logger for a specific module."""
    if parent_logger:
        return parent_logger.getChild(module_name)
    return logging.getLogger(module_name)
