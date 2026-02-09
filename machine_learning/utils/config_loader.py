"""
Configuration loader for the machine learning pipeline.
Provides centralized access to all configuration parameters.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default configuration file path (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class PipelineConfig:
    """Centralized configuration management for the ML pipeline."""

    _instance: Optional["PipelineConfig"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls, config_path: Optional[Union[str, Path]] = None):
        """Singleton pattern to ensure consistent config across modules."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = self._get_defaults()
            return

        try:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self._config = self._get_defaults()

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Return default configuration values.

        These must stay in sync with config.yaml. They serve as a fallback
        only when the YAML file cannot be loaded.
        """
        return {
            "paths": {
                "data_dir": "data/pigGTEx",
                "metadata_file": "data/PigGTEx_v0.MetaTable.csv",
                "expression_pattern": "data/pigGTEx/{tissue_name}.expr_tpm.txt.gz",
            },
            "preprocessing": {
                "log_transform": True,
                "min_variance_percentile": 0,
                "standardize": False,
            },
            "splitting": {
                "n_cv_folds": 5,
                "seed": 42,
            },
            "stage_selection": {
                "min_samples_4class": 40,
                "min_samples_3class": 30,
                "min_samples_2class": 15,
                "min_total_2class": 40,
            },
            "stage_mapping": {
                "Infant": {"min_days": 0, "max_days": 20},
                "Early childhood": {"min_days": 21, "max_days": 59},
                "Pre-pubertal": {"min_days": 60, "max_days": 149},
                "Post-pubertal": {"min_days": 150, "max_days": 365},
                "Adult": {"min_days": 366, "max_days": None},
            },
            "age_conversion": {
                "days": 1,
                "weeks": 7,
                "months": 30,
                "years": 365,
            },
        }

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation or multiple keys.

        Examples:
            config.get("model", "lightgbm", "learning_rate")
            config.get("splitting", "seed")
        """
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to top-level config sections."""
        return self._config.get(key, {})

    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get("paths", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get("preprocessing", {})

    @property
    def splitting(self) -> Dict[str, Any]:
        """Get train/test splitting configuration."""
        return self._config.get("splitting", {})

    @property
    def stage_selection(self) -> Dict[str, Any]:
        """Get stage selection thresholds."""
        return self._config.get("stage_selection", {})

    @property
    def stage_mapping(self) -> Dict[str, Dict[str, int]]:
        """Get developmental stage age boundaries."""
        return self._config.get("stage_mapping", {})

    @classmethod
    def reload(cls, config_path: Optional[Union[str, Path]] = None) -> "PipelineConfig":
        """Force reload of configuration (useful for testing)."""
        cls._instance = None
        return cls(config_path)


# Convenience function for getting the singleton config instance
def get_config(config_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Get the pipeline configuration singleton."""
    return PipelineConfig(config_path)
