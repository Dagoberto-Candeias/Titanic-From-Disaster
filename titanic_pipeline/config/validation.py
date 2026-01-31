"""
Configuration validation and management for Titanic ML Pipeline.
"""

import logging
from typing import Dict, Any, Optional
from .default import (
    DEFAULT_CONFIG,
    DEFAULT_LOGGING_CONFIG,
    DEFAULT_EXPECTED_TRAIN_SCHEMA,
    DEFAULT_EXPECTED_TEST_SCHEMA,
)


class ConfigManager:
    """Manages configuration loading, validation, and merging."""

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = {}
        self.logging_config = {}
        self.train_schema = {}
        self.test_schema = {}
        self.load_config(config_override)

    def load_config(self, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load and merge configuration from defaults and external file."""
        # Start with defaults
        self.config = DEFAULT_CONFIG.copy()
        self.logging_config = DEFAULT_LOGGING_CONFIG.copy()
        self.train_schema = DEFAULT_EXPECTED_TRAIN_SCHEMA.copy()
        self.test_schema = DEFAULT_EXPECTED_TEST_SCHEMA.copy()

        # Try to load external config
        try:
            # Tenta importar de src.config se disponível no path
            import sys
            if 'src' not in sys.modules:
                try:
                    from src.config import CONFIG as IMPORTED_CONFIG
                    from src.config import EXPECTED_TRAIN_SCHEMA as IMPORTED_TRAIN_SCHEMA
                    from src.config import EXPECTED_TEST_SCHEMA as IMPORTED_TEST_SCHEMA
                    from src.config import LOGGING_CONFIG as IMPORTED_LOGGING_CONFIG
                    
                    # Merge configurations
                    self.config.update(IMPORTED_CONFIG)
                    self.train_schema.update(IMPORTED_TRAIN_SCHEMA)
                    self.test_schema.update(IMPORTED_TEST_SCHEMA)
                    self.logging_config.update(IMPORTED_LOGGING_CONFIG)
                    logging.info("✅ External configuration loaded successfully from src.config")
                except ImportError:
                    pass
        except Exception as e:
            logging.warning(f"⚠️  External config load skipped: {e}")

        # Apply runtime overrides
        if config_override:
            self.config.update(config_override)
            logging.info(f"🔧 Configuration overridden: {config_override}")

        # Apply platform-specific adjustments
        self._apply_platform_adjustments()

        # Validate configuration
        self._validate_config()

        return {
            "config": self.config,
            "logging_config": self.logging_config,
            "train_schema": self.train_schema,
            "test_schema": self.test_schema,
        }

    def get_config(self) -> Dict[str, Any]:
        """Returns the current configuration dictionary."""
        return self.config

    def get_schemas(self) -> Dict[str, Any]:
        """Returns the data schemas."""
        return {
            "train": self.train_schema,
            "test": self.test_schema
        }

    def _apply_platform_adjustments(self):
        """Apply platform-specific configuration adjustments."""
        import platform

        if platform.system() == "Windows":
            # Limit parallel jobs on Windows to avoid subprocess issues
            self.config["parallel_jobs"] = min(self.config.get("parallel_jobs", 4), 1)
            logging.info("🔧 Windows detected - limiting parallel_jobs to 1")

    def _validate_config(self):
        """Validate configuration values."""
        # Validate parallel_jobs
        if not isinstance(self.config.get("parallel_jobs"), int) or self.config["parallel_jobs"] < 1:
            self.config["parallel_jobs"] = 1
            logging.warning("⚠️  Invalid parallel_jobs, setting to 1")

        # Validate cv_folds
        if not isinstance(self.config.get("cv_folds"), int) or self.config["cv_folds"] < 2:
            self.config["cv_folds"] = 5
            logging.warning("⚠️  Invalid cv_folds, setting to 5")

        # Validate random_state
        if not isinstance(self.config.get("random_state"), int):
            self.config["random_state"] = 42
            logging.warning("⚠️  Invalid random_state, setting to 42")

        # Validate optuna_trials
        if self.config.get("fast_mode", False):
            self.config["optuna_trials"] = 0
            logging.info("⚡ FAST MODE: Disabling Optuna optimization")

    def get_config_summary(self) -> str:
        """Get a summary of the current configuration."""
        summary = []
        summary.append("=== CONFIGURATION SUMMARY ===")
        summary.append(f"Debug Mode: {self.config.get('debug_mode', False)}")
        summary.append(f"Parallel Jobs: {self.config.get('parallel_jobs', 1)}")
        summary.append(f"CV Folds: {self.config.get('cv_folds', 5)}")
        summary.append(f"Random State: {self.config.get('random_state', 42)}")
        summary.append(f"Cache Enabled: {self.config.get('cache_enabled', True)}")
        summary.append(f"Fast Mode: {self.config.get('fast_mode', False)}")
        summary.append(f"Optuna Trials: {self.config.get('optuna_trials', 0)}")
        summary.append("=" * 30)

        return "\n".join(summary)
