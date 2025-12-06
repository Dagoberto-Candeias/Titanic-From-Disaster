"""
Configuration management for Titanic ML Pipeline.
"""

from .default import DEFAULT_CONFIG, DEFAULT_LOGGING_CONFIG
from .validation import ConfigManager

__all__ = ["DEFAULT_CONFIG", "DEFAULT_LOGGING_CONFIG", "ConfigManager"]
