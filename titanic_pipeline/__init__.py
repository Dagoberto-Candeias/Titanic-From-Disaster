"""
Titanic ML Pipeline Package.

A comprehensive machine learning pipeline for the Titanic Kaggle competition,
featuring automated feature engineering, model training, hyperparameter optimization,
ensemble methods, and report generation.
"""

# Lazy import para evitar circular imports
def __getattr__(name):
    if name == "TitanicPipeline":
        from .core import TitanicPipeline
        return TitanicPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = "Dagoberto Candeias de Moraes"
__email__ = "118550@ufv.br"

__all__ = ["TitanicPipeline"]
