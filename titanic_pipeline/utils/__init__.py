"""
Utilities for Titanic ML Pipeline.
"""

import logging
import random
import numpy as np

from .cache import CacheManager
from .validation import DataValidator
from .parallel import ParallelProcessor

logger = logging.getLogger(__name__)

def set_global_seeds(seed: int):
    """Set seeds for reproducibility across random, numpy and os where applicable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    logger.info(f"🔒 Global seeds set to {seed}")

def ensure_feature_cols_intersection(train_cols, test_cols, feature_cols):
    """Ensure feature columns are present in both train and test datasets."""
    common_cols = set(train_cols) & set(test_cols)
    feature_cols = [col for col in feature_cols if col in common_cols]
    return feature_cols

__all__ = ["CacheManager", "DataValidator", "ParallelProcessor", "set_global_seeds", "ensure_feature_cols_intersection"]
