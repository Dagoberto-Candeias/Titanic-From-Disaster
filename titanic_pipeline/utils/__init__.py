"""
Titanic Pipeline Utilities
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Import from memory module
from .memory import (
    optimize_memory_usage,
    force_garbage_collection,
    chunked_dataframe_processing,
    get_memory_stats,
    set_global_seeds,
)

# Import from validation module
from .validation import (
    DataValidator,
    ensure_feature_cols_intersection,
)

# Import from cache module
from .cache import (
    get_cache_key,
    cache_result,
    load_cached_result,
)

# Import from parallel module
from .parallel import (
    safe_parallel_map,
    is_tree_model,
    robust_pickle_dump,
    robust_pickle_load,
)

# Import from utils module
from .utils import (
    validate_data_schema,
)

# Re-export commonly used functions for convenience
__all__ = [
    # Memory utilities
    "optimize_memory_usage",
    "force_garbage_collection",
    "chunked_dataframe_processing",
    "get_memory_stats",
    "set_global_seeds",

    # Validation utilities
    "DataValidator",
    "ensure_feature_cols_intersection",

    # Cache utilities
    "get_cache_key",
    "cache_result",
    "load_cached_result",

    # Parallel utilities
    "safe_parallel_map",
    "is_tree_model",
    "robust_pickle_dump",
    "robust_pickle_load",

    # General utilities
    "validate_data_schema",
]
