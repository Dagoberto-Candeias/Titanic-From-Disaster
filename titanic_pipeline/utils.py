
"""
titanic_pipeline.utils
Utilities for Titanic pipeline - Part 1 (safety & caching helpers)

Functions:
- get_cache_key(data_hash, operation)
- robust_pickle_dump(obj, path)
- robust_pickle_load(path)
- safe_check_is_fitted(estimator, X_sample=None)
- safe_parallel_map(func, iterable, max_workers=None)
- ensure_feature_cols_intersection(train_cols, test_cols, feature_cols)
- is_tree_model(estimator)
- set_global_seeds(seed)
"""

import os
import pickle
import hashlib
import logging
import random
import json
from typing import Any, Callable, Iterable, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.utils.validation import check_is_fitted
except Exception:
    check_is_fitted = None

# joblib optional
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# multiprocessing fallback
import multiprocessing

def get_cache_key(data_hash: str, operation: str) -> str:
    """
    Generate a stable cache key based on data hash and operation.
    Note: intentionally does NOT include timestamp to allow cache reuse across runs
    for the same data.
    """
    key = hashlib.md5(f"{data_hash}_{operation}".encode()).hexdigest()
    return key

def robust_pickle_dump(obj: Any, path: str) -> None:
    """Safely dump an object to a pickle file, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✅ Saved pickle: {path}")
    except Exception as e:
        logger.exception(f"❌ Failed to save pickle {path}: {e}")

def robust_pickle_load(path: str) -> Optional[Any]:
    """Safely load a pickle file. Returns None on failure."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"✅ Loaded pickle: {path}")
        return obj
    except Exception as e:
        logger.warning(f"⚠️  Could not load pickle {path}: {e}")
        return None

def safe_check_is_fitted(estimator: Any, X_sample=None) -> bool:
    """
    Try to check if an estimator/preprocessor is fitted.
    If sklearn.check_is_fitted is unavailable or fails, try to call transform on a sample.
    Returns True if seems fitted, False otherwise.
    """
    try:
        if check_is_fitted is not None:
            check_is_fitted(estimator)
            return True
    except Exception:
        pass

    # Fallback: try transform on a small sample if possible
    if X_sample is not None:
        try:
            _ = estimator.transform(X_sample[:1])
            return True
        except Exception:
            return False

    # Last resort: heuristic - presence of attributes like 'n_features_in_' or 'classes_'
    heuristics = ["n_features_in_", "classes_", "feature_names_in_"]
    for attr in heuristics:
        if hasattr(estimator, attr):
            return True
    return False

def safe_parallel_map(func: Callable, iterable: Iterable, max_workers: Optional[int] = None, use_joblib: bool = True):
    """
    Execute func over iterable in parallel with joblib or ProcessPoolExecutor fallback.
    Returns list of results in order.
    """
    iterable = list(iterable)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    if use_joblib and JOBLIB_AVAILABLE:
        try:
            results = Parallel(n_jobs=max_workers)(delayed(func)(item) for item in iterable)
            return results
        except Exception as e:
            logger.warning(f"⚠️ joblib parallel failed, falling back to sequential. Error: {e}")

    # Fallback to sequential map
    results = []
    for item in iterable:
        try:
            results.append(func(item))
        except Exception as e:
            logger.exception(f"❌ Error in safe_parallel_map for item {item}: {e}")
            results.append(None)
    return results

def ensure_feature_cols_intersection(train_cols: Iterable[str], test_cols: Iterable[str], feature_cols: Iterable[str]) -> List[str]:
    """
    Ensure feature_cols are present in both train and test.
    Returns a list with the intersection preserving order from feature_cols.
    """
    train_set = set(train_cols)
    test_set = set(test_cols)
    final = [c for c in feature_cols if c in train_set and c in test_set]
    missing = [c for c in feature_cols if c not in final]
    if missing:
        logger.warning(f"⚠️ The following feature_cols were removed because missing in train/test: {missing}")
    return final

def is_tree_model(estimator: Any) -> bool:
    """Simple heuristic to check if estimator is tree-based (so TreeExplainer can be used)."""
    try:
        cls_name = estimator.__class__.__name__.lower()
        return any(k in cls_name for k in ("xgb", "lgbm", "randomforest", "extratrees", "lightgbm", "catboost", "decisiontree"))
    except Exception:
        return False

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
