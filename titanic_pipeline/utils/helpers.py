"""Helper utilities re-exported by the `titanic_pipeline.utils` package.

This module provides a small subset of functions that used to live in
`titanic_pipeline/utils.py` (module). When the utils package was added the
module got shadowed; we keep copies here for compatibility and easy imports.
"""
import logging
import os
import pickle
import multiprocessing
from typing import Any, Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


def robust_pickle_dump(obj: Any, path: str) -> None:
    """Safely dump an object to a pickle file, creating dirs if needed."""
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
        logger.warning(f"⚠️ Could not load pickle {path}: {e}")
    return None


def safe_parallel_map(func: Callable, iterable: Iterable, max_workers: Optional[int] = None, use_joblib: bool = True):
    """Naive parallel mapping with graceful fallback to sequential execution."""
    try:
        from joblib import Parallel, delayed  # optional
        JOBLIB_AVAILABLE = True
    except Exception:
        JOBLIB_AVAILABLE = False

    iterable = list(iterable)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    if use_joblib and JOBLIB_AVAILABLE:
        try:
            results = Parallel(n_jobs=max_workers)(delayed(func)(item) for item in iterable)
            return results
        except Exception as e:
            logger.warning(f"⚠️ joblib parallel failed, falling back to sequential. Error: {e}")

    results = []
    for item in iterable:
        try:
            results.append(func(item))
        except Exception as e:
            logger.exception(f"❌ Error in safe_parallel_map for item {item}: {e}")
            results.append(None)
    return results


def is_tree_model(estimator: Any) -> bool:
    """Heuristic: return True if estimator class name hints at tree models."""
    try:
        cls_name = estimator.__class__.__name__.lower()
        return any(k in cls_name for k in ("xgb", "lgbm", "randomforest", "extratrees", "lightgbm", "catboost", "decisiontree"))
    except Exception:
        return False
