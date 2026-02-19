
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
from typing import Any, Callable, Iterable, List, Optional
import pandas as pd
import numpy as np
import multiprocessing


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


def get_cache_key(data_hash: str, operation: str) -> str:
    """
    Generate a stable cache key based on data hash and operation.

    Note: intentionally does NOT include timestamp to allow cache reuse
    across runs for the same data.
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
        logger.warning(f"⚠️ Could not load pickle {path}: {e}")
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
            logger.warning(
                f"⚠️ joblib parallel failed, falling back to sequential. Error: {e}"
            )

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
        logger.warning(
            f"⚠️ The following feature_cols were removed because missing "
            f"in train/test: {missing}"
        )
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

def optimize_memory_usage(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """
    Itera sobre todas as colunas de um dataframe e modifica o tipo de dado
    para reduzir o uso de memória.
    """
    start_mem = df.memory_usage(deep=deep).sum() / 1024**2
    logger.info(f"Uso de memória do dataframe é {start_mem:.2f} MB")

    # Normalize pandas 'string' dtype to plain object strings for
    # consistent behavior across pandas versions (so low-cardinality
    # detection converts to 'category').
    try:
        str_cols = df.select_dtypes(include=["string"]).columns.tolist()
        for c in str_cols:
            df[c] = df[c].astype(object)
    except Exception:
        pass

    for col in df.columns:
        col_series = df[col]
        try:
            from pandas.api import types as pd_types
        except Exception:
            pd_types = None

        # Datetime should remain unchanged
        if (pd_types and pd_types.is_datetime64_any_dtype(col_series)) or 'datetime' in str(col_series.dtype):
            continue

        # Integer types
        if pd_types and pd_types.is_integer_dtype(col_series):
            c_min = col_series.min()
            c_max = col_series.max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = col_series.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = col_series.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = col_series.astype(np.int32)
            else:
                df[col] = col_series.astype(np.int64)

        # Float types (including ints converted to float due to NaNs)
        elif pd_types and pd_types.is_float_dtype(col_series):
            c_min = col_series.min()
            c_max = col_series.max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = col_series.astype(np.float32)
            else:
                df[col] = col_series.astype(np.float64)

        # Boolean types -> convert to int8 for compactness
        elif pd_types and pd_types.is_bool_dtype(col_series):
            df[col] = col_series.astype(np.int8)

        # Object / string-like columns
        else:
            is_str_like = False
            if pd_types:
                is_str_like = pd_types.is_object_dtype(col_series) or pd_types.is_string_dtype(col_series)
            # Fallback for pandas versions where StringDtype detection is different
            if not is_str_like:
                dt_name = str(col_series.dtype).lower()
                if 'string' in dt_name or 'stringdtype' in dt_name:
                    is_str_like = True
            else:
                is_str_like = (col_series.dtype == object)

            # For any non-numeric/non-datetime column, check cardinality and
            # convert to 'category' when it is low (saves memory).
            try:
                unique_ratio = len(col_series.dropna().unique()) / len(col_series)
            except Exception:
                unique_ratio = len(col_series.unique()) / len(col_series)

            if unique_ratio < 0.5:
                df[col] = col_series.astype('category')

    end_mem = df.memory_usage(deep=deep).sum() / 1024**2
    logger.info(f"Uso de memória após otimização: {end_mem:.2f} MB")
    logger.info(
        f"Redução de {100 * (start_mem - end_mem) / start_mem:.1f}%"
    )
    # Ensure pandas 'string' dtype is converted to 'category' when low-cardinality
    try:
        from pandas.api import types as pd_types
        for col in df.columns:
            dt_name = str(df[col].dtype).lower()
            if 'string' in dt_name and not pd_types.is_categorical_dtype(df[col]):
                try:
                    unique_ratio = len(df[col].dropna().unique()) / len(df[col])
                except Exception:
                    unique_ratio = len(df[col].unique()) / len(df[col])
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
    except Exception:
        pass
    return df
