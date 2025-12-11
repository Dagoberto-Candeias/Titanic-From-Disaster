"""
Memory optimization utilities for Titanic ML Pipeline.
Enhanced with monitoring, garbage collection, and chunked processing.
"""

import gc
import logging
import psutil
import os
from typing import Dict, Iterator
import numpy as np
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
    logger.info(
        f"   💾 Memory usage: {final_memory:.1f} MB"
        f" (Δ{memory_delta:+.1f} MB)"
    )


def get_max_decimal_places(series: pd.Series) -> int:
    """Get the maximum number of decimal places in the series."""
    max_dec = 0
    for val in series:
        if pd.isna(val):
            continue
        str_val = str(val)
        if '.' in str_val:
            dec = len(str_val.split('.')[-1])
            max_dec = max(max_dec, dec)
    return max_dec


def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame by downcasting numeric types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized memory usage
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Handle object columns - convert to category if low cardinality
        if col_type == object:
            # Check cardinality
            unique_values = df[col].nunique()
            # Convert to category if less than 10 unique values (matching test expectations)
            if unique_values < 10:
                df[col] = df[col].astype('category')
            continue

        # Skip datetime columns
        if 'datetime' in str(col_type):
            continue

        # Skip if column has NaN values for float optimization
        has_nan = df[col].isna().any()

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type)[:3] == 'int':
            if (c_min > np.iinfo(np.int8).min and
                    c_max < np.iinfo(np.int8).max):
                df[col] = df[col].astype(np.int8)
            elif (c_min > np.iinfo(np.int16).min and
                  c_max < np.iinfo(np.int16).max):
                df[col] = df[col].astype(np.int16)
            elif (c_min > np.iinfo(np.int32).min and
                  c_max < np.iinfo(np.int32).max):
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif str(col_type) == 'bool':
            # Convert boolean columns to int8 as expected by tests
            df[col] = df[col].astype(np.int8)
        else:
            # For float columns, be more conservative
            range_ok = (c_min > np.finfo(np.float32).min and
                        c_max < np.finfo(np.float32).max)
            if has_nan:
                # With NaN, downcast to float32 if possible
                if range_ok:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            else:
                # No NaN, check decimal places for precision
                decimal_places = get_max_decimal_places(df[col])
                if range_ok and decimal_places <= 5:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.info(
        f"   💾 Memory usage decreased from {start_mem:.2f} MB to "
        f"{end_mem:.2f} MB ("
        f"{reduction:.1f}% reduction)"
    )

    return df


def force_garbage_collection() -> None:
    """Force garbage collection and log memory cleanup."""
    process = psutil.Process(os.getpid())
    before_gc = process.memory_info().rss / 1024 / 1024  # MB

    gc.collect()

    after_gc = process.memory_info().rss / 1024 / 1024  # MB
    freed_memory = before_gc - after_gc

    logger.info(f"   🗑️  Garbage collection: {freed_memory:+.1f} MB freed")


def chunked_dataframe_processing(df: pd.DataFrame,
                                 chunk_size: int = 1000,
                                 processor_func=None) -> Iterator[pd.DataFrame]:
    """
    Process DataFrame in chunks to handle large datasets efficiently.

    Args:
        df: Input DataFrame
        chunk_size: Size of each chunk
        processor_func: Optional function to apply to each chunk

    Yields:
        Processed DataFrame chunks
    """
    total_rows = len(df)

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx].copy()

        if processor_func:
            chunk = processor_func(chunk)

        yield chunk

        # Force garbage collection between chunks
        if start_idx % (chunk_size * 10) == 0:
            force_garbage_collection()


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }


def set_global_seeds(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    # Set Python random seed
    random.seed(seed)

    # Set numpy random seed
    np.random.seed(seed)

    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"   🎲 Global random seeds set to {seed}")
