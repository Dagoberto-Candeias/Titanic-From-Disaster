import pandas as pd
from pandas.api import types as pd_types
import numpy as np
import pytest
import sys
import os

# Adjust path to import from titanic_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from titanic_pipeline.utils import optimize_memory_usage

@pytest.fixture
def clean_numeric_dataframe() -> pd.DataFrame:
    """Creates a DataFrame with numeric types without NaNs for precise optimization checks."""
    data = {
        'int_small': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Max 127 -> int8
        'int_medium': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], # Max 32767 -> int16
        'int_large': [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], # Max 2147483647 -> int32
        'float_small': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0], # Precision for float32
        'float_large': [1.123456789, 2.123456789, 3.123456789, 4.123456789, 5.123456789, 6.123456789, 7.123456789, 8.123456789, 9.123456789, 10.123456789], # Precision for float64
        'object_low_card': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'], # Should become category
        'object_high_card': [f'item_{i}' for i in range(10)], # Should remain object (high cardinality)
        'bool_col': [True, False, True, False, True, False, True, False, True, False], # Should become int8
        'datetime_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10']) # Should remain datetime
    }
    return pd.DataFrame(data)

@pytest.fixture
def dataframe_with_nans() -> pd.DataFrame:
    """Creates a DataFrame with NaNs in numeric columns."""
    data = {
        'int_with_nan': [1, 2, np.nan, 4, 5], # Contains NaN, so will be float type
        'float_with_nan': [1.1, np.nan, 3.3, 4.4, 5.5], # Contains NaN
        'object_with_nan': ['A', 'B', np.nan, 'C', 'A'] # Contains NaN, low cardinality
    }
    return pd.DataFrame(data)

def test_optimize_memory_usage_clean_numeric(clean_numeric_dataframe: pd.DataFrame):
    """Tests if optimize_memory_usage correctly converts data types and reduces memory for clean numeric data."""
    df_original = clean_numeric_dataframe.copy()
    initial_memory = df_original.memory_usage(deep=True).sum()

    df_optimized = optimize_memory_usage(df_original)
    optimized_memory = df_optimized.memory_usage(deep=True).sum()

    # Assert memory usage has decreased
    assert optimized_memory < initial_memory

    # Assert specific type conversions for clean numeric data
    assert df_optimized['int_small'].dtype == 'int8'
    assert df_optimized['int_medium'].dtype == 'int16'
    assert df_optimized['int_large'].dtype == 'int32'
    assert df_optimized['float_small'].dtype == 'float32'
    assert df_optimized['float_large'].dtype == 'float64' # Should remain float64 due to precision
    assert isinstance(df_optimized['object_low_card'].dtype, pd.CategoricalDtype)
    assert pd_types.is_object_dtype(df_optimized['object_high_card']) or pd_types.is_string_dtype(df_optimized['object_high_card'])
    assert df_optimized['bool_col'].dtype == 'int8' # Booleans are often converted to int8
    assert df_optimized['datetime_col'].dtype == 'datetime64[ns]' # Datetime should remain unchanged

def test_optimize_memory_usage_with_nans(dataframe_with_nans: pd.DataFrame):
    """Tests if optimize_memory_usage correctly handles data with NaNs."""
    df_original = dataframe_with_nans.copy()
    initial_memory = df_original.memory_usage(deep=True).sum()

    df_optimized = optimize_memory_usage(df_original)
    optimized_memory = df_optimized.memory_usage(deep=True).sum()

    # Assert memory usage has decreased
    assert optimized_memory < initial_memory

    # Columns with NaNs will be float types, and should be downcasted to float32 if range allows
    assert df_optimized['int_with_nan'].dtype == 'float32'
    assert df_optimized['float_with_nan'].dtype == 'float32'
    assert isinstance(df_optimized['object_with_nan'].dtype, pd.CategoricalDtype)

def test_optimize_memory_usage_no_change_if_not_optimizable():
    """Tests that memory usage doesn't change if no optimization is possible."""
    df_original = pd.DataFrame({'id': range(1000), 'text': [f'long_string_{i}' for i in range(1000)]})
    initial_memory = df_original.memory_usage(deep=True).sum()
    df_optimized = optimize_memory_usage(df_original)
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    # Expect memory to be very close, or slightly less if 'id' was int64 and became int16/int32
    assert optimized_memory <= initial_memory
    assert pd_types.is_object_dtype(df_optimized['text']) or pd_types.is_string_dtype(df_optimized['text'])
    assert df_optimized['id'].dtype == 'int16' # Should be optimized from int64 to int16