import pandas as pd
import numpy as np
from titanic_pipeline.utils import optimize_memory_usage


def test_datetime_columns_unchanged():
    df = pd.DataFrame({
        'date_col': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
        'num_col': [1, 2, 3]
    })
    out = optimize_memory_usage(df.copy())
    assert out['date_col'].dtype == 'datetime64[ns]'


def test_no_float16_produced_for_safe_values():
    # Values that could fit in float16 but we prefer float32/float64 to preserve precision
    df = pd.DataFrame({
        'f1': [1.123456, 2.123456, 3.123456],
        'f2': [10000.0, 20000.0, 30000.0]
    })
    out = optimize_memory_usage(df.copy())
    # Ensure no float16 dtypes are present
    assert not any(str(dtype).startswith('float16') for dtype in out.dtypes), (
        f"Found float16 dtype, got: {out.dtypes.to_dict()}"
    )


def test_columns_with_nans_become_float32():
    df = pd.DataFrame({
        'int_with_nan': [1, 2, np.nan, 4, 5],
        'float_with_nan': [1.1, np.nan, 3.3, 4.4, 5.5],
    })
    out = optimize_memory_usage(df.copy())
    assert str(out['int_with_nan'].dtype) == 'float32'
    assert str(out['float_with_nan'].dtype) == 'float32'


def test_object_with_nan_becomes_category():
    df = pd.DataFrame({
        'obj': ['A', 'B', np.nan, 'A', 'B']
    })
    out = optimize_memory_usage(df.copy())
    from pandas.api import types as pd_types
    assert pd_types.is_categorical_dtype(out['obj']) or pd_types.is_string_dtype(out['obj'])


def test_bool_converted_to_int8():
    df = pd.DataFrame({'b': [True, False, True, False]})
    out = optimize_memory_usage(df.copy())
    assert str(out['b'].dtype) == 'int8'
