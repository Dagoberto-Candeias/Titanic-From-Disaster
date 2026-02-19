"""Compatibility wrappers for feature preprocessing utilities.

These wrappers provide the older, convenience APIs used in tests and
notebooks while delegating heavy work to `titanic_pipeline.preprocessing`.
"""
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from titanic_pipeline import preprocessing as pp


def create_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    imputer_strategy: str = 'median',
    scaler: str = 'standard'
) -> object:
    """Create a simple sklearn ColumnTransformer compatible pipeline.

    Note: This wrapper keeps a lightweight, test-friendly signature used
    in older notebooks. For full configuration use
    `titanic_pipeline.preprocessing.create_feature_pipeline`.
    """
    # Build a minimal DataFrame to let the underlying implementation infer
    # dtypes when needed. The wrapper will not fit on data here; callers
    # should call `fit`/`transform` on returned pipeline with actual data.
    dummy_cols = numeric_features + categorical_features
    dummy_df = pd.DataFrame(columns=dummy_cols)
    # Delegate to main implementation with sane defaults
    return pp.create_feature_pipeline(
        df=dummy_df,
        feature_cols=dummy_cols,
        numerical_imputer='simple' if imputer_strategy == 'median' else imputer_strategy,
        categorical_encoder='onehot',
        scaler=(scaler in ['standard', 'minmax'])
    )


def advanced_missing_imputation(df: pd.DataFrame, strategy: str = 'auto', report: bool = True):
    """Delegate to central advanced_missing_imputation implementation."""
    return pp.advanced_missing_imputation(df, strategy=strategy, report=report)


def select_features_via_model(X, y, model_type: str = 'rf', k: int = 10):
    """Select top-k features using model importances.

    Accepts either a numpy array or a pandas DataFrame for X.
    Returns (X_selected, selected_feature_names).
    """
    from sklearn.ensemble import RandomForestClassifier

    # Normalize X to DataFrame for column handling
    if isinstance(X, np.ndarray):
        n_cols = X.shape[1]
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_cols)])
    else:
        X_df = pd.DataFrame(X)

    # Fit a simple RF (or XGB if requested and available)
    if model_type.lower() in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        except Exception:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_df, y)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not expose feature_importances_.")

    ranked_idx = importances.argsort()[::-1]
    top_idx = ranked_idx[:min(k, len(ranked_idx))]
    selected_features = [X_df.columns[i] for i in top_idx]
    X_selected = X_df.iloc[:, top_idx].to_numpy(dtype=np.float64, na_value=np.nan)

    return X_selected, selected_features
