#!/usr/bin/env python3
"""
Manual reporting tests (moved out of top-level to avoid pytest collection conflicts).
Run this script directly for ad-hoc checks:
    python scripts/manual_reporting_test.py
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to path if running from project subdir
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from titanic_pipeline.core.reporting import (
    generate_roc_curves,
    generate_feature_correlation_heatmap,
    generate_model_performance_timeline,
    generate_changelog_and_manifest,
    save_timing_report,
    generate_shap_comparison_plot,
    generate_model_calibration_plots,
    generate_permutation_importance,
)


def create_mock_data():
    """Create mock data for testing."""
    model_results = {
        'RandomForest': {
            'mean_score': 0.85,
            'std_score': 0.02,
            'cv_scores': [0.83, 0.86, 0.84, 0.87, 0.85],
            'trained_model': None
        },
        'XGBoost': {
            'mean_score': 0.82,
            'std_score': 0.03,
            'cv_scores': [0.80, 0.83, 0.81, 0.84, 0.82],
            'trained_model': None
        }
    }

    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))

    train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': y_train
    })

    feature_cols = ['feature1', 'feature2', 'feature3']

    return model_results, X_train, y_train, train, feature_cols


def main():
    # Clean up any existing output
    if os.path.exists('output'):
        shutil.rmtree('output')

    model_results, X_train, y_train, train, feature_cols = create_mock_data()

    generate_roc_curves(model_results, X_train, y_train, feature_cols)
    generate_feature_correlation_heatmap(train, feature_cols)
    generate_model_performance_timeline(model_results)
    generate_changelog_and_manifest(feature_cols, model_results, 120.5)
    save_timing_report(120.5, model_results)

    print('Manual reporting test run complete. Check output/graficos and output/relatorios')


if __name__ == '__main__':
    main()
