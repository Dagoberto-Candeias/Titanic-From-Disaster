#!/usr/bin/env python3
"""
Test script for reporting functions to verify directory creation and functionality.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to path to import titanic_pipeline
sys.path.insert(0, '.')

from titanic_pipeline.core.reporting import (
    generate_roc_curves,
    generate_feature_correlation_heatmap,
    generate_model_performance_timeline,
    generate_changelog_and_manifest,
    save_timing_report,
    generate_shap_comparison_plot,
    generate_model_calibration_plots,
    generate_permutation_importance
)

def create_mock_data():
    """Create mock data for testing."""
    # Mock model results
    model_results = {
        'RandomForest': {
            'mean_score': 0.85,
            'std_score': 0.02,
            'cv_scores': [0.83, 0.86, 0.84, 0.87, 0.85],
            'trained_model': None  # Will be set if needed
        },
        'XGBoost': {
            'mean_score': 0.82,
            'std_score': 0.03,
            'cv_scores': [0.80, 0.83, 0.81, 0.84, 0.82],
            'trained_model': None
        }
    }

    # Mock training data
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))

    # Mock train DataFrame
    train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': y_train
    })

    feature_cols = ['feature1', 'feature2', 'feature3']

    return model_results, X_train, y_train, train, feature_cols

def test_directory_creation():
    """Test that directories are created correctly."""
    print("🧪 Testing directory creation...")

    # Clean up any existing output
    if os.path.exists('output'):
        shutil.rmtree('output')

    # Create mock data
    model_results, X_train, y_train, train, feature_cols = create_mock_data()

    # Test generate_roc_curves
    try:
        generate_roc_curves(model_results, X_train, y_train, feature_cols)
        assert os.path.exists('output/graficos/roc_curves'), "ROC curves directory not created"
        assert os.path.exists('output/graficos/roc_curves/04_roc_curve.png'), "ROC curve file not created"
        print("   ✅ ROC curves directory and file created")
    except Exception as e:
        print(f"   ❌ ROC curves test failed: {e}")

    # Test generate_feature_correlation_heatmap
    try:
        generate_feature_correlation_heatmap(train, feature_cols)
        assert os.path.exists('output/graficos/correlation'), "Correlation directory not created"
        assert os.path.exists('output/graficos/correlation/09_feature_correlation_heatmap.png'), "Correlation heatmap file not created"
        print("   ✅ Correlation heatmap directory and file created")
    except Exception as e:
        print(f"   ❌ Correlation heatmap test failed: {e}")

    # Test generate_model_performance_timeline
    try:
        generate_model_performance_timeline(model_results)
        assert os.path.exists('output/graficos/timeline'), "Timeline directory not created"
        assert os.path.exists('output/graficos/timeline/10_model_performance_timeline.png'), "Timeline file not created"
        print("   ✅ Timeline directory and file created")
    except Exception as e:
        print(f"   ❌ Timeline test failed: {e}")

    # Test generate_changelog_and_manifest
    try:
        generate_changelog_and_manifest(feature_cols, model_results, 120.5)
        assert os.path.exists('output/changelog'), "Changelog directory not created"
        assert os.path.exists('output/changelog/manifest.json'), "Manifest file not created"
        assert os.path.exists('output/changelog/CHANGELOG.md'), "Changelog file not created"
        print("   ✅ Changelog and manifest directories and files created")
    except Exception as e:
        print(f"   ❌ Changelog and manifest test failed: {e}")

    # Test save_timing_report
    try:
        save_timing_report(120.5, model_results)
        assert os.path.exists('output/relatorios'), "Reports directory not created"
        assert os.path.exists('output/relatorios/timing_report.json'), "Timing report file not created"
        print("   ✅ Timing report directory and file created")
    except Exception as e:
        print(f"   ❌ Timing report test failed: {e}")

    print("🧪 Directory creation tests completed.\n")

def test_error_handling():
    """Test error handling in functions."""
    print("🧪 Testing error handling...")

    # Test with empty model results
    try:
        generate_roc_curves({}, pd.DataFrame(), pd.Series(), [])
        print("   ✅ Empty model results handled gracefully")
    except Exception as e:
        print(f"   ❌ Empty model results test failed: {e}")

    # Test with missing dependencies (SHAP)
    try:
        # This should fail gracefully if shap is not installed
        generate_shap_comparison_plot([], pd.DataFrame(), [])
        print("   ✅ SHAP missing dependency handled gracefully")
    except Exception as e:
        print(f"   ⚠️  SHAP test (expected to fail if shap not installed): {e}")

    print("🧪 Error handling tests completed.\n")

def main():
    """Run all tests."""
    print("🚀 Starting comprehensive reporting tests...\n")

    test_directory_creation()
    test_error_handling()

    print("✅ All tests completed!")

    # List created directories
    print("\n📁 Created directory structure:")
    for root, dirs, files in os.walk('output'):
        level = root.replace('output', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    main()
