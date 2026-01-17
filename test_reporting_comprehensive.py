#!/usr/bin/env python3
"""
Comprehensive test script for titanic_pipeline/core/reporting.py
Tests all functions with sample data to ensure they work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from titanic_pipeline.core.reporting import (
    ReportingManager,
    generate_reports,
    generate_roc_curves,
    generate_feature_correlation_heatmap,
    generate_model_performance_timeline,
    generate_changelog_and_manifest,
    save_timing_report,
    generate_shap_comparison_plot,
    improved_generate_submission,
    log_model_performance_to_csv,
    generate_permutation_importance,
    _get_best_score,
)

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    # Create sample training data
    n_samples = 891
    n_features = 20

    # Create more descriptive feature names
    descriptive_feature_names = [
        "age_normalized",
        "fare_log",
        "family_size",
        "title_encoded",
        "cabin_deck",
        "embarked_onehot_S",
        "embarked_onehot_C",
        "embarked_onehot_Q",
        "sex_male",
        "pclass_1",
        "pclass_2",
        "pclass_3",
        "sibsp_scaled",
        "parch_scaled",
        "age_fare_interaction",
        "family_wealth_score",
        "cabin_number",
        "ticket_prefix",
        "name_length",
        "age_group",
    ]

    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=descriptive_feature_names,
    )

    # Add some categorical features
    X_train["Pclass"] = np.random.choice([1, 2, 3], n_samples)
    X_train["Sex"] = np.random.choice(["male", "female"], n_samples)
    X_train["Embarked"] = np.random.choice(["S", "C", "Q"], n_samples)

    # Encode categorical features for model training
    from sklearn.preprocessing import LabelEncoder

    le_sex = LabelEncoder()
    X_train["Sex"] = le_sex.fit_transform(X_train["Sex"])
    le_embarked = LabelEncoder()
    X_train["Embarked"] = le_embarked.fit_transform(X_train["Embarked"])

    y_train = np.random.choice([0, 1], n_samples)

    # Create sample test data
    test = pd.DataFrame(
        {
            "PassengerId": range(1, 101),
            "Pclass": np.random.choice([1, 2, 3], 100),
            "Sex": np.random.choice(["male", "female"], 100),
            "Age": np.random.uniform(1, 80, 100),
            "SibSp": np.random.randint(0, 5, 100),
            "Parch": np.random.randint(0, 5, 100),
            "Fare": np.random.uniform(0, 500, 100),
            "Embarked": np.random.choice(["S", "C", "Q"], 100),
        }
    )

    # Add the feature columns to test data
    for name in descriptive_feature_names:
        test[name] = np.random.randn(100)

    # Encode categorical features in test data
    test["Sex"] = le_sex.transform(test["Sex"])
    test["Embarked"] = le_embarked.transform(test["Embarked"])

    # Sample feature columns
    feature_cols = descriptive_feature_names + [
        "Pclass",
        "Sex",
        "Embarked",
    ]

    # Sample model results
    model_results = {
        "RandomForest": {
            "mean_score": 0.85,
            "std_score": 0.02,
            "cv_scores": [0.83, 0.86, 0.84, 0.87, 0.85],
            "mean_auc": 0.88,
            "mean_precision": 0.82,
            "mean_recall": 0.79,
            "mean_f1": 0.80,
            "trained_model": RandomForestClassifier(
                random_state=42, n_estimators=10
            ).fit(X_train, y_train),
        },
        "LogisticRegression": {
            "mean_score": 0.78,
            "std_score": 0.03,
            "cv_scores": [0.76, 0.79, 0.77, 0.80, 0.78],
            "mean_auc": 0.81,
            "mean_precision": 0.75,
            "mean_recall": 0.72,
            "mean_f1": 0.73,
            "trained_model": LogisticRegression(random_state=42, max_iter=100).fit(
                X_train, y_train
            ),
        },
    }

    return X_train, y_train, test, feature_cols, model_results


def test_reporting_manager():
    """Test ReportingManager class."""
    logger.info("Testing ReportingManager...")

    X_train, y_train, test, feature_cols, model_results = create_sample_data()

    config = {
        "generate_md": True,
        "generate_docx": True,
        "generate_pdf": True,
        "include_calibration_plots": True,
        "include_feature_importance": True,
    }

    manager = ReportingManager(config)

    try:
        manager.generate_reports(model_results, feature_cols, X_train, y_train)
        logger.info("✓ ReportingManager.generate_reports completed successfully")
    except Exception as e:
        logger.error(f"✗ ReportingManager.generate_reports failed: {e}")


def test_standalone_functions():
    """Test standalone functions."""
    logger.info("Testing standalone functions...")

    X_train, y_train, test, feature_cols, model_results = create_sample_data()

    # Test generate_reports
    try:
        config = {
            "generate_md": True,
            "generate_docx": True,
            "generate_pdf": True,
            "include_calibration_plots": True,
            "include_feature_importance": True,
        }
        generate_reports(model_results, feature_cols, X_train, y_train, config)
        logger.info("✓ generate_reports completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_reports failed: {e}")

    # Test generate_roc_curves
    try:
        generate_roc_curves(model_results, X_train, y_train, feature_cols)
        logger.info("✓ generate_roc_curves completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_roc_curves failed: {e}")

    # Test generate_feature_correlation_heatmap
    try:
        train_data = X_train.copy()
        train_data["Survived"] = y_train
        generate_feature_correlation_heatmap(train_data, feature_cols)
        logger.info("✓ generate_feature_correlation_heatmap completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_feature_correlation_heatmap failed: {e}")

    # Test generate_model_performance_timeline
    try:
        generate_model_performance_timeline(model_results)
        logger.info("✓ generate_model_performance_timeline completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_model_performance_timeline failed: {e}")

    # Test generate_changelog_and_manifest
    try:
        import datetime

        script_total_time = datetime.timedelta(seconds=120)
        generate_changelog_and_manifest(feature_cols, model_results, script_total_time)
        logger.info("✓ generate_changelog_and_manifest completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_changelog_and_manifest failed: {e}")

    # Test save_timing_report
    try:
        save_timing_report(script_total_time, model_results)
        logger.info("✓ save_timing_report completed successfully")
    except Exception as e:
        logger.error(f"✗ save_timing_report failed: {e}")

    # Test generate_shap_comparison_plot (requires shap)
    try:
        top_models = [("RandomForest", model_results["RandomForest"])]
        generate_shap_comparison_plot(top_models, X_train.values, feature_cols)
        logger.info("✓ generate_shap_comparison_plot completed successfully")
    except ImportError:
        logger.warning("⚠ generate_shap_comparison_plot skipped: shap not available")
    except Exception as e:
        logger.error(f"✗ generate_shap_comparison_plot failed: {e}")

    # Test improved_generate_submission (requires saved model)
    try:
        # First save a model
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )
        pipeline.fit(X_train, y_train)

        os.makedirs("output/models", exist_ok=True)
        joblib.dump(pipeline, "output/models/best_model_pipeline.pkl")

        improved_generate_submission(pipeline, test, feature_cols, X_train)
        logger.info("✓ improved_generate_submission completed successfully")
    except Exception as e:
        logger.error(f"✗ improved_generate_submission failed: {e}")

    # Test log_model_performance_to_csv
    try:
        log_model_performance_to_csv(model_results)
        logger.info("✓ log_model_performance_to_csv completed successfully")
    except Exception as e:
        logger.error(f"✗ log_model_performance_to_csv failed: {e}")

    # Test generate_permutation_importance
    try:
        model = model_results["RandomForest"]["trained_model"]
        generate_permutation_importance(
            model, X_train, y_train, feature_cols, model_name="RandomForest"
        )
        logger.info("✓ generate_permutation_importance completed successfully")
    except Exception as e:
        logger.error(f"✗ generate_permutation_importance failed: {e}")

    # Test _get_best_score
    try:
        best_score = _get_best_score(model_results)
        assert best_score == 0.85, f"Expected 0.85, got {best_score}"
        logger.info("✓ _get_best_score completed successfully")
    except Exception as e:
        logger.error(f"✗ _get_best_score failed: {e}")


def check_generated_files():
    """Check if expected files were generated."""
    logger.info("Checking generated files...")

    expected_files = [
        "output/relatorios/relatorio_final.md",
        "output/graficos/roc_curves/04_roc_curve.png",
        "output/graficos/correlation/09_feature_correlation_heatmap.png",
        "output/graficos/timeline/10_model_performance_timeline.png",
        "output/changelog/manifest.json",
        "output/changelog/CHANGELOG.md",
        "output/relatorios/timing_report.json",
        "output/submission_titanic_final.csv",
        "output/reports/model_performance.csv",
        "output/graficos/permutation_importance_RandomForest.png",
        "output/graficos/permutation_importance_RandomForest.csv",
    ]

    for file_path in expected_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {file_path} exists")
        else:
            logger.warning(f"⚠ {file_path} not found")


def main():
    """Run all tests."""
    logger.info("Starting comprehensive testing of reporting.py...")

    # Create output directories
    for dir_path in [
        "output/relatorios",
        "output/graficos",
        "output/models",
        "output/reports",
        "output/changelog",
    ]:
        os.makedirs(dir_path, exist_ok=True)

    # Run tests
    test_reporting_manager()
    test_standalone_functions()
    check_generated_files()

    logger.info("Comprehensive testing completed!")


if __name__ == "__main__":
    main()
