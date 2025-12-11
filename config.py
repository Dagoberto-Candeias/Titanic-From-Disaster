"""
External configuration file for Titanic ML Pipeline
ELT579 - UFV
Author: Dagoberto Candeias de Moraes (118550)
"""

import multiprocessing
import logging

# System Configuration
CONFIG = {
    "debug_mode": False,
    "random_state": 42,
    "parallel_jobs": 1,
    "cv_folds": 20,
    "cache_enabled": True,  # Enabled for better performance
    "fast_mode": False,
    "feature_selection": True,  # Enable feature selection
    "run_smoke_tests": True,  # Run basic tests
    "optuna_trials": 150,  # Increased trials for better optimization
    "use_knn_imputation": True,  # Enable KNN imputation option
    "enhanced_balance": True,  # Use SMOTE + undersampling
    "generate_all_plots": True,  # Generate all possible plots
    "log_level": logging.INFO,  # Configurable log level
    "max_features_for_shap": 100,  # Limit for SHAP analysis
    # New keys for improvements
    "kfold_te_splits": 5,  # Splits for K-Fold Target Encoding
    "smote_k": 5,  # k_neighbors for SMOTE
    "smote_strategy": "auto",  # Sampling strategy for SMOTE
    "calibration_method": "isotonic",  # Method for calibration
    "calibration_cv": 3,  # CV folds for calibration
    "permutation_repeats": 5,  # Repeats for permutation importance
    "randomized_n_iter": 100,  # Iterations for RandomizedSearchCV fallback
    "report_include_images": True,  # Include images in reports
    "selection_threshold": 0.01,  # Threshold for feature selection
    "selection_model": "RandomForest",  # Model for feature selection
    "te_prior": 10,  # Prior for target encoding smoothing
}

# Model configurations with optimized defaults
MODEL_CONFIGS = {
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": CONFIG["random_state"],
        "n_jobs": 1,
    },
    "XGBoost": {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 0.2,
        "random_state": CONFIG["random_state"],
        "eval_metric": "logloss",
        "early_stopping_rounds": 10,
        "verbosity": 0,
        "n_jobs": -1,
    },
    "LightGBM": {
        "n_estimators": 1500,
        "learning_rate": 0.015,
        "num_leaves": 40,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": CONFIG["random_state"],
        "early_stopping_round": 10,
        "n_jobs": -1,
        "verbosity": -1,
    },
    "CatBoost": {
        "iterations": 800,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3,
        "random_strength": 0.8,
        "border_count": 128,
        "verbose": 0,
        "random_state": CONFIG["random_state"],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": (100, 75, 50, 25, 5),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": CONFIG["random_state"],
        "early_stopping": True,
    },
    "GaussianProcessClassifier": {
        "random_state": CONFIG["random_state"],
        "max_iter_predict": 100,
    },
    "BaggingClassifier": {
        "n_estimators": 50,
        "max_samples": 0.8,
        "max_features": 0.8,
        "bootstrap": True,
        "bootstrap_features": False,
        "random_state": CONFIG["random_state"],
        "n_jobs": -1,
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "create_feat_prefix": True,  # Add 'feat_' prefix to new features
    "advanced_interactions": True,
    "calibration_enabled": True,  # Enable model calibration
    "polynomial_features": True,
    "target_encoding": True,
    "demographic_features": True,
    # New keys for improvements
    "kfold_te": True,  # Enable K-Fold Target Encoding
    "missing_indicators": True,  # Add missing indicators
    "bins_categories": True,  # Create bins and categories
}

# Report configuration
REPORT_CONFIG = {
    "generate_md": True,
    "generate_docx": True,
    "generate_pdf": True,
    "include_table_images": True,  # Generate table images
    "include_calibration_plots": True,
    "include_feature_importance": True,
    # New keys for improvements
    "include_shap_comparison": True,  # Include SHAP comparison plots
}

# Schema validation with data types
EXPECTED_TRAIN_SCHEMA = {
    "PassengerId": "int64",
    "Survived": "int64",
    "Pclass": "int64",
    "Name": "object",
    "Sex": "object",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "object",
    "Fare": "float64",
    "Cabin": "object",
    "Embarked": "object",
}

EXPECTED_TEST_SCHEMA = {
    "PassengerId": "int64",
    "Pclass": "int64",
    "Name": "object",
    "Sex": "object",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "object",
    "Fare": "float64",
    "Cabin": "object",
    "Embarked": "object",
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "titanic_ml.log",
            "formatter": "detailed",
            "encoding": "utf-8",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "root": {
        "level": CONFIG["log_level"],
        "handlers": ["file", "console"],
    },
}

# Cache configuration
CACHE_CONFIG = {
    "cache_dir": "output/cache",
    "cache_hyperparams": True,  # Cache optimized hyperparameters
    "cache_models": True,
    "cache_features": True,
    "cache_plots": True,
}

# Test configuration
TEST_CONFIG = {
    "run_unit_tests": True,
    "run_integration_tests": True,
    "test_accuracy_threshold": 0.78,
    "test_feature_count": 25,
    # New keys for improvements
    "smoke_coverage": ["smote", "calibration", "feature_select"],
}
